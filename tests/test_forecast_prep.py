"""prep turns a spec + source file into cached parquet tables; these tests check the tables'
shape against the fixtures' known answers, not against a second implementation of the sql"""

from __future__ import annotations

import csv
import json
import math

import pytest
from forecast_fixtures import BRANCHES, write_panel, write_rows

duckdb = pytest.importorskip("duckdb")

from smolsmort.forecast.prep import prepare  # noqa: E402
from smolsmort.forecast.spec import (  # noqa: E402
    ROW,
    Censor,
    Column,
    PrepSpec,
)


def _row_spec(source, **changes):
    base = {
        "source": str(source),
        "mode": "row",
        "task": "regression",
        "columns": (
            Column("created", "anchor"),
            Column("branch", "dimension"),
            Column("size", "measure"),
            Column("planned_offset", "measure", known=False),
            Column("leak_plus1", "measure", known=False),
            Column("lead_weeks", "target"),
        ),
        "censor": Censor("created"),
    }
    return PrepSpec(**{**base, **changes})


def _panel_spec(source, **changes):
    base = {
        "source": str(source),
        "mode": "series",
        "task": "regression",
        "columns": (
            Column("day", "time"),
            Column("project", "dimension"),
            Column("product", "dimension"),
            Column("orders", "measure", aggregation="last"),
            Column("units", "target", aggregation="sum"),
        ),
        "horizon": 8,
    }
    return PrepSpec(**{**base, **changes})


def _read(con, path, sql):
    return con.execute(sql.format(p=str(path))).fetchall()


# ---------------------------------------------------------------------------
# series mode
# ---------------------------------------------------------------------------


def test_series_prep_has_twelve_series_on_monday_steps(tmp_path):
    truth = write_panel(tmp_path)
    prepared = prepare(_panel_spec(truth.path), tmp_path / "cache")
    con = duckdb.connect()

    series_count = _read(
        con,
        prepared.folder / "panel.parquet",
        "SELECT count(DISTINCT __series) FROM read_parquet('{p}')",
    )[0][0]
    assert series_count == truth.series

    weekdays = _read(
        con,
        prepared.folder / "panel.parquet",
        "SELECT DISTINCT dayofweek(__step) FROM read_parquet('{p}')",
    )
    assert {row[0] for row in weekdays} == {1}  # duckdb: 1 == monday


def test_series_prep_fills_the_hole_with_zero_units(tmp_path):
    truth = write_panel(tmp_path)
    prepared = prepare(_panel_spec(truth.path), tmp_path / "cache")
    con = duckdb.connect()

    series, start, stop = truth.gap
    rows = con.execute(
        f"SELECT __step, units FROM read_parquet('{prepared.folder / 'panel.parquet'}') "
        f"WHERE __series = '{series}' ORDER BY __step"
    ).fetchall()
    hole_weeks = rows[start:stop]
    assert len(hole_weeks) == stop - start
    assert all(units == 0.0 for _step, units in hole_weeks)


def test_series_prep_ends_project_c_at_its_own_last_week(tmp_path):
    truth = write_panel(tmp_path)
    prepared = prepare(_panel_spec(truth.path), tmp_path / "cache")
    con = duckdb.connect()

    c_max = con.execute(
        f"SELECT max(__step) FROM read_parquet('{prepared.folder / 'panel.parquet'}') "
        "WHERE __series LIKE 'C|%'"
    ).fetchone()[0]
    global_max = con.execute(
        f"SELECT max(__step) FROM read_parquet('{prepared.folder / 'panel.parquet'}')"
    ).fetchone()[0]
    assert c_max.date() < global_max.date()
    assert (global_max.date() - c_max.date()).days // 7 >= truth.weeks - truth.ended_week - 1


def test_series_prep_sums_the_two_weekly_transaction_rows(tmp_path):
    truth = write_panel(tmp_path)
    prepared = prepare(_panel_spec(truth.path), tmp_path / "cache")
    con = duckdb.connect()

    first_week_units = con.execute(
        f"SELECT __step, units FROM read_parquet('{prepared.folder / 'panel.parquet'}') "
        "WHERE __series = 'A|p1' ORDER BY __step LIMIT 1"
    ).fetchone()
    raw_rows = con.execute(
        f"SELECT day, units FROM read_csv('{truth.path}', header = true) "
        "WHERE project = 'A' AND product = 'p1' ORDER BY day LIMIT 2"
    ).fetchall()
    assert first_week_units[1] == pytest.approx(raw_rows[0][1] + raw_rows[1][1])


def test_series_prep_forward_fills_the_last_aggregation_across_the_hole(tmp_path):
    truth = write_panel(tmp_path)
    prepared = prepare(_panel_spec(truth.path), tmp_path / "cache")
    con = duckdb.connect()

    series, start, stop = truth.gap
    rows = con.execute(
        f"SELECT __step, orders FROM read_parquet('{prepared.folder / 'panel.parquet'}') "
        f"WHERE __series = '{series}' ORDER BY __step"
    ).fetchall()
    before_gap = rows[start - 1][1]
    hole_values = [orders for _step, orders in rows[start:stop]]
    assert hole_values == [before_gap] * (stop - start)


def test_series_prep_scaffolds_h_steps_after_the_global_last_for_every_series(tmp_path):
    truth = write_panel(tmp_path)
    prepared = prepare(_panel_spec(truth.path), tmp_path / "cache")
    con = duckdb.connect()

    scaffold_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{prepared.folder / 'scaffold.parquet'}')"
    ).fetchone()[0]
    assert scaffold_rows == truth.series * 8

    global_last, scaffold_min = con.execute(
        f"SELECT (SELECT max(__step) FROM read_parquet('{prepared.folder / 'panel.parquet'}')), "
        f"(SELECT min(__step) FROM read_parquet('{prepared.folder / 'scaffold.parquet'}'))"
    ).fetchone()
    assert scaffold_min > global_last


# ---------------------------------------------------------------------------
# row mode
# ---------------------------------------------------------------------------


def test_row_prep_writes_every_row(tmp_path):
    truth = write_rows(tmp_path, n=600)
    prepared = prepare(_row_spec(truth.path), tmp_path / "cache")
    con = duckdb.connect()

    total = con.execute(
        f"SELECT count(*) FROM read_parquet('{prepared.folder / 'rows.parquet'}')"
    ).fetchone()[0]
    assert total == 600


def test_row_prep_bounds_open_and_known_rows_correctly(tmp_path):
    truth = write_rows(tmp_path, n=600)
    prepared = prepare(_row_spec(truth.path), tmp_path / "cache")
    con = duckdb.connect()
    rows_path = prepared.folder / "rows.parquet"

    as_of = con.execute(f"SELECT CAST({prepared.summary['as_of']!r} AS DATE)").fetchone()[0]

    open_rows = con.execute(
        f"SELECT __anchor, __lower, __upper FROM read_parquet('{rows_path}') WHERE lead_weeks IS NULL"
    ).fetchall()
    assert len(open_rows) == truth.open_rows
    for anchor, lower, upper in open_rows:
        assert lower == pytest.approx((as_of - anchor).days / 7)
        assert math.isinf(upper) and upper > 0

    known_rows = con.execute(
        f"SELECT lead_weeks, __lower, __upper FROM read_parquet('{rows_path}') "
        "WHERE lead_weeks IS NOT NULL"
    ).fetchall()
    assert known_rows  # the fixture always has some closed lines
    for target, lower, upper in known_rows:
        assert lower == target == upper


def test_row_prep_predict_matches_the_open_count(tmp_path):
    truth = write_rows(tmp_path, n=600)
    prepared = prepare(_row_spec(truth.path), tmp_path / "cache")
    con = duckdb.connect()

    predict_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{prepared.folder / 'predict.parquet'}')"
    ).fetchone()[0]
    assert predict_rows == truth.open_rows


def test_row_prep_classification_target_gets_null_bounds(tmp_path):
    truth = write_rows(tmp_path, n=600)
    spec = _row_spec(
        truth.path,
        task="classification",
        columns=(
            Column("created", "anchor"),
            Column("size", "measure"),
            Column("branch", "target"),
        ),
        censor=None,
    )
    prepared = prepare(spec, tmp_path / "cache")
    con = duckdb.connect()

    bad = con.execute(
        f"SELECT count(*) FROM read_parquet('{prepared.folder / 'rows.parquet'}') "
        "WHERE __lower IS NOT NULL OR __upper IS NOT NULL"
    ).fetchone()[0]
    assert bad == 0
    branches = {
        row[0]
        for row in con.execute(
            f"SELECT DISTINCT branch FROM read_parquet('{prepared.folder / 'rows.parquet'}')"
        ).fetchall()
    }
    assert branches == set(BRANCHES)


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_prepare_reuses_an_unchanged_spec_and_source(tmp_path):
    truth = write_rows(tmp_path, n=200)
    spec = _row_spec(truth.path)
    cache = tmp_path / "cache"

    first = prepare(spec, cache)
    mtime = (first.folder / "rows.parquet").stat().st_mtime
    assert not first.reused

    second = prepare(spec, cache)
    assert second.reused
    assert second.folder == first.folder
    assert (second.folder / "rows.parquet").stat().st_mtime == mtime


def test_prepare_gets_a_new_folder_when_an_aggregation_changes(tmp_path):
    truth = write_panel(tmp_path)
    cache = tmp_path / "cache"
    spec_a = _panel_spec(truth.path)
    spec_b = _panel_spec(
        truth.path,
        columns=(
            Column("day", "time"),
            Column("project", "dimension"),
            Column("product", "dimension"),
            Column("orders", "measure", aggregation="mean"),
            Column("units", "target", aggregation="sum"),
        ),
    )
    folder_a = prepare(spec_a, cache).folder
    folder_b = prepare(spec_b, cache).folder
    assert folder_a != folder_b


def test_prepare_gets_a_new_folder_when_the_source_changes_by_one_byte(tmp_path):
    truth = write_rows(tmp_path, n=200)
    cache = tmp_path / "cache"
    spec = _row_spec(truth.path)
    folder_before = prepare(spec, cache).folder

    # flip one character's case, well inside an existing row - same row count, one changed byte
    text = truth.path.read_text()
    assert "north" in text
    truth.path.write_text(text.replace("north", "North", 1))

    folder_after = prepare(spec, cache).folder
    assert folder_before != folder_after


# ---------------------------------------------------------------------------
# encoding and json
# ---------------------------------------------------------------------------


def test_latin1_encoded_column_values_come_back_intact(tmp_path):
    path = tmp_path / "accounts.csv"
    rows = [
        {"created": "2024-01-01", "Account: Account Name": "Müller GmbH", "value": "1.5"},
        {"created": "2024-01-02", "Account: Account Name": "Käufer & Söhne", "value": "2.5"},
    ]
    with path.open("w", newline="", encoding="latin-1") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    spec = PrepSpec(
        source=str(path),
        mode="row",
        task="regression",
        encoding="latin-1",
        columns=(
            Column("created", "anchor"),
            Column("Account: Account Name", "dimension"),
            Column("value", "target"),
        ),
    )
    prepared = prepare(spec, tmp_path / "cache")
    con = duckdb.connect()
    values = {
        row[0]
        for row in con.execute(
            f"SELECT \"Account: Account Name\" FROM read_parquet('{prepared.folder / 'rows.parquet'}')"
        ).fetchall()
    }
    assert values == {"Müller GmbH", "Käufer & Söhne"}


def test_json_source_gives_the_same_rows_as_the_equivalent_csv(tmp_path):
    records = [
        {"created": "2024-01-01", "branch": "north", "value": 1.5},
        {"created": "2024-01-02", "branch": "south", "value": 2.5},
        {"created": "2024-01-03", "branch": "east", "value": None},
    ]
    csv_path = tmp_path / "rows.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["created", "branch", "value"])
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "value": "" if record["value"] is None else record["value"]})

    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps(records))

    def _spec(source):
        return PrepSpec(
            source=str(source),
            mode="row",
            task="regression",
            columns=(
                Column("created", "anchor"),
                Column("branch", "dimension"),
                Column("value", "target"),
            ),
        )

    csv_prepared = prepare(_spec(csv_path), tmp_path / "cache_csv")
    json_prepared = prepare(_spec(json_path), tmp_path / "cache_json")
    con = duckdb.connect()
    csv_rows = con.execute(
        f"SELECT __anchor, branch, value FROM read_parquet('{csv_prepared.folder / 'rows.parquet'}') "
        "ORDER BY __anchor"
    ).fetchall()
    json_rows = con.execute(
        f"SELECT __anchor, branch, value FROM read_parquet('{json_prepared.folder / 'rows.parquet'}') "
        "ORDER BY __anchor"
    ).fetchall()
    assert csv_rows == json_rows


# ---------------------------------------------------------------------------
# sql
# ---------------------------------------------------------------------------


def test_the_stored_sql_contains_the_quoted_column_names(tmp_path):
    truth = write_rows(tmp_path, n=100)
    prepared = prepare(_row_spec(truth.path), tmp_path / "cache")
    assert '"created"' in prepared.sql
    assert '"branch"' in prepared.sql
    assert '"lead_weeks"' in prepared.sql
    assert f'"{ROW}"' in prepared.sql


def test_running_the_stored_sql_by_hand_reproduces_the_table(tmp_path, monkeypatch):
    """replays prep.sql verbatim against the same connection and diffs the parquet bytes,
    rather than re-parsing the text into a separate output path - the COPY targets in the
    stored sql are the real cache files, so a faithful replay overwrites them with themselves"""
    truth = write_rows(tmp_path, n=100)
    prepared = prepare(_row_spec(truth.path), tmp_path / "cache")
    rows_path = prepared.folder / "rows.parquet"
    before = rows_path.read_bytes()

    # the stored sql's copy targets are relative filenames, meant to be run from the cache folder
    monkeypatch.chdir(prepared.folder)
    con = duckdb.connect()
    con.execute("SET threads=1")
    for statement in prepared.sql.split("\n\n"):
        con.execute(statement)

    assert rows_path.read_bytes() == before


def test_the_predict_filter_reads_source_columns_the_spec_leaves_out(tmp_path):
    """a user filters on the file's own columns; most of them never become spec columns"""
    source = tmp_path / "lines.csv"
    source.write_text(
        "created,branch,lead,is_open\n"
        "2026-01-05,a,3.0,0\n2026-01-12,a,,1\n2026-01-19,b,,0\n2026-01-26,b,,1\n"
    )
    spec = PrepSpec(
        str(source),
        "row",
        "regression",
        (Column("created", "anchor"), Column("branch", "dimension"), Column("lead", "target")),
        predict_where="is_open = 1",
    )
    prepared = prepare(spec, tmp_path / "cache")
    rows = duckdb.sql(f"SELECT * FROM '{prepared.folder}/rows.parquet'").columns
    predicted = duckdb.sql(f"SELECT count(*) FROM '{prepared.folder}/predict.parquet'").fetchone()[
        0
    ]
    assert predicted == 2
    assert "__predict" not in rows and "is_open" not in rows


def test_series_history_ends_before_the_step_holding_the_export_date(tmp_path):
    truth = write_panel(tmp_path)
    full = prepare(_panel_spec(truth.path), tmp_path / "cache")
    # a wednesday: its week is still running
    prepared = prepare(_panel_spec(truth.path, as_of="2025-06-04"), tmp_path / "cache")
    steps = duckdb.sql(f"SELECT max(__step) FROM '{prepared.folder}/panel.parquet'").fetchone()[0]
    assert str(steps)[:10] == "2025-05-26", "the week of 2025-06-02 is still running"
    assert prepared.summary["dropped_after_as_of"] > 0
    assert prepared.summary["panel_rows"] < full.summary["panel_rows"]
