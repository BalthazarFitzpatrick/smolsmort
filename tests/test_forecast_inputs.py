"""categorical inputs retain source-level detail without changing the output grain"""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from smolsmort.forecast.prep import prepare
from smolsmort.forecast.spec import FILES, Column, PrepSpec, SpecError

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def input_source(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "day,group,site,segment,units\n"
        "2025-01-06,a,north,101,2\n"
        "2025-01-06,a,north,102,3\n"
        "2025-01-06,a,south,101,5\n"
        "2025-01-06,b,south,103,7\n"
        "2025-01-07,a,,101,11\n"
        "2025-01-07,a,north,102,13\n"
        "2025-01-07,b,south,,17\n",
        encoding="utf-8",
    )
    return source


def _make_spec(source, **changes):
    spec = PrepSpec(
        str(source),
        "series",
        "regression",
        (
            Column("day", "time"),
            Column("group", "dimension"),
            Column("site", "input"),
            Column("units", "target", "sum"),
        ),
        step="day",
    )
    return replace(spec, **changes)


def test_inputs_preserve_group_grain_and_reconcile_target_and_rows(input_source, tmp_path):
    prepared = prepare(_make_spec(input_source), tmp_path / "cache")
    with duckdb.connect() as con:
        con.read_parquet(str(prepared.folder / "panel.parquet")).create_view("panel")
        con.read_parquet(str(prepared.folder / "inputs.parquet")).create_view("inputs")
        assert con.sql("SELECT DISTINCT __series FROM panel ORDER BY __series").fetchall() == [
            ("a",),
            ("b",),
        ]
        assert con.sql("SELECT count(*) FROM panel").fetchone() == (4,)
        assert con.sql("DESCRIBE panel").fetchall()[0][0] == "__step"
        assert "site" not in con.table("panel").columns
        assert con.table("inputs").columns == [
            "__series",
            "__step",
            "__column",
            "__level",
            "rows",
            "target",
        ]
        assert {row[0]: row[1] for row in con.sql("DESCRIBE inputs").fetchall()} == {
            "__series": "VARCHAR",
            "__step": "DATE",
            "__column": "VARCHAR",
            "__level": "VARCHAR",
            "rows": "BIGINT",
            "target": "DOUBLE",
        }
        totals = con.sql(
            "SELECT __series, strftime(__step, '%Y-%m-%d'), sum(rows), sum(target) "
            "FROM inputs GROUP BY __series, __step ORDER BY __series, __step"
        ).fetchall()
        assert totals == [
            ("a", "2025-01-06", 3, 10.0),
            ("a", "2025-01-07", 2, 24.0),
            ("b", "2025-01-06", 1, 7.0),
            ("b", "2025-01-07", 1, 17.0),
        ]
        panel_targets = con.sql(
            "SELECT __series, strftime(__step, '%Y-%m-%d'), units "
            "FROM panel ORDER BY __series, __step"
        ).fetchall()
        assert [(series, step, target) for series, step, _, target in totals] == panel_targets
        assert con.sql(
            "SELECT __column, rows, target FROM inputs WHERE __level = '(none)'"
        ).fetchall() == [("site", 1, 11.0)]


def test_inputs_never_appear_in_scaffold(input_source, tmp_path):
    prepared = prepare(_make_spec(input_source, horizon=2), tmp_path / "cache")
    with duckdb.connect() as con:
        scaffold = con.read_parquet(str(prepared.folder / "scaffold.parquet"))
        assert scaffold.columns == ["__step", "__series", "group"]
        scaffold.create_view("scaffold")
        assert con.sql(
            "SELECT __series, count(*) FROM scaffold GROUP BY __series ORDER BY __series"
        ).fetchall() == [("a", 2), ("b", 2)]


def test_multiple_inputs_are_independent_long_tables(input_source, tmp_path):
    spec = _make_spec(input_source)
    spec = replace(spec, columns=(*spec.columns, Column("segment", "input")))
    prepared = prepare(spec, tmp_path / "cache")
    with duckdb.connect() as con:
        con.read_parquet(str(prepared.folder / "inputs.parquet")).create_view("inputs")
        assert con.sql(
            "SELECT __column, sum(rows), sum(target) FROM inputs "
            "GROUP BY __column ORDER BY __column"
        ).fetchall() == [("segment", 7, 58.0), ("site", 7, 58.0)]
        assert con.sql(
            "SELECT DISTINCT __level FROM inputs WHERE __column = 'segment' ORDER BY __level"
        ).fetchall() == [("(none)",), ("101",), ("102",), ("103",)]
        assert con.sql(
            "SELECT rows, target FROM inputs "
            "WHERE __series = 'a' AND __step = DATE '2025-01-06' "
            "AND __column = 'site' AND __level = 'north'"
        ).fetchall() == [(2, 5.0)]


def test_no_inputs_keep_legacy_digest():
    spec = PrepSpec(
        "source.csv",
        "series",
        "regression",
        (Column("day", "time"), Column("group", "dimension"), Column("units", "target", "sum")),
        step="day",
    )
    assert spec.digest() == "0ce3495b28183342"


def test_no_inputs_keep_original_files_and_reuse_cache(input_source, tmp_path):
    spec = _make_spec(input_source)
    spec = replace(spec, columns=tuple(column for column in spec.columns if column.role != "input"))
    prepared = prepare(spec, tmp_path / "cache")
    assert not prepared.reused
    assert set(FILES["series"]) == {"panel.parquet", "scaffold.parquet"}
    assert {path.name for path in prepared.folder.iterdir()} == {
        "panel.parquet",
        "scaffold.parquet",
        "prep.sql",
        "prep.json",
    }
    reused = prepare(spec, tmp_path / "cache")
    assert reused.reused
    assert reused.folder == prepared.folder
    assert not (reused.folder / "inputs.parquet").exists()


def test_input_cache_rebuilds_when_inputs_file_is_missing(input_source, tmp_path):
    spec = _make_spec(input_source)
    cache = tmp_path / "cache"
    prepared = prepare(spec, cache)
    assert not prepared.reused
    assert prepare(spec, cache).reused
    (prepared.folder / "inputs.parquet").unlink()
    rebuilt = prepare(spec, cache)
    assert not rebuilt.reused
    assert rebuilt.folder == prepared.folder
    assert (rebuilt.folder / "inputs.parquet").is_file()
    assert prepare(spec, cache).reused


def test_row_mode_rejects_input_role():
    with pytest.raises(SpecError, match="input.*series"):
        PrepSpec(
            "source.csv",
            "row",
            "regression",
            (Column("day", "anchor"), Column("site", "input"), Column("units", "target")),
        )


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("sum", 20.0),
        ("mean", 20 / 3),
        ("median", 3.0),
        ("min", 1.0),
        ("max", 16.0),
        ("count", 3),
        ("last", 16.0),
    ],
)
def test_input_target_uses_first_target_aggregation(tmp_path, aggregation, expected):
    source = tmp_path / "source.csv"
    source.write_text(
        "day,site,units,secondary\n"
        "2025-01-06,north,1,100\n"
        "2025-01-07,north,3,200\n"
        "2025-01-08,north,16,300\n"
        "2025-01-09,north,,400\n",
        encoding="utf-8",
    )
    spec = PrepSpec(
        str(source),
        "series",
        "regression",
        (
            Column("day", "time"),
            Column("site", "input"),
            Column("units", "target", aggregation),
            Column("secondary", "target", "sum"),
        ),
        step="week",
    )
    prepared = prepare(spec, tmp_path / "cache")
    with duckdb.connect() as con:
        con.read_parquet(str(prepared.folder / "inputs.parquet")).create_view("inputs")
        series, column, level, rows, target = con.sql(
            "SELECT __series, __column, __level, rows, target FROM inputs"
        ).fetchone()
        assert (series, column, level, rows) == ("all", "site", "north", 4)
        assert target == pytest.approx(expected)
        con.read_parquet(str(prepared.folder / "panel.parquet")).create_view("panel")
        assert con.sql("SELECT units FROM panel").fetchone()[0] == pytest.approx(expected)


def test_input_count_target_accepts_text_and_counts_only_nonnull_values(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "day,site,order_id\n"
        "2025-01-06,north,order-a\n"
        "2025-01-07,north,\n"
        "2025-01-08,south,order-b\n",
        encoding="utf-8",
    )
    spec = PrepSpec(
        str(source),
        "series",
        "regression",
        (Column("day", "time"), Column("site", "input"), Column("order_id", "target", "count")),
        step="week",
    )
    prepared = prepare(spec, tmp_path / "cache")
    with duckdb.connect() as con:
        con.read_parquet(str(prepared.folder / "inputs.parquet")).create_view("inputs")
        assert con.sql("SELECT __level, rows, target FROM inputs ORDER BY __level").fetchall() == [
            ("north", 2, 1),
            ("south", 1, 1),
        ]


def test_input_sum_with_null_targets_reconciles_with_zero_panel_target(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "day,group,site,units\n2025-01-06,a,north,\n2025-01-06,a,north,\n2025-01-06,a,south,\n",
        encoding="utf-8",
    )
    prepared = prepare(_make_spec(source), tmp_path / "cache")
    with duckdb.connect() as con:
        con.read_parquet(str(prepared.folder / "inputs.parquet")).create_view("inputs")
        con.read_parquet(str(prepared.folder / "panel.parquet")).create_view("panel")
        assert con.sql("SELECT sum(rows), sum(target) FROM inputs").fetchone() == (3, 0.0)
        assert con.sql("SELECT units FROM panel").fetchone() == (0.0,)


def test_input_names_are_quoted_as_identifiers_and_labels(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        'day,group,"site\'s ""name""",units\n2025-01-06,a,north,2\n', encoding="utf-8"
    )
    input_name = 'site\'s "name"'
    spec = _make_spec(source)
    spec = replace(
        spec,
        columns=tuple(
            replace(column, name=input_name) if column.role == "input" else column
            for column in spec.columns
        ),
    )
    prepared = prepare(spec, tmp_path / "cache")
    with duckdb.connect() as con:
        con.read_parquet(str(prepared.folder / "inputs.parquet")).create_view("inputs")
        assert con.sql("SELECT __column, __level, rows, target FROM inputs").fetchall() == [
            (input_name, "north", 1, 2.0)
        ]


def test_inputs_apply_source_and_history_filters(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "day,group,site,units,included\n"
        "2025-01-06,a,north,2,1\n"
        "2025-01-06,a,south,100,0\n"
        "2025-01-07,a,south,200,1\n"
        ",a,unknown,300,1\n",
        encoding="utf-8",
    )
    prepared = prepare(
        _make_spec(source, where="included = 1", as_of="2025-01-07"), tmp_path / "cache"
    )
    with duckdb.connect() as con:
        con.read_parquet(str(prepared.folder / "inputs.parquet")).create_view("inputs")
        assert con.sql("SELECT __level, rows, target FROM inputs").fetchall() == [("north", 1, 2.0)]
    assert prepared.summary["dropped_null_time"] == 1
    assert prepared.summary["dropped_after_as_of"] == 1


@pytest.mark.parametrize("mode", ["series", "row"])
def test_parse_spec_accepts_inputs_only_in_series_mode(input_source, tmp_path, mode):
    from smolsmort.forecast.tab import _parse_spec
    from smolsmort.review.routes import RequestError

    app = SimpleNamespace(state=SimpleNamespace(bases={"forecast": tmp_path}))
    data = asdict(_make_spec(input_source))
    data["source"] = input_source.name
    data["mode"] = mode
    if mode == "row":
        data["columns"][0]["role"] = "anchor"
        with pytest.raises(RequestError, match="input.*series"):
            _parse_spec(app, {"spec": data})
    else:
        assert _parse_spec(app, {"spec": data}).named("input") == ["site"]
