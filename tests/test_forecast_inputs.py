"""categorical inputs retain source-level detail without changing the output grain"""

import csv
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import pipeline
from smolsmort.forecast.features import LAGS, family_sort_key, timeseries_features
from smolsmort.forecast.pipeline import load_workspace, make_genome
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.screen import screen_families
from smolsmort.forecast.search import DEFAULTS, Budget, _select_birth_families, search
from smolsmort.forecast.spec import FILES, SERIES, STEP, Column, PrepSpec, SpecError

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
        "timeseries",
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
        "timeseries",
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
    assert set(FILES["timeseries"]) == {"panel.parquet", "scaffold.parquet"}
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


def test_entity_mode_rejects_input_role():
    with pytest.raises(SpecError, match="input.*series"):
        PrepSpec(
            "source.csv",
            "entity",
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
        "timeseries",
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
        "timeseries",
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


@pytest.mark.parametrize("mode", ["timeseries", "entity"])
def test_parse_spec_accepts_inputs_only_in_timeseries_mode(input_source, tmp_path, mode):
    from smolsmort.forecast.tab import _parse_spec
    from smolsmort.review.routes import RequestError

    app = SimpleNamespace(state=SimpleNamespace(bases={"forecast": tmp_path}))
    data = asdict(_make_spec(input_source))
    data["source"] = input_source.name
    data["mode"] = mode
    if mode == "entity":
        data["columns"][0]["role"] = "anchor"
        with pytest.raises(RequestError, match="input.*series"):
            _parse_spec(app, {"spec": data})
    else:
        assert _parse_spec(app, {"spec": data}).named("input") == ["site"]


def make_input_panel():
    steps = np.datetime64("2025-01-06") + np.arange(12) * np.timedelta64(7, "D")
    spec = PrepSpec(
        "synthetic.csv",
        "timeseries",
        "regression",
        (
            Column("day", "time"),
            Column("group", "dimension"),
            Column("site", "input"),
            Column("units", "target", "sum"),
        ),
        step="week",
        horizon=2,
    )
    panel = {
        SERIES: np.repeat(["a", "b"], len(steps)),
        STEP: np.tile(steps, 2),
        "group": np.repeat(["a", "b"], len(steps)),
        "units": np.arange(len(steps) * 2, dtype=float) + 10,
    }
    scaffold = {
        SERIES: np.repeat(["a", "b"], 2),
        STEP: np.tile(steps[-1] + np.arange(1, 3) * np.timedelta64(7, "D"), 2),
    }
    records = [("a", 0, level, 10.0) for level in ("alpha", "beta", "charlie", "delta", "epsilon")]
    records += [
        ("b", 0, "zeta", 100.0),
        ("a", 1, "alpha", 0.0),
        ("a", 1, "zeta", 0.0),
        ("a", 3, "alpha", 2.0),
        ("a", 3, "zeta", 6.0),
        ("a", 3, "epsilon", 2.0),
        ("a", 4, "alpha", 2.0),
        ("a", 4, "zeta", 3.0),
        ("a", 4, "epsilon", 4.0),
        ("a", 4, "other", 1.0),
        ("a", 4, "new", 10.0),
        ("b", 4, "new", 10_000.0),
    ]
    inputs = {
        SERIES: np.array([row[0] for row in records]),
        STEP: np.array([steps[row[1]] for row in records]),
        "__column": np.full(len(records), "site"),
        "__level": np.array([row[2] for row in records]),
        "rows": np.ones(len(records), dtype=int),
        "target": np.array([row[3] for row in records]),
    }
    return spec, panel, scaffold, inputs, panel[STEP] < steps[3]


@pytest.mark.parametrize("bucket", [(1, 1), (2, 2)])
def test_input_features_keep_grain_and_use_origin_lags(bucket):
    spec, panel, scaffold, inputs, train = make_input_panel()
    original = timeseries_features(spec, panel, scaffold, bucket, "week", train)
    frame = timeseries_features(spec, panel, scaffold, bucket, "week", train, inputs=inputs)
    for name in ("y", "series", "step", "origin", "future", "base"):
        np.testing.assert_array_equal(getattr(frame, name), getattr(original, name))
    assert set(frame.features.families) - set(original.features.families) == {
        "site@day:mix",
        "site@day:spread",
    }
    input_columns = [
        index
        for index, family in enumerate(frame.features.families)
        if family in {"site@day:mix", "site@day:spread"}
    ]
    assert len(input_columns) == (6 + 2) * len(LAGS)
    assert all(frame.features.types[index] == "q" for index in input_columns)
    names = frame.features.names
    origin = np.datetime64("2025-02-03")
    row = np.flatnonzero((frame.series == "a") & (frame.origin == origin))[0]
    expected = {
        0: (0.1, 0.15, 0.75, 5.0, 0.325),
        1: (0.2, 0.6, 0.2, 3.0, 0.44),
        2: (np.nan, np.nan, np.nan, np.nan, np.nan),
        3: (0.0, 0.0, 0.0, 2.0, 0.0),
    }
    columns = (
        "mix:site:level:alpha",
        "mix:site:level:zeta",
        "mix:site:other",
        "spread:site:count",
        "spread:site:hhi",
    )
    for lag, values in expected.items():
        actual = [
            frame.features.x[row, names.index(f"{name}_lag_{bucket[1] + lag}")] for name in columns
        ]
        np.testing.assert_allclose(actual, values, rtol=1e-6, equal_nan=True)
    other_series = np.flatnonzero((frame.series == "b") & (frame.origin == origin))[0]
    assert frame.features.x[other_series, names.index(f"mix:site:other_lag_{bucket[1]}")] == 1


def test_input_levels_freeze_globally_on_train_rows_with_name_ties():
    spec, panel, scaffold, inputs, train = make_input_panel()
    frame = timeseries_features(spec, panel, scaffold, (1, 1), "week", train, inputs=inputs)
    names = frame.features.names
    selected = {
        name.removeprefix("mix:site:level:").removesuffix("_lag_1")
        for name in names
        if name.startswith("mix:site:level:") and name.endswith("_lag_1")
    }
    assert selected == {"zeta", "alpha", "beta", "charlie", "delta"}
    assert not any(":new_" in name or ":epsilon_" in name for name in names)
    changed = {name: values.copy() for name, values in inputs.items()}
    changed["target"][changed[STEP] >= np.datetime64("2025-01-27")] *= 1000
    rebuilt = timeseries_features(spec, panel, scaffold, (1, 1), "week", train, inputs=changed)
    assert rebuilt.features.names == names


def test_input_level_named_other_does_not_collide_with_other_bucket():
    spec, panel, scaffold, inputs, train = make_input_panel()
    inputs["__level"][inputs["__level"] == "zeta"] = "other"
    frame = timeseries_features(spec, panel, scaffold, (1, 1), "week", train, inputs=inputs)
    assert "mix:site:level:other_lag_1" in frame.features.names
    assert "mix:site:other_lag_1" in frame.features.names
    assert len(frame.features.names) == len(set(frame.features.names))


def test_input_features_stay_missing_beyond_an_ended_series():
    spec, panel, scaffold, inputs, train = make_input_panel()
    keep = (panel[SERIES] != "a") | (panel[STEP] <= np.datetime64("2025-02-03"))
    panel = {name: values[keep] for name, values in panel.items()}
    frame = timeseries_features(spec, panel, scaffold, (1, 1), "week", train[keep], inputs=inputs)
    ended_future = (frame.series == "a") & frame.future
    assert ended_future.any()
    values = frame.features.pick({"site@day:mix", "site@day:spread"})[0]
    assert np.isnan(values[ended_future]).all()


def test_workspace_freezes_input_levels_at_first_validation_fold(monkeypatch):
    spec, _, _, _, _ = make_input_panel()
    steps = np.datetime64("2025-01-06") + np.arange(54) * np.timedelta64(7, "D")
    panel = {
        SERIES: np.full(52, "a"),
        STEP: steps[:52],
        "group": np.full(52, "a"),
        "units": np.arange(52, dtype=float) + 100,
    }
    inputs = {
        SERIES: panel[SERIES],
        STEP: panel[STEP],
        "__column": np.full(52, "site"),
        "__level": np.where(np.arange(52) < 26, "known", "unseen"),
        "rows": np.ones(52, dtype=int),
        "target": np.where(np.arange(52) < 26, 1.0, 100_000.0),
    }
    tables = {
        "panel.parquet": panel,
        "inputs.parquet": inputs,
        "scaffold.parquet": {SERIES: np.full(2, "a"), STEP: steps[52:]},
    }
    monkeypatch.setattr(pipeline, "read_table", lambda path, **kwargs: tables[path.name])
    workspace = load_workspace(spec, Path("synthetic"), {"step": "week"})
    cutoff = workspace.series_folds[0][0]
    assert cutoff == steps[26]
    assert {"site@day:mix", "site@day:spread"} <= set(workspace.families)
    for bucket, frame in workspace.frames.items():
        assert not any(":level:unseen_" in name for name in frame.features.names)
        rows = frame.origin >= cutoff
        other = frame.features.names.index(f"mix:site:other_lag_{bucket[1]}")
        np.testing.assert_array_equal(frame.features.x[rows, other], 1)


def test_input_column_and_level_names_cannot_overwrite_another_family():
    spec, panel, scaffold, inputs, train = make_input_panel()
    spec = replace(spec, columns=(*spec.columns, Column("site:level:x", "input")))
    first = {name: values.copy() for name, values in inputs.items()}
    first["__level"] = np.full(len(inputs[STEP]), "x:level:y", dtype=object)
    second = {name: values.copy() for name, values in inputs.items()}
    second["__column"] = np.full(len(inputs[STEP]), "site:level:x", dtype=object)
    second["__level"] = np.full(len(inputs[STEP]), "y", dtype=object)
    # one source level per series and step matches the prepared table's unique grain
    keep = np.array([0, 5])
    combined = {name: np.concatenate([first[name][keep], second[name][keep]]) for name in inputs}
    frame = timeseries_features(spec, panel, scaffold, (1, 1), "week", train, inputs=combined)
    assert len(frame.features.pick({"site@day:mix"})[1]) == 2 * len(LAGS)
    assert len(frame.features.pick({"site%3Alevel%3Ax@day:mix"})[1]) == 2 * len(LAGS)
    assert len(frame.features.names) == len(set(frame.features.names))


def test_input_families_are_available_to_births_without_screen_priors():
    workspace = SimpleNamespace(
        mode="timeseries",
        unit="week",
        families=[
            "units@day:lag+0",
            "site@day:mix",
            "site@day:spread",
            "segment@day:mix",
            "segment@day:spread",
        ],
    )
    assert _select_birth_families(workspace) == sorted(workspace.families, key=family_sort_key)


def write_leading_input_source(folder):
    rng = np.random.default_rng(42)
    steps = np.datetime64("2023-01-02") + np.arange(156) * np.timedelta64(7, "D")
    source = folder / "leading-input.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["day", "group", "site", "noise_0", "noise_1", "noise_2", "units"])
        for series in range(6):
            shares = rng.uniform(0.1, 0.9, (len(steps) + 2, 4))
            for step, date in enumerate(steps):
                target = 100 + 100 * (shares[step, 0] - 0.5)
                for levels in itertools.product((0, 1), repeat=4):
                    weights = [
                        shares[step + 2, col] if level == 0 else 1 - shares[step + 2, col]
                        for col, level in enumerate(levels)
                    ]
                    writer.writerow(
                        [
                            str(date),
                            f"series_{series}",
                            *(f"s{level + 1}" for level in levels),
                            target * np.prod(weights),
                        ]
                    )
    return PrepSpec(
        str(source),
        "timeseries",
        "regression",
        (
            Column("day", "time"),
            Column("group", "dimension"),
            Column("site", "input"),
            *(Column(f"noise_{index}", "input") for index in range(3)),
            Column("units", "target", "sum"),
        ),
        step="week",
        horizon=2,
    )


def test_leading_site_mix_ranks_near_top_and_wins_short_search(tmp_path):
    pytest.importorskip("xgboost")
    started = time.perf_counter()
    spec = write_leading_input_source(tmp_path)
    prepared = prepare(spec, tmp_path / "cache")
    workspace = load_workspace(spec, prepared.folder, prepared.summary)
    ranking = screen_families(workspace)
    assert "site@day:mix" in [row["family"] for row in ranking[:3]]
    priors = {row["family"]: row["p_on"] for row in ranking}
    without_site = make_genome(
        [name for name in workspace.families if name != "site@day:mix"],
        workspace.profile["genes"]["objectives"][0],
        DEFAULTS,
    )
    result = search(
        workspace,
        Budget(population=4, max_generations=2, time_cap=40, seed=42, nthread=1),
        priors=priors,
        warm=[without_site.to_dict()],
    )
    assert result["best"] is not None
    assert "site@day:mix" in result["best"]["families"]
    report = {
        "screen_top_five": [
            {key: row[key] for key in ("family", "score", "p_on")} for row in ranking[:5]
        ],
        "winner_families": result["best"]["families"],
        "seconds": time.perf_counter() - started,
    }
    report_path = tmp_path / "input-search-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(report_path)}))
