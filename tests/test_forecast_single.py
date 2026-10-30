"""synthetic stacked forecasts use fresh origins and keep every lead"""

from pathlib import Path

import numpy as np
import pytest

from smolsmort.forecast import evaluate as ev
from smolsmort.forecast import pipeline
from smolsmort.forecast.families import get_family
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec


def build_workspace(
    monkeypatch, *, weeks=165, series=2, holdout="seasonal", mode="single", values=None
):
    positions = np.arange(weeks)
    steps = np.datetime64("2020-01-06") + positions * np.timedelta64(7, "D")
    if values is None:
        values = np.concatenate(
            [
                40 + index * 5 + positions / 10 + 8 * np.sin(positions * 2 * np.pi / 52)
                for index in range(series)
            ]
        )
    panel = {
        SERIES: np.repeat([f"s{index}" for index in range(series)], weeks),
        STEP: np.tile(steps, series),
        "y": np.asarray(values),
        "market": np.tile(100 + positions, series).astype(float),
    }
    future = steps[-1] + np.arange(1, 9) * np.timedelta64(7, "D")
    scaffold = {SERIES: np.repeat(np.unique(panel[SERIES]), 8), STEP: np.tile(future, series)}
    tables = {"panel.parquet": panel, "scaffold.parquet": scaffold}
    monkeypatch.setattr(pipeline, "read_table", lambda path, **kwargs: tables[path.name])
    spec = PrepSpec(
        "synthetic.csv",
        "timeseries",
        "regression",
        (
            Column("day", "time"),
            Column("y", "target", "sum"),
            Column("market", "measure", "mean"),
        ),
        step="week",
        horizon=8,
    )
    workspace = pipeline.load_workspace(
        spec, Path("synthetic"), {"step": "week"}, holdout=holdout, horizon_mode=mode
    )
    return workspace


def candidate(workspace):
    return pipeline.make_genome(
        ["y@day:lag+0", "y@day:lag+1", "y@day:season"],
        "squared",
        get_family("ridge").defaults(),
        "ridge",
    )


@pytest.mark.parametrize("holdout", ["seasonal", "block"])
def test_stacked_slice_fits_scores_and_forecasts(monkeypatch, holdout):
    ws = build_workspace(monkeypatch, holdout=holdout)
    assert list(ws.frames) == [(1, 8)]
    genome = candidate(ws)
    scored = pipeline.score(ws, genome)
    final = pipeline.finish(ws, genome, validation=scored.prediction)
    assert np.isfinite(scored.fitness) and np.isfinite(final["model_error"])
    assert scored.metrics["fold_count"] == (4 if holdout == "seasonal" else 3)
    forecast = final["forecast"]
    for key in np.unique(forecast["series"]):
        rows = forecast["series"] == key
        assert forecast["weeks_ahead"][rows].tolist() == list(range(1, 9))
        assert len(np.unique(forecast["origin"][rows])) == 1
        assert np.all(
            forecast["step"][rows]
            == forecast["origin"][rows] + forecast["weeks_ahead"][rows] * np.timedelta64(7, "D")
        )
    frame = ws.frames[(1, 8)]
    for phase, windows, blocks in (
        ("val", ws.series_folds, pipeline._validation_masks(ws, (1, 8))),
        ("test", ws.test_folds, pipeline._test_windows(ws, (1, 8))),
    ):
        for (start, end), (train, evaluate) in zip(windows, blocks, strict=True):
            assert np.all(frame.step[train] < start)
            assert np.all((frame.origin[evaluate] >= start) & (frame.origin[evaluate] < end))
        rows = final["validation" if phase == "val" else "test"]
        assert len(rows["actual"]) == int(ws.series_masks[(1, 8)][phase].sum())
    assert final["horizon_mode"] == "single"
    assert final["origin_blocks"]["validation"][0]["origin_count"] > 1


def test_every_origin_has_every_known_lead(monkeypatch):
    ws = build_workspace(monkeypatch)
    frame = ws.frames[(1, 8)]
    last = ws.panel[STEP].max()
    for origin in np.unique(frame.origin[~frame.future]):
        rows = (frame.origin == origin) & (frame.series == "s0") & ~frame.future
        remaining = int((last - origin) / np.timedelta64(7, "D"))
        assert sorted(frame.weeks_ahead[rows]) == list(range(1, min(8, remaining) + 1))


@pytest.mark.parametrize("replacement", [np.nan, 1e12])
@pytest.mark.parametrize("origin_offset", [0, 6, 12])
def test_future_data_cannot_change_origin_features_or_predictions(
    monkeypatch, replacement, origin_offset
):
    ws = build_workspace(monkeypatch)
    frame = ws.frames[(1, 8)]
    start = ws.series_folds[0][0]
    origin = start + origin_offset * np.timedelta64(7, "D")
    train, _ = next(pipeline._validation_masks(ws, (1, 8)))
    rows = (frame.origin == origin) & ~frame.future
    genome = candidate(ws)
    expected = pipeline._timeseries_predict(ws, genome, frame, train, rows, (1, 8), 1)
    original_features = frame.features.x[rows].copy()
    for name in ("y", "market"):
        ws.panel[name][ws.panel[STEP] > origin] = replacement
    changed = pipeline.load_workspace(
        ws.spec, ws.folder, ws.summary, holdout="seasonal", horizon_mode="single"
    )
    changed_frame = changed.frames[(1, 8)]
    changed_rows = (changed_frame.origin == origin) & ~changed_frame.future
    changed_train = (changed_frame.step < start) & np.isfinite(changed_frame.y)
    np.testing.assert_allclose(
        changed_frame.features.x[changed_rows], original_features, equal_nan=True
    )
    actual = pipeline._timeseries_predict(
        changed, genome, changed_frame, changed_train, changed_rows, (1, 8), 1
    )
    np.testing.assert_allclose(actual, expected)


def test_single_design_always_has_lead_and_ridge_one_hot(monkeypatch):
    ws = build_workspace(monkeypatch)
    frame = ws.frames[(1, 8)]
    _, names, _ = pipeline._timeseries_design(candidate(ws), frame)
    assert "weeks_ahead" in names
    assert all(f"weeks_ahead_{lead}" in names for lead in range(1, 9))
    rows = (frame.weeks_ahead == 1) & ~frame.future
    index = frame.features.names.index("lag_0")
    np.testing.assert_allclose(frame.features.x[rows, index], frame.base[rows], rtol=1e-6)


def test_seasonal_features_follow_target_and_guard_short_periods(monkeypatch):
    from smolsmort.forecast.features import stacked_timeseries_features

    ws = build_workspace(monkeypatch)
    panel = ws.panel
    scaffold = {SERIES: np.array([], dtype=object), STEP: np.array([], dtype="datetime64[D]")}
    frame = stacked_timeseries_features(
        ws.spec, panel, scaffold, "week", panel[STEP] < ws.series_folds[0][0], period=3
    )
    origin = np.unique(panel[STEP])[100]
    rows = (frame.series == "s0") & (frame.origin == origin)
    target_positions = 100 + frame.weeks_ahead[rows]
    values = panel["y"][panel[SERIES] == "s0"]

    def column(name):
        return frame.features.x[rows, frame.features.names.index(name)]

    np.testing.assert_allclose(column("absolute_lag_1"), values[99], rtol=1e-6)
    np.testing.assert_allclose(
        column("target_season_lag"), values[target_positions - 52], rtol=1e-6
    )
    expected_mean = np.mean([values[target_positions - lag] for lag in (51, 52, 53)], axis=0)
    np.testing.assert_allclose(column("target_season_mean"), expected_mean, rtol=1e-6)
    period = column("target_period_lag_3")
    np.testing.assert_allclose(period[:3], values[target_positions[:3] - 3], rtol=1e-6)
    assert np.isnan(period[3:]).all()


def test_lead_error_arithmetic_matches_hand_calculation():
    errors, table = ev.lead_errors(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([8.0, 23.0, 35.0, 34.0]),
        np.array([10.0, 15.0, 30.0, 50.0]),
        np.array([1, 1, 2, 2]),
        "test",
    )
    assert errors["1"] == {"model": 5 / 30, "naive": 5 / 30, "rows": 2}
    assert errors["2"] == {"model": 11 / 70, "naive": 10 / 70, "rows": 2}
    np.testing.assert_allclose(table["ratio"], [1, 1.1])
    np.testing.assert_allclose(table["mae"], [2.5, 5.5])


def test_lead_bands_keep_residual_scales_separate():
    actual = np.array([9.0, 11.0, 9.0, 11.0, 6.0, 14.0, 6.0, 14.0])
    prediction = np.full(8, 10.0)
    leads = np.repeat([1, 8], 4)
    bands = ev.fit_lead_bands(actual, prediction, leads)
    lower, upper = ev.apply_lead_bands(bands, prediction, leads)
    np.testing.assert_allclose(lower, [9] * 4 + [6] * 4)
    np.testing.assert_allclose(upper, [11] * 4 + [14] * 4)
    table = ev.lead_band_summary(
        {
            "series": np.full(8, "s0"),
            "actual": actual,
            "lower": lower,
            "upper": upper,
            "weeks_ahead": leads,
        },
        "test",
    )
    overall = table["series"] == "all"
    assert table["weeks_ahead"][overall].tolist() == [0, 1, 8]
    assert table["rows"][overall].tolist() == [8, 4, 4]
    np.testing.assert_allclose(table["coverage"][overall], [1, 1, 1])
    np.testing.assert_allclose(table["band_width_pct"][overall], [0.5, 0.2, 0.8])


def test_saved_outputs_preserve_rows_leads_and_band_arithmetic(monkeypatch, tmp_path):
    import json

    from smolsmort.forecast.tables import read_table
    from smolsmort.forecast.worker import _write_result, _write_tables

    pytest.importorskip("duckdb")
    ws = build_workspace(monkeypatch)
    genome = candidate(ws)
    final = pipeline.finish(ws, genome)
    _write_result(tmp_path, ws.spec, {"best": genome.to_dict(), "stopped": "synthetic"}, final)
    _write_tables(tmp_path, ws.spec, final, prefix="winner")
    _write_tables(tmp_path, ws.spec, final, prefix="ensemble")
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["horizon_mode"] == "single" and saved["horizon"] == 8
    assert set(saved["lead_errors"]) == {str(lead) for lead in range(1, 9)}
    assert saved["band_summary"]["test"]["all"]["rows"] == len(final["test"]["actual"])
    for name in ("validation", "test", "forecast"):
        table = read_table(tmp_path / f"{name}.parquet")
        assert len(table["prediction"]) == len(final[name]["prediction"])
        assert {"origin", "weeks_ahead", "bucket"} <= table.keys()
        for lead in range(1, 9):
            rows = table["weeks_ahead"] == lead
            band = ev.Band(**saved["bands"][str(lead)])
            lower, upper = ev.apply_band(band, table["prediction"][rows], nonnegative=True)
            np.testing.assert_allclose(table["lower"][rows], lower)
            np.testing.assert_allclose(table["upper"][rows], upper)
    errors = read_table(tmp_path / "lead_errors.parquet")
    assert len(errors["weeks_ahead"]) == 16
    assert set(errors["phase"]) == {"validation", "test"}
    for prefix in ("winner", "ensemble"):
        assert (tmp_path / f"{prefix}_lead_errors.parquet").exists()
    bands = read_table(tmp_path / "band.parquet")
    assert set(bands["weeks_ahead"]) == set(range(9))


@pytest.mark.parametrize("transform", ["none", "log1p", "logdiff"])
def test_single_transforms_and_statistical_baseline_use_raw_outputs(monkeypatch, transform):
    ws = build_workspace(monkeypatch)
    genome = pipeline.make_genome([], "squared", {}, "snaive", transform)
    scored = pipeline.score(ws, genome)
    final = pipeline.finish(ws, genome, validation=scored.prediction)
    assert np.isfinite(final["model_error"])
    assert final["model_error"] == pytest.approx(final["baseline_error"])
    assert final["test"]["prediction"].min() > 10


def test_input_shares_and_spread_ignore_values_after_each_origin(monkeypatch):
    from dataclasses import replace

    from smolsmort.forecast.features import stacked_timeseries_features

    ws = build_workspace(monkeypatch)
    spec = replace(ws.spec, columns=(*ws.spec.columns, Column("channel", "input")))
    inputs = {
        SERIES: np.repeat(ws.panel[SERIES], 2),
        STEP: np.repeat(ws.panel[STEP], 2),
        "__column": np.full(2 * len(ws.panel[STEP]), "channel"),
        "__level": np.tile(["a", "b"], len(ws.panel[STEP])),
        "target": np.tile([3.0, 7.0], len(ws.panel[STEP])),
    }
    scaffold = {SERIES: np.array([], dtype=object), STEP: np.array([], dtype="datetime64[D]")}
    train = ws.panel[STEP] < ws.series_folds[0][0]
    original = stacked_timeseries_features(spec, ws.panel, scaffold, "week", train, inputs=inputs)
    origin = ws.series_folds[0][0] + np.timedelta64(42, "D")
    inputs["target"][inputs[STEP] > origin] = 1e12
    changed = stacked_timeseries_features(spec, ws.panel, scaffold, "week", train, inputs=inputs)
    rows = original.origin == origin
    np.testing.assert_allclose(changed.features.x[rows], original.features.x[rows], equal_nan=True)
    for name, expected in (("mix:channel:level:a_lag_0", 0.3), ("spread:channel:hhi_lag_0", 0.58)):
        column = original.features.names.index(name)
        np.testing.assert_allclose(original.features.x[rows, column], expected)


def test_candidate_search_and_ensemble_keep_stacked_rows(monkeypatch):
    from smolsmort.forecast.search import Budget, search

    ws = build_workspace(monkeypatch)
    ws.families = ["y@day:lag+0", "y@day:lag+1", "y@day:season", "weeks_ahead"]
    searched = search(
        ws, Budget(population=2, max_generations=1), family="ridge", defaults_only=True
    )
    genome = pipeline.genome_from_dict(searched["best"])
    scored = pipeline.score(ws, genome)
    alternate = pipeline.make_genome(genome.families, "squared", {"alpha": 10.0}, "ridge")
    other = pipeline.score(ws, alternate)
    prediction = (scored.prediction + other.prediction) / 2
    final = pipeline.finish(ws, genome, members=[genome, alternate], validation=prediction)
    np.testing.assert_allclose(final["validation"]["prediction"], prediction)
    assert len(final["test"]["prediction"]) == int(ws.series_masks[(1, 8)]["test"].sum())
    assert set(final["lead_errors"]) == {str(lead) for lead in range(1, 9)}
    expected = []
    for start, end in ws.series_folds:
        rows = (final["validation"]["origin"] >= start) & (final["validation"]["origin"] < end)
        expected.append(ev.wape(final["validation"]["actual"][rows], prediction[rows]))
    assert final["val_error"] == pytest.approx(np.mean(expected))
