"""seasonal holdout: rolling origins a quarter apart, no row read at or after its origin"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from smolsmort.forecast import evaluate, pipeline
from smolsmort.forecast.families import get_family
from smolsmort.forecast.pipeline import finish, load_workspace, make_genome, score
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec
from smolsmort.forecast.splits import SplitError, seasonal_cuts
from smolsmort.forecast.worker import _write_result


def build_workspace(monkeypatch, holdout, weeks=156, horizon=4):
    positions = np.arange(weeks)
    steps = np.datetime64("2023-01-02") + positions * np.timedelta64(7, "D")
    values = 20 + positions / 5 + 3 * np.sin(2 * np.pi * positions / 13)
    panel = {
        SERIES: np.repeat(["a", "b"], weeks),
        STEP: np.tile(steps, 2),
        "y": np.concatenate([values, 2 * values + 5]),
    }
    future = steps[-1] + np.arange(1, horizon + 1) * np.timedelta64(7, "D")
    scaffold = {SERIES: np.repeat(["a", "b"], horizon), STEP: np.tile(future, 2)}
    tables = {"panel.parquet": panel, "scaffold.parquet": scaffold}
    monkeypatch.setattr(pipeline, "read_table", lambda path, **kwargs: tables[path.name])
    spec = PrepSpec(
        "synthetic.csv",
        "timeseries",
        "regression",
        (Column("day", "time"), Column("y", "target", "sum")),
        step="week",
        horizon=horizon,
    )
    ws = load_workspace(spec, Path("synthetic"), {"step": "week"}, holdout=holdout)
    ws.families = ["y@day:lag+0", "y@day:lag+1"]
    return ws


@pytest.fixture
def seasonal(monkeypatch):
    return build_workspace(monkeypatch, "seasonal")


def candidate(ws):
    return make_genome(ws.families, "squared", get_family("ridge").defaults(), "ridge")


def test_origins_land_one_per_season_and_end_inside_the_history():
    cuts = seasonal_cuts(156, 4)
    assert len(cuts.val) == len(cuts.test) == 4
    starts = cuts.val + cuts.test
    assert np.diff(starts).tolist() == [13] * 7
    # the newest horizon ends exactly at the history end
    assert cuts.test[-1] + cuts.horizon == 156
    # four origins a quarter apart: each set sees every quarter of the year once
    assert len({(start // 13) % 4 for start in cuts.val}) == 4
    assert len({(start // 13) % 4 for start in cuts.test}) == 4
    assert max(cuts.val) < min(cuts.test)
    assert not cuts.fell_back


def test_monthly_origins_are_three_steps_apart():
    cuts = seasonal_cuts(80, 3, unit="month")
    assert np.diff(cuts.val + cuts.test).tolist() == [3] * 7


def test_a_long_horizon_widens_the_stride_so_windows_never_overlap():
    cuts = seasonal_cuts(400, 26)
    assert cuts.stride == 26
    assert min(np.diff(cuts.val + cuts.test)) >= cuts.horizon


@pytest.mark.parametrize(("steps", "origins"), [(156, 8), (100, 6), (70, 4)])
def test_short_history_falls_back_to_fewer_origins(steps, origins):
    cuts = seasonal_cuts(steps, 4)
    assert len(cuts.val) + len(cuts.test) == origins
    assert len(cuts.val) == len(cuts.test) >= 2
    assert cuts.val[0] >= 26
    assert cuts.fell_back == (origins < 8)


def test_history_too_short_for_four_origins_is_refused():
    with pytest.raises(SplitError, match="cannot hold"):
        seasonal_cuts(50, 4)


def test_no_training_row_reaches_its_origin(seasonal):
    ws = seasonal
    assert len(ws.series_folds) == len(ws.test_folds) == 4
    for bucket, frame in ws.frames.items():
        windows = [
            *zip(ws.series_folds, pipeline._validation_masks(ws, bucket), strict=True),
            *zip(ws.test_folds, pipeline._test_windows(ws, bucket), strict=True),
        ]
        for (start, end), (fit, evaluate_rows) in windows:
            assert fit.any() and evaluate_rows.any()
            assert np.all(frame.step[fit] < start)
            assert np.all((frame.step[evaluate_rows] >= start) & (frame.step[evaluate_rows] < end))
            # a forecast is made before its origin: the row's own origin is a step earlier
            assert np.all(frame.origin[evaluate_rows] < start)
        masks = ws.series_masks[bucket]
        assert not np.any(masks["val"] & masks["test"])
        assert frame.step[masks["val"]].max() < frame.step[masks["test"]].min()
        # the first validation fit sees only steps before the first origin
        assert np.all(frame.step[masks["train"]] < ws.series_folds[0][0])


def test_block_holdout_is_the_default_and_unchanged(monkeypatch, seasonal):
    block = build_workspace(monkeypatch, "block")
    default = build_workspace(monkeypatch, "block")
    assert block.holdout_info is None and block.test_folds == ()
    assert len(block.series_folds) == 3
    for bucket in block.frames:
        assert block.series_masks[bucket].keys() == {"train", "val", "test", "future"}
        for name, mask in block.series_masks[bucket].items():
            np.testing.assert_array_equal(mask, default.series_masks[bucket][name])
    assert block.series_folds != seasonal.series_folds


def test_seasonal_score_is_the_mean_over_validation_origins(seasonal):
    scored = score(seasonal, candidate(seasonal))
    assert scored.metrics["fold_count"] == 4
    assert scored.fitness == pytest.approx(np.mean(scored.metrics["fold_scores"]))


def test_seasonal_finish_refits_per_test_origin_and_records_bands(seasonal, tmp_path):
    ws = seasonal
    fits = []
    original = pipeline._timeseries_fit

    def spy(ws_, genome, frame, rows, nthread):
        fits.append(frame.step[rows].max())
        return original(ws_, genome, frame, rows, nthread)

    pipeline._timeseries_fit = spy
    try:
        final = finish(ws, candidate(ws))
    finally:
        pipeline._timeseries_fit = original
    starts = [start for start, _ in ws.test_folds]
    # every test origin got a model that never saw its own window
    for start in starts:
        assert any(last < start for last in fits)
    assert final["holdout"]["holdout"] == "seasonal"
    assert final["holdout"]["test_origins"] == [str(start) for start in starts]
    test = final["test"]
    assert {"lower", "upper"} <= test.keys()
    assert np.all(test["lower"] <= test["upper"])
    steps = test["step"].astype("datetime64[D]")
    assert all(any(start <= day < end for start, end in ws.test_folds) for day in steps[::7])
    assert set(final["band_table"]["series"]) == {"all", "a", "b"}
    assert "band_width_pct" in final["forecast"]
    spec = ws.spec
    _write_result(tmp_path, spec, {"best": candidate(ws).to_dict(), "stopped": "x"}, final)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["holdout"]["fell_back"] is False
    assert result["band_summary"]["test"]["all"]["rows"] == len(test["actual"])
    assert (tmp_path / "band.parquet").exists()
    checks = {check["check"] for check in final["verdict"]["checks"]}
    assert "band width vs actuals" in checks


def test_band_summary_matches_hand_computed_widths():
    series = np.array(["a", "a", "b", "b"])
    actual = np.array([10.0, 20.0, 0.0, 40.0])
    lower = np.array([8.0, 15.0, 0.0, 30.0])
    upper = np.array([12.0, 25.0, 5.0, 50.0])
    table = evaluate.band_summary(series, actual, lower, upper)
    assert table["series"].tolist() == ["all", "a", "b"]
    assert table["rows"].tolist() == [4, 2, 2]
    # widths 4, 10, 5, 20 over actuals 10, 20, 0, 40
    np.testing.assert_allclose(table["band_width_pct"], [39 / 70, 14 / 30, 25 / 40])
    # rows with an actual of zero are left out of the median: 0.4, 0.5, 0.5
    np.testing.assert_allclose(table["median_width_pct"], [0.5, 0.45, 0.5])
    # a: both inside; b: 0 inside [0, 5], 40 inside [30, 50]
    np.testing.assert_allclose(table["coverage"], [1.0, 1.0, 1.0])


def test_request_validates_holdout():
    base = {"version": 2, "time_budget_s": 2, "families": [{"name": "snaive"}]}
    assert parse_request(base)["holdout"] == "block"
    assert parse_request({**base, "holdout": "seasonal"})["holdout"] == "seasonal"
    with pytest.raises(ValueError, match="holdout must be block or seasonal"):
        parse_request({**base, "holdout": "yearly"})


def test_seasonal_finish_runs_a_raw_series_family(seasonal):
    genome = make_genome([], "squared", get_family("snaive").defaults(), "snaive")
    scored = score(seasonal, genome)
    final = finish(seasonal, genome, validation=scored.prediction)
    assert scored.metrics["fold_count"] == 4
    assert np.isfinite(final["model_error"]) and len(final["test"]["actual"])
