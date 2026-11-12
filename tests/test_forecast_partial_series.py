"""short-series gaps stay local through scores and available-member blends"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_forecast_family_stats import workspace as workspace

from smolsmort.forecast import evaluate, pipeline, tiles
from smolsmort.forecast.horizon_view import blend_rows
from smolsmort.forecast.metrics import read_test_metrics
from smolsmort.forecast.plan_sets import pick_ensemble
from smolsmort.forecast.tables import read_table, write_table


def test_partial_scores_exclude_missing_predictions_and_keep_period_alignment():
    truth = np.array([10.0, 100.0, 20.0])
    prediction = np.array([12.0, np.nan, 18.0])
    ws = SimpleNamespace(spec=SimpleNamespace(task="regression"), mode="timeseries")
    scored = pipeline._fitness(ws, prediction, truth, np.array([1, 2, 3]))
    assert scored.fitness == pytest.approx(4 / 30)
    assert scored.metrics["wape"] == pytest.approx(4 / 30)
    assert scored.metrics["mae"] == 2
    np.testing.assert_allclose(scored.contrib, [2 / 30, 0, 2 / 30])
    assert np.isnan(evaluate.wape([100], [np.nan]))
    assert np.isnan(evaluate.mae([100], [np.nan]))


def test_empty_coverage_fold_does_not_poison_other_fold_scores():
    days = np.arange("2025-01-01", "2025-01-05", dtype="datetime64[D]")
    ws = SimpleNamespace(
        spec=SimpleNamespace(task="regression"),
        mode="timeseries",
        series_folds=[(days[0], days[2]), (days[2], days[-1] + np.timedelta64(1, "D"))],
    )
    scored = pipeline._fitness(
        ws, np.array([11, 9, np.nan, np.nan]), np.full(4, 10), days.astype(int)
    )
    assert scored.fitness == pytest.approx(0.1)
    assert scored.metrics["fold_count"] == 1
    np.testing.assert_allclose(scored.contrib, [0.05, 0.05, 0, 0])


def test_mean_and_horizon_blend_keep_other_available_predictions():
    mean = evaluate.mean_predictions([[10, np.nan, np.nan], [20, 40, np.nan]])
    np.testing.assert_allclose(mean, [15, 40, np.nan], equal_nan=True)
    table = {
        "series": np.array(["long", "long", "short", "short"]),
        "step": np.array([1, 1, 1, 1]),
        "weeks_ahead": np.array([1, 2, 1, 2]),
        "prediction": np.array([10, np.nan, np.nan, np.nan]),
    }
    blended = blend_rows(table, {1: 1, 2: 8})
    np.testing.assert_allclose(blended["prediction"], [10, np.nan], equal_nan=True)


def test_saved_metrics_and_coverage_exclude_short_series(tmp_path):
    pytest.importorskip("duckdb")
    table = {
        "series": np.array(["long", "long", "short"]),
        "step": np.array([1, 2, 1]),
        "actual": np.array([10, 20, 100]),
        "prediction": np.array([12, 18, np.nan]),
    }
    path = write_table(tmp_path / "test.parquet", table)
    metrics = read_test_metrics(path)
    assert metrics["wape"]["value"] == pytest.approx(4 / 30)
    assert metrics["mae"]["value"] == 2
    assert metrics["wape"]["excluded_rows"] == 1
    assert tiles.table_coverage(table) == {"covered": 1, "total": 2}
    np.testing.assert_allclose(read_table(path)["prediction"], table["prediction"], equal_nan=True)


def test_set_ensemble_keeps_short_series_and_uses_its_other_member(tmp_path):
    pytest.importorskip("duckdb")
    for family, prediction in (("stats", [10, np.nan]), ("ridge", [12, 20])):
        folder = tmp_path / family
        folder.mkdir()
        for split in ("validation", "test", "forecast"):
            write_table(
                folder / f"winner_{split}.parquet",
                {
                    "series": np.array(["long", "short"]),
                    "step": np.array([1, 1]),
                    "actual": np.array([11, 20]),
                    "prediction": np.array(prediction),
                },
            )
    selected, parts = pick_ensemble(tmp_path, {"stats": "stats", "ridge": "ridge"})
    assert selected == ("stats", "ridge")
    for table in parts.values():
        assert list(table["series"]) == ["long", "short"]
        np.testing.assert_allclose(table["prediction"], [11, 20])


def test_tile_ensemble_scores_other_family_on_short_series():
    tables = {}
    for name, prediction in (("stats", [10, np.nan]), ("ridge", [12, 20])):
        table = {
            "series": np.array(["long", "short"]),
            "step": np.array([1, 1]),
            "actual": np.array([11, 20]),
            "prediction": np.array(prediction),
        }
        tables[name] = {"validation": table, "test": table}
    scored = tiles.score_ensembles([{"tile_id": name} for name in tables], tables)
    assert scored[0]["metrics"]["wape"] == {"validation": 0, "test": 0}


def test_tile_ensemble_ratio_uses_the_same_covered_rows():
    table = {
        "series": np.array(["long", "short"]),
        "step": np.array([1, 1]),
        "actual": np.array([10, 100]),
        "prediction": np.array([11, np.nan]),
        "naive_prediction": np.array([12, 0]),
    }
    tables = {name: {"validation": table, "test": table} for name in ("ets", "snaive")}
    scored = tiles.score_ensembles([{"tile_id": name} for name in tables], tables)
    assert scored[0]["error_metrics"]["test_ratio"]["value"] == pytest.approx(0.5)


def test_missing_baseline_keeps_model_score_but_disables_ensemble_ratio():
    table = {
        "series": np.array(["long", "short"]),
        "step": np.array([1, 1]),
        "actual": np.array([10, 100]),
        "prediction": np.array([11, 101]),
        "naive_prediction": np.array([12, np.nan]),
    }
    tables = {name: {"validation": table, "test": table} for name in ("ets", "ridge")}
    scored = tiles.score_ensembles([{"tile_id": name} for name in tables], tables)
    assert scored[0]["metrics"]["wape"]["test"] == pytest.approx(2 / 110)
    assert scored[0]["error_metrics"]["test_ratio"]["value"] is None


def test_missing_baseline_disables_lead_ratio_without_hiding_model_error():
    errors, _ = evaluate.lead_errors(
        np.array([10, 100]),
        np.array([11, 101]),
        np.array([12, np.nan]),
        np.array([1, 1]),
        "test",
    )
    assert errors["1"]["model"] == pytest.approx(2 / 110)
    assert np.isnan(errors["1"]["naive"])


def test_finish_requires_baseline_for_every_covered_row(workspace, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "_timeseries_predict",
        lambda ws, genome, frame, fit_rows, eval_rows, *args, **kwargs: np.ones(
            int(eval_rows.sum())
        ),
    )

    def baseline(frame, rows, *args):
        values = np.ones(int(rows.sum()))
        values[0] = np.nan
        return values

    monkeypatch.setattr(pipeline, "_timeseries_baseline", baseline)
    genome = pipeline.make_genome([], "squared", {}, family="snaive")
    result = pipeline.finish(workspace, genome)
    assert np.isfinite(result["model_error"])
    assert np.isnan(result["baseline_error"])
    assert np.isnan(result["verdict"]["accuracy"]["ratio"])
    assert "baseline comparison unavailable" in result["verdict"]["summary"]
    assert "naive baseline n/a" in result["verdict"]["accuracy"]["text"]


def test_set_requires_baseline_for_every_covered_row(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    from smolsmort.forecast.plan_sets import create_set, update_set

    folder = tmp_path / "ridge"
    folder.mkdir()
    for split in ("validation", "test", "forecast"):
        write_table(
            folder / f"winner_{split}.parquet",
            {
                "series": np.array(["long", "short"]),
                "step": np.array([1, 1]),
                "actual": np.array([10, 100]),
                "prediction": np.array([11, 101]),
                "naive_prediction": np.array([12, np.nan]),
            },
        )
    monkeypatch.setattr(tiles, "ensemble_rows", lambda folder: [])
    monkeypatch.setattr(tiles, "write_ensemble_views", lambda folder, rows: None)
    create_set(tmp_path, "combined", "test", {}, ["ridge"])
    result_path = update_set(tmp_path, "combined", {"ridge": "ridge"}) / "result.json"
    result = json.loads(result_path.read_text())
    assert result["ensemble"]["model_error"] == pytest.approx(2 / 110)
    assert result["ensemble"]["baseline_error"] is None
    assert result["ensemble"]["verdict"]["accuracy"]["ratio"] is None
