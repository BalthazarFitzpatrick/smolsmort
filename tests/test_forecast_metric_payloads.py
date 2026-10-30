"""saved synthetic rows keep test, validation, winner and ensemble metrics separate"""

import json

import numpy as np
import pytest

from smolsmort.forecast import metrics, runs, views
from smolsmort.forecast.tables import write_table

# write_table needs duckdb, which ci does not install
pytest.importorskip("duckdb")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_predictions(path, actual, prediction, *, actual_column="actual"):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_table(
        path,
        {
            "step": np.array(["2025-01-06"] * len(actual), dtype="datetime64[D]"),
            "series": np.array(["synthetic"] * len(actual)),
            actual_column: np.array(actual, dtype=float),
            "prediction": np.array(prediction, dtype=float),
        },
    )


def test_saved_metrics_use_raw_rows_and_count_mape_exclusions(tmp_path):
    path = tmp_path / "test.parquet"
    write_predictions(path, [10, -10, 0, 20, np.nan], [12, -8, 4, 10, 2])
    values = metrics.read_test_metrics(path)
    assert values["wape"]["value"] == pytest.approx(18 / 40)
    assert values["mae"]["value"] == pytest.approx(18 / 4)
    assert values["mape"]["value"] == pytest.approx(0.35)
    assert values["mape"]["excluded_rows"] == 3
    assert all(item["scope"] == "test" for item in values.values())


def test_zero_actuals_and_undefined_predictions_remain_honest(tmp_path):
    path = tmp_path / "test.parquet"
    write_predictions(path, [0, 0], [2, 4], actual_column="truth")
    values = metrics.read_test_metrics(path)
    assert values["wape"]["value"] is None and "zero" in values["wape"]["reason"]
    assert values["mae"]["value"] == 3
    assert values["mape"]["value"] is None and values["mape"]["excluded_rows"] == 2
    write_predictions(path, [10, 20, 0], [12, np.nan, 0])
    values = metrics.read_test_metrics(path)
    for name in ("wape", "mae"):
        assert values[name]["value"] is None and "undefined predictions" in values[name]["reason"]
    assert values["mape"]["value"] == pytest.approx(0.2)
    assert values["mape"]["excluded_rows"] == 2


def test_prediction_cache_refreshes_after_atomic_replacement(tmp_path):
    path = tmp_path / "test.parquet"
    write_predictions(path, [10], [12])
    assert metrics.read_test_metrics(path)["mae"]["value"] == 2
    replacement = tmp_path / "replacement.parquet"
    write_predictions(replacement, [10, 20], [10, 20])
    replacement.replace(path)
    values = metrics.read_test_metrics(path)
    assert values["mae"]["value"] == 0
    values["mae"]["value"] = 99
    assert metrics.read_test_metrics(path)["mae"]["value"] == 0


@pytest.fixture
def saved_metrics_run(tmp_path):
    folder = tmp_path / "synthetic-run"
    request = {
        "version": 2,
        "spec": {"mode": "timeseries", "task": "regression"},
        "families": [{"name": name} for name in ("ridge", "xgboost", "snaive")],
        "time_budget_s": 60,
    }
    status = {
        "state": "done",
        "families": {name: {"state": "done"} for name in ("ridge", "xgboost", "snaive")},
    }
    result = {
        "val_error": 0.08,
        "verdict": {"accuracy": {"ratio": 0.8}},
        "winner": {"family": "ridge", "fitness": 0.1, "verdict": {"accuracy": {"ratio": 1.2}}},
        "ensemble": {"members": ["ridge", "xgboost"], "fitness": 0.08},
        "families": {
            "ridge": {"fitness": 0.1},
            "xgboost": {"fitness": 0.3, "verdict": {"accuracy": {"ratio": 1.5}}},
            "snaive": {"fitness": 0.5},
        },
    }
    write_json(folder / "request.json", request)
    write_json(folder / "status.json", status)
    write_json(folder / "result.json", result)
    write_json(folder / "recipe.json", {"family": "ensemble", "genome": {"family": "ridge"}})
    for name, saved in result["families"].items():
        write_json(
            folder / "families" / name / "leaderboard.json",
            [{"fitness": saved["fitness"], "metrics": {"wape": 0.11, "mae": 3}}],
        )
    write_predictions(folder / "winner_test.parquet", [10, 20], [12, 24])
    write_predictions(folder / "test.parquet", [10, 20], [11, 22])
    write_predictions(folder / "families/xgboost/test.parquet", [10, 20], [15, 30])
    return folder, request, status


def test_every_payload_keeps_family_and_candidate_scopes_separate(saved_metrics_run):
    folder, request, status = saved_metrics_run
    state = views.multi_run_state(folder, request, status)
    details = views.result_details(folder, request, {"model": "winner"})
    assert state["metric_registry"] == details["metric_registry"]
    assert state["error_metrics"]["wape"]["value"] == pytest.approx(0.1)
    assert state["error_metrics"]["validation_error"]["value"] == 0.08
    assert details["error_metrics"]["wape"]["value"] == pytest.approx(0.2)
    assert details["winner"]["error_metrics"] == details["error_metrics"]
    assert details["ensemble"]["error_metrics"] == state["error_metrics"]
    for name, expected in (("ridge", 0.2), ("xgboost", 0.5), ("snaive", None)):
        row = state["families"][name]
        values = row["error_metrics"]
        assert values == row["tile"]["error_metrics"] == details["families"][name]["error_metrics"]
        assert values["wape"]["value"] == expected
        if expected is None:
            assert values["wape"]["reason"]
    candidate = state["leaderboard"][0]["error_metrics"]
    assert candidate["wape"]["value"] == 0.11 and candidate["wape"]["scope"] == "validation"
    assert candidate["mape"]["value"] is None
    listed = runs.list_runs(folder.parent)[0]
    assert listed["error_metrics"] == state["error_metrics"]
    assert listed["metric_registry"] == state["metric_registry"]
    assert all("tile" not in row for row in listed["families"].values())


def test_missing_winner_predictions_never_borrow_ensemble_metrics(saved_metrics_run):
    folder, request, status = saved_metrics_run
    (folder / "winner_test.parquet").unlink()
    tile = views.family_tiles(folder, request, status)["ridge"]
    assert tile["error_metrics"]["wape"]["value"] is None
    assert tile["error_metrics"]["test_ratio"]["value"] == 1.2
    assert views.result_details(folder, request, {})["error_metrics"]["wape"]["value"] == 0.1


def test_missing_ensemble_artifacts_never_borrow_winner_metrics(saved_metrics_run):
    folder, request, _ = saved_metrics_run
    result = json.loads((folder / "result.json").read_text())
    result["ensemble"].pop("fitness")
    result.pop("val_error")
    write_json(folder / "result.json", result)
    write_json(folder / "recipe.json", {"family": "ridge", "genome": {"family": "ridge"}})
    payload = views.result_details(folder, request, {"model": "ensemble"})
    for item in payload["error_metrics"].values():
        assert item["value"] is None and item["reason"]
    assert payload["winner"]["error_metrics"]["wape"]["value"] == 0.2


def test_legacy_run_list_and_leaderboard_have_the_same_registry(tmp_path):
    folder = tmp_path / "legacy"
    write_json(folder / "request.json", {"spec": {"mode": "timeseries", "task": "regression"}})
    write_json(folder / "status.json", {"state": "done"})
    write_json(folder / "result.json", {"val_error": 0.3, "verdict": {"accuracy": {"ratio": 0.9}}})
    write_json(folder / "leaderboard.json", [{"fitness": 0.3, "metrics": {"mae": 2}}])
    write_predictions(folder / "test.parquet", [10], [11])
    run = runs.list_runs(tmp_path)[0]
    assert run["error_metrics"]["wape"]["value"] == 0.1
    assert run["error_metrics"]["validation_error"]["value"] == 0.3
    assert run["metric_registry"] == metrics.metric_registry({})
    assert views.read_leaderboard(folder)[0]["error_metrics"]["mae"]["scope"] == "validation"
    (folder / "leaderboard.json").unlink()
    request = views.read_request(folder)
    tile = views.family_tiles(folder, request, {"state": "done"})["xgboost"]
    assert tile["error_metrics"]["validation_error"]["value"] == 0.3
    write_json(folder / "leaderboard.json", [{"fitness": 0.3, "metrics": {"mae": 2}}])
    write_json(folder / "result.json", {"verdict": {"accuracy": {"ratio": 0.9}}})
    assert runs.list_runs(tmp_path)[0]["error_metrics"]["validation_error"]["value"] == 0.3
    (folder / "leaderboard.json").write_text("[{", encoding="utf-8")
    assert runs.list_runs(tmp_path)[0]["error_metrics"]["validation_error"]["value"] is None


def test_partial_predictions_and_missing_columns_do_not_break_polling(tmp_path):
    path = tmp_path / "test.parquet"
    path.write_bytes(b"incomplete")
    values = metrics.read_test_metrics(path)
    assert all(value["value"] is None and value["reason"] for value in values.values())
    write_table(path, {"prediction": np.array([2.0])})
    values = metrics.read_test_metrics(path)
    assert all("no actual" in value["reason"] for value in values.values())
    json.dumps(values, allow_nan=False)
