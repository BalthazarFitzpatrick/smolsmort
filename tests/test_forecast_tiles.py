"""tile payloads use saved synthetic artifacts without fitting models"""

import json

import numpy as np
import pytest

from smolsmort.forecast import views
from smolsmort.forecast.tables import write_table

# write_table needs duckdb, which ci does not install
pytest.importorskip("duckdb")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_predictions(path, prediction=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_table(
        path,
        {
            "step": np.array(["2025-01-06", "2025-01-06", "2025-01-13"], dtype="datetime64[D]"),
            "series": np.array(["a", "b", "a"]),
            "actual": np.array([4.0, 6.0, 12.0]),
            "prediction": np.array([prediction, prediction + 1, prediction + 2], dtype=float),
            "lower": np.array([1.0, 2.0, 3.0]),
            "upper": np.array([9.0, 10.0, 11.0]),
        },
    )


@pytest.fixture
def saved_run(tmp_path):
    request = {
        "version": 2,
        "spec": {"mode": "timeseries", "task": "regression"},
        "families": [{"name": name} for name in ("ridge", "xgboost", "snaive")],
        "time_budget_s": 600,
    }
    states = {"ridge": "done", "xgboost": "done", "snaive": "pruned"}
    status = {
        "state": "done",
        "families": {name: {"state": state} for name, state in states.items()},
    }
    for index, (name, state) in enumerate(states.items()):
        genome = {"family": name, "params": {"alpha": 1} if name == "ridge" else {}}
        target = tmp_path / "families" / name
        write_json(target / "status.json", {"state": state, "elapsed": 60 + index, "generation": 4})
        write_json(
            target / "leaderboard.json",
            [
                {
                    "fitness": 0.2 + index / 10,
                    "genome": genome,
                    "metrics": {"fold_scores": [0.2, 0.3, 0.1]},
                }
            ],
        )
        write_json(target / "winner.json", genome)
    write_json(tmp_path / "request.json", request)
    write_json(tmp_path / "status.json", status)
    write_json(tmp_path / "recipe.json", {"family": "ensemble", "genome": {"family": "ridge"}})
    write_json(
        tmp_path / "result.json",
        {
            "model_error": 0.1,
            "verdict": {"accuracy": {"ratio": 0.5, "baseline_error": 0.2}},
            "winner": {
                "family": "ridge",
                "fitness": 0.2,
                "model_error": 0.182,
                "fold_scores": [0.21, 0.19, 0.2],
                "verdict": {"accuracy": {"ratio": 0.71, "metric": "wape", "baseline_error": 0.256}},
            },
            "ensemble": {"members": ["ridge", "xgboost"]},
            "families": {
                name: {"state": state, "fitness": 0.2 + index / 10}
                for index, (name, state) in enumerate(states.items())
            },
        },
    )
    write_json(
        tmp_path / "families/ridge/importance.json",
        {"method": "coefficient", "features": [{"feature": "lag_1", "importance": 1.0}]},
    )
    write_predictions(tmp_path / "winner_test.parquet")
    write_predictions(tmp_path / "test.parquet", prediction=100)
    return tmp_path, request, status


def test_tiles_keep_family_winner_and_ensemble_test_artifacts_separate(saved_run):
    folder, request, status = saved_run
    state = views.multi_run_state(folder, request, status)
    tile = state["families"]["ridge"]["tile"]
    assert tile["test_ratio"] == 0.71
    assert tile["test_error"] == 0.182 and tile["test_error_metric"] == "wape"
    assert tile["test_naive_wape"] == 0.256
    assert tile["chart"] == [
        {"step": "2025-01-06", "actual": 10.0, "prediction": 21.0, "lower": 3.0, "upper": 19.0},
        {"step": "2025-01-13", "actual": 12.0, "prediction": 12.0, "lower": 3.0, "upper": 11.0},
    ]
    assert tile["importance"]["features"] == [{"feature": "lag_1", "importance": 1.0}]
    assert tile["genome"]["params"] == {"alpha": 1}
    assert tile["elapsed"] == 60 and tile["generation"] == 4
    for name in ("xgboost", "snaive"):
        loser = state["families"][name]["tile"]
        assert loser["test_ratio"] is None and loser["test_error"] is None
        assert loser["test_naive_wape"] is None
        assert loser["chart"] == [] and loser["chart_reason"]
    assert tile["folds"] == [
        {"label": "fold 1", "error": 0.2, "ratio": None},
        {"label": "fold 2", "error": 0.3, "ratio": None},
        {"label": "fold 3", "error": 0.1, "ratio": None},
    ]
    assert tile["folds_note"] == "wape per validation window; mark = naive on test"


@pytest.mark.parametrize("baseline", [None, 0, 1.4])
def test_tiles_keep_each_familys_saved_test_wape_and_baseline(saved_run, baseline):
    folder, request, status = saved_run
    result = json.loads((folder / "result.json").read_text())
    result["families"]["xgboost"].update(
        {
            "test_error": 0.24,
            "verdict": {"accuracy": {"metric": "wape", "baseline_error": baseline}},
        }
    )
    tiles = views.family_tiles(folder, request, status, result=result)
    assert tiles["xgboost"]["test_error"] == 0.24
    assert tiles["xgboost"]["test_naive_wape"] == baseline
    assert tiles["ridge"]["test_naive_wape"] == 0.256
    assert tiles["snaive"]["test_naive_wape"] is None


def test_tiles_fall_back_to_own_saved_folds_and_baseline(saved_run):
    folder, request, status = saved_run
    result = json.loads((folder / "result.json").read_text())
    result["families"]["xgboost"].update({"fold_scores": [0.6, 0.4], "baseline_error": 0.7})
    write_json(folder / "families/xgboost/leaderboard.json", [])
    tile = views.family_tiles(folder, request, status, result=result)["xgboost"]
    assert [fold["error"] for fold in tile["folds"]] == [0.6, 0.4]
    assert tile["test_naive_wape"] == 0.7


def test_tiles_never_label_a_different_metric_as_naive_wape(saved_run):
    folder, request, status = saved_run
    result = json.loads((folder / "result.json").read_text())
    result["winner"]["verdict"]["accuracy"].update({"metric": "ibs", "baseline_error": 0.2})
    tile = views.family_tiles(folder, request, status, result=result)["ridge"]
    assert tile["test_naive_wape"] is None


def test_tiles_label_seasonal_origins_and_refresh_on_poll(saved_run):
    folder, request, status = saved_run
    result = json.loads((folder / "result.json").read_text())
    result["holdout"] = {"validation_origins": ["2023-12-04", "2024-01-01", "2024-02-05"]}
    write_json(folder / "result.json", result)
    first = views.multi_run_state(folder, request, status)
    assert [fold["label"] for fold in first["families"]["ridge"]["tile"]["folds"]] == result[
        "holdout"
    ]["validation_origins"]
    write_predictions(folder / "winner_test.parquet", prediction=20)
    next_state = views.multi_run_state(
        folder, request, status, since_generation=first["generation"]
    )
    assert "eval_history" not in next_state
    assert next_state["families"]["ridge"]["tile"]["chart"][0]["prediction"] == 41


def test_tile_chart_averages_horizon_buckets_before_summing_series(tmp_path):
    path = tmp_path / "test.parquet"
    write_table(
        path,
        {
            "step": np.array(["2025-01-06"] * 4, dtype="datetime64[D]"),
            "series": np.array(["a", "a", "b", "b"]),
            "bucket": np.array(["1-4", "5-13", "1-4", "5-13"]),
            "actual": np.array([4.0, 4.0, 6.0, 6.0]),
            "prediction": np.array([2.0, 6.0, 9.0, np.nan]),
            "lower": np.array([1.0, 3.0, 8.0, np.nan]),
            "upper": np.array([5.0, 7.0, 10.0, np.nan]),
        },
    )
    assert views._read_tile_chart(path) == [
        {"step": "2025-01-06", "actual": 10.0, "prediction": 13.0, "lower": 10.0, "upper": 16.0}
    ]


def test_tiles_never_substitute_ensemble_for_missing_winner_chart(saved_run):
    folder, request, status = saved_run
    (folder / "winner_test.parquet").unlink()
    tile = views.family_tiles(folder, request, status)["ridge"]
    assert tile["chart"] == []
    assert tile["test_ratio"] == 0.71


def test_summary_poll_skips_family_chart_reads(saved_run, monkeypatch):
    folder, request, status = saved_run

    def reject_chart(_path):
        pytest.fail("run summaries must not read tile charts")

    monkeypatch.setattr(views, "_read_tile_chart", reject_chart)
    summary = views.multi_run_state(
        folder, request, status, since_generation=2**63 - 1, since_events=2**63 - 1
    )
    assert all("tile" not in family for family in summary["families"].values())


def test_tiles_read_family_score_without_requiring_a_test_artifact(saved_run):
    folder, request, status = saved_run
    result = json.loads((folder / "result.json").read_text())
    result["families"]["xgboost"]["verdict"] = {"accuracy": {"ratio": 1.1, "model_error": 0.3}}
    write_json(folder / "result.json", result)
    tile = views.family_tiles(folder, request, status)["xgboost"]
    assert tile["test_ratio"] == 1.1 and tile["chart"] == []
    write_predictions(folder / "families/xgboost/test.parquet", prediction=30)
    tile = views.family_tiles(folder, request, status)["xgboost"]
    assert tile["test_ratio"] == 1.1 and tile["test_error"] == 0.3
    assert tile["chart"][0]["prediction"] == 61


def test_tiles_use_default_winner_verdict_for_older_partial_summaries(saved_run):
    folder, request, status = saved_run
    write_json(folder / "recipe.json", {"family": "ridge", "genome": {"family": "ridge"}})
    result = json.loads((folder / "result.json").read_text())
    del result["winner"]["verdict"]
    write_json(folder / "result.json", result)
    tile = views.family_tiles(folder, request, status)["ridge"]
    assert tile["test_ratio"] == 0.5


@pytest.mark.parametrize("family_state", ["running", "paused"])
def test_running_tiles_preserve_candidate_metrics_and_active_slice_time(
    tmp_path, monkeypatch, family_state
):
    request = {"version": 2, "families": [{"name": "ridge"}], "spec": {"mode": "timeseries"}}
    status = {
        "state": family_state,
        "families": {
            "ridge": {
                "state": family_state,
                "generation": 5,
                "slice_history": [
                    {"wall_s": 60},
                    {"started_at": 100, **({"ended_at": 120} if family_state == "paused" else {})},
                ],
            }
        },
    }
    write_json(tmp_path / "families/ridge/status.json", {"state": family_state, "elapsed": 10})
    write_json(tmp_path / "families/ridge/winner.json", {"params": {"alpha": 1}})
    write_json(
        tmp_path / "families/ridge/leaderboard.json",
        [
            {
                "fitness": 0.3,
                "genome": {"params": {"alpha": 2}},
                "metrics": {"fold_scores": [0.2, 0.4]},
            }
        ],
    )
    monkeypatch.setattr(views.time, "time", lambda: 120)
    tile = views.family_tiles(tmp_path, request, status)["ridge"]
    assert tile["state"] == family_state and tile["generation"] == 5
    assert tile["elapsed"] == 80 and tile["validation_error"] == 0.3
    assert tile["genome"]["params"] == {"alpha": 2}
    assert tile["test_ratio"] is None and tile["chart"] == []
    assert [fold["error"] for fold in tile["folds"]] == [0.2, 0.4]


def test_legacy_result_details_expose_a_single_tile_with_metric(tmp_path):
    request = {"spec": {"mode": "entity", "task": "regression"}}
    write_json(tmp_path / "status.json", {"state": "done", "generation": 3, "elapsed": 10})
    write_json(tmp_path / "recipe.json", {"genome": {"family": "xgboost"}})
    write_json(
        tmp_path / "result.json",
        {"model_error": 0.2, "verdict": {"accuracy": {"ratio": 0.8, "metric": "ibs"}}},
    )
    result = views.result_details(tmp_path, request, {})
    assert list(result["tiles"]) == ["xgboost"]
    tile = result["tiles"]["xgboost"]
    assert tile["test_error_metric"] == "ibs" and tile["test_ratio"] == 0.8
    assert tile["chart"] == [] and tile["folds"] == []


def test_partial_parquet_does_not_break_polling(saved_run):
    folder, request, status = saved_run
    (folder / "winner_test.parquet").write_bytes(b"partial")
    tiles = views.family_tiles(folder, request, status)
    assert tiles["ridge"]["chart"] == []
    json.dumps(tiles, allow_nan=False)


def test_nonrecord_slice_history_does_not_break_family_poll(saved_run):
    folder, request, status = saved_run
    status["families"]["ridge"]["slice_history"] = ["private history"]
    state = views.multi_run_state(folder, request, status)
    assert state["families"]["ridge"]["tile"]["elapsed"] == 60
    assert "slice_history" not in state["families"]["ridge"]
