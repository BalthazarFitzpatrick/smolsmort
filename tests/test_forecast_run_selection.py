"""saved validation chooses the run headline and initial results model"""

import json

import numpy as np
import pytest

from smolsmort.forecast.runs import list_runs
from smolsmort.forecast.tables import write_table
from smolsmort.forecast.views import _prediction_path, _select_model, result_details


@pytest.mark.parametrize(
    ("ensemble_fitness", "recipe_family", "selected"),
    [
        (0.09, "snaive", "ensemble"),
        (0.1, "snaive", "winner"),
        (0.11, "snaive", "winner"),
        (None, "snaive", "winner"),
        (0.11, "ensemble", "ensemble"),
    ],
)
@pytest.mark.parametrize("nested_winner", ["absent", "family", "complete"])
def test_saved_selection(tmp_path, ensemble_fitness, recipe_family, selected, nested_winner):
    run = tmp_path / "run"
    run.mkdir()
    request = {"spec": {"mode": "timeseries", "task": "regression", "columns": []}}
    winner = {
        "family": "snaive",
        "fitness": 0.1,
        "model_error": 0.2,
        "verdict": {"accuracy": {"ratio": 1.0}},
    }
    ensemble = {
        "fitness": ensemble_fitness,
        "model_error": 0.18,
        "members": ["lightgbm", "snaive"],
        "verdict": {"accuracy": {"ratio": 0.9}},
    }
    result = {
        "val_error": 0.1,
        "model_error": 0.2,
        "verdict": winner["verdict"],
        "ensemble": ensemble,
    }
    if nested_winner == "complete":
        result["winner"] = winner
    elif nested_winner == "family":
        result["winner"] = {"family": "snaive"}
    for name, data in {
        "request": request,
        "result": result,
        "recipe": {"family": recipe_family},
        "status": {"state": "done"},
    }.items():
        (run / f"{name}.json").write_text(json.dumps(data))
    for model, prediction in [("winner", 8.0), ("ensemble", 8.2)]:
        write_table(
            run / f"{model}_test.parquet",
            {"actual": np.array([10.0]), "prediction": np.array([prediction])},
        )
    summary = list_runs(tmp_path)[0]
    details = result_details(run, request, {})
    assert details["model"] == selected
    assert _select_model(run, result, {})["default_model"] == selected
    assert (
        details["family"]
        == summary["selected_family"]
        == ("ensemble" if selected == "ensemble" else "snaive")
    )
    assert summary["test_error"] == (0.18 if selected == "ensemble" else 0.2)
    for payload in (summary, details):
        metrics = payload["error_metrics"]
        assert metrics["validation_error"]["value"] == (
            ensemble_fitness if selected == "ensemble" else 0.1
        )
        assert metrics["test_ratio"]["value"] == (0.9 if selected == "ensemble" else 1.0)
        assert metrics["wape"]["value"] == pytest.approx(0.18 if selected == "ensemble" else 0.2)
    assert _prediction_path(run, {}, "test") == run / f"{selected}_test.parquet"
    assert result_details(run, request, {"model": "winner"})["model"] == "winner"


def test_selection_keeps_prediction_ownership(tmp_path):
    result = {"winner": {"family": "snaive", "fitness": 0.1}, "ensemble": {"fitness": 0.09}}
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "test.parquet").touch()
    with pytest.raises(ValueError, match="no saved ensemble test"):
        _prediction_path(tmp_path, {}, "test")
    assert _prediction_path(tmp_path, {"model": "winner"}, "test") == tmp_path / "test.parquet"
    result.pop("ensemble")
    assert _select_model(tmp_path, result, {})["model"] == "winner"
