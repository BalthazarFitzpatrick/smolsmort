"""winner, ensemble, and importance fits share the coordinator's thread budget"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import ensemble


@pytest.mark.parametrize(
    ("coordinator_threads", "request_threads", "cpu_threads", "expected_threads"),
    [
        (8, None, 12, 8),
        (8, 4, 12, 8),
        (None, 4, 12, 4),
        (None, None, 12, 12),
        (None, None, None, 1),
    ],
)
def test_finish_and_importance_use_resolved_threads(
    tmp_path, monkeypatch, coordinator_threads, request_threads, cpu_threads, expected_threads
):
    request = {"spec": {}, "prepared": str(tmp_path), "summary": {}}
    if request_threads is not None:
        request["nthread"] = request_threads
    coordinator = {"families": {name: {"state": "done"} for name in ("xgboost", "lightgbm")}}
    if coordinator_threads is not None:
        coordinator["nthread"] = coordinator_threads
    ranked = [
        {"family": name, "fitness": 0.2, "prediction": [9, 11], "genome": {"family": name}}
        for name in coordinator["families"]
    ]
    for name, data in (
        ("request.json", request),
        ("coordinator.json", coordinator),
        ("finalists.json", ranked),
    ):
        (tmp_path / name).write_text(json.dumps(data))
    for name in coordinator["families"]:
        (tmp_path / "families" / name).mkdir(parents=True)

    frame = SimpleNamespace(y=np.array([10, 10]), step=np.array([1, 2]))
    workspace = SimpleNamespace(
        mode="timeseries",
        frames={"series": frame},
        series_masks={"series": {"val": np.array([True, True])}},
    )
    fit_calls = []
    importance_calls = []

    def finish_candidate(ws, genome, **kwargs):
        assert ws is workspace
        fit_calls.append(("ensemble" if kwargs.get("members") else "winner", kwargs["nthread"]))
        return {"model_error": 0.2}

    def fit_family_importance(ws, genome, *, nthread):
        assert ws is workspace
        importance_calls.append((genome.key(), nthread))
        return {"lag": 1.0}

    def write_result(folder, spec, result, final):
        (folder / "recipe.json").write_text("{}")
        (folder / "result.json").write_text("{}")

    monkeypatch.setattr(ensemble, "watch_parent", lambda pid: None)
    monkeypatch.setattr(ensemble.os, "cpu_count", lambda: cpu_threads)
    monkeypatch.setattr(ensemble, "spec_from_dict", lambda data: data)
    monkeypatch.setattr(ensemble, "load_workspace", lambda *args, **kwargs: workspace)
    monkeypatch.setattr(
        ensemble,
        "genome_from_dict",
        lambda data, **kwargs: SimpleNamespace(key=lambda: data["family"], to_dict=lambda: data),
    )
    monkeypatch.setattr(ensemble, "finish", finish_candidate)
    monkeypatch.setattr(ensemble, "fit_importance", fit_family_importance)
    monkeypatch.setattr(ensemble, "combine_importance", lambda name, importance: importance)
    monkeypatch.setattr(ensemble, "_fitness", lambda *args: SimpleNamespace(fitness=0.1))
    monkeypatch.setattr(ensemble, "_write_result", write_result)

    ensemble.run(tmp_path, 123)

    assert fit_calls == [("winner", expected_threads), ("ensemble", expected_threads)]
    assert importance_calls == [("xgboost", expected_threads), ("lightgbm", expected_threads)]
    assert json.loads((tmp_path / "finish_complete.json").read_text()) == {"family": "ensemble"}
