"""the worker persists one thread budget before any preparation or fitting"""

import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from smolsmort.forecast import coordinator, worker
from smolsmort.forecast.spec import Column, PrepSpec


@pytest.mark.parametrize(
    ("stored_threads", "request_threads", "expected_threads"),
    [(None, None, 12), (None, 4, 4), (8, None, 8), (8, 4, 8)],
)
def test_worker_persists_threads_before_screen_and_recipe_refit(
    tmp_path, monkeypatch, stored_threads, request_threads, expected_threads
):
    spec = PrepSpec(
        "unused.csv",
        "timeseries",
        "regression",
        (Column("day", "time"), Column("units", "target", aggregation="sum")),
    )
    request = {
        "version": 2,
        "spec": asdict(spec),
        "prepared": str(tmp_path),
        "summary": {},
        "time_budget_s": 60,
        "families": [{"name": "ridge"}],
        "recipe": {"family": "ridge", "transform": "none"},
        "budget": {"seed": 7, "nthread": 2},
    }
    if request_threads is not None:
        request["nthread"] = request_threads
    (tmp_path / "request.json").write_text(json.dumps(request))
    if stored_threads is not None:
        (tmp_path / "coordinator.json").write_text(
            json.dumps(
                {
                    "started_at": 100,
                    "deadline": 160,
                    "phase": "broad",
                    "families": {},
                    "nthread": stored_threads,
                    "machine": {"resolved_threads": stored_threads},
                }
            )
        )
    workspace = SimpleNamespace(mode="timeseries", excluded_series=[], profile={})
    calls = []
    cpu_calls = []
    lock = SimpleNamespace(try_acquire=lambda: True, close=lambda: calls.append("closed"))

    def read_threads():
        state = json.loads((tmp_path / "coordinator.json").read_text())
        assert state["nthread"] == expected_threads
        assert state["machine"]["resolved_threads"] == expected_threads

    def count_cpus():
        cpu_calls.append(True)
        return 12

    def load_workspace(*args, **kwargs):
        read_threads()
        calls.append("load")
        return workspace

    def screen_families(ws, *, seed, nthread, pinned):
        read_threads()
        assert ws is workspace
        assert seed == 7
        assert pinned == frozenset()
        assert nthread == expected_threads
        calls.append("screen")
        return {"ranked": []}

    def finish_recipe(ws, genome, *, nthread, importance_cache):
        assert ws is workspace
        assert nthread == expected_threads
        assert importance_cache == {}
        calls.append("finish")
        return {"verdict": {}, "model_error": 0.2, "baseline_error": 0.4}

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(worker, "RunLock", lambda: lock)
    monkeypatch.setattr(coordinator.os, "cpu_count", count_cpus)
    monkeypatch.setattr(
        coordinator, "describe_machine", lambda nthread, cap: {"resolved_threads": nthread}
    )
    monkeypatch.setattr(worker, "load_workspace", load_workspace)
    monkeypatch.setattr(worker, "screen_families", screen_families)
    monkeypatch.setattr(
        worker, "genome_from_dict", lambda data, **kwargs: SimpleNamespace(to_dict=lambda: data)
    )
    monkeypatch.setattr(worker, "finish", finish_recipe)

    assert worker.run(tmp_path) == 0

    assert calls == ["load", "screen", "finish", "closed"]
    assert len(cpu_calls) == int(stored_threads is None and request_threads is None)
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "done"
    assert json.loads((tmp_path / "recipe.json").read_text())["family"] == "ridge"
    state = json.loads((tmp_path / "coordinator.json").read_text())
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["machine"] == state["machine"]
    assert result["phase_history"] == state["phase_history"]
    assert [item["phase"] for item in result["phase_history"]] == [
        "broad",
        "finish",
        "finish_start",
        "finish_end",
    ]
    assert result["elapsed"] >= 0
