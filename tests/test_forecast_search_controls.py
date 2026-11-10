"""training controls reach durable requests and family search without changing old requests"""

import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from smolsmort.forecast import family_worker, runs, tab
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.search import Budget
from smolsmort.forecast.spec import Column, PrepSpec


def make_request(**changes):
    return {
        "version": 2,
        "time_budget_s": 1200,
        "families": [{"name": "snaive"}],
        **changes,
    }


@pytest.mark.parametrize("field", ["population", "max_generations"])
@pytest.mark.parametrize("value", [1, 16, 60])
def test_request_accepts_positive_search_controls(field, value):
    request = make_request(**{field: value})
    parsed = parse_request(request)
    assert parsed[field] == value
    assert parse_request(json.loads(json.dumps(parsed))) == parsed


@pytest.mark.parametrize("field", ["population", "max_generations"])
@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "16", [], {}])
def test_request_rejects_invalid_search_controls(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        parse_request(make_request(**{field: value}))


def test_old_request_keeps_its_exact_normalized_shape():
    request = make_request()
    expected = {
        **request,
        "aggregate": "none",
        "transform": "none",
        "holdout": "block",
        "horizon_mode": "single",
        "ensemble": {"enabled": True, "top": 3},
        "max_concurrent_families": 4,
        "_max_concurrent_families_explicit": False,
        "candidate_limit_fraction": 0.25,
        "plateau": 5,
        "min_generations": 0,
        "prune_sigma": 1.0,
    }
    parsed = parse_request(request)
    expected["max_concurrent_families"] = parsed["max_concurrent_families"]
    assert parsed == expected
    assert parse_request(json.loads(json.dumps(parsed))) == expected


@pytest.fixture
def route_context(tmp_path, monkeypatch):
    spec = PrepSpec(
        str(tmp_path / "data.csv"),
        "timeseries",
        "regression",
        (Column("day", "time"), Column("target", "target", aggregation="sum")),
        step="week",
        horizon=8,
    )
    prepared = SimpleNamespace(folder=tmp_path / "cache", summary={}, sql="", reused=False)
    root = tmp_path / "runs"
    monkeypatch.setattr(tab, "_runs_root", lambda *args: root)
    monkeypatch.setattr(tab, "_parse_spec", lambda *args: spec)
    monkeypatch.setattr(tab, "_prepare_spec", lambda *args: prepared)
    monkeypatch.setattr(
        tab, "_validate_selections", lambda payload, spec: parse_request({"version": 2, **payload})
    )
    monkeypatch.setattr(tab.views, "save_prepared_spec", lambda *args: None)
    monkeypatch.setattr(runs.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=123))
    monkeypatch.setattr(runs, "_PROCESSES", {})
    return root, spec


def test_route_did_not_add_optional_keys_to_old_requests(route_context):
    root, _ = route_context
    result = tab._start(None, make_request())
    saved = json.loads((root / result["run_id"] / "request.json").read_text())
    assert "population" not in saved
    assert "max_generations" not in saved
    assert parse_request(saved) == saved


@pytest.mark.parametrize("warm", [False, True])
def test_route_persisted_controls_include_legacy_warm_budget(route_context, monkeypatch, warm):
    root, spec = route_context
    controls = {"population": 8, "plateau": 7, "max_generations": 22}
    payload = make_request(**controls, max_concurrent_families=1)
    if warm:
        source = root / "previous"
        source.mkdir(parents=True)
        (source / "request.json").write_text(json.dumps(make_request(spec=asdict(spec))))
        monkeypatch.setattr(tab, "_run_dir", lambda *args: source)
        monkeypatch.setattr(tab.views, "winning_genome", lambda *args: {"family": "snaive"})
        payload = {"warm_from": "previous", "budget": {**controls, "time_cap": 1200}}
    result = tab._start(None, payload)
    saved = json.loads((root / result["run_id"] / "request.json").read_text())
    assert {key: saved[key] for key in controls} == controls
    assert saved["time_budget_s"] == 1200
    assert parse_request(saved) == saved


@pytest.mark.parametrize("controls", [{}, {"population": 8, "max_generations": 22}])
def test_family_search_uses_controls_and_retains_old_defaults(route_context, monkeypatch, controls):
    root, spec = route_context
    root.mkdir()
    request = make_request(spec=asdict(spec), prepared=str(root), summary={}, **controls)
    (root / "request.json").write_text(json.dumps(request))
    budgets = []
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(
        family_worker, "get_family", lambda *args: SimpleNamespace(available=lambda: (True, ""))
    )
    monkeypatch.setattr(family_worker, "load_workspace", lambda *args, **kwargs: SimpleNamespace())

    def capture_search(workspace, budget, **kwargs):
        budgets.append(budget)
        return {
            "leaderboard": [{"fitness": 1}],
            "population": [],
            "best": {},
            "elapsed": 0,
            "stopped": "max_generations",
        }

    monkeypatch.setattr(family_worker, "search", capture_search)
    assert family_worker.run(root, "snaive") == 0
    assert budgets[0].population == controls.get("population", Budget().population)
    assert budgets[0].max_generations == controls.get("max_generations", 2**31 - 1)
