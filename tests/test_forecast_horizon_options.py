"""the selected horizon mode survives validation, saved requests, and mocked launches"""

import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from smolsmort.forecast import family_worker, runs, tab
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.spec import Column, PrepSpec
from smolsmort.review.routes import RequestError


def make_request(**changes):
    return {
        "version": 2,
        "time_budget_s": 60,
        "families": [{"name": "snaive"}],
        **changes,
    }


def test_horizon_mode_defaults_without_mutating_request():
    request = make_request()

    assert parse_request(request)["horizon_mode"] == "grouped"
    assert "horizon_mode" not in request


@pytest.mark.parametrize("horizon_mode", ["grouped", "single"])
def test_request_preserves_horizon_mode(horizon_mode):
    assert parse_request(make_request(horizon_mode=horizon_mode))["horizon_mode"] == horizon_mode


@pytest.mark.parametrize("horizon_mode", ["mixed", "", None, True, 1, [], {}])
@pytest.mark.parametrize("versioned", [False, True])
def test_request_rejects_unknown_horizon_mode(horizon_mode, versioned):
    request = make_request() if versioned else {}
    with pytest.raises(ValueError, match="horizon_mode must be grouped or single"):
        parse_request({**request, "horizon_mode": horizon_mode})


def test_unversioned_request_retains_identity():
    request = {"budget": {"population": 3}}

    assert parse_request(request) is request
    assert "horizon_mode" not in request


@pytest.fixture
def mock_run(tmp_path, monkeypatch):
    spec = PrepSpec(
        str(tmp_path / "synthetic.csv"),
        "timeseries",
        "regression",
        (Column("day", "time"), Column("target", "target", aggregation="sum")),
        step="week",
        horizon=8,
    )
    prepared = SimpleNamespace(folder=tmp_path / "cache", summary={}, sql="", reused=False)
    starts = []

    def start_worker(command, **kwargs):
        starts.append(command)
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(runs.subprocess, "Popen", start_worker)
    monkeypatch.setattr(runs, "_PROCESSES", {})
    return tmp_path / "runs", spec, prepared, starts


@pytest.mark.parametrize(
    "changes, expected", [({}, "grouped"), ({"horizon_mode": "single"}, "single")]
)
def test_start_run_writes_horizon_mode(mock_run, changes, expected):
    root, spec, prepared, starts = mock_run

    run_id = runs.start_run(root, spec, prepared, **make_request(**changes))

    request = json.loads((root / run_id / "request.json").read_text())
    assert request["horizon_mode"] == expected
    assert len(starts) == 1


def test_unversioned_start_run_preserves_explicit_single(mock_run):
    root, spec, prepared, starts = mock_run

    run_id = runs.start_run(root, spec, prepared, horizon_mode="single")

    request = json.loads((root / run_id / "request.json").read_text())
    assert request["horizon_mode"] == "single"
    assert "version" not in request
    assert len(starts) == 1


def test_start_run_rejects_unknown_horizon_mode_before_launch(mock_run):
    root, spec, prepared, starts = mock_run

    with pytest.raises(ValueError, match="horizon_mode must be grouped or single"):
        runs.start_run(root, spec, prepared, **make_request(horizon_mode="mixed"))

    assert not root.exists()
    assert starts == []


@pytest.fixture
def mock_routes(mock_run, monkeypatch):
    root, spec, prepared, starts = mock_run
    monkeypatch.setattr(tab, "_parse_spec", lambda *args: spec)
    monkeypatch.setattr(tab, "_prepare_spec", lambda *args: prepared)
    monkeypatch.setattr(tab, "_runs_root", lambda app: root)
    monkeypatch.setattr(
        tab, "_validate_selections", lambda payload, spec: parse_request({"version": 2, **payload})
    )
    monkeypatch.setattr(tab.views, "save_prepared_spec", lambda *args: None)
    return root, spec, prepared, starts


def test_start_route_forwards_horizon_mode(mock_routes):
    root, _, _, starts = mock_routes

    result = tab._start(None, make_request(horizon_mode="single"))

    saved = json.loads((root / result["run_id"] / "request.json").read_text())
    assert saved["horizon_mode"] == "single"
    assert len(starts) == 1


@pytest.mark.parametrize("action", ["refit", "warm"])
def test_refit_and_legacy_warm_preserve_horizon_mode(mock_routes, monkeypatch, action):
    root, spec, _, starts = mock_routes
    source = root / "saved"
    source.mkdir(parents=True)
    previous = make_request(spec=asdict(spec), horizon_mode="single")
    (source / "request.json").write_text(json.dumps(previous))
    (source / "recipe.json").write_text(json.dumps({"genome": {"family": "snaive"}}))
    monkeypatch.setattr(tab, "_run_dir", lambda *args: source)
    monkeypatch.setattr(tab.views, "winning_genome", lambda *args: {"family": "snaive"})

    result = (
        tab._refit(None, {"run_id": source.name})
        if action == "refit"
        else tab._start(None, {"warm_from": source.name})
    )

    saved = json.loads((root / result["run_id"] / "request.json").read_text())
    assert saved["horizon_mode"] == "single"
    assert len(starts) == 1


@pytest.mark.parametrize("horizon_mode", ["grouped", "single"])
def test_legacy_warm_preserves_saved_horizon_mode(mock_routes, monkeypatch, horizon_mode):
    root, spec, prepared, starts = mock_routes
    source = root / "saved"
    source.mkdir(parents=True)
    previous = {"spec": asdict(spec), "horizon_mode": horizon_mode}
    (source / "request.json").write_text(json.dumps(previous))
    monkeypatch.setattr(tab, "_run_dir", lambda *args: source)
    monkeypatch.setattr(tab, "_spec_from_payload", lambda *args: asdict(spec))
    monkeypatch.setattr(tab, "_cache_root", lambda *args: prepared.folder)
    monkeypatch.setattr(tab, "prepare", lambda *args: prepared)

    result = tab._start(None, {"warm_from": source.name})

    saved = json.loads((root / result["run_id"] / "request.json").read_text())
    assert saved.get("horizon_mode", "grouped") == horizon_mode
    assert "version" not in saved
    assert ("horizon_mode" in saved) == (horizon_mode == "single")
    assert len(starts) == 1


def test_family_catalog_route_forwards_horizon_mode(mock_routes, monkeypatch):
    modes = []

    def family_catalog(*args, **kwargs):
        modes.append(kwargs["horizon_mode"])
        return []

    monkeypatch.setattr(tab.views, "family_catalog", family_catalog)

    assert tab._families(None, {"horizon_mode": "single"}) == {"families": []}
    assert modes == ["single"]


def test_family_catalog_route_rejects_unknown_horizon_mode():
    with pytest.raises(RequestError, match="horizon_mode must be grouped or single"):
        tab._families(None, {"horizon_mode": "mixed"})


@pytest.mark.parametrize("horizon_mode", ["grouped", "single"])
def test_family_worker_forwards_horizon_mode(mock_run, monkeypatch, horizon_mode):
    root, spec, prepared, _ = mock_run
    root.mkdir()
    request = make_request(
        spec=asdict(spec),
        prepared=str(prepared.folder),
        summary={},
        horizon_mode=horizon_mode,
    )
    (root / "request.json").write_text(json.dumps(request))
    modes = []

    def load_workspace(*args, **kwargs):
        modes.append(kwargs["horizon_mode"])
        return SimpleNamespace(mode="timeseries")

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(family_worker, "load_workspace", load_workspace)
    monkeypatch.setattr(family_worker, "read_priors", lambda *args: None)
    monkeypatch.setattr(family_worker, "read_top_families", lambda *args: None)
    monkeypatch.setattr(
        family_worker,
        "get_family",
        lambda name: SimpleNamespace(available=lambda: (True, "")),
    )
    monkeypatch.setattr(
        family_worker,
        "search",
        lambda *args, **kwargs: {
            "leaderboard": [{"fitness": 0.1}],
            "population": [],
            "best": {},
            "stopped": "single candidate",
            "elapsed": 0,
        },
    )

    assert family_worker.run(root, "snaive") == 0
    assert modes == [horizon_mode]
