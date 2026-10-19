"""pruning thresholds survive request validation and the run's durable request"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from smolsmort.forecast import runs
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.spec import Column, PrepSpec

INVALID_SIGMAS = [True, False, None, "2", [], {}, 0, -1, float("nan"), float("inf"), -float("inf")]


def make_request(**changes):
    return {
        "version": 2,
        "time_budget_s": 86400,
        "families": [{"name": "snaive"}],
        **changes,
    }


def test_prune_sigma_defaults_without_mutating_request():
    request = make_request()

    assert parse_request(request)["prune_sigma"] == 1.0
    assert "prune_sigma" not in request


@pytest.mark.parametrize("sigma", [0.1, 1, 1.5, 2.0])
def test_prune_sigma_accepts_positive_finite_numbers(sigma):
    assert parse_request(make_request(prune_sigma=sigma))["prune_sigma"] == sigma


@pytest.mark.parametrize("sigma", INVALID_SIGMAS)
def test_prune_sigma_rejects_invalid_thresholds(sigma):
    with pytest.raises(ValueError, match="prune_sigma must be positive and finite"):
        parse_request(make_request(prune_sigma=sigma))


@pytest.fixture
def mock_run(tmp_path, monkeypatch):
    spec = PrepSpec(
        str(tmp_path / "synthetic.csv"),
        "row",
        "regression",
        (Column("day", "anchor"), Column("target", "target")),
    )
    prepared = SimpleNamespace(folder=tmp_path / "cache", summary={})
    starts = []

    def start_worker(command, **_kwargs):
        starts.append(command)
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(runs.subprocess, "Popen", start_worker)
    monkeypatch.setattr(runs, "_PROCESSES", {})
    return tmp_path / "runs", spec, prepared, starts


@pytest.mark.parametrize("changes, expected", [({}, 1.0), ({"prune_sigma": 2.5}, 2.5)])
def test_start_run_writes_prune_sigma(mock_run, changes, expected):
    root, spec, prepared, starts = mock_run

    run_id = runs.start_run(root, spec, prepared, **make_request(**changes))

    request = json.loads((root / run_id / "request.json").read_text())
    assert request["prune_sigma"] == expected
    assert len(starts) == 1


@pytest.mark.parametrize("sigma", INVALID_SIGMAS)
def test_start_run_rejects_prune_sigma_before_starting_worker(mock_run, sigma):
    root, spec, prepared, starts = mock_run

    with pytest.raises(ValueError, match="prune_sigma must be positive and finite"):
        runs.start_run(root, spec, prepared, **make_request(prune_sigma=sigma))

    assert not root.exists()
    assert starts == []


def test_unversioned_request_retains_legacy_validation():
    request = {"budget": {"population": 3}}

    assert parse_request(request) is request
    assert "prune_sigma" not in request
