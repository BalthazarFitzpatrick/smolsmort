"""the family library probe runs once per server and never caches a failure"""

import subprocess

import pytest

from smolsmort.forecast import views


@pytest.fixture(autouse=True)
def empty_cache():
    views._probe_cache.clear()
    yield
    views._probe_cache.clear()


def test_a_good_probe_runs_once_and_returns_copies(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args, 0, stdout='{"ridge": [true, ""]}')

    monkeypatch.setattr(views.subprocess, "run", fake_run)
    first = views.probe_families()
    first["ridge"] = "changed"
    assert views.probe_families() == {"ridge": [True, ""]}
    assert calls == [views._PROBE_BUDGET_S]


def test_a_timeout_raises_and_is_not_cached(monkeypatch):
    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(views.subprocess, "run", slow)
    with pytest.raises(views.ViewError, match="could not check family libraries"):
        views.probe_families()
    assert views._probe_cache == {}
