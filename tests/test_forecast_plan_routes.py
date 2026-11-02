"""saved search plan routes without workers or training"""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from smolsmort.forecast import plan, processes
from smolsmort.forecast.spec import Column, PrepSpec
from smolsmort.forecast.tab import forecast_tab
from smolsmort.review.routes import RequestError


@pytest.fixture
def app(tmp_path):
    return SimpleNamespace(state=SimpleNamespace(bases={"forecast": str(tmp_path)}))


@pytest.fixture
def config():
    spec = PrepSpec(
        "source.csv",
        "timeseries",
        "regression",
        (Column("day", "time"), Column("value", "target", aggregation="sum")),
    )
    return {
        "hours": 2,
        "families": [{"name": "ridge", "budget_s": 600}],
        "sets": [{"label": "clients", "horizon": 8, "spec": json.loads(json.dumps(asdict(spec)))}],
    }


def test_get_missing_plan_returns_idle(app):
    result = forecast_tab().get["/api/forecast-plan"](app, {})
    assert result["plan"] is None
    assert result["status"]["state"] == "idle"
    assert result["running"] is False
    assert "error" not in result


def test_save_and_get_plan_with_status(app, tmp_path, config):
    tab = forecast_tab()
    assert tab.post["/api/forecast-plan"](app, config) == {"plan": config}
    status = {"state": "stopped", "jobs_done": 1, "jobs_total": 2}
    (tmp_path / "forecast-plan-status.json").write_text(json.dumps(status))
    result = tab.get["/api/forecast-plan"](app, {})
    assert result == {"plan": config, "status": status, "running": False}
    assert plan.load_plan(tmp_path) == config


@pytest.mark.parametrize("route", ["/api/forecast-plan", "/api/forecast-plan-start"])
def test_invalid_plan_has_plain_message(app, route):
    with pytest.raises(RequestError) as caught:
        forecast_tab().post[route](app, {})
    assert str(caught.value) == "choose exactly one of deadline or hours"
    assert caught.value.status == 400


@pytest.mark.parametrize("contents", ['{"hours": 0}', "not json"])
def test_bad_saved_plan_returns_null_and_error(app, tmp_path, contents):
    (tmp_path / "forecast-plan.json").write_text(contents)
    result = forecast_tab().get["/api/forecast-plan"](app, {})
    assert result["plan"] is None
    assert result["error"]
    assert result["status"]["state"] == "idle"


def test_start_and_stop_use_forecast_root(app, tmp_path, monkeypatch):
    calls = []

    def start(root, runs_root, cache_root):
        calls.append((root, runs_root, cache_root))
        return 12345

    monkeypatch.setattr(plan, "start_plan", start)
    monkeypatch.setattr(plan, "stop_plan", lambda root: calls.append(root))
    tab = forecast_tab()
    assert tab.post["/api/forecast-plan-start"](app, {}) == {"pid": 12345}
    assert calls == [(tmp_path, tmp_path / ".forecast-runs", tmp_path / ".forecast-cache")]
    assert tab.post["/api/forecast-plan-stop"](app, {}) == {"ok": True}
    assert calls[-1] == tmp_path


def test_running_plan_refuses_save_and_start(app, tmp_path, monkeypatch, config):
    (tmp_path / "forecast-plan.pid").write_text("12345")
    monkeypatch.setattr(processes, "is_process_alive", lambda pid: pid == 12345)
    monkeypatch.setattr(plan, "start_plan", lambda *args: pytest.fail("must not start"))
    tab = forecast_tab()
    assert tab.get["/api/forecast-plan"](app, {})["running"] is True
    for route, message in (
        ("/api/forecast-plan", "stop the search plan before saving changes"),
        ("/api/forecast-plan-start", "search plan is already running"),
    ):
        with pytest.raises(RequestError) as caught:
            tab.post[route](app, config)
        assert str(caught.value) == message
        assert caught.value.status == 409
    assert not (tmp_path / "forecast-plan.json").exists()


@pytest.mark.parametrize("pid", ["12345", "invalid", "0", "-1"])
def test_stale_or_invalid_pid_allows_save(app, tmp_path, monkeypatch, config, pid):
    (tmp_path / "forecast-plan.pid").write_text(pid)
    monkeypatch.setattr(processes, "is_process_alive", lambda pid: False)
    tab = forecast_tab()
    assert tab.get["/api/forecast-plan"](app, {})["running"] is False
    assert tab.post["/api/forecast-plan"](app, config) == {"plan": config}
