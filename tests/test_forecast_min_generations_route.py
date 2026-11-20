"""the v2 start route forwards min_generations to the run starter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from forecast_fixtures import write_panel

from smolsmort.forecast import runs
from smolsmort.forecast.tab import forecast_tab

SPEC = {
    "source": "panel.csv",
    "mode": "timeseries",
    "task": "regression",
    "step": "week",
    "horizon": 4,
    "columns": [
        {"name": "day", "role": "time"},
        {"name": "project", "role": "dimension"},
        {"name": "product", "role": "dimension"},
        {"name": "orders", "role": "measure", "aggregation": "sum"},
        {"name": "units", "role": "target", "aggregation": "sum"},
    ],
}


def start(tmp_path, monkeypatch, **changes):
    pytest.importorskip("duckdb")
    write_panel(tmp_path, weeks=60)
    app = SimpleNamespace(state=SimpleNamespace(bases={"forecast": str(tmp_path)}))
    calls = []
    monkeypatch.setattr(runs, "start_run", lambda *args, **kwargs: calls.append(kwargs) or "new")
    payload = {"spec": SPEC, "families": [{"name": "ridge"}], "time_budget_s": 30, **changes}
    forecast_tab().post["/api/forecast-runs"](app, payload)
    return calls


def test_min_generations_reaches_start_run(tmp_path, monkeypatch):
    calls = start(tmp_path, monkeypatch, min_generations=10)
    assert calls[0]["min_generations"] == 10


def test_min_generations_defaults_to_zero(tmp_path, monkeypatch):
    calls = start(tmp_path, monkeypatch)
    assert calls[0]["min_generations"] == 0
