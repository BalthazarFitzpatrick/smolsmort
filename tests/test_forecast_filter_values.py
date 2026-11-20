"""filter menu values cover the whole saved result, not one page of the table"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import views
from smolsmort.forecast.tab import forecast_tab
from smolsmort.forecast.tables import write_table

# parquet fixtures need duckdb, which ci does not install
pytest.importorskip("duckdb")

SERIES = 2001


@pytest.fixture
def wide_run(tmp_path):
    run = tmp_path / ".forecast-runs" / "wide"
    prepared = tmp_path / ".forecast-cache" / "wide"
    prepared.mkdir(parents=True)
    run.mkdir(parents=True)
    request = {
        "version": 2,
        "prepared": str(prepared),
        "families": [{"name": "ridge"}],
        "spec": {
            "mode": "timeseries",
            "task": "regression",
            "step": "week",
            "horizon": 1,
            "horizon_mode": "single",
            "columns": [
                {"name": "day", "role": "time"},
                {"name": "region", "role": "dimension"},
                {"name": "units", "role": "target", "aggregation": "sum"},
            ],
        },
    }
    result = {"val_error": 0.2, "winner": {"family": "ridge", "fitness": 0.2}}
    for name, data in (
        ("request", request),
        ("result", result),
        ("status", {"state": "done", "families": {}}),
        ("recipe", {"family": "ridge", "genome": {"family": "ridge"}}),
    ):
        (run / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    series = np.array([f"s{i:04d}" for i in range(SERIES)])
    # the last series is the only one in region "zulu", past row 2000
    region = np.array(["alpha"] * (SERIES - 1) + ["zulu"])
    step = np.array(["2025-01-06"] * SERIES, dtype="datetime64[D]")
    write_table(
        prepared / "panel.parquet",
        {"__series": series, "__step": step, "region": region, "units": np.ones(SERIES)},
    )
    forecast = {
        "series": series,
        "step": step + np.timedelta64(7, "D"),
        "origin": step,
        "weeks_ahead": np.ones(SERIES, dtype=int),
        "bucket": np.array(["h1"] * SERIES),
        "prediction": np.ones(SERIES),
        "lower": np.zeros(SERIES),
        "upper": np.full(SERIES, 2.0),
    }
    write_table(run / "forecast.parquet", forecast)
    write_table(run / "winner_forecast.parquet", forecast)
    return run, request, SimpleNamespace(state=SimpleNamespace(bases={"forecast": str(tmp_path)}))


def test_filter_values_include_values_past_the_first_table_page(wide_run):
    run, request, _ = wide_run
    assert views.filter_values(run, request, {}) == {"values": {"region": ["alpha", "zulu"]}}


def test_filter_values_route_returns_the_same_values(wide_run):
    run, _, app = wide_run
    tab = forecast_tab()
    handler = tab.get["/api/forecast-filter-values"]
    assert handler(app, {"id": run.name}) == {"values": {"region": ["alpha", "zulu"]}}
