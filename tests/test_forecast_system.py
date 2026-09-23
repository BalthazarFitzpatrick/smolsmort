"""the assembled forecast topics end to end, through the routes the page calls: one real search per
mode on the synthetic fixtures, then what a user would check - the leak refused, the baseline beaten,
the band honest, every open row or future step predicted, the export carrying the verdict"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from forecast_fixtures import write_panel, write_rows
from test_forecast_routes import _row_spec, _run_to_done, _series_spec

from smolsmort.forecast.features import buckets_for
from smolsmort.forecast.tables import read_table

pytest.importorskip("xgboost")
pytest.importorskip("duckdb")

BUDGET = {"population": 10, "max_generations": 6, "plateau": 3, "nthread": 1, "seed": 0}


def _outcome(app, run_id):
    run = Path(app.state.bases["forecast"]) / ".forecast-runs" / run_id
    request = json.loads((run / "request.json").read_text())
    return run, request, json.loads((run / "result.json").read_text())


def _export(app, tab, run_id) -> list[str]:
    getter, content_type = tab.images["/api/forecast-export/"]
    assert content_type == "text/csv"
    return getter(app, f"{run_id}.csv").decode().splitlines()


@pytest.fixture(scope="module")
def row_system(tmp_path_factory):
    yield _run_to_done(
        tmp_path_factory.mktemp("row_system"),
        lambda root: write_rows(root, n=3000),
        _row_spec,
        "rows.csv",
        BUDGET,
    )


@pytest.fixture(scope="module")
def series_system(tmp_path_factory):
    yield _run_to_done(
        tmp_path_factory.mktemp("series_system"),
        write_panel,
        _series_spec,
        "panel.csv",
        BUDGET,
    )


def test_row_mode_refuses_the_leak_and_beats_the_baseline(row_system):
    app, tab, run_id, _ = row_system
    run, _, result = _outcome(app, run_id)
    profile = json.loads((run / "profile.json").read_text())
    assert "leak_plus1" in profile["leaks"]
    for entry in json.loads((run / "leaderboard.json").read_text()):
        assert not [f for f in entry["genome"]["families"] if "leak_plus1" in f]
    assert result["model_error"] < result["baseline_error"]
    calibrated = {c["check"]: c for c in result["verdict"]["checks"]}["band is calibrated"]
    assert calibrated["passed"], calibrated["detail"]


def test_row_mode_predicts_every_open_row_and_exports_the_verdict(row_system):
    app, tab, run_id, _ = row_system
    run, request, result = _outcome(app, run_id)
    forecast = read_table(run / "forecast.parquet")
    assert len(forecast["prediction"]) == request["summary"]["predict_rows"] > 0
    lines = _export(app, tab, run_id)
    assert lines[0].startswith("# verdict:")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[1:]))))
    assert len(rows) == len(forecast["prediction"])
    assert {"prediction", "lower", "upper", "verdict"} <= set(rows[0])
    expected = "trusted" if result["verdict"]["trusted"] else "not trusted"
    assert {r["verdict"] for r in rows} == {expected}


def test_series_mode_forecasts_every_combination_and_beats_seasonal_naive(series_system):
    app, tab, run_id, _ = series_system
    run, request, result = _outcome(app, run_id)
    horizon = request["spec"]["horizon"]
    forecast = read_table(run / "forecast.parquet")
    assert len(forecast["prediction"]) == request["summary"]["series"] * horizon
    assert set(forecast["bucket"]) == {f"{lo}-{hi}" for lo, hi in buckets_for(horizon)}
    assert result["model_error"] < result["baseline_error"]


def test_series_mode_view_and_export_agree_with_the_run(series_system):
    app, tab, run_id, _ = series_system
    _, _, result = _outcome(app, run_id)
    view = tab.get["/api/forecast-view"](app, {"id": run_id, "breakdown": "project"})
    assert view["verdict"] == result["verdict"]
    assert {s["id"].split(":")[0] for s in view["series"]} >= {"actual", "forecast"}
    assert view["bands"] and view["markers"]
    lines = _export(app, tab, run_id)
    assert lines[0] == f"# verdict: {result['verdict']['summary']}"
