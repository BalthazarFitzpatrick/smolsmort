"""short series skip raw models without hiding the family's covered series"""

from types import SimpleNamespace

import numpy as np
import pytest
from test_forecast_family_stats import make_task
from test_forecast_family_stats import workspace as workspace

from smolsmort.forecast.families import get_family
from smolsmort.forecast.families.stats import series_eligibility
from smolsmort.forecast.pipeline import PipelineError, finish, make_genome, score
from smolsmort.forecast.spec import SERIES, STEP
from smolsmort.forecast.splits import timeseries_cuts
from smolsmort.forecast.views import _series_coverage, _suitability_reason, family_catalog


def make_eligibility_workspace():
    steps = np.datetime64("2024-01-01") + np.arange(100).astype("timedelta64[D]")
    cut = timeseries_cuts(100, 1).val_start
    starts = {"long": 0, "zero": cut, "short": cut - 7, "minimum": cut - 8}
    panel = {
        SERIES: np.concatenate([np.repeat(key, 100 - start) for key, start in starts.items()]),
        STEP: np.concatenate([steps[start:] for start in starts.values()]),
        "y": np.ones(sum(100 - start for start in starts.values())),
    }
    return SimpleNamespace(
        panel=panel,
        unit="day",
        profile={"period": 4},
        spec=SimpleNamespace(
            horizon=1, named=lambda role: ["y"], task="regression", mode="timeseries"
        ),
    )


def test_eligibility_counts_zero_short_boundary_and_long():
    ws = make_eligibility_workspace()
    result = series_eligibility(ws.panel, "y", ws.unit, ws.profile, 1)
    assert result.counts == {"long": 61, "minimum": 8, "short": 7, "zero": 0}
    assert result.eligible == {"long", "minimum"}
    assert _suitability_reason(get_family("snaive"), ws) is None
    assert _series_coverage(get_family("snaive"), ws) == {
        "skipped_series": 2,
        "total_series": 4,
        "coverage_note": "n/a for 2 of 4 series: needs two seasons",
    }


def test_all_ineligible_keeps_original_reason():
    ws = make_eligibility_workspace()
    ws.profile = {"period": 100}
    assert _suitability_reason(get_family("snaive"), ws) == (
        "needs two seasons (200 training steps per series); shortest series has 0"
    )


@pytest.mark.parametrize(
    "unit,period", [("day", 7), ("week", 52), ("month", 12), ("quarter", 4), ("year", 1)]
)
def test_default_period_table_unchanged(unit, period):
    ws = make_eligibility_workspace()
    result = series_eligibility(ws.panel, "y", unit, {"genes": {"period": 2}}, 1)
    assert result.period == period


def test_catalog_keeps_mixed_family_selectable(tmp_path, monkeypatch):
    from smolsmort.forecast import pipeline, views

    ws = make_eligibility_workspace()
    monkeypatch.setattr(pipeline, "load_workspace", lambda *args, **kwargs: ws)
    monkeypatch.setattr(views, "probe_families", lambda: {"snaive": (True, "")})
    monkeypatch.setattr(views, "registered_families", lambda: [get_family("snaive")])
    entry = family_catalog(tmp_path, ws.spec, SimpleNamespace(folder=tmp_path, summary={}))[0]
    assert entry["suitable"] is True and entry["reason"] is None
    assert entry["skipped_series"] == 2
    ws.profile["period"] = 100
    entry = family_catalog(tmp_path, ws.spec, SimpleNamespace(folder=tmp_path, summary={}))[0]
    assert entry["suitable"] is False and entry["skipped_series"] == 4
    assert entry["reason"] == (
        "needs two seasons (200 training steps per series); shortest series has 0"
    )


def shorten_second_series(ws):
    steps = np.unique(ws.panel[STEP])
    cut = steps[timeseries_cuts(len(steps), ws.spec.horizon).val_start]
    keep = (ws.panel[SERIES] != "b") | (ws.panel[STEP] >= cut)
    ws.panel = {name: values[keep] for name, values in ws.panel.items()}
    return series_eligibility(ws.panel, "y", ws.unit, ws.profile, ws.spec.horizon)


@pytest.mark.parametrize("name", ["snaive", "ets", "theta", "arima"])
def test_fit_skips_only_ineligible_series(workspace, monkeypatch, name):
    from smolsmort.forecast.families import stats

    eligibility = shorten_second_series(workspace)
    task = make_task(workspace)
    task.eligibility = eligibility
    if name != "snaive":
        monkeypatch.setattr(stats.StatsFamily, "available", lambda self: (True, ""))

        def forecast(covered, *args):
            assert set(covered.frame.series[covered.rows]) == {"a"}
            return np.ones(int(covered.rows.sum()))

        monkeypatch.setattr(stats, "forecast_rows", forecast)
    predicted = get_family(name).fit(task, objective="squared").predict(task)
    assert np.isfinite(predicted[task.frame.series[task.rows] == "a"]).all()
    assert np.isnan(predicted[task.frame.series[task.rows] == "b"]).all()


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_eligible_nonfinite_predictions_still_fail(workspace, monkeypatch, value):
    from smolsmort.forecast.families import stats

    task = make_task(workspace)
    task.eligibility = shorten_second_series(workspace)
    monkeypatch.setattr(stats.StatsFamily, "available", lambda self: (True, ""))
    monkeypatch.setattr(stats, "forecast_rows", lambda task, *args: np.full(task.rows.sum(), value))
    with pytest.raises(PipelineError, match="nonfinite predictions"):
        get_family("ets").fit(task, objective="squared")


def test_short_series_score_and_finish_preserve_n_a(workspace):
    shorten_second_series(workspace)
    genome = make_genome([], "squared", {}, family="snaive")
    scored = score(workspace, genome)
    assert np.isfinite(scored.fitness)
    result = finish(workspace, genome)
    assert np.isfinite(result["model_error"])
    assert result["model_error"] == result["baseline_error"]
    forecast = result["forecast"]
    assert np.isnan(forecast["prediction"][forecast["series"] == "b"]).all()
    assert np.isfinite(forecast["prediction"][forecast["series"] == "a"]).all()


@pytest.mark.parametrize("name", ["ets", "theta", "arima"])
def test_real_stats_preserve_covered_predictions(workspace, name):
    pytest.importorskip("statsforecast")
    task = make_task(workspace)
    expected = get_family(name).fit(task, objective="squared").predict(task)
    task.eligibility = shorten_second_series(workspace)
    task.panel = workspace.panel
    predicted = get_family(name).fit(task, objective="squared").predict(task)
    covered = task.frame.series[task.rows] == "a"
    np.testing.assert_allclose(predicted[covered], expected[covered])
    assert np.isnan(predicted[~covered]).all()


@pytest.mark.parametrize("transform", ["log1p", "logdiff"])
def test_skipped_series_do_not_enter_target_transforms(workspace, transform):
    shorten_second_series(workspace)
    workspace.panel["y"][workspace.panel[SERIES] == "b"] = -1
    genome = make_genome([], "squared", {}, family="snaive", transform=transform)
    assert np.isfinite(score(workspace, genome).fitness)
