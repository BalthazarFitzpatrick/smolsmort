"""single statistical forecasts use each origin's complete, observed history"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast.families import get_family
from smolsmort.forecast.families import stats as stats_module
from smolsmort.forecast.families.stats import RawSeriesTask, build_model, forecast_single_rows
from smolsmort.forecast.pipeline import PipelineError
from smolsmort.forecast.spec import SERIES, STEP


def make_task(origins=(70, 71, 72, 73, 74, 75), *, future=False):
    count, horizon = 90, 8
    days = np.datetime64("2024-01-01") + (7 * np.arange(count)).astype("timedelta64[D]")
    values = 50 + np.arange(count) / 10 + 8 * np.sin(2 * np.pi * np.arange(count) / 13)
    panel = {
        SERIES: np.repeat(["a", "b"], count),
        STEP: np.tile(days, 2),
        "y": np.concatenate([values, values * 2 + 3]),
    }
    records = [
        (key, days[origin], lead)
        for key in ("b", "a")
        for origin in origins
        for lead in range(1, horizon + 1)
        if future or origin + lead < count
    ]
    keys, origin_dates, leads = map(np.asarray, zip(*records, strict=True))
    frame = SimpleNamespace(
        series=keys,
        origin=origin_dates,
        weeks_ahead=leads,
        step=origin_dates + (7 * leads).astype("timedelta64[D]"),
    )
    return RawSeriesTask(
        panel,
        "y",
        "week",
        frame,
        np.ones(len(keys), dtype=bool),
        (1, horizon),
        13,
        horizon_mode="single",
    )


def test_single_stats_reused_fit_updates_each_origin_and_preserves_row_order(monkeypatch):
    pd = pytest.importorskip("pandas")
    calls = []

    class Model:
        def fit(self, values):
            self.values = values
            calls.append(("fit", values.copy()))
            return self

        def predict(self, h):
            return {"mean": self.values[-1] + np.arange(1, h + 1)}

        def forward(self, y, h):
            calls.append(("forward", y.copy()))
            return {"mean": y[-1] + np.arange(1, h + 1)}

    monkeypatch.setattr(stats_module, "build_model", lambda *args: Model())
    task = make_task()
    actual = forecast_single_rows(task, "ets")
    for index, (key, origin, lead) in enumerate(
        zip(task.frame.series, task.frame.origin, task.frame.weeks_ahead, strict=True)
    ):
        selected = (task.panel[SERIES] == key) & (task.panel[STEP] == origin)
        assert actual[index] == task.panel["y"][selected][0] + lead
    assert [kind for kind, _ in calls] == [
        "fit",
        "forward",
        "forward",
        "forward",
        "fit",
        "forward",
    ] * 2
    assert [len(values) for _, values in calls] == list(range(71, 77)) * 2
    assert pd.Timestamp(task.frame.origin.max()) < pd.Timestamp(task.panel[STEP].max())


@pytest.mark.parametrize("name", ["ets", "theta", "arima"])
@pytest.mark.parametrize("replacement", [np.nan, 1e12])
def test_single_stats_never_reads_after_origin(name, replacement):
    pytest.importorskip("statsforecast")
    task = make_task(origins=(70,))
    expected = get_family(name).fit(task, objective="squared").predict(task)
    changed = copy.deepcopy(task)
    changed.panel["y"][changed.panel[STEP] > task.frame.origin.max()] = replacement
    actual = get_family(name).fit(changed, objective="squared").predict(changed)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", ["ets", "theta", "arima"])
def test_single_stats_refits_every_four_origins_and_updates_state(name):
    pytest.importorskip("statsforecast")
    task = make_task()
    actual = get_family(name).fit(task, objective="squared").predict(task)
    for key in ("a", "b"):
        values = task.panel["y"][task.panel[SERIES] == key]
        fitted = None
        for number, origin in enumerate(range(70, 76)):
            if number % 4 == 0:
                fitted = build_model(name, 13).fit(values[: origin + 1])
                forecast = fitted.predict(h=8)["mean"]
            else:
                forecast = fitted.forward(y=values[: origin + 1], h=8)["mean"]
            origin_date = task.panel[STEP][origin]
            selected = (task.frame.series == key) & (task.frame.origin == origin_date)
            np.testing.assert_allclose(actual[selected], forecast, rtol=1e-12)


@pytest.mark.parametrize("name", ["ets", "theta", "arima"])
def test_single_stats_tail_and_future_rows_keep_all_available_leads(name):
    pytest.importorskip("statsforecast")
    task = make_task(origins=(86, 87, 88))
    prediction = get_family(name).fit(task, objective="squared").predict(task)
    assert len(prediction) == 2 * (3 + 2 + 1)
    assert np.isfinite(prediction).all()
    task = make_task(origins=(89,), future=True)
    prediction = get_family(name).fit(task, objective="squared").predict(task)
    assert len(prediction) == 16
    assert np.isfinite(prediction).all()


def test_single_stats_rejects_unobserved_origin():
    pytest.importorskip("statsforecast")
    task = make_task(origins=(70,))
    task.panel = {
        name: values[task.panel[STEP] < task.frame.origin.max()]
        for name, values in task.panel.items()
    }
    with pytest.raises(PipelineError, match="unobserved origin"):
        get_family("ets").fit(task, objective="squared")
