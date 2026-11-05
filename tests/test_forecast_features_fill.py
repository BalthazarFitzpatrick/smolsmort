"""shutdown history stays causal while labels retain observed demand"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast.features import (
    fill_shutdown_weeks,
    stacked_timeseries_features,
    timeseries_features,
)
from smolsmort.forecast.pipeline import _timeseries_baseline, _timeseries_raw_fit, make_genome
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec
from smolsmort.forecast.transform import get_transform

FILL = {"below": 0.5, "median_weeks": 13, "min_level": 10}


def test_shutdown_fill_excludes_consecutive_shutdowns_and_preserves_input():
    values = np.array([100, 110, 120, 130, 10, 5, 70, 60, np.nan, np.inf])
    original = values.copy()
    filled, mask = fill_shutdown_weeks(values, **FILL)
    np.testing.assert_array_equal(mask, [False] * 4 + [True, True] + [False] * 4)
    np.testing.assert_array_equal(filled[:8], [100, 110, 120, 130, 115, 115, 70, 60])
    np.testing.assert_array_equal(values, original)
    assert np.isnan(filled[8]) and np.isinf(filled[9])
    changed = values.copy()
    changed[6:] = 10000
    np.testing.assert_array_equal(fill_shutdown_weeks(changed, **FILL)[0][:6], filled[:6])


def test_fill_requires_four_normal_finite_weeks_and_minimum_level():
    assert not fill_shutdown_weeks([100, np.nan, 100, 100, 0], **FILL)[1].any()
    assert not fill_shutdown_weeks([1, 1, 1, 1, 0], **FILL)[1].any()
    filled, mask = fill_shutdown_weeks([10, 10, 10, 10, 5, 4], **FILL)
    np.testing.assert_array_equal(mask, [False] * 5 + [True])
    assert filled[-1] == 10


def test_fill_uses_only_the_trailing_calendar_window():
    values = [1000] * 5 + [600] * 4 + [200]
    filled, mask = fill_shutdown_weeks(values, 0.5, 4, 10)
    assert not mask[:-1].any() and mask[-1] and filled[-1] == 600


def make_panel():
    days = np.datetime64("2024-01-01") + np.arange(20) * 7
    spec = PrepSpec(
        "synthetic.csv",
        "timeseries",
        "regression",
        (Column("day", "time"), Column("y", "target", "sum"), Column("m", "measure", "sum")),
        horizon=2,
        history_fill=FILL,
    )
    panel = {
        STEP: np.tile(days, 2),
        SERIES: np.repeat(["a", "b"], 20),
        "y": np.concatenate(([100.0] * 13 + [10.0, 5.0] + [100.0] * 5, [1.0] * 20)),
        "m": np.arange(40, dtype=float),
    }
    scaffold = {STEP: np.array([], dtype="datetime64[D]"), SERIES: np.array([], dtype=str)}
    return spec, panel, scaffold


@pytest.mark.parametrize("stacked", [False, True])
def test_builders_fill_lags_and_logdiff_base_but_preserve_labels_and_measures(stacked):
    spec, panel, scaffold = make_panel()
    original = panel["y"].copy()
    train = np.ones(40, dtype=bool)

    def build(spec, panel):
        if stacked:
            return stacked_timeseries_features(spec, panel, scaffold, "week", train)
        return timeseries_features(spec, panel, scaffold, (1, 1), "week", train)

    frame = build(spec, panel)
    raw = build(replace(spec, history_fill=None), panel)
    np.testing.assert_array_equal(frame.y, raw.y)
    np.testing.assert_array_equal(panel["y"], original)
    inside = (frame.series == "a") & (frame.origin == panel[STEP][13])
    assert inside.any() and np.all(frame.base[inside] == 100)
    assert np.any(frame.y[inside] == 5)
    lag = "lag_0" if stacked else "lag_1"
    np.testing.assert_array_equal(frame.features.x[inside, frame.features.names.index(lag)], 100)
    second = (frame.series == "a") & (frame.origin == panel[STEP][14])
    for offset in (0, 1):
        name = f"lag_{offset + (0 if stacked else 1)}"
        np.testing.assert_array_equal(
            frame.features.x[second, frame.features.names.index(name)], 100
        )
    np.testing.assert_array_equal(_timeseries_baseline(frame, inside, (1, 1), 1), 100)
    tiny = frame.series == "b"
    np.testing.assert_array_equal(frame.features.x[tiny], raw.features.x[tiny])
    for index, name in enumerate(frame.features.names):
        if name.startswith("m_"):
            np.testing.assert_array_equal(frame.features.x[:, index], raw.features.x[:, index])
    for name in ("log1p", "logdiff"):
        transform = get_transform(name)
        scaled = transform.fwd(frame.y[inside], frame.base[inside])
        expected = np.log1p(frame.y[inside]) - (np.log1p(100) if name == "logdiff" else 0)
        np.testing.assert_allclose(scaled, expected)
        np.testing.assert_allclose(transform.inv(scaled, frame.base[inside]), frame.y[inside])
    changed = {**panel, "y": original.copy()}
    changed["y"][15:20] = 10000
    future_changed = build(spec, changed)
    np.testing.assert_array_equal(future_changed.features.x[inside], frame.features.x[inside])


@pytest.mark.parametrize("transform", ["none", "log1p", "logdiff"])
@pytest.mark.parametrize("horizon_mode", ["mixed", "single"])
def test_raw_stats_fill_before_transform_without_mutating_panel(
    monkeypatch, transform, horizon_mode
):
    spec, panel, scaffold = make_panel()
    frame = timeseries_features(spec, panel, scaffold, (1, 1), "week", np.ones(40, dtype=bool))
    rows = (frame.series == "a") & (frame.step == panel[STEP][14])
    ws = SimpleNamespace(spec=spec, panel=panel, unit="week", profile={}, horizon_mode=horizon_mode)
    family = SimpleNamespace(fit=lambda task, **kwargs: task)
    monkeypatch.setattr("smolsmort.forecast.pipeline.get_family", lambda name: family)
    genome = make_genome((), "squared", {}, family="ets", transform=transform)
    _, task = _timeseries_raw_fit(ws, genome, frame, rows, (1, 1), 1)
    clock = frame.origin if horizon_mode == "single" else frame.step
    assert task.panel[STEP].max() == clock[rows].max()
    for key, level in (("a", 100), ("b", 1)):
        expected = level if transform == "none" else np.log1p(level)
        np.testing.assert_allclose(task.panel["y"][task.panel[SERIES] == key], expected)
    assert panel["y"][13] == 10 and panel["y"][14] == 5
