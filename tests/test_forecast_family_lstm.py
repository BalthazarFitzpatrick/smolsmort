"""lstm family: global torch model on raw weekly history, tiny panel for speed"""

from __future__ import annotations

import builtins
import copy
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smolsmort.forecast.families.lstm import LstmFamily  # noqa: E402
from smolsmort.forecast.families.stats import (  # noqa: E402
    RawSeriesTask,
    season_length,
    series_eligibility,
)
from smolsmort.forecast.features import buckets_for, timeseries_features  # noqa: E402
from smolsmort.forecast.pipeline import Workspace  # noqa: E402
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec  # noqa: E402

COUNT, HORIZON, LATE_START = 120, 4, 112
PARAMS = {"hidden": 8.0, "layers": 1.0, "lookback": 13.0, "dropout": 0.0, "learning_rate": 5e-3}


def build_panel(shift: float = 0.0, after: int | None = None):
    weeks = np.datetime64("2023-01-02") + 7 * np.arange(COUNT).astype("timedelta64[D]")
    t = np.arange(COUNT)
    base = 50 + t / 10 + 8 * np.sin(2 * np.pi * t / 4)
    third = np.where(t >= LATE_START, base, np.nan)
    values = [base, base * 2 + 3, third]
    if after is not None:
        values = [np.where(t > after, v + shift, v) for v in values]
    panel = {
        SERIES: np.repeat(["a", "b", "c"], COUNT),
        STEP: np.tile(weeks, 3),
        "y": np.concatenate(values),
    }
    return panel, weeks


def build_task(panel, weeks, phase="val"):
    future = weeks[-1] + 7 * np.arange(1, HORIZON + 1).astype("timedelta64[D]")
    scaffold = {SERIES: np.repeat(["a", "b", "c"], HORIZON), STEP: np.tile(future, 3)}
    spec = PrepSpec(
        source="synthetic.csv",
        mode="timeseries",
        task="regression",
        step="week",
        horizon=HORIZON,
        columns=(Column("week", "time"), Column("y", "target", "sum")),
    )
    ws = Workspace(spec, None, {}, {"genes": {"period": 4}, "period": 4}, [], "week")
    bucket = buckets_for(HORIZON)[0]
    frame = timeseries_features(
        spec, panel, scaffold, bucket, "week", panel[STEP] < weeks[81], period=4
    )
    cuts = {
        "val": ~frame.future & (frame.step >= weeks[81]) & (frame.step < weeks[107]),
        "test": ~frame.future & (frame.step >= weeks[107]),
    }
    return RawSeriesTask(
        panel,
        "y",
        "week",
        frame,
        cuts[phase],
        bucket,
        season_length(ws.profile),
        series_eligibility(panel, "y", "week", ws.profile, HORIZON),
    )


@pytest.fixture(scope="module")
def task():
    panel, weeks = build_panel()
    return build_task(panel, weeks)


@pytest.fixture(scope="module")
def fitted(task):
    return LstmFamily().fit(task, objective="squared", params=PARAMS, seed=3)


def test_contract():
    family = LstmFamily()
    assert family.needs == "raw_series" and family.pip_extra == "neural"
    names = [p.name for p in family.space()]
    assert names == ["hidden", "layers", "lookback", "dropout", "learning_rate"]
    assert family.defaults() == {
        "hidden": 32,
        "layers": 1,
        "lookback": 26,
        "dropout": 0.1,
        "learning_rate": 1e-3,
    }
    assert family.objectives("regression") == ("squared",)
    with pytest.raises(ValueError):
        family.objectives("classification")
    assert family.cost((100, 3), {"hidden": 64.0}) > family.cost((100, 3), {"hidden": 8.0})


def test_predictions_shape_and_nan_for_short_series(task, fitted):
    prediction = fitted.predict(task)
    rows = task.rows
    assert prediction.shape == (int(rows.sum()),)
    series = task.frame.series[rows]
    short = series == "c"
    assert short.any() and (~short).any()
    assert np.isnan(prediction[short]).all()
    assert np.isfinite(prediction[~short]).all()


def test_eval_history_has_epoch_curves(fitted):
    history = fitted.eval_history
    assert history["metric"] == "squared_error"
    assert 1 <= len(history["training"]) <= 100
    assert len(history["training"]) == len(history["validation"])


def test_no_leakage_after_origin(task):
    origins = np.unique(task.frame.origin[task.rows])
    first = origins[0]
    only = task.rows & (task.frame.origin == first)
    base = RawSeriesTask(**{**task.__dict__, "rows": only})
    weeks = np.unique(task.panel[STEP])
    cut = int(np.searchsorted(weeks, first, side="right")) - 1
    shifted_panel, _ = build_panel(shift=500.0, after=cut)
    shifted = build_task(shifted_panel, weeks)
    shifted = RawSeriesTask(**{**shifted.__dict__, "rows": only})
    family = LstmFamily()
    one = family.fit(base, objective="squared", params=PARAMS, seed=1).predict(base)
    two = family.fit(shifted, objective="squared", params=PARAMS, seed=1).predict(shifted)
    keep = np.isfinite(one)
    assert keep.any()
    np.testing.assert_array_equal(one[keep], two[keep])


def test_fixed_seed_is_deterministic(task, fitted):
    again = LstmFamily().fit(task, objective="squared", params=PARAMS, seed=3)
    np.testing.assert_array_equal(fitted.predict(task), again.predict(task))


def test_thread_count_restored(task):
    before = torch.get_num_threads()
    LstmFamily().fit(task, objective="squared", params=PARAMS, nthread=2)
    assert torch.get_num_threads() == before


def test_unavailable_library_hint(monkeypatch):
    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.setattr("smolsmort.forecast.families.base.import_module", refuse)
    ok, hint = LstmFamily().available()
    assert ok is False and "uv sync --extra neural" in hint


def test_writes_no_files(task, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    LstmFamily().fit(copy.copy(task), objective="squared", params=PARAMS)
    assert os.listdir(tmp_path) == []
