"""statistical families on exact rolling origins, with optional imports kept lazy"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from smolsmort.forecast.families.base import Param, probe_import
from smolsmort.forecast.spec import SERIES, STEP


def season_length(profile: dict) -> int:
    period = profile.get("genes", {}).get("period", profile.get("period"))
    return int(period) if period is not None and period > 0 else 1


@dataclass
class RawSeriesTask:
    panel: dict
    target: str
    unit: str
    frame: object
    rows: np.ndarray
    bucket: tuple[int, int]
    season_length: int


@dataclass
class StatsFitted:
    prediction: np.ndarray
    season_length: int
    eval_history: dict | None = None
    warnings: tuple[str, ...] = ()

    def predict(self, task: RawSeriesTask) -> np.ndarray:
        return self.prediction.copy()


def join_predictions(output, expected, alias: str) -> np.ndarray:
    """refuse incomplete or ambiguous joins and preserve the requested frame order"""
    keys = ["unique_id", "ds", "cutoff"]
    if expected.duplicated(keys).any() or output.duplicated(keys).any():
        raise ValueError("duplicate series/step/origin matches in statistical forecasts")
    joined = expected.merge(output[keys + [alias]], on=keys, how="left", validate="one_to_one")
    prediction = joined[alias].to_numpy(dtype=float)
    if len(prediction) != len(expected) or not np.isfinite(prediction).all():
        raise ValueError("missing or nonfinite series/step/origin matches in statistical forecasts")
    return prediction


def build_model(name: str, period: int):
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    if name == "ets":
        return AutoETS(season_length=period)
    if name == "theta":
        return AutoTheta(season_length=period)
    return AutoARIMA(
        season_length=period,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        max_order=4,
        nmodels=20,
        approximation=True,
    )


def forecast_rows(task: RawSeriesTask, name: str, nthread: int) -> np.ndarray:
    import pandas as pd
    from statsforecast import StatsForecast

    frame, rows = task.frame, task.rows
    expected = pd.DataFrame(
        {
            "unique_id": frame.series[rows],
            "ds": np.asarray(frame.step[rows], dtype="datetime64[ns]"),
            "cutoff": np.asarray(frame.origin[rows], dtype="datetime64[ns]"),
        }
    )
    if expected.empty:
        return np.empty(0, dtype=float)
    frequencies = {"day": "D", "week": "W-MON", "month": "MS", "quarter": "QS", "year": "YS"}
    freq = frequencies[task.unit]
    panel = pd.DataFrame(
        {
            "unique_id": task.panel[SERIES],
            "ds": np.asarray(task.panel[STEP], dtype="datetime64[ns]"),
            "y": task.panel[task.target],
        }
    )
    panel = panel[panel.unique_id.isin(expected.unique_id) & (panel.ds <= expected.ds.max())]
    parts = []
    for key, targets in expected.groupby("unique_id", sort=False):
        history = panel[(panel.unique_id == key) & (panel.ds <= targets.ds.max())].sort_values("ds")
        earliest = targets.cutoff.min()
        count = int((history.ds <= earliest).sum())
        if count < 2:
            raise ValueError(
                f"insufficient history for series {key!r} at {earliest}: {count} observations"
            )
        if history.ds.duplicated().any() or not np.isfinite(history.y).all():
            raise ValueError(f"duplicate steps or nonfinite history for series {key!r}")
        regular = pd.date_range(history.ds.min(), history.ds.max(), freq=freq)
        if not np.array_equal(history.ds.to_numpy(), regular.to_numpy()):
            raise ValueError(f"irregular history for series {key!r}; expected {freq} steps")
        # future targets extend the cv tail only; every evaluated origin stays in observed history
        if targets.cutoff.max() > history.ds.max():
            raise ValueError(f"unobserved origin for series {key!r}: {targets.cutoff.max()}")
        tail = pd.date_range(history.ds.max(), targets.ds.max(), freq=freq)[1:]
        if len(tail):
            padding = pd.DataFrame({"unique_id": key, "ds": tail, "y": 0.0})
            history = pd.concat([history, padding], ignore_index=True)
        parts.append(history)
    model = build_model(name, task.season_length)
    engine = StatsForecast(models=[model], freq=freq, n_jobs=max(1, nthread))
    output = engine.cross_validation(
        df=pd.concat(parts, ignore_index=True),
        h=task.bucket[1],
        step_size=1,
        n_windows=int(expected.ds.nunique()),
        refit=False,
    )
    return join_predictions(output, expected, str(model))


class StatsFamily:
    needs = "raw_series"
    pip_extra = "stats"

    def __init__(self, name: str):
        self.name = self.label = name

    def available(self) -> tuple[bool, str]:
        if self.name == "snaive":
            return True, ""
        return probe_import("statsforecast", self.pip_extra)

    def space(self) -> list[Param]:
        return []

    def defaults(self) -> dict:
        return {}

    def objectives(self, task: str) -> tuple[str, ...]:
        if task != "regression":
            raise ValueError(f"family {self.name!r} requires regression")
        return ("squared",)

    def cost(self, shape: tuple[int, int], params: dict) -> float:
        rows, series = shape
        return float(rows * series * (20 if self.name == "arima" else 1))

    def fit(
        self,
        x: RawSeriesTask,
        *,
        objective: str,
        y=None,
        feature_types=None,
        params: dict | None = None,
        seed: int = 0,
        nthread: int = 1,
        **options,
    ) -> StatsFitted:
        from smolsmort.forecast.pipeline import PipelineError, _series_baseline

        available, reason = self.available()
        if not available:
            raise PipelineError(f"family {self.name!r} is unavailable; {reason}")
        if params:
            raise PipelineError(f"family {self.name!r} accepts no parameters")
        notices = []
        try:
            if self.name == "snaive":
                prediction = _series_baseline(x.frame, x.rows, x.bucket, x.season_length)
            else:
                with warnings.catch_warnings(record=True) as notices:
                    prediction = forecast_rows(x, self.name, nthread)
            if not np.isfinite(prediction).all():
                raise ValueError("insufficient history: nonfinite predictions")
        except (
            ValueError,
            IndexError,
            KeyError,
            NotImplementedError,
            ZeroDivisionError,
            RuntimeError,
            FloatingPointError,
        ) as exc:
            raise PipelineError(
                f"family {self.name!r} failed in bucket {x.bucket}: "
                f"{type(exc).__name__}: {exc}; check history at the earliest origin"
            ) from exc
        messages = tuple(sorted({str(notice.message) for notice in notices}))
        return StatsFitted(prediction, x.season_length, warnings=messages)
