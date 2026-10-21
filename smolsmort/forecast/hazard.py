"""weekly person-period exposures and conditional survival for entity forecasts"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smolsmort.forecast.model import Fitted, ModelError, predict

FEATURE_NAMES = ["hazard_week", "hazard_calendar_week"]
# tail extrapolation stops this many weeks past its start; further means "not within reach"
TAIL_LIMIT_WEEKS = 104


@dataclass
class PeriodRows:
    x: np.ndarray
    y: np.ndarray | None
    entity: np.ndarray
    week: np.ndarray
    date: np.ndarray
    cap: int


@dataclass
class HazardFitted:
    fitted: Fitted
    cap: int

    @property
    def eval_history(self):
        return self.fitted.eval_history


def expand_periods(x, lower, upper, anchors, *, cap=None, entity=None, prediction=False):
    """expand weekly bounds; the final bucket pools the tail, keeping entity ids for every row.
    infer a cap only from training input; all other phases must reuse that cap"""
    x = np.asarray(x, dtype=np.float32)
    anchors = np.asarray(anchors).astype("datetime64[D]")
    entity = np.arange(len(x)) if entity is None else np.asarray(entity)
    if prediction:
        if cap is None:
            raise ModelError("hazard prediction needs the training cap")
    else:
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        usable = np.isfinite(lower) & (lower >= 0) & ~np.isnan(upper) & ~np.isnat(anchors)
        if cap is None:
            if not usable.any():
                raise ModelError("hazard needs finite non-negative training durations")
            cap = int(np.floor(np.percentile(lower[usable], 95)))
    if not np.isfinite(cap) or cap < 0 or cap != int(cap):
        raise ModelError("hazard cap must be a non-negative whole week")
    cap = int(cap)
    if prediction:
        counts = np.full(len(x), cap + 1, dtype=int)
    else:
        closed = np.isfinite(upper)
        duration = np.floor(np.where(usable, lower, 0))
        counts = np.where(usable, np.minimum(duration + closed, cap + 1), 0).astype(int)
    positions = np.repeat(np.arange(len(x)), counts)
    week = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    dates = anchors[positions] + (week * 7).astype("timedelta64[D]")
    # the iso week belongs to the year containing that week's thursday
    thursday = dates + (3 - (dates.astype(int) + 3) % 7).astype("timedelta64[D]")
    year_start = thursday.astype("datetime64[Y]").astype("datetime64[D]")
    calendar_week = (thursday - year_start).astype(int) // 7 + 1
    features = np.column_stack([x[positions], week, calendar_week]).astype(np.float32)
    labels = None
    if not prediction:
        labels = (closed[positions] & (week == np.minimum(duration[positions], cap))).astype(
            np.float32
        )
    return PeriodRows(features, labels, entity[positions], week, dates, cap)


def predict_hazard(fitted, x, anchors, *, waited=None):
    """return weekly hazards, conditional survival and capped duration quantiles in weeks"""
    periods = expand_periods(x, None, None, anchors, cap=fitted.cap, prediction=True)
    week = np.arange(fitted.cap + 1)
    shape = (len(x), len(week))
    hazards = predict(fitted.fitted, periods.x).reshape(shape) if len(x) else np.empty(shape)
    waited = np.zeros(len(x)) if waited is None else np.asarray(waited, dtype=float)
    waiting_week = np.floor(np.maximum(waited, 0))
    remaining = week[None, :] >= waiting_week[:, None]
    # summing only remaining log hazards avoids underflow in s_k / s_(w-1)
    log_survival = np.log1p(-np.clip(hazards.astype(float), 0, 1 - np.finfo(float).eps))
    survival = np.exp(np.cumsum(np.where(remaining, log_survival, 0), axis=1))
    survival_start = np.column_stack([np.ones(len(x)), survival[:, :-1]])

    # past the cap the pooled tail hazard repeats, so quantiles never precede the observed wait
    eps = np.finfo(float).eps
    tail = np.clip(hazards[:, -1].astype(float), eps, 1 - eps)
    start = np.maximum(waiting_week, fitted.cap + 1)
    start_survival = np.where(waiting_week > fitted.cap, 1.0, survival[:, -1])

    def find_quantile(probability):
        crossed = survival <= 1 - probability
        with np.errstate(divide="ignore"):
            extra = np.ceil(np.log((1 - probability) / start_survival) / np.log1p(-tail))
        limited = ~crossed.any(axis=1) & (extra > TAIL_LIMIT_WEEKS)
        beyond = start + np.clip(extra, 1, TAIL_LIMIT_WEEKS) - 1
        weeks = np.where(crossed.any(axis=1), crossed.argmax(axis=1), beyond).astype(float)
        return weeks, limited

    (prediction, median_limited), (lower, _), (upper, upper_limited) = (
        find_quantile(probability) for probability in (0.5, 0.1, 0.9)
    )
    return {
        "week": week,
        "hazard": hazards,
        "survival": survival,
        "survival_start": survival_start,
        "prediction": prediction,
        "lower": lower,
        "upper": upper,
        # median or p90 hit the tail limit: report as not within reach, not as a date
        "beyond_limit": median_limited | upper_limited,
    }
