"""forecast errors against naive, an 80% band from validation residuals, and history facts"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc

import numpy as np


def wape(y, p) -> float:
    """sum of absolute errors over the sum of actuals, on rows with a known actual"""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    known = np.isfinite(y)
    total = np.abs(y[known]).sum()
    return float(np.abs(y[known] - p[known]).sum() / total) if total else float("nan")


def mae(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    known = np.isfinite(y)
    return float(np.abs(y[known] - p[known]).mean()) if known.any() else float("nan")


def log_loss(codes, probs) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    codes = np.asarray(codes, dtype=int)
    return float(-np.log(probs[np.arange(len(codes)), codes]).mean())


def brier(codes, probs) -> float:
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(codes)), np.asarray(codes, dtype=int)] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def open_consistency(p, lower) -> float:
    """share of still-open rows predicted at or beyond what they have already waited"""
    p, lower = np.asarray(p, dtype=float), np.asarray(lower, dtype=float)
    return float((p >= lower).mean()) if len(p) else float("nan")


def _survival_bounds(lower, upper):
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("survival bounds must be matching one-dimensional arrays")
    closed = np.isfinite(upper) & (lower == upper)
    if not np.all(np.isfinite(lower) & (lower >= 0) & (closed | np.isposinf(upper))):
        raise ValueError("survival metrics need exact events or finite right-censoring times")
    return lower, upper, closed


def concordance(lower, upper, risk) -> float:
    """harrell's c; risk is a predicted duration, so larger values mean later events"""
    lower, _, closed = _survival_bounds(lower, upper)
    risk = np.asarray(risk, dtype=float)
    if risk.shape != lower.shape or not np.isfinite(risk).all():
        raise ValueError("concordance needs one finite predicted duration per entity")
    correct, pairs = 0.0, 0
    for i in np.flatnonzero(closed):
        # event/event time ties are unknown order; an event precedes censoring at the same time
        later = (lower > lower[i]) | ((lower == lower[i]) & ~closed)
        correct += np.count_nonzero(risk[later] > risk[i])
        correct += 0.5 * np.count_nonzero(risk[later] == risk[i])
        pairs += np.count_nonzero(later)
    return float(correct / pairs) if pairs else float("nan")


def _step_survival(times, survival, query, *, left=False):
    positions = np.searchsorted(times, query, side="left" if left else "right")
    return np.r_[1.0, survival][positions]


def kaplan_meier(lower, upper, weeks, *, censoring=False) -> np.ndarray:
    """survival at each requested time; censoring=True fits the reverse km distribution"""
    lower, _, closed = _survival_bounds(lower, upper)
    times, inverse, counts = np.unique(lower, return_inverse=True, return_counts=True)
    events = np.bincount(inverse, weights=closed, minlength=len(times))
    at_risk = len(lower) - np.cumsum(counts) + counts
    if censoring:
        # events precede censoring at ties, so they leave the reverse km risk set first
        at_risk = at_risk - events
        events = counts - events
    hazard = np.divide(events, at_risk, out=np.zeros(len(times)), where=at_risk > 0)
    survival = np.cumprod(1 - hazard)
    return _step_survival(times, survival, np.asarray(weeks, dtype=float))


def integrated_brier(lower, upper, survival_curves, weeks, censor_km) -> float:
    """ipcw brier score integrated by trapezoids over weeks 1..h (one week uses its score).
    censor_km is (all train time knots, reverse km survival at those knots), preserving g(t-)
    for fractional event times; censored rows stay in the denominator, never become events"""
    lower, _, closed = _survival_bounds(lower, upper)
    weeks = np.asarray(weeks, dtype=float)
    survival = np.asarray(survival_curves, dtype=float)
    if weeks.ndim != 1 or survival.shape != (len(lower), len(weeks)):
        raise ValueError("survival curves must have one row per entity and one column per week")
    if not np.all(np.isfinite(weeks) & (weeks > 0)) or np.any(np.diff(weeks) <= 0):
        raise ValueError("scoring weeks must be finite, positive and increasing")
    if not np.all(np.isfinite(survival) & (survival >= 0) & (survival <= 1)):
        raise ValueError("survival probabilities must be finite and between zero and one")
    if not len(lower) or not len(weeks):
        return float("nan")
    times, probabilities = (np.asarray(values, dtype=float) for values in censor_km)
    if (
        times.ndim != 1
        or probabilities.shape != times.shape
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1))
        or np.any(np.diff(probabilities) > 0)
    ):
        raise ValueError("censor_km needs sorted train knots and non-increasing survival")
    event_weights = _step_survival(times, probabilities, lower, left=True)
    week_weights = _step_survival(times, probabilities, weeks)
    occurred = closed[:, None] & (lower[:, None] <= weeks)
    alive = lower[:, None] > weeks
    if np.any(week_weights <= 0) or np.any((event_weights <= 0) & occurred.any(axis=1)):
        raise ValueError("scoring horizon exceeds positive train censoring survival")
    errors = np.divide(
        survival**2, event_weights[:, None], out=np.zeros_like(survival), where=occurred
    )
    errors += np.where(alive, (1 - survival) ** 2 / week_weights, 0)
    scores = errors.mean(axis=0)
    if len(weeks) == 1:
        return float(scores[0])
    return float(np.trapezoid(scores, weeks) / (weeks[-1] - weeks[0]))


def group_kaplan_meier(keys_train, lower, upper, keys_test, weeks, *, waited=None):
    """group survival with pooled fallback; condition on completed weeks as hazard does"""
    lower, upper, _ = _survival_bounds(lower, upper)
    keys_train, keys_test = (
        np.asarray(keys_train, dtype=object),
        np.asarray(keys_test, dtype=object),
    )
    weeks = np.asarray(weeks, dtype=float)
    waited = np.zeros(len(keys_test)) if waited is None else np.floor(np.maximum(waited, 0))
    result = np.empty((len(keys_test), len(weeks)))
    for key in dict.fromkeys(keys_test):
        train, test = keys_train == key, keys_test == key
        if not train.any():
            train = np.ones(len(lower), dtype=bool)
        survival = kaplan_meier(lower[train], upper[train], weeks)
        start = kaplan_meier(lower[train], upper[train], np.nextafter(waited[test], -np.inf))
        # an exhausted empirical tail has no remaining observed events; keep it flat at one
        conditional = np.divide(
            survival[None, :],
            start[:, None],
            out=np.ones((test.sum(), len(weeks))),
            where=start[:, None] > 0,
        )
        result[test] = np.where(weeks < waited[test, None], 1, np.minimum(conditional, 1))
    return result


def aft_survival(prediction, weeks, *, scale=1.0, distribution="normal", waited=None):
    """convert xgboost's exp(mu) predictions to conditional duration survival, in matching units"""
    prediction, weeks = np.asarray(prediction, dtype=float), np.asarray(weeks, dtype=float)
    waited = np.zeros(len(prediction)) if waited is None else np.asarray(waited, dtype=float)
    if scale is None:
        return (weeks[None, :] < np.maximum(prediction, waited)[:, None]).astype(float)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("aft survival needs a positive distribution scale")

    def log_survival(times):
        with np.errstate(divide="ignore"):
            z = (np.log(times) - np.log(prediction)[:, None]) / scale
        if distribution == "normal":
            tail = np.fromiter((0.5 * erfc(v / np.sqrt(2)) for v in z.flat), float).reshape(z.shape)
            with np.errstate(divide="ignore"):
                return np.log(tail)
        if distribution == "extreme":
            return -np.exp(np.minimum(z, 700))
        if distribution == "logistic":
            return -np.logaddexp(0, z)
        raise ValueError(f"unknown aft distribution {distribution!r}")

    log_curve, log_start = log_survival(weeks[None, :]), log_survival(waited[:, None])
    with np.errstate(invalid="ignore"):
        conditional = np.exp(np.minimum(log_curve - log_start, 0))
    # if even the starting tail underflows, the duration step is the available fallback
    step = (weeks[None, :] < np.maximum(prediction, waited)[:, None]).astype(float)
    return np.where(
        weeks[None, :] < waited[:, None], 1, np.where(np.isnan(conditional), step, conditional)
    )


@dataclass(frozen=True)
class Band:
    lo: float
    hi: float
    relative: bool
    floor: float
    n: int


def fit_band(y, p, level: float = 0.8, relative: bool | None = None) -> Band | None:
    """quantiles of validation residuals; relative residuals scale with the prediction so a band
    learned on quiet weeks still widens for busy ones. none under 4 residuals"""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    known = np.isfinite(y) & np.isfinite(p)
    y, p = y[known], p[known]
    if len(y) < 4:
        return None
    if relative is None:
        # a ratio needs a positive scale: for a target centred at zero the floor collapses to
        # almost nothing and a near-zero prediction blows the band up
        relative = bool(y.min() >= 0 and np.median(y) > 0)
    # floor stops a near-zero prediction turning one residual into a huge ratio - unmeasured guess
    floor = max(0.1 * float(np.median(np.abs(y))), 1e-6) if relative else 1.0
    scale = np.maximum(np.abs(p), floor) if relative else 1.0
    resid = (y - p) / scale
    if len(resid) >= 20:
        lo, hi = np.quantile(resid, [(1 - level) / 2, 1 - (1 - level) / 2])
    else:
        # too few for two tails: one symmetric width from the absolute residuals
        width = float(np.quantile(np.abs(resid), level))
        lo, hi = -width, width
    return Band(float(lo), float(hi), relative, floor, len(resid))


def apply_band(band: Band | None, p, nonnegative: bool = False):
    """lower and upper edges around predictions; nan edges when there is no band"""
    p = np.asarray(p, dtype=float)
    if band is None:
        return np.full_like(p, np.nan), np.full_like(p, np.nan)
    scale = np.maximum(np.abs(p), band.floor) if band.relative else 1.0
    lo, hi = p + band.lo * scale, p + band.hi * scale
    if nonnegative:
        lo = np.maximum(lo, 0.0)
    return lo, hi


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    known = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return (
        float(((y[known] >= lo[known]) & (y[known] <= hi[known])).mean())
        if known.any()
        else float("nan")
    )


def group_median(keys_train, y_train, keys_test) -> np.ndarray:
    """the naive baseline: each test row gets its group's median, unseen groups the overall one"""
    y_train = np.asarray(y_train, dtype=float)
    known = np.isfinite(y_train)
    overall = float(np.median(y_train[known]))
    groups: dict = {}
    for key, value in zip(np.asarray(keys_train, dtype=object)[known], y_train[known], strict=True):
        groups.setdefault(key, []).append(value)
    medians = {key: float(np.median(values)) for key, values in groups.items()}
    return np.array([medians.get(key, overall) for key in keys_test], dtype=float)


def judge(
    *,
    model_error: float,
    baseline_error: float,
    band_coverage: float | None,
    test_rows: int,
    test_steps: int | None = None,
    needs_band: bool = True,
    band_width: float | None = None,
    spread: float | None = None,
    metric: str = "wape",
    val_error: float | None = None,
) -> dict:
    """compare untouched test error with naive and report band and history facts"""
    checks = []
    ratio = model_error / baseline_error if baseline_error else float("inf")
    accuracy = {
        "metric": metric,
        "model_error": model_error,
        "val_error": val_error,
        "val_test_gap": model_error - val_error if val_error is not None else None,
        "baseline_error": baseline_error,
        "ratio": ratio,
        "improvement": 1 - ratio if np.isfinite(ratio) else None,
        "text": _accuracy_text(metric, model_error, baseline_error, ratio),
    }
    checks.append(
        {
            "check": "naive baseline comparison",
            "group": "accuracy",
            "detail": f"error {model_error:.3g} vs baseline {baseline_error:.3g} "
            f"(ratio {ratio:.2f})",
        }
    )
    has_band = band_coverage is not None and np.isfinite(band_coverage)
    if needs_band and not has_band:
        checks.append(
            {
                "check": "band coverage",
                "group": "band",
                "detail": "no band could be fitted",
            }
        )
    elif needs_band:
        checks.append(
            {
                "check": "band coverage",
                "group": "band",
                "detail": f"80% band covered {band_coverage:.0%} of test rows",
            }
        )
    if needs_band and band_width is not None and spread is not None:
        checks.append(
            {
                "check": "band width and target spread",
                "group": "band",
                "detail": f"median 80% band {band_width:.3g} wide vs the target's own "
                f"10-90% range {spread:.3g}",
            }
        )
    steps = "" if test_steps is None else f", {test_steps} steps"
    checks.append(
        {
            "check": "test history",
            "group": "history",
            "detail": f"{test_rows} test rows{steps}",
        }
    )
    comparison = "better" if ratio < 1 else "worse"
    summary = f"ratio {ratio:.2f}, {comparison} than naive: {accuracy['text']}"
    return {
        "summary": summary,
        "accuracy": accuracy,
        "checks": checks,
    }


def _accuracy_text(metric: str, model: float, baseline: float, ratio: float) -> str:
    if metric == "wape":
        model_text, baseline_text = f"{model:.1%}", f"{baseline:.1%}"
    else:
        model_text, baseline_text = f"{model:.3g}", f"{baseline:.3g}"
    gain = (
        f"{abs(1 - ratio):.0%} {'better' if ratio < 1 else 'worse'}" if np.isfinite(ratio) else ""
    )
    return f"{metric} {model_text} vs naive baseline {baseline_text}" + (
        f", {gain}" if gain else ""
    )
