"""how good a forecast is on rows it never saw: errors, an 80% band from validation residuals,
and the trust verdict that tells the user when a forecast is not worth acting on"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# trust thresholds - from the plan agreed 2026-09-23, not yet measured against real outcomes
BASELINE_RATIO = 0.9
COVERAGE_RANGE = (0.65, 0.95)
MIN_TEST_ROWS = 30
MIN_TEST_STEPS = 13


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
) -> dict:
    """the trust verdict on the untouched test split; every failed check is named with its numbers"""
    checks = []
    ratio = model_error / baseline_error if baseline_error else float("inf")
    checks.append(
        {
            "check": "beats the naive baseline",
            "passed": bool(ratio <= BASELINE_RATIO),
            "detail": f"error {model_error:.3g} vs baseline {baseline_error:.3g} "
            f"(ratio {ratio:.2f}, needs <= {BASELINE_RATIO})",
        }
    )
    has_band = band_coverage is not None and np.isfinite(band_coverage)
    if needs_band and not has_band:
        checks.append(
            {"check": "band is calibrated", "passed": False, "detail": "no band could be fitted"}
        )
    elif needs_band:
        lo, hi = COVERAGE_RANGE
        checks.append(
            {
                "check": "band is calibrated",
                "passed": bool(lo <= band_coverage <= hi),
                "detail": f"80% band covered {band_coverage:.0%} of test rows (needs {lo:.0%}-{hi:.0%})",
            }
        )
    if needs_band and band_width is not None and spread is not None:
        # a band no narrower than the target's own 10-90% range says only "the usual range"
        checks.append(
            {
                "check": "band is sharper than the data's own spread",
                "passed": bool(np.isfinite(band_width) and band_width < spread),
                "detail": f"median 80% band {band_width:.3g} wide vs the target's own "
                f"10-90% range {spread:.3g}",
            }
        )
    enough = test_rows >= MIN_TEST_ROWS and (test_steps is None or test_steps >= MIN_TEST_STEPS)
    steps = "" if test_steps is None else f", {test_steps} steps (needs {MIN_TEST_STEPS})"
    checks.append(
        {
            "check": "enough history to judge",
            "passed": bool(enough),
            "detail": f"{test_rows} test rows (needs {MIN_TEST_ROWS}){steps}",
        }
    )
    trusted = all(c["passed"] for c in checks)
    failed = [c["check"] for c in checks if not c["passed"]]
    summary = (
        "the forecast held up on data it never saw"
        if trusted
        else "do not act on this forecast - it failed: " + ", ".join(failed)
    )
    return {"trusted": trusted, "summary": summary, "checks": checks}
