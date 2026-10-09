"""what the training rows look like, computed from them alone: types, gaps, leaks, censoring,
season and lead-lag - and the starting genes the search seeds itself with"""

from __future__ import annotations

import numpy as np

from smolsmort.forecast.spec import LOWER, SERIES, STEP, UPPER, PrepSpec
from smolsmort.forecast.tables import is_text

# minimum evidence for a seasonal peak
SEASON_ACF = 0.3
MAX_LAG = 26
# an input this close to the target is the target in disguise, e.g. a value set only afterwards
LEAK_CORRELATION = 0.99


def _stats(values: np.ndarray) -> dict:
    if is_text(values):
        present = values[[v is not None and v == v and v != "" for v in values]]
        levels, counts = np.unique(present.astype(str), return_counts=True)
        top = sorted(zip(counts.tolist(), levels.tolist(), strict=True), reverse=True)[:5]
        return {
            "kind": "text",
            "missing": float(1 - len(present) / max(len(values), 1)),
            "levels": int(len(levels)),
            "top": [[level, count] for count, level in top],
        }
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"kind": "number", "missing": 1.0}
    return {
        "kind": "number",
        "missing": float(1 - len(finite) / len(values)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "zeros": float((finite == 0).mean()),
    }


def _leak(values: np.ndarray, target: np.ndarray) -> str | None:
    both = np.isfinite(values) & np.isfinite(target)
    if both.sum() < 10 or np.std(target[both]) == 0:
        return None
    gap = values[both] - target[both]
    if np.std(gap) <= 0.01 * np.std(target[both]):
        return f"equals the target plus {np.mean(gap):.3g} on {int(both.sum())} rows"
    if np.std(values[both]) > 0:
        corr = float(np.corrcoef(values[both], target[both])[0, 1])
        if abs(corr) >= LEAK_CORRELATION:
            return f"correlates {corr:.3f} with the target"
    return None


def _acf(values: np.ndarray, lags: range) -> dict[int, float]:
    x = np.arange(len(values))
    values = values - np.polyval(np.polyfit(x, values, 1), x) if len(values) > 2 else values
    denom = float((values * values).sum())
    if denom == 0:
        return {}
    return {k: float((values[k:] * values[:-k]).sum() / denom) for k in lags if k < len(values)}


def _detect_periods(values, unit):
    """calendar periods supported by nearby acf peaks, with two full cycles of history"""
    candidates = {"day": (7, 365), "week": (52,), "month": (12,), "quarter": (4,), "year": ()}
    detected = {}
    residual = np.asarray(values, dtype=float).copy()
    for period in candidates[unit]:
        if len(values) < 2 * period:
            continue
        radius = max(1, period // 26)
        acf = _acf(residual, range(max(1, period - radius - 1), period + radius + 2))
        peaks = [
            acf[k]
            for k in range(period - radius, period + radius + 1)
            if k in acf and acf[k] >= acf.get(k - 1, 0) and acf[k] >= acf.get(k + 1, 0)
        ]
        strength = max(peaks, default=0.0)
        if strength >= SEASON_ACF:
            detected[period] = strength
            # remove shorter cycles before testing a yearly cycle, so their harmonics do not qualify
            phase = np.arange(len(residual)) % period
            means = np.array([residual[phase == i].mean() for i in range(period)])
            residual -= means[phase]
    return detected


def _step_unit(spec, steps):
    if spec.step:
        return spec.step
    spacing = np.median(np.diff(steps.astype("datetime64[D]")).astype(int)) if len(steps) > 1 else 7
    return next(
        unit
        for limit, unit in (
            (2, "day"),
            (14, "week"),
            (45, "month"),
            (120, "quarter"),
            (float("inf"), "year"),
        )
        if spacing <= limit
    )


def _series_arrays(table, name, train):
    """one array per series of `name`, over its training steps in order"""
    keys = table[SERIES][train]
    order = np.lexsort((table[STEP][train], keys))
    keys, values = keys[order], table[name][train][order]
    return [values[keys == key] for key in np.unique(keys)]


def _lead_lag(measure_series, target_series) -> tuple[int, float, int]:
    """the lag at which a measure's week-on-week changes best match the target's, pooled over
    series; differences strip a shared season that would make every lag look alike"""
    best = (0, 0.0, 0)
    for k in range(1, MAX_LAG + 1):
        xs, ys = [], []
        for m, y in zip(measure_series, target_series, strict=True):
            dm, dy = np.diff(m), np.diff(y)
            if len(dy) > k:
                xs.append(dm[:-k])
                ys.append(dy[k:])
        if not xs:
            break
        x, y = np.concatenate(xs), np.concatenate(ys)
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() < 10 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
            continue
        corr = float(np.corrcoef(x[keep], y[keep])[0, 1])
        if abs(corr) > abs(best[1]):
            best = (k, corr, int(keep.sum()))
    return best


def build_profile(spec: PrepSpec, table: dict, train: np.ndarray) -> dict:
    """the profile of `table`'s training rows; nothing outside `train` is ever read. row mode:
    pass bounds relabelled as of the train cut (splits.labels_as_of), as training will see them"""
    train = np.asarray(train, dtype=bool)
    target = spec.named("target")[0]
    y = table[target][train]
    profile = {"mode": spec.mode, "rows": int(train.sum()), "columns": {}, "leaks": {}}
    for col in spec.columns:
        if col.role in ("dimension", "measure", "target") and col.name in table:
            profile["columns"][col.name] = {"role": col.role, **_stats(table[col.name][train])}
    numeric_target = not is_text(y)
    if numeric_target:
        for name in spec.named("measure"):
            reason = _leak(table[name][train], y) if name in table else None
            if reason:
                profile["leaks"][name] = reason
    if spec.mode == "row":
        if LOWER in table:
            open_rows = np.isfinite(table[LOWER][train]) & np.isinf(table[UPPER][train])
            profile["censored"] = float(open_rows.mean()) if len(open_rows) else 0.0
    else:
        profile.update(_series_profile(spec, table, train, target))
    profile["genes"] = suggest_genes(spec, profile)
    return profile


def _series_totals(table, train, target):
    """training totals in step order, shared by profiling and feature-period selection"""
    steps = np.unique(table[STEP][train])
    totals = np.zeros(len(steps))
    for key in np.unique(table[SERIES][train]):
        mine = train & (table[SERIES] == key)
        index = np.searchsorted(steps, table[STEP][mine])
        np.add.at(totals, index, np.nan_to_num(table[target][mine]))
    return steps, totals


def _series_profile(spec, table, train, target) -> dict:
    target_series = _series_arrays(table, target, train)
    steps, totals = _series_totals(table, train, target)
    unit = _step_unit(spec, steps)
    detected = _detect_periods(totals, unit)
    period = next(iter(detected), None)
    strength = detected.get(period, 0.0)
    zeros = [float((s == 0).mean()) for s in target_series if len(s)]
    slope = np.polyfit(np.arange(len(totals)), totals, 1)[0] if len(totals) > 2 else 0.0
    leads = {}
    for name in spec.named("measure"):
        if name in table and name not in spec.named("target"):
            lag, corr, n = _lead_lag(_series_arrays(table, name, train), target_series)
            leads[name] = {"lag": lag, "corr": corr, "n": n}
    return {
        "steps": int(len(steps)),
        "series": int(len(target_series)),
        "period": int(period) if period is not None and strength >= SEASON_ACF else None,
        "periods": list(detected),
        "period_acf": float(strength),
        "zero_share": {"median": float(np.median(zeros)), "max": float(max(zeros))}
        if zeros
        else {},
        "trend_per_step": float(slope / totals.mean()) if totals.mean() else 0.0,
        "leads": leads,
    }


def suggest_genes(spec: PrepSpec, profile: dict) -> dict:
    """the search's generation-0 seed: objectives that fit the target, the season, useful lags"""
    target = profile["columns"].get(spec.named("target")[0], {})
    if spec.task == "classification":
        objectives = ["logistic"]
    else:
        objectives = ["squared", "absolute"]
        nonnegative = target.get("min", -1) >= 0
        sparse = profile.get("zero_share", {}).get("median", target.get("zeros", 0)) > 0.3
        if nonnegative and sparse:
            objectives += ["tweedie", "poisson"]
        if profile.get("censored", 0) > 0:
            objectives.append("aft")
    genes = {"objectives": objectives, "exclude": sorted(profile["leaks"])}
    if spec.mode == "series":
        genes["period"] = profile.get("period")
        # outside the ~95% band a correlation of pure noise stays inside: 2 / sqrt(n)
        genes["measure_lags"] = {
            name: lead["lag"]
            for name, lead in profile.get("leads", {}).items()
            if lead["n"] and abs(lead["corr"]) > 2 / np.sqrt(lead["n"])
        }
    return genes
