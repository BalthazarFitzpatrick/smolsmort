"""the effort setting: low and high rescale the ranges of the params that make a model costly"""

from __future__ import annotations

import math

EFFORTS = ("low", "normal", "high")
LOW_SHARE = 0.4
HIGH_FACTOR = 1.5

# per family and param name: the ceiling that high may not pass
COSTLY = {
    "xgboost": {"max_depth": 12},
    "lightgbm": {"num_leaves": 512, "max_depth": 20},
    "catboost": {"depth": 12},
    "extratrees": {"n_estimators": 1000, "max_depth": 32},
    "mlp": {"n_layers": 6, "width": 1024},
    "lstm": {"hidden": 256, "layers": 3, "lookback": 156},
}


def check_effort(effort: str) -> str:
    if effort not in EFFORTS:
        raise ValueError("effort must be low, normal or high")
    return effort


def scale_range(family: str, name: str, lo, hi, kind: str, effort: str = "normal"):
    """the (lo, hi) for an effort level; params that are not costly come back unchanged"""
    ceiling = COSTLY.get(family, {}).get(name)
    if effort == "normal" or ceiling is None or kind not in ("int", "log", "float", "lin"):
        return lo, hi
    if effort == "low":
        if kind == "log":
            new = math.exp(math.log(lo) + LOW_SHARE * (math.log(hi) - math.log(lo)))
        else:
            new = lo + LOW_SHARE * (hi - lo)
        if kind == "int":
            new = max(lo + 1, round(new))
        return lo, min(new, hi)
    if kind in ("int", "log"):
        new = min(hi * HIGH_FACTOR, ceiling)
        if kind == "int":
            new = int(new)
        return lo, max(new, hi)
    return lo, hi


def clip_default(value, lo, hi, kind: str):
    """a default outside the scaled range moves to the nearest bound"""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    clipped = min(max(value, lo), hi)
    return int(clipped) if kind == "int" else clipped
