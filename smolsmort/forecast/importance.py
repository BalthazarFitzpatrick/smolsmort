"""feature contributions from fitted winners, named as the input pipeline names them"""

from __future__ import annotations

import numpy as np

METHODS = {
    "xgboost": "gain",
    "lightgbm": "gain",
    "ridge": "absolute_standardized_coefficient",
}


def measure_importance(family, fitted, names, training=None) -> dict[str, float] | None:
    """normalise one fit, retaining all columns until horizon buckets have been combined"""
    if family == "xgboost":
        model = getattr(fitted, "fitted", fitted)
        scores = model.booster.get_score(importance_type="gain")
        values = np.array([scores.get(f"f{i}", 0.0) for i in range(len(names))])
    elif family == "lightgbm":
        values = fitted.booster.feature_importance(importance_type="gain")
    elif family == "ridge":
        # scale indicators as well as numeric columns, then reunite each input's encodings
        scaled = np.abs(fitted.coefficients) * fitted.transform(training).std(axis=0)
        values = np.zeros(len(names))
        offset = len(fitted.numeric)
        values[fitted.numeric] = scaled[:offset]
        count = len(fitted.missing)
        values[fitted.missing] += scaled[offset : offset + count]
        offset += count
        for column, levels in fitted.categories:
            count = len(levels) + 1
            values[column] += scaled[offset : offset + count].sum()
            offset += count
    else:
        return None
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values) & (values > 0), values, 0)
    total = values.sum()
    return dict(zip(names, values / total if total else values, strict=True))


def combine_importance(family, buckets) -> dict | None:
    """give each horizon fit equal weight, then normalise its twenty largest contributions"""
    if family not in METHODS:
        return None
    totals = {}
    for scores in buckets:
        for name, value in scores.items():
            totals[name] = totals.get(name, 0.0) + value
    top = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:20]
    total = sum(value for _, value in top)
    return {
        "method": METHODS[family],
        "features": [
            {"feature": name, "importance": float(value / total)}
            for name, value in top
            if total and value > 0
        ],
    }
