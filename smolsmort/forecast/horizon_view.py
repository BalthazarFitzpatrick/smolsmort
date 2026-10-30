"""select an observed lead or blend overlapping predictions without inventing a band"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


def _validate_lead(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("lead must be a positive integer")


def build_blend_weights(horizon: int, weights: dict[int, int] | None = None) -> dict[int, int]:
    """every lead contributes, with unspecified weeks counting once"""
    _validate_lead(horizon)
    if weights is None:
        weights = {}
    if not isinstance(weights, dict):
        raise ValueError("blend_weights must map weeks to weights")
    complete = dict.fromkeys(range(1, horizon + 1), 1)
    for lead, weight in weights.items():
        if isinstance(lead, bool) or not isinstance(lead, Integral) or not 1 <= lead <= horizon:
            raise ValueError(f"blend_weights weeks must be integers from 1 to {horizon}")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, Integral)
            or weight not in (1, 2, 4, 8)
        ):
            raise ValueError("blend_weights values must be 1, 2, 4, or 8")
        complete[int(lead)] = int(weight)
    return complete


def validate_blend_weights(weights: dict[int, float]) -> dict[int, float]:
    if not isinstance(weights, dict) or not weights:
        raise ValueError("weights must be a nonempty mapping of leads to weights")
    validated = {}
    for lead, weight in weights.items():
        _validate_lead(lead)
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise ValueError("weights must be finite and nonnegative")
        try:
            weight = float(weight)
        except (OverflowError, ValueError, TypeError) as exc:
            raise ValueError("weights must be finite and nonnegative") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weights must be finite and nonnegative")
        validated[int(lead)] = weight
    if not any(weight > 0 for weight in validated.values()):
        raise ValueError("weights must contain at least one positive value")
    return validated


def lead_rows(table: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """keep exactly lead n, including its fitted band; old tables pass through"""
    if "weeks_ahead" not in table:
        return table
    _validate_lead(n)
    selected = np.asarray(table["weeks_ahead"]) == n
    return {name: np.asarray(values)[selected] for name, values in table.items()}


def blend_rows(
    table: dict[str, np.ndarray], weights: dict[int, float] | None = None
) -> dict[str, np.ndarray]:
    """blend all available predictions per target with explicit, locally normalized weights"""
    if "weeks_ahead" not in table:
        return table
    validated = validate_blend_weights(weights) if weights is not None else None
    columns = {name: np.asarray(values) for name, values in table.items()}
    groups = {}
    for index, target in enumerate(zip(columns["series"], columns["step"], strict=True)):
        groups.setdefault(target, []).append(index)
    first_rows = np.asarray([indices[0] for indices in groups.values()], dtype=int)
    # a blend has no single forecast origin, lead, horizon bucket, or fitted band
    omitted = {"origin", "weeks_ahead", "bucket", "lower", "upper"}
    result = {name: values[first_rows] for name, values in columns.items() if name not in omitted}
    averaged = [name for name in ("prediction", "naive_prediction") if name in result]
    for name in averaged:
        result[name] = np.empty(len(groups), dtype=float)
    for target_index, indices in enumerate(groups.values()):
        if validated is None:
            target_weights = np.ones(len(indices), dtype=float)
        else:
            target_weights = np.asarray(
                [validated.get(lead, 0.0) for lead in columns["weeks_ahead"][indices]],
                dtype=float,
            )
        largest = float(target_weights.max())
        if largest <= 0:
            raise ValueError("weights must include a positive value for every target week")
        # scaling before normalizing keeps finite weights from overflowing their sum
        target_weights /= largest
        target_weights /= target_weights.sum()
        positive = target_weights > 0
        for name in averaged:
            values = columns[name][indices].astype(float)
            result[name][target_index] = np.dot(values[positive], target_weights[positive])
    return result
