"""optional calendar summaries of per-entity weekly survival curves"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# z of the 90th percentile of a standard normal; the band is the 10th to 90th percentile
Z_80 = 1.2815515655446004

Table = dict[str, np.ndarray]


def build_curve_table(curves: dict) -> Table:
    """flatten the fitted weekly grid without changing its conditional survival values"""
    weeks = np.asarray(curves["week"])
    row = np.repeat(curves["row"], len(weeks))
    anchor = np.repeat(np.asarray(curves["anchor"]).astype("datetime64[D]"), len(weeks))
    week = np.tile(weeks, len(curves["row"]))
    table = {
        "row": row,
        "anchor": anchor,
        "week": week,
        "date": anchor + (week * 7).astype("timedelta64[D]"),
        "hazard": np.asarray(curves["hazard"]).reshape(-1),
        "survival": np.asarray(curves["survival"]).reshape(-1),
    }
    if "weight" in curves:
        table["weight"] = np.repeat(np.asarray(curves["weight"], dtype=float), len(weeks))
    return table


def expected_events(curve_table: Table, unit: str) -> Table:
    """sum survival drops by monday calendar week; curve dates already encode the target unit.
    with a `weight` column each drop is scaled by it, so the sum is expected units, and the sum
    gets a 10-90% band that treats every entity-week as an independent draw"""
    order = np.lexsort((curve_table["week"], curve_table["row"]))
    row = np.asarray(curve_table["row"])[order]
    survival = np.asarray(curve_table["survival"], dtype=float)[order]
    survival_start = np.ones(len(row))
    same_entity = row[1:] == row[:-1]
    survival_start[1:] = np.where(same_entity, survival[:-1], 1.0)
    dates = np.asarray(curve_table["date"])[order].astype("datetime64[D]")
    monday = dates - ((dates.astype(np.int64) + 3) % 7).astype("timedelta64[D]")
    steps, positions = np.unique(monday, return_inverse=True)
    drop = survival_start - survival
    if "weight" not in curve_table:
        totals = np.bincount(positions, weights=drop, minlength=len(steps))
        return {"step": steps, "prediction": totals}
    weight = np.asarray(curve_table["weight"], dtype=float)[order]
    chance = np.clip(drop, 0.0, 1.0)
    totals = np.bincount(positions, weights=weight * chance, minlength=len(steps))
    variance = np.bincount(
        positions, weights=weight**2 * chance * (1 - chance), minlength=len(steps)
    )
    spread = Z_80 * np.sqrt(variance)
    return {
        "step": steps,
        "prediction": totals,
        "lower": np.maximum(totals - spread, 0.0),
        "upper": totals + spread,
    }


def skip_aggregation(curve_table: Table, unit: str) -> None:
    """keep the entity curves without a calendar summary"""
    return None


REGISTRY: dict[str, Callable[[Table, str], Table | None]] = {
    "none": skip_aggregation,
    "expected_events": expected_events,
}


def get_aggregator(name: str) -> Callable[[Table, str], Table | None]:
    if not isinstance(name, str) or name not in REGISTRY:
        raise ValueError(f"aggregate must be one of: {', '.join(REGISTRY)}")
    return REGISTRY[name]
