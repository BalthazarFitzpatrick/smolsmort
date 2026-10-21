"""measure default forecast candidates and choose a bounded family allocation"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from statistics import median
from time import perf_counter

import numpy as np

DEFAULT_THREADS = (1, 2, 3, 5, 10)


def build_calibration_panel() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """weekly lag features for 20 series, with 2,000 training and 240 held-out rows"""
    rng = np.random.default_rng(79)
    history, train_weeks, predict_weeks, series_count = 52, 100, 12, 20
    total_weeks = history + train_weeks + predict_weeks
    weeks = np.arange(total_weeks, dtype=float)[:, None]
    series = np.arange(series_count, dtype=float)[None, :]
    levels = rng.uniform(20, 80, size=(1, series_count))
    trends = rng.uniform(-0.03, 0.08, size=(1, series_count))
    season = np.sin(2 * np.pi * weeks / 52 + series / series_count)
    measures = levels * (1 + 0.25 * season) + rng.normal(0, 3, (total_weeks, series_count))
    target = np.maximum(
        0,
        levels
        + trends * weeks
        + 0.35 * measures
        + 8 * season
        + rng.normal(0, 4, (total_weeks, series_count)),
    )
    indices = np.arange(history, total_weeks)
    columns = [target[indices - lag] for lag in range(1, 27)]
    for window in (4, 8, 13, 26, 52):
        previous = target[indices[:, None] - np.arange(1, window + 1)]
        columns.extend(
            (
                previous.mean(axis=1),
                previous.std(axis=1),
                previous.min(axis=1),
                previous.max(axis=1),
            )
        )
    columns.extend(measures[indices - lag] for lag in range(1, 9))
    shape = (len(indices), series_count)
    phase = 2 * np.pi * indices[:, None] / 52
    columns.extend(
        np.broadcast_to(values, shape)
        for values in (
            np.sin(phase),
            np.cos(phase),
            indices[:, None] / 52,
            series,
            np.sin(4 * phase),
            np.cos(4 * phase),
        )
    )
    features = np.stack(columns, axis=-1).astype(np.float32)
    train_x = features[:train_weeks].reshape(-1, 60)
    train_y = target[history : history + train_weeks].reshape(-1)
    predict_x = features[train_weeks:].reshape(-1, 60)
    return train_x, train_y, predict_x


def _check_positive_integer(value: int, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be an integer of at least 1")


def _check_threads(threads: Iterable[int]) -> list[int]:
    values = list(threads)
    if not values:
        raise ValueError("threads must contain at least one thread count")
    for value in values:
        _check_positive_integer(value, "threads")
    return sorted(set(values))


def _measure_candidate(family, objective, params, panel, feature_types, count) -> float:
    train_x, train_y, predict_x = panel

    def fit_predict():
        fitted = family.fit(
            train_x,
            objective=objective,
            y=train_y,
            feature_types=feature_types,
            params=params,
            seed=79,
            nthread=count,
        )
        fitted.predict(predict_x)

    fit_predict()
    durations = []
    for _ in range(3):
        started = perf_counter()
        fit_predict()
        durations.append(perf_counter() - started)
    return median(durations)


def measure_candidates(threads: Iterable[int]) -> dict[int, float]:
    """sum each family's median fit and prediction time after one warm-up"""
    from smolsmort.forecast.families import get_family
    from smolsmort.forecast.search import DEFAULTS

    counts = _check_threads(threads)
    candidates = []
    for name, objective in (("xgboost", "squared"), ("lightgbm", "regression_l1")):
        family = get_family(name)
        available, reason = family.available()
        if not available:
            if name == "xgboost":
                raise ValueError(f"calibration needs xgboost; {reason}")
            print(f"skipping lightgbm: {reason}")
            continue
        params = dict(DEFAULTS) if name == "xgboost" else family.defaults()
        candidates.append((family, objective, params))

    panel = build_calibration_panel()
    feature_types = ["q"] * panel[0].shape[1]
    timings = {}
    for count in counts:
        timings[count] = sum(
            _measure_candidate(family, objective, params, panel, feature_types, count)
            for family, objective, params in candidates
        )
    return timings


def choose_parallelism(
    timings: Mapping[int, float], cores: int, max_families: int = 3
) -> dict[str, int]:
    """compare bounded family counts using measured or nearest-lower thread timings"""
    _check_positive_integer(cores, "cores")
    _check_positive_integer(max_families, "max_families")
    counts = _check_threads(timings)
    for count, seconds in timings.items():
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(f"timing for {count} threads must be finite positive seconds")
    best_cap, best_throughput = 1, -math.inf
    for cap in range(1, min(cores, max_families) + 1):
        family_threads = cores // cap
        lower = [count for count in counts if count <= family_threads]
        if not lower:
            raise ValueError(f"no measured thread count at or below {family_threads}; include 1")
        throughput = cap / timings[lower[-1]]
        # caps are visited in ascending order, so equal throughput keeps fewer families
        if throughput > best_throughput:
            best_cap, best_throughput = cap, throughput
    return {"max_concurrent_families": best_cap, "threads_per_family": cores // best_cap}


def run_calibration(
    threads: Iterable[int] = DEFAULT_THREADS, *, max_families: int = 3, write: bool = False
) -> int:
    """print measured timings and optionally save the selected machine defaults"""
    try:
        _check_positive_integer(max_families, "max_families")
        counts = _check_threads(threads)
        cores = os.cpu_count() or 1
        counts = sorted({1, *(min(count, cores) for count in counts)})
        timings = measure_candidates(counts)
        selected = choose_parallelism(timings, cores, max_families)
        print("threads  median seconds")
        for count, seconds in sorted(timings.items()):
            print(f"{count:>7}  {seconds:>14.3f}")
        print(
            f"selected max_concurrent_families={selected['max_concurrent_families']} "
            f"threads_per_family={selected['threads_per_family']} on {cores} cores"
        )
        if write:
            from smolsmort.forecast.machine_config import write_machine_config

            path = write_machine_config(
                {
                    **selected,
                    "calibrated_at": datetime.now(UTC).isoformat(),
                    "calibration": {str(count): seconds for count, seconds in timings.items()},
                }
            )
            print(f"wrote {path}")
    except (OSError, ValueError) as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 1
    return 0
