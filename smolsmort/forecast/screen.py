"""train-only feature-family priors, saved before search for future bouquet sampling"""

from __future__ import annotations

import importlib

import numpy as np

from smolsmort.forecast.families.lightgbm import LightgbmFamily
from smolsmort.forecast.features import STEP_FEATURES, step_index
from smolsmort.forecast.importance import measure_importance
from smolsmort.forecast.pipeline import Workspace, _validation_masks
from smolsmort.forecast.spec import SERIES, STEP


def _rank_values(values: np.ndarray) -> np.ndarray:
    """average ranks preserve ties without requiring scipy"""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts + 1) / 2)[inverse]


def _rank_correlation(values, residual) -> float:
    keep = np.isfinite(values) & np.isfinite(residual)
    if keep.sum() < 3:
        return 0.0
    x, y = _rank_values(values[keep]), _rank_values(residual[keep])
    x, y = x - x.mean(), y - y.mean()
    scale = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(abs(x @ y) / scale, 0, 1)) if scale else 0.0


def _training_data(ws, max_rows, rng):
    candidates = set(ws.families)
    columns = sorted(
        {
            (family, name, kind)
            for frame in ws.frames.values()
            for family, name, kind in zip(
                frame.features.families, frame.features.names, frame.features.types, strict=True
            )
            if family in candidates
        }
    )
    column_index = {column: index for index, column in enumerate(columns)}
    xs, ys = [], []
    period = ws.profile["genes"].get("period") or STEP_FEATURES[ws.unit][3]
    cutoff = (
        ws.series_folds[0][0]
        if ws.series_folds
        else min(
            frame.step[ws.series_masks[bucket]["val"]].min() for bucket, frame in ws.frames.items()
        )
    )
    training_rows = {}
    for bucket_index, (bucket, frame) in enumerate(sorted(ws.frames.items())):
        masks = ws.series_masks[bucket]
        train, _ = next(_validation_masks(ws, bucket))
        assert not np.any(train & (masks["val"] | masks["test"] | frame.future)), (
            "validation and test targets are sealed during screening"
        )
        assert np.all(frame.step[train] < cutoff), (
            "screening rows must precede the first validation fold"
        )
        rows = np.flatnonzero(train)
        row_limit = max_rows // len(ws.frames) + (bucket_index < max_rows % len(ws.frames))
        if len(rows) > row_limit:
            rows = np.sort(rng.choice(rows, row_limit, replace=False))
        training_rows[bucket] = rows
    panel_rows = ws.panel[STEP] < cutoff
    history = dict(
        zip(
            zip(
                ws.panel[SERIES][panel_rows],
                step_index(ws.panel[STEP][panel_rows], ws.unit),
                strict=True,
            ),
            ws.panel[ws.spec.named("target")[0]][panel_rows],
            strict=True,
        )
    )
    for bucket, rows in training_rows.items():
        frame = ws.frames[bucket]
        features = frame.features
        names = features.names
        # use the most recent same-phase value available at each bucket's origin
        lag = ((bucket[1] + period - 1) // period) * period if period else bucket[1]
        baseline = np.array(
            [
                history.get((key, step - lag), np.nan)
                for key, step in zip(
                    frame.series[rows], step_index(frame.step[rows], ws.unit), strict=True
                )
            ]
        )
        residual = frame.y[rows] - baseline
        usable = np.isfinite(residual)
        x = np.full((int(usable.sum()), len(columns)), np.nan, dtype=np.float32)
        for index, column in enumerate(zip(features.families, names, features.types, strict=True)):
            if column in column_index:
                x[:, column_index[column]] = features.x[rows[usable], index]
        xs.append(x)
        ys.append(residual[usable])
    return np.concatenate(xs), np.concatenate(ys), columns


def _fit_gain(x, residual, columns, seed):
    gains = np.zeros(len(columns))
    if len(residual) < 20 or not len(columns):
        return None
    if np.ptp(residual) == 0:
        return gains
    try:
        importlib.import_module("lightgbm")
    except ImportError:
        return None
    fitted = LightgbmFamily().fit(
        x,
        y=residual,
        objective="regression",
        feature_types=[kind for _, _, kind in columns],
        params={"num_leaves": 15, "max_depth": 4},
        seed=seed,
        nthread=1,
        max_rounds=40,
        patience=0,
    )
    names = [str(index) for index in range(len(columns))]
    importance = measure_importance("lightgbm", fitted, names)
    return np.array([importance[name] for name in names])


def screen_families(ws: Workspace, *, seed: int = 0, max_rows: int = 20_000) -> list[dict]:
    """rank families on pre-validation rows by blended max rank correlation and gain share.
    priors are 0.1 + 0.8 * score / top_score, clipped to [0.1, 0.9]; zero scores start at 0.1.
    a seeded 10% of families from the bottom half get priors raised to at least 0.5"""
    if ws.mode != "series":
        raise ValueError("feature screening requires series mode")
    if max_rows < 1:
        raise ValueError("screening max_rows must be positive")
    if not ws.families:
        return []
    x, residual, columns = _training_data(ws, max_rows, np.random.default_rng(seed))
    gains = _fit_gain(x, residual, columns, seed)
    has_gain = gains is not None
    if gains is None:
        gains = np.zeros(len(columns))
    scores = {
        family: {"family": family, "score": 0.0, "rank_corr": 0.0, "gain": 0.0}
        for family in sorted(set(ws.families))
    }
    for index, (family, _, _) in enumerate(columns):
        entry = scores[family]
        entry["rank_corr"] = max(entry["rank_corr"], _rank_correlation(x[:, index], residual))
        entry["gain"] += float(gains[index])
    for entry in scores.values():
        entry["score"] = (
            (entry["rank_corr"] + entry["gain"]) / 2 if has_gain else entry["rank_corr"]
        )
    ranked = sorted(scores.values(), key=lambda entry: (-entry["score"], entry["family"]))
    top_score = ranked[0]["score"]
    for entry in ranked:
        entry["p_on"] = (
            float(np.clip(0.1 + 0.8 * (entry["score"] / top_score), 0.1, 0.9))
            if entry["score"] > 0 and top_score > 0
            else 0.1
        )
        entry["wildcard"] = False
    count = int(np.floor(len(ranked) * 0.1 + 0.5))
    bottom = np.arange(len(ranked) // 2, len(ranked))
    for index in np.random.default_rng(seed).choice(bottom, count, replace=False):
        ranked[index]["p_on"] = max(0.5, ranked[index]["p_on"])
        ranked[index]["wildcard"] = True
    return ranked
