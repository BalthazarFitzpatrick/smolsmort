"""cursor-free forecast tiles; all values are raw and the ui handles normalization

read_tiles(folder) returns [{tile_id, kind, family, recipe_key, recipe, metrics,
curves, state, best_generation, running_generation, members (ensemble only)}]
metrics maps ape/mape/wape/ae/mae/wae to {validation, test}; missing values are null
curves maps generation to {train_loss: [], validation: {metric: []}, loss: {}}
one-shot curves contain one value; legacy files may omit metrics and training
model ids use the saved genome key; ensemble ids hash sorted member tile ids
tiles rank by validation wape only; all scored subsets remain in ensembles.json
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from smolsmort.forecast.metrics import CURVE_METRICS, prediction_metrics
from smolsmort.forecast.tables import read_table


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_records(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def recipe_key(recipe):
    from smolsmort.forecast.pipeline import genome_from_dict

    try:
        return genome_from_dict(recipe).key()
    except (KeyError, ValueError, TypeError):
        return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]


def read_predictions(path):
    if not path.exists():
        return None
    try:
        import duckdb
    except ImportError:
        return None

    try:
        table = read_table(path)
        return table if "prediction" in table and ("actual" in table or "truth" in table) else None
    except (OSError, ValueError, duckdb.Error):
        return None


def table_metrics(table):
    if table is None:
        return dict.fromkeys(CURVE_METRICS)
    return prediction_metrics(table.get("actual", table.get("truth")), table["prediction"])


def align_tables(tables):
    """align prediction tables by series, target step and lead, preserving first-table order"""
    keys = [key for key in ("series", "step", "weeks_ahead") if key in tables[0]]
    indexes = [
        {tuple(str(table[key][i]) for key in keys): i for i in range(len(table["prediction"]))}
        for table in tables
    ]
    common = [key for key in indexes[0] if all(key in index for index in indexes)]
    return [
        {name: values[[index[key] for key in common]] for name, values in table.items()}
        for table, index in zip(tables, indexes, strict=True)
    ]


def score_ensembles(models, tables):
    """score every subset of two or more families on aligned rows; test never ranks"""
    available = sorted(tile["tile_id"] for tile in models if tile["tile_id"] in tables)
    rows = []
    for count in range(2, len(available) + 1):
        for members in itertools.combinations(available, count):
            metrics = {}
            for split in ("validation", "test"):
                parts = [tables[member].get(split) for member in members]
                if any(part is None for part in parts):
                    values = dict.fromkeys(CURVE_METRICS)
                else:
                    parts = align_tables(parts)
                    if not len(parts[0]["prediction"]):
                        metrics.update(
                            {name: {**metrics.get(name, {}), split: None} for name in CURVE_METRICS}
                        )
                        continue
                    actuals = [part.get("actual", part.get("truth")) for part in parts]
                    if not all(
                        np.array_equal(actuals[0], actual, equal_nan=True) for actual in actuals
                    ):
                        raise ValueError("ensemble members disagree on actuals")
                    prediction = np.mean([part["prediction"] for part in parts], axis=0)
                    values = prediction_metrics(actuals[0], prediction)
                for name, value in values.items():
                    metrics.setdefault(name, {})[split] = value
            if metrics.get("wape", {}).get("validation") is None:
                continue
            key = hashlib.sha256(json.dumps(members).encode()).hexdigest()[:20]
            rows.append(
                {
                    "tile_id": f"ensemble:{key}",
                    "kind": "ensemble",
                    "family": "ensemble",
                    "recipe_key": key,
                    "members": list(members),
                    "recipe": {"weight": "equal", "members": list(members)},
                    "metrics": metrics,
                    "curves": {},
                    "state": "done",
                    "best_generation": None,
                    "running_generation": None,
                }
            )
    return sorted(rows, key=rank_tile)


def rank_tile(tile):
    value = tile["metrics"]["wape"]["validation"]
    return (value if value is not None else float("inf"), tile["tile_id"])


def ensemble_rows(folder):
    """full scored ensemble table for persistence after each completed family"""
    models = read_tiles(folder, include_ensembles=False)
    tables = {}
    for tile in models:
        target = Path(folder) / "families" / tile["family"]
        parts = {
            split: read_predictions(target / f"{split}.parquet") for split in ("validation", "test")
        }
        if parts["validation"] is not None:
            tables[tile["tile_id"]] = parts
    return score_ensembles(models, tables)


def read_tiles(folder: Path, *, include_ensembles=True):
    folder = Path(folder)
    request = read_json(folder / "request.json", {})
    status = read_json(folder / "status.json", {})
    saved_recipe = read_json(folder / "recipe.json", {})
    names = [item["name"] for item in request.get("families", [])]
    if not names:
        names = [(saved_recipe.get("genome") or {}).get("family", "xgboost")]
    models, tables = [], {}
    for family in names:
        target = folder / "families" / family
        board = read_json(target / "leaderboard.json", [])
        best = min(
            (entry for entry in board if entry.get("fitness") is not None),
            key=lambda entry: entry["fitness"],
            default={},
        )
        recipe = best.get("genome") or read_json(target / "winner.json", {})
        if not recipe and (saved_recipe.get("genome") or {}).get("family", "xgboost") == family:
            recipe = saved_recipe.get("genome") or request.get("recipe") or {}
        key = recipe_key(recipe) if recipe else "pending"
        tile_id = f"model:{family}:{key}"
        progress = {
            **status.get("families", {}).get(family, {}),
            **read_json(target / "status.json", {}),
        }
        records = read_records(target / "generation_curves.jsonl")
        if not records and len(names) == 1:
            records = read_records(folder / "generation_curves.jsonl")
        curves = {}
        for record in records:
            candidates = record.get("candidates", [record])
            candidate = next((item for item in candidates if item.get("key", key) == key), None)
            if candidate is None:
                continue
            validation = dict(candidate.get("curves") or {})
            legacy = candidate.get("validation")
            if isinstance(legacy, list):
                validation.setdefault(candidate.get("metric", "rmse"), legacy)
            for name in CURVE_METRICS:
                value = candidate.get("validation_metrics", candidate.get("metrics", {})).get(name)
                missing = [None] * max(1, len(legacy) if isinstance(legacy, list) else 1)
                validation.setdefault(name, [value] if value is not None else missing)
            curves[str(record["generation"])] = {
                "train_loss": candidate.get("train_loss", candidate.get("training", [])),
                "validation": validation,
                "loss": candidate.get("loss", {}),
            }
        split_tables = {}
        for split in ("validation", "test"):
            path = target / f"{split}.parquet"
            if not path.exists() and len(names) == 1:
                path = folder / f"winner_{split}.parquet"
                if not path.exists():
                    path = folder / f"{split}.parquet"
            split_tables[split] = read_predictions(path)
        scores = {split: table_metrics(table) for split, table in split_tables.items()}
        metrics = {
            name: {split: values[name] for split, values in scores.items()}
            for name in CURVE_METRICS
        }
        for name in CURVE_METRICS:
            if metrics[name]["validation"] is None:
                saved_metrics = best.get("metrics", {})
                metrics[name]["validation"] = saved_metrics.get(
                    "validation_metrics", saved_metrics
                ).get(name)
        generations = [int(generation) for generation in curves]
        running = (
            status.get("state") not in ("done", "failed", "cancelled")
            and not (folder / "cancel").exists()
            and progress.get("state", status.get("state"))
            in ("pending", "starting", "running", "paused", "queued")
        )
        models.append(
            {
                "tile_id": tile_id,
                "kind": "model",
                "family": family,
                "recipe_key": key,
                "recipe": recipe,
                "metrics": metrics,
                "curves": curves,
                "state": "running" if running else "done",
                "best_generation": best.get("generation", min(generations, default=None)),
                "running_generation": progress.get("generation") if running else None,
            }
        )
        if split_tables["validation"] is not None:
            tables[tile_id] = split_tables
    ensembles = score_ensembles(models, tables)[:5] if include_ensembles else []
    return sorted(models + ensembles, key=rank_tile)
