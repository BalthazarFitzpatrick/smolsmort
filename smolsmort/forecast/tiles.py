"""cursor-free forecast tiles; all values are raw and the ui handles normalization

read_tiles(folder) returns [{tile_id, kind, family, recipe_key, recipe, metrics,
curves, state, best_generation, running_generation, members (ensemble only)}]
metrics maps ape/mape/wape/ae/mae/wae to {validation, test}; missing values are null
error_metrics keeps model search fitness or pooled ensemble validation wape for the picker
curves maps generation to {train_loss: [], validation: {metric: []}, loss: {}}
one-shot curves contain one value; legacy files may omit metrics and training
model ids use the saved genome key; ensemble ids hash sorted member tile ids
tiles rank by validation wape only; all scored subsets remain in ensembles.json
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
from pathlib import Path

import numpy as np

from smolsmort.forecast.metrics import CURVE_METRICS, error_metrics, prediction_metrics
from smolsmort.forecast.tables import read_table

_ENSEMBLE_LOCK = threading.RLock()


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


def _join_indexes(tables, keys):
    """join keys once; retain last duplicate values and first occurrence order"""
    lengths = [len(table["prediction"]) for table in tables]
    if keys:
        records = [
            np.rec.fromarrays([np.asarray(table[key]).astype(str) for key in keys])
            for table in tables
        ]
        _, inverse = np.unique(np.concatenate(records), return_inverse=True)
    else:
        inverse = np.zeros(sum(lengths), dtype=int)
    size = int(inverse.max()) + 1 if len(inverse) else 0
    indexes, orders = [], []
    offset = 0
    for length in lengths:
        joined = inverse[offset : offset + length]
        index = np.full(size, -1, dtype=int)
        order = np.full(size, length, dtype=int)
        np.maximum.at(index, joined, np.arange(length))
        np.minimum.at(order, joined, np.arange(length))
        indexes.append(index)
        orders.append(order)
        offset += length
    return indexes, orders


def align_tables(tables):
    """align prediction tables by shared keys, preserving first-table order"""
    keys = [
        key for key in ("series", "step", "weeks_ahead") if all(key in table for table in tables)
    ]
    indexes, orders = _join_indexes(tables, keys)
    common = np.all(np.asarray(indexes) >= 0, axis=0)
    order = np.argsort(orders[0][common], kind="stable")
    return [
        {name: values[index[common][order]] for name, values in table.items()}
        for table, index in zip(tables, indexes, strict=True)
    ]


def score_ensembles(models, tables):
    """score every subset of two or more families on aligned rows; test never ranks"""
    available = sorted(tile["tile_id"] for tile in models if tile["tile_id"] in tables)
    aligned = {}
    for split in ("validation", "test"):
        parts = {
            member: tables[member][split]
            for member in available
            if tables[member].get(split) is not None
        }
        if not parts:
            continue
        aligned[split] = (parts, {})
    rows = []
    for count in range(2, len(available) + 1):
        for members in itertools.combinations(available, count):
            metrics = {}
            for split in ("validation", "test"):
                parts = [tables[member].get(split) for member in members]
                if any(part is None for part in parts):
                    values = dict.fromkeys(CURVE_METRICS)
                else:
                    keys = tuple(
                        key
                        for key in ("series", "step", "weeks_ahead")
                        if all(key in part for part in parts)
                    )
                    split_parts, joins = aligned[split]
                    if keys not in joins:
                        eligible = {
                            member: part
                            for member, part in split_parts.items()
                            if all(key in part for key in keys)
                        }
                        indexes, orders = _join_indexes(list(eligible.values()), keys)
                        joins[keys] = dict(
                            zip(eligible, zip(indexes, orders, strict=True), strict=True)
                        )
                    indexes = [joins[keys][member][0] for member in members]
                    common = np.all(np.asarray(indexes) >= 0, axis=0)
                    order = np.argsort(joins[keys][members[0]][1][common], kind="stable")
                    parts = [
                        {
                            name: values[index[common][order]]
                            for name, values in part.items()
                            if name in ("actual", "truth", "prediction")
                        }
                        for part, index in zip(parts, indexes, strict=True)
                    ]
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
    return _set_ensemble_error_metrics(sorted(rows, key=rank_tile))


def _set_ensemble_error_metrics(rows):
    for tile in rows:
        # derive picker fitness on read so older ensemble files remain usable
        metrics = tile.setdefault("error_metrics", {})
        metrics.setdefault(
            "validation_error",
            error_metrics({}, fitness=tile["metrics"]["wape"]["validation"])["validation_error"],
        )
    return rows


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


def read_ensemble_rows(folder):
    """reuse saved scores until a family prediction table changes"""
    from smolsmort.forecast.views import write_json_atomic

    folder = Path(folder)
    path = folder / "ensembles.json"
    with _ENSEMBLE_LOCK:
        sources = [
            path
            for split in ("validation", "test")
            for path in (folder / "families").glob(f"*/{split}.parquet")
        ]
        if path.exists() and all(
            path.stat().st_mtime_ns >= source.stat().st_mtime_ns for source in sources
        ):
            saved = read_json(path, None)
            if isinstance(saved, list):
                return _set_ensemble_error_metrics(sorted(saved, key=rank_tile))
        rows = ensemble_rows(folder)
        write_json_atomic(path, rows)
        return rows


def write_ensemble_views(folder, rows):
    """persist the kept ensembles with bands fitted on validation rows"""
    from smolsmort.forecast.evaluate import apply_lead_bands, fit_lead_bands
    from smolsmort.forecast.tables import write_table

    folder = Path(folder)
    for row in sorted(rows, key=rank_tile)[:5]:
        parts = {}
        for split in ("validation", "test", "forecast"):
            paths = [
                folder / "families" / member.split(":")[1] / f"{split}.parquet"
                for member in row["members"]
            ]
            if not all(path.exists() for path in paths):
                break
            tables = align_tables([read_table(path) for path in paths])
            for name in ("actual", "truth"):
                if name in tables[0] and not all(
                    np.array_equal(tables[0][name], table[name], equal_nan=True) for table in tables
                ):
                    raise ValueError("ensemble members disagree on actuals")
            parts[split] = dict(tables[0])
            parts[split]["prediction"] = np.mean([table["prediction"] for table in tables], axis=0)
        if len(parts) != 3 or not len(parts["validation"]["prediction"]):
            continue
        validation = parts["validation"]
        buckets = {
            value: index + 1 for index, value in enumerate(np.unique(validation.get("bucket", [])))
        }
        leads = {
            split: table.get(
                "weeks_ahead",
                np.asarray([buckets.get(value, 1) for value in table["bucket"]])
                if "bucket" in table
                else np.ones(len(table["prediction"]), dtype=int),
            )
            for split, table in parts.items()
        }
        for level, suffix in ((0.8, ""), (0.5, "_50")):
            bands = fit_lead_bands(
                validation.get("actual", validation.get("truth")),
                validation["prediction"],
                leads["validation"],
                validation.get("series"),
                level=level,
            )
            for split, table in parts.items():
                table[f"lower{suffix}"], table[f"upper{suffix}"] = apply_lead_bands(
                    bands, table["prediction"], leads[split], table.get("series")
                )
        for split, table in parts.items():
            target = folder / "tile_views" / row["recipe_key"] / f"{split}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(
                f"{split}.{os.getpid()}.{threading.get_ident()}.tmp.parquet"
            )
            try:
                write_table(temporary, table)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def read_tiles(folder: Path, *, include_ensembles=True):
    folder = Path(folder)
    request = read_json(folder / "request.json", {})
    status = read_json(folder / "status.json", {})
    saved_recipe = read_json(folder / "recipe.json", {})
    result = read_json(folder / "result.json", {})
    winner = result.get("winner") or {}
    names = [item["name"] for item in request.get("families", [])]
    if not names:
        names = [(saved_recipe.get("genome") or {}).get("family", "xgboost")]
    winner_family = winner.get("family") or (saved_recipe.get("genome") or {}).get("family")
    if not winner_family and saved_recipe.get("family") != "ensemble":
        winner_family = saved_recipe.get("family") or (names[0] if len(names) == 1 else None)
    models = []
    for family in names:
        target = folder / "families" / family
        board = read_json(target / "leaderboard.json", [])
        if not board and request.get("version") != 2:
            board = read_json(folder / "leaderboard.json", [])
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
            candidate = (
                record
                if record.get("key") == key
                else next((item for item in candidates if item.get("key", key) == key), None)
            )
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
        split_tables, split_paths = {}, {}
        for split in ("validation", "test"):
            path = target / f"{split}.parquet"
            if not path.exists() and family == winner_family:
                path = folder / f"winner_{split}.parquet"
                if not path.exists() and saved_recipe.get("family") != "ensemble":
                    path = folder / f"{split}.parquet"
            split_paths[split] = path
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
        summary = result.get("families", {}).get(family, {})
        if family == winner_family:
            summary = {**result, **winner} if saved_recipe.get("family") != "ensemble" else winner
        fitness = best.get(
            "fitness", summary.get("fitness", progress.get("best_error", summary.get("val_error")))
        )
        models.append(
            {
                "tile_id": tile_id,
                "kind": "model",
                "family": family,
                "recipe_key": key,
                "recipe": recipe,
                "metrics": metrics,
                "error_metrics": error_metrics(
                    request.get("spec", {}),
                    fitness=fitness,
                    verdict=summary.get("verdict"),
                    path=split_paths["test"],
                ),
                "curves": curves,
                "state": "running" if running else "done",
                "best_generation": best.get("generation", min(generations, default=None)),
                "running_generation": progress.get("generation") if running else None,
            }
        )
    ensembles = read_ensemble_rows(folder)[:5] if include_ensembles else []
    return sorted(models + ensembles, key=rank_tile)
