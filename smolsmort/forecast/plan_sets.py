"""incremental set runs and validation-selected equal-weight ensembles"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from smolsmort.forecast.evaluate import apply_lead_bands, fit_lead_bands
from smolsmort.forecast.tables import read_table, write_table
from smolsmort.forecast.views import write_json_atomic


def _read(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def create_set(runs_root: Path, run_id: str, label: str, spec: dict, families: list[str]):
    """create the pending family tiles before the first part starts"""
    folder = Path(runs_root) / run_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "request.json").exists():
        return folder
    write_json_atomic(
        folder / "request.json",
        {
            "version": 2,
            "kind": "set",
            "label": label,
            "spec": spec,
            "families": [{"name": name, "method": "genetic"} for name in families],
            "time_budget_s": 0,
            "combined_from": {},
        },
    )
    pending = {name: {"state": "pending"} for name in families}
    write_json_atomic(
        folder / "status.json",
        {"state": "running", "started_at": time.time(), "families": pending},
    )
    write_json_atomic(folder / "coordinator.json", {"families": pending})
    write_json_atomic(folder / "result.json", {"families": {}})
    write_json_atomic(folder / "leaderboard.json", [])
    return folder


def _validation_error(folder):
    result = _read(folder / "result.json", {})
    value = (result.get("verdict") or {}).get("accuracy", {}).get("val_error")
    if value is None:
        value = result.get("fitness")
    return float(value) if value is not None else math.inf


def _wape(prediction, actual):
    denominator = float(np.abs(actual).sum())
    error = float(np.abs(prediction - actual).sum())
    return error / denominator if denominator else (0.0 if error == 0 else math.inf)


def pick_ensemble(runs_root: Path, members: dict[str, str]):
    """choose the equal-weight subset with the lowest aligned validation wape"""
    parts = {}
    stacks = {}
    for part in ("validation", "test", "forecast"):
        tables = {
            family: read_table(Path(runs_root) / run_id / f"winner_{part}.parquet")
            for family, run_id in members.items()
        }
        first = next(iter(tables.values()))
        keys = [name for name in ("series", "step", "weeks_ahead") if name in first]
        indexes = {
            family: {
                tuple(str(table[key][i]) for key in keys): i
                for i in range(len(table["prediction"]))
            }
            for family, table in tables.items()
        }
        ordered = [
            key
            for key in next(iter(indexes.values()))
            if all(key in index for index in indexes.values())
        ]
        if not ordered:
            raise ValueError(f"ensemble {part} tables have no matching rows")
        aligned = {
            family: np.asarray(
                [table["prediction"][indexes[family][key]] for key in ordered], dtype=float
            )
            for family, table in tables.items()
        }
        finite = np.all(np.isfinite(np.column_stack(list(aligned.values()))), axis=1)
        first_index = next(iter(indexes.values()))
        rows = [first_index[key] for key in ordered]
        parts[part] = {name: values[rows][finite].copy() for name, values in first.items()}
        stacks[part] = {family: values[finite] for family, values in aligned.items()}
    actual = parts["validation"]["actual"]
    best = min(
        (
            group
            for count in range(1, len(members) + 1)
            for group in itertools.combinations(members, count)
        ),
        key=lambda group: _wape(
            np.mean([stacks["validation"][name] for name in group], axis=0), actual
        ),
    )
    for part, table in parts.items():
        table["prediction"] = np.mean([stacks[part][name] for name in best], axis=0)
    validation = parts["validation"]
    if "weeks_ahead" in validation:
        bands = fit_lead_bands(
            validation["actual"], validation["prediction"], validation["weeks_ahead"]
        )
        for part in ("test", "forecast"):
            table = parts[part]
            table["lower"], table["upper"] = apply_lead_bands(
                bands, table["prediction"], table["weeks_ahead"]
            )
    return best, parts


def update_set(
    runs_root: Path, run_id: str, members: dict[str, str], *, state="running", family_states=None
):
    """refresh real family artifacts and ensemble after a completed part"""
    runs_root = Path(runs_root)
    folder = runs_root / run_id
    request = _read(folder / "request.json", {})
    previous = request.get("combined_from", {})
    if any(name in previous and previous[name] != part_id for name, part_id in members.items()):
        (folder / "route-cursors.json").unlink(missing_ok=True)
    status = _read(folder / "status.json", {})
    names = [item["name"] for item in request["families"]]
    progress = {name: {"state": (family_states or {}).get(name, "pending")} for name in names}
    coordinator = {}
    summaries = {}
    board, events = [], []
    elapsed, budget = 0.0, 0.0
    result = {}
    if members:
        best = min(members, key=lambda name: _validation_error(runs_root / members[name]))
        source = runs_root / members[best]
        result = _read(source / "result.json", {})
        for path in source.iterdir():
            if path.is_file() and (
                path.suffix == ".parquet"
                or path.name in {"recipe.json", "importance.json", "eval_history.json"}
            ):
                shutil.copy2(path, folder / path.name)
        for key in ("prepared", "horizon_mode", "holdout", "transform", "target_transform"):
            source_request = _read(source / "request.json", {})
            if key in source_request:
                request[key] = source_request[key]
    for family, part_id in members.items():
        source = runs_root / part_id
        target = folder / "families" / family
        if (source / "families" / family).exists():
            shutil.copytree(source / "families" / family, target, dirs_exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        for part in ("validation", "test", "forecast"):
            winner = source / f"winner_{part}.parquet"
            if winner.exists() and not (source / "families" / family / f"{part}.parquet").exists():
                shutil.copy2(winner, target / f"{part}.parquet")
        part_status = _read(source / "status.json", {})
        progress[family] = part_status.get("families", {}).get(
            family, {"state": part_status.get("state", "done")}
        )
        coordinator[family] = (
            _read(source / "coordinator.json", {}).get("families", {}).get(family, progress[family])
        )
        part_result = _read(source / "result.json", {})
        summaries[family] = part_result.get("families", {}).get(family, part_result)
        board.extend(_read(source / "leaderboard.json", []))
        event_path = source / "events.jsonl"
        if event_path.exists():
            events.extend(event_path.read_text(encoding="utf-8").splitlines())
        elapsed += float(part_result.get("elapsed") or 0)
        budget += float(_read(source / "request.json", {}).get("time_budget_s") or 0)
    request.update(combined_from=dict(members), time_budget_s=budget)
    result.update(families=summaries, elapsed=elapsed)
    available = {
        family: part_id
        for family, part_id in members.items()
        if all(
            (runs_root / part_id / f"winner_{part}.parquet").exists()
            for part in ("validation", "test", "forecast")
        )
    }
    if available:
        selected, parts = pick_ensemble(runs_root, available)
        validation_error = _wape(parts["validation"]["prediction"], parts["validation"]["actual"])
        test_error = _wape(parts["test"]["prediction"], parts["test"]["actual"])
        baseline = result.get("baseline_error")
        ratio = test_error / baseline if baseline else None
        result["ensemble"] = {
            "members": list(selected),
            "fitness": validation_error,
            "model_error": test_error,
            "baseline_error": baseline,
            "verdict": {
                "summary": f"ensemble of {', '.join(selected)}",
                "accuracy": {
                    "metric": "wape",
                    "model_error": test_error,
                    "val_error": validation_error,
                    "baseline_error": baseline,
                    "ratio": ratio,
                },
            },
        }
        for part, table in parts.items():
            write_table(folder / f"ensemble_{part}.parquet", table)
    status.update(state=state, families=progress, elapsed=elapsed)
    if state == "stopped":
        status["reason"] = "stopped by the user"
    write_json_atomic(folder / "request.json", request)
    write_json_atomic(folder / "coordinator.json", {"families": {**progress, **coordinator}})
    write_json_atomic(folder / "result.json", result)
    write_json_atomic(folder / "leaderboard.json", board)
    (folder / "events.jsonl").write_text(
        "\n".join(events) + ("\n" if events else ""), encoding="utf-8"
    )
    write_json_atomic(folder / "status.json", status)
    return folder
