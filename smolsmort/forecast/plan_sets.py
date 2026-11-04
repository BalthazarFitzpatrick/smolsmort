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
    from smolsmort.forecast.tiles import align_tables

    parts = {}
    stacks = {}
    for part in ("validation", "test", "forecast"):
        tables = {
            family: read_table(Path(runs_root) / run_id / f"winner_{part}.parquet")
            for family, run_id in members.items()
        }
        aligned_tables = dict(zip(tables, align_tables(list(tables.values())), strict=True))
        first = next(iter(aligned_tables.values()))
        if not len(first["prediction"]):
            raise ValueError(f"ensemble {part} tables have no matching rows")
        aligned = {
            family: np.asarray(table["prediction"], dtype=float)
            for family, table in aligned_tables.items()
        }
        finite = np.all(np.isfinite(np.column_stack(list(aligned.values()))), axis=1)
        parts[part] = {name: values[finite].copy() for name, values in first.items()}
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
        summaries[family] = dict(part_result.get("families", {}).get(family, part_result))
        for key in ("verdict", "model_error", "baseline_error", "val_error"):
            if key in part_result:
                summaries[family][key] = part_result[key]
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
    if state == "cancelled":
        status.update(reason="cancelled", stopped="cancelled")
    write_json_atomic(folder / "request.json", request)
    write_json_atomic(folder / "coordinator.json", {"families": {**progress, **coordinator}})
    write_json_atomic(folder / "result.json", result)
    write_json_atomic(folder / "leaderboard.json", board)
    (folder / "events.jsonl").write_text(
        "\n".join(events) + ("\n" if events else ""), encoding="utf-8"
    )
    write_json_atomic(folder / "status.json", status)
    from smolsmort.forecast.tiles import ensemble_rows

    write_json_atomic(folder / "ensembles.json", ensemble_rows(folder))
    return folder


def build_refit_run(
    runs_root,
    spec,
    prepared,
    family_recipes,
    *,
    run_id=None,
    label="refit",
    nthread=1,
    transform=None,
    holdout="block",
    horizon_mode="single",
    start_run=None,
    wait_run=None,
):
    """synchronously refit saved genomes, publishing each tile before the next family

    call from a background job; returns the set run id; no family search is launched
    recipes accept saved recipe.json wrappers or plain genomes
    """
    import uuid
    from dataclasses import asdict

    from smolsmort.forecast import runs

    runs_root = Path(runs_root)
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    genomes = {}
    for name, recipe in family_recipes.items():
        genome = dict(recipe.get("genome", recipe))
        genome.setdefault("family", name)
        genome.setdefault("transform", recipe.get("transform", "none"))
        if transform is not None and genome["transform"] != transform:
            raise ValueError("refit recipe transform differs from requested transform")
        genomes[name] = genome
    if not genomes or any(genome["family"] != name for name, genome in genomes.items()):
        raise ValueError("refit recipes must match their family")
    folder = create_set(runs_root, run_id, label, asdict(spec), list(genomes))
    for family, genome in genomes.items():
        target = folder / "families" / family
        target.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target / "winner.json", genome)
    start_run = start_run or runs.start_run
    members, states = {}, {}
    state = "done"
    for family, genome in genomes.items():
        if (folder / "cancel").exists():
            state = "cancelled"
            break
        states[family] = "running"
        update_set(runs_root, run_id, members, family_states=states)
        part_id = start_run(
            runs_root,
            spec,
            prepared,
            recipe=genome,
            version=2,
            families=[{"name": family}],
            time_budget_s=1,
            nthread=nthread,
            transform=genome["transform"],
            holdout=holdout,
            horizon_mode=horizon_mode,
            ensemble={"enabled": False},
            part_of=run_id,
            label=f"{label} - {family}",
        )
        if wait_run is not None:
            status = wait_run(runs_root, part_id)
        else:
            while True:
                if (folder / "cancel").exists():
                    runs.cancel_run(runs_root, part_id)
                status = runs.run_state(runs_root, part_id)
                if status["state"] in ("done", "failed", "cancelled"):
                    break
                time.sleep(0.2)
        states[family] = status["state"]
        if status["state"] == "done":
            members[family] = part_id
        elif status["state"] == "cancelled":
            state = "cancelled"
            break
        else:
            state = "failed"
        update_set(runs_root, run_id, members, family_states=states)
    update_set(runs_root, run_id, members, state=state, family_states=states)
    return run_id
