"""persistent sequential forecast searches driven by one saved plan"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from smolsmort.forecast import runs
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.processes import is_process_alive
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.spec import spec_from_dict

OVERHEAD_S = 150
MIN_BUDGET_S = 240
_PROCESSES = {}


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")


def _spec(root, item):
    data = item.get("spec")
    if isinstance(data, str):
        path = Path(data)
        data = _read(path if path.is_absolute() else Path(root) / path).get("spec")
    if not isinstance(data, dict):
        raise ValueError("set spec must be an object or a saved request path")
    try:
        return replace(spec_from_dict(data), horizon=item["horizon"])
    except (TypeError, KeyError) as exc:
        raise ValueError(f"invalid set spec: {exc}") from exc


def _validate(root, data):
    if not isinstance(data, dict):
        raise ValueError("plan must be an object")
    if ("deadline" in data) == ("hours" in data):
        raise ValueError("choose exactly one of deadline or hours")
    if "hours" in data:
        _positive(data["hours"], "hours")
    else:
        try:
            datetime.strptime(data["deadline"], "%Y-%m-%d %H:%M")
        except (TypeError, ValueError) as exc:
            raise ValueError("deadline must be yyyy-mm-dd hh:mm") from exc
    families = data.get("families")
    sets = data.get("sets")
    if (
        not isinstance(families, list)
        or not families
        or not all(isinstance(f, dict) for f in families)
    ):
        raise ValueError("families must be a non-empty list of objects")
    if not isinstance(sets, list) or not sets or not all(isinstance(s, dict) for s in sets):
        raise ValueError("sets must be a non-empty list of objects")
    names = []
    for family in families:
        _positive(family.get("budget_s"), "budget_s")
        parse_request(
            {
                "version": 2,
                "families": [{"name": family.get("name")}],
                "time_budget_s": family["budget_s"],
                "plateau": family.get("patience", 6),
                "min_generations": family.get("min_generations", 0),
                "transform": data.get("transform", "log1p"),
                "holdout": data.get("holdout", "seasonal"),
                "prune_sigma": data.get("prune_sigma", 2.0),
                "ensemble": data.get("ensemble", {"enabled": False}),
                "max_concurrent_families": 1,
            }
        )
        names.append(family["name"])
    if len(set(names)) != len(names):
        raise ValueError("family names must be unique")
    labels = []
    for item in sets:
        if not isinstance(item.get("label"), str) or not item["label"].strip():
            raise ValueError("set label must be a non-empty string")
        labels.append(item["label"])
        if type(item.get("horizon")) is not int or item["horizon"] < 1:
            raise ValueError("set horizon must be a positive integer")
        _spec(root, item)
        if not isinstance(item.get("budgets", {}), dict):
            raise ValueError("set budgets must be an object")
        for name, budget in item.get("budgets", {}).items():
            if name not in names:
                raise ValueError("set budgets must name selected families")
            _positive(budget, "budget override")
    if len(set(labels)) != len(labels):
        raise ValueError("set labels must be unique")
    warm = data.get("warm", {})
    if not isinstance(warm, dict) or any(
        label not in labels
        or not isinstance(entries, dict)
        or any(
            name not in names or not isinstance(run, str) or not run
            for name, run in entries.items()
        )
        for label, entries in warm.items()
    ):
        raise ValueError("warm must map set labels and families to run ids")
    passes = data.get("passes", {})
    if not isinstance(passes, dict) or type(passes.get("second_pass", False)) is not bool:
        raise ValueError("passes.second_pass must be a boolean")
    selected = passes.get("families", names)
    if (
        not isinstance(selected, list)
        or not selected
        or not all(isinstance(name, str) for name in selected)
        or len(set(selected)) != len(selected)
        or any(name not in names for name in selected)
    ):
        raise ValueError("passes.families must select unique plan families")
    for name, minimum, default in (("keep", 1, 8), ("min_generations", 0, 8), ("patience", 1, 6)):
        value = passes.get(name, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"passes.{name} must be an integer of at least {minimum}")
    return data


def load_plan(root) -> dict:
    return _validate(root, _read(Path(root) / "forecast-plan.json"))


def save_plan(root, data) -> dict:
    data = _validate(root, data)
    _write(Path(root) / "forecast-plan.json", data)
    return data


def plan_status(root) -> dict:
    status = _read(
        Path(root) / "forecast-plan-status.json",
        {
            "state": "idle",
            "jobs_done": 0,
            "jobs_total": 0,
            "current": None,
            "seconds_left": 0,
            "sets": {},
            "warnings": [],
        },
    )
    state = _read(Path(root) / "forecast-plan-state.json")
    if "deadline" in state:
        status["seconds_left"] = max(0, state["deadline"] - time.time())
    return status


def start_plan(root, runs_root, cache_root) -> int:
    root = Path(root).resolve()
    load_plan(root)
    pid_path = root / "forecast-plan.pid"
    if pid_path.exists():
        pid = int(pid_path.read_text())
        process = _PROCESSES.get(pid)
        alive = process.poll() is None if process is not None else is_process_alive(pid)
        if alive:
            return pid
    (root / "forecast-plan.stop").unlink(missing_ok=True)
    with (root / "forecast-plan.log").open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "smolsmort.forecast.plan",
                str(root),
                str(Path(runs_root).resolve()),
                str(Path(cache_root).resolve()),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(process.pid))
    _PROCESSES[process.pid] = process
    return process.pid


def stop_plan(root):
    root = Path(root)
    (root / "forecast-plan.stop").touch()
    state = _read(root / "forecast-plan-state.json")
    current = state.get("current")
    if current and current.get("run_id"):
        runs.cancel_run(Path(state["runs_root"]), current["run_id"])
    status = plan_status(root)
    _write(
        root / "forecast-plan-status.json",
        {**status, "state": "stopped", "stopped": "stopped by the user"},
    )


def _slug(label):
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "set"


def _budget(item, family):
    return item.get("budgets", {}).get(family["name"], family["budget_s"])


def _make_warm(runs_root, item, family, source, keep=None):
    source = Path(runs_root) / source / "families" / family
    population = _read(source / "population.json", [])
    if not population:
        return None
    if keep is not None:
        checkpoint = _read(source / "checkpoint.json")
        entries = checkpoint.get("entries", [])
        scores = {
            json.dumps(entry["genome"], sort_keys=True): entry.get("fitness") for entry in entries
        }
        population = sorted(
            population,
            key=lambda genome: (
                scores.get(json.dumps(genome, sort_keys=True))
                if scores.get(json.dumps(genome, sort_keys=True)) is not None
                else float("inf")
            ),
        )[:keep]
    name = f"warm-{_slug(item['label'])}-{family}"
    folder = Path(runs_root) / name
    _write(folder / "families" / family / "population.json", population)
    _write(folder / "recipe.json", {"genome": {"family": family}})
    return name


def run_plan(root, runs_root, cache_root, *, start_run=runs.start_run, wait_run=runs.wait_run):
    from smolsmort.forecast.plan_sets import create_set, update_set

    root, runs_root, cache_root = Path(root), Path(runs_root), Path(cache_root)
    plan = load_plan(root)
    path = root / "forecast-plan-state.json"
    state = _read(path)
    deadline = state.get("deadline") or (
        datetime.strptime(plan["deadline"], "%Y-%m-%d %H:%M").timestamp()
        if "deadline" in plan
        else time.time() + plan["hours"] * 3600
    )
    state.update(deadline=deadline, runs_root=str(runs_root.resolve()))
    state.setdefault("finished", {})
    state.setdefault("members", {})
    state.setdefault("sets", {})
    state.setdefault("pass", 1)
    state.setdefault("first_pass", {})
    families = {family["name"]: family for family in plan["families"]}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for index, item in enumerate(plan["sets"]):
        label = item["label"]
        if label not in state["sets"]:
            run_id = f"set-{stamp}-{_slug(label)}-{index}"
            state["sets"][label] = run_id
            create_set(runs_root, run_id, label, asdict(_spec(root, item)), list(families))
        state["members"].setdefault(label, {})
    _write(path, state)

    def publish(jobs, current=None, status="running"):
        need = sum(
            _budget(item, families[name]) + OVERHEAD_S
            for key, item, name in jobs
            if key not in state["finished"]
        )
        left = max(0, deadline - time.time())
        _write(
            root / "forecast-plan-status.json",
            {
                "state": status,
                "jobs_done": len(state["finished"]),
                "jobs_total": len(set(state["finished"]) | {key for key, _, _ in jobs}),
                "current": current,
                "seconds_left": left,
                "sets": {
                    label: {"families_done": len(members), "families_total": len(families)}
                    for label, members in state["members"].items()
                },
                "warnings": ["remaining planned need is less than time left; search may be idle"]
                if need < left
                else [],
            },
        )

    stopped = False
    while True:
        number = state["pass"]
        selected = (
            list(families)
            if number == 1
            else plan.get("passes", {}).get("families", list(families))
        )
        jobs = [
            (f"{number}:{index}:{name}", item, name)
            for index, item in enumerate(plan["sets"])
            for name in selected
        ]
        for position, (key, item, name) in enumerate(jobs):
            if key in state["finished"]:
                continue
            if (root / "forecast-plan.stop").exists():
                stopped = True
                break
            left = deadline - time.time()
            if left < MIN_BUDGET_S + OVERHEAD_S:
                break
            todo = sum(
                _budget(s, families[f]) + OVERHEAD_S
                for k, s, f in jobs[position:]
                if k not in state["finished"]
            )
            scale = min(1.0, left / todo)
            family = families[name]
            label = item["label"]
            current = {"set": label, "family": name, "pass": number}
            publish(jobs, current)
            run_id = None
            try:
                spec = _spec(root, item)
                source = state["members"][label].get(name) or plan.get("warm", {}).get(
                    label, {}
                ).get(name)
                if number == 1:
                    comparable = [
                        (other["horizon"], state["members"][other["label"]].get(name))
                        for other in plan["sets"]
                        if other["horizon"] < item["horizon"]
                        and asdict(replace(_spec(root, other), horizon=item["horizon"]))
                        == asdict(spec)
                    ]
                    fresh = [entry for entry in comparable if entry[1]]
                    if fresh:
                        source = max(fresh)[1]
                passes = plan.get("passes", {})
                if number > 1:
                    source = state["first_pass"].get(label, {}).get(name, source)
                warm = (
                    _make_warm(
                        runs_root, item, name, source, passes.get("keep", 8) if number > 1 else None
                    )
                    if source
                    else None
                )
                pending = state.get("current", {})
                run_id = pending.get("run_id") if pending.get("key") == key else None
                budget = max(MIN_BUDGET_S, int(_budget(item, family) * scale))
                if run_id is None:
                    run_id = start_run(
                        runs_root,
                        spec,
                        prepare(spec, cache_root),
                        version=2,
                        families=[{"name": name, "method": "genetic"}],
                        time_budget_s=budget,
                        warm_from=warm,
                        plateau=passes.get("patience", 6)
                        if number > 1
                        else family.get("patience", 6),
                        min_generations=passes.get("min_generations", 8)
                        if number > 1
                        else family.get("min_generations", 0),
                        prune_sigma=plan.get("prune_sigma", 2.0),
                        transform=plan.get("transform", "log1p"),
                        holdout=plan.get("holdout", "seasonal"),
                        ensemble=plan.get("ensemble", {"enabled": False}),
                        max_concurrent_families=1,
                        part_of=state["sets"][label],
                        label=f"{label} - {name}",
                    )
                    state["current"] = {**current, "key": key, "run_id": run_id}
                    _write(path, state)
                result = wait_run(runs_root, run_id, timeout=budget * 3 + 1800)
                if (root / "forecast-plan.stop").exists() or result.get("state") in (
                    "stopped",
                    "cancelled",
                ):
                    stopped = True
                    state.pop("current", None)
                    break
                state["finished"][key] = {
                    "run_id": run_id,
                    "state": result.get("state"),
                    "scale": scale,
                    "budget_s": budget,
                }
                with (root / "forecast-plan.log").open("a") as log:
                    log.write(json.dumps({**current, **state["finished"][key]}) + "\n")
                if result.get("state") == "done":
                    state["members"][label][name] = run_id
                    if number == 1:
                        state["first_pass"].setdefault(label, {})[name] = run_id
                _write(path, state)
                update_set(
                    runs_root,
                    state["sets"][label],
                    state["members"][label],
                    state="done" if len(state["members"][label]) == len(families) else "running",
                )
            except Exception as exc:
                if run_id is not None and runs._alive(runs_root / run_id):
                    runs.cancel_run(runs_root, run_id)
                state["finished"][key] = {
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                with (root / "forecast-plan.log").open("a") as log:
                    log.write(json.dumps({**current, **state["finished"][key]}) + "\n")
            state.pop("current", None)
            _write(path, state)
            publish(jobs)
        if stopped or any(key not in state["finished"] for key, _, _ in jobs):
            break
        if not plan.get("passes", {}).get("second_pass", False) or deadline - time.time() < (
            MIN_BUDGET_S + OVERHEAD_S
        ) * len(plan["sets"]) * len(plan.get("passes", {}).get("families", families)):
            break
        state["pass"] += 1
        _write(path, state)
    for label, run_id in state["sets"].items():
        family_states = {
            name: "failed"
            for key, item, name in jobs
            if item["label"] == label and state["finished"].get(key, {}).get("state") == "failed"
        }
        complete = all(key in state["finished"] for key, item, _ in jobs if item["label"] == label)
        update_set(
            runs_root,
            run_id,
            state["members"][label],
            state="stopped" if stopped else "done" if complete else "running",
            family_states=family_states,
        )
    publish(jobs, status="stopped" if stopped else "done")
    _write(path, state)
    return plan_status(root)


if __name__ == "__main__":
    try:
        run_plan(*sys.argv[1:4])
    finally:
        pid_path = Path(sys.argv[1]) / "forecast-plan.pid"
        if pid_path.exists() and pid_path.read_text() == str(os.getpid()):
            pid_path.unlink()
