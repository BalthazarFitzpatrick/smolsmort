"""one resumable search per process; optional libraries stay isolated"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

from smolsmort.forecast.families import get_family
from smolsmort.forecast.pipeline import load_workspace
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.search import Budget, search
from smolsmort.forecast.spec import spec_from_dict
from smolsmort.forecast.worker import _Run, _warm_start


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def run(folder: Path, name: str) -> int:
    if "torch" in sys.modules:
        raise SystemExit("the forecast worker must not share a process with torch")
    folder = Path(folder)
    request = parse_request(read_json(folder / "request.json"))
    selection = next(item for item in request["families"] if item["name"] == name)
    target = folder / "families" / name
    target.mkdir(parents=True, exist_ok=True)
    if read_json(target / "status.json", {}).get("state") == "done":
        return 0
    job = _Run(target)
    started_at = time.time()
    progress = {"generation": 0, "best_error": None, "candidates_done": 0}
    for filename, default in (
        ("leaderboard.json", []),
        ("eval_history.json", None),
        ("winner.json", None),
        ("population.json", []),
    ):
        if not (target / filename).exists():
            job.save(filename, default)
    for filename in ("events.jsonl", "generation_curves.jsonl"):
        (target / filename).touch(exist_ok=True)
    for line in (target / "generation_curves.jsonl").read_text().splitlines():
        try:
            job.recorded_generations.add(json.loads(line)["generation"])
        except (json.JSONDecodeError, KeyError):
            continue

    def cancelled():
        return job.cancelled or (folder / "cancel").exists()

    def checkpoint(state):
        job.save("checkpoint.json", state)
        job.save("population.json", state["population"])
        progress["generation"] = state["generation"]
        progress["candidates_done"] = len(state["entries"])
        errors = [entry["fitness"] for entry in state["entries"] if entry["fitness"] is not None]
        progress["best_error"] = min(errors) if errors else None
        job.status("running", started_at=started_at, **progress)

    def event(payload):
        job.event({**payload, "family": name})

    try:
        job.status("running", started_at=started_at, **progress)
        family = get_family(name)
        available, reason = family.available()
        if not available:
            raise ValueError(f"family {name!r} is unavailable; {reason}")
        spec = spec_from_dict(request["spec"])
        if spec.mode == "row" and name != "xgboost":
            raise ValueError("per-row runs support xgboost only")
        ws = load_workspace(spec, Path(request["prepared"]), request["summary"])
        nthread = max(
            1, (request.get("nthread") or os.cpu_count() or 1) // len(request["families"])
        )
        warm_path = (
            Path(request["warm_from"]) / "families" / name if request.get("warm_from") else None
        )
        warm = _warm_start(warm_path) if warm_path else None
        result = search(
            ws,
            Budget(time_cap=request["time_budget_s"], nthread=nthread, max_generations=2**31 - 1),
            family=family,
            method=selection.get("method", "genetic"),
            space=selection.get("space"),
            warm=warm,
            resume=read_json(target / "checkpoint.json"),
            on_checkpoint=checkpoint,
            on_event=event,
            on_best=job.record_best,
            on_generation=job.record_generation,
            on_leaderboard=job.record_leaderboard,
            stop=cancelled,
        )
        job.save("leaderboard.json", result["leaderboard"])
        job.save("population.json", result["population"])
        job.save("winner.json", result["best"])
        if cancelled():
            state, reason = "cancelled", "cancelled"
        elif not any(entry["fitness"] is not None for entry in result["leaderboard"]):
            state = "failed"
            reason = (
                "; ".join(entry["note"] for entry in result["leaderboard"])
                or "no candidate could be fitted"
            )
        else:
            state, reason = "done", result["stopped"]
        job.status(
            state, reason=reason, elapsed=result["elapsed"], started_at=started_at, **progress
        )
        event({"event": state, "stopped": reason})
        return 1 if state == "failed" else 0
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        job.status("failed", reason=reason, trace=traceback.format_exc(), **progress)
        event({"event": "failed", "reason": reason})
        return 1


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]), sys.argv[2]))
