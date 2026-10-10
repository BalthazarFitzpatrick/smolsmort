"""the search in its own process: `python -m smolsmort.forecast.worker <run folder>`. torch's
openmp and xgboost's abort each other in one process (U0a, 2026-09-23), so this one never loads
torch and xgboost gets every core"""

from __future__ import annotations

import json
import os
import signal as signal
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from smolsmort.forecast.pipeline import finish, genome_from_dict, load_workspace
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.search import Budget, search
from smolsmort.forecast.spec import spec_from_dict
from smolsmort.forecast.tables import write_table


class _Run:
    def __init__(self, folder: Path):
        self.folder = folder
        self._cancelled = False
        self.recorded_generations: set[int] = set()

    @property
    def cancelled(self):
        return self._cancelled or (self.folder / "cancel").exists()

    def _cancel(self, *_):
        self._cancelled = True

    def event(self, payload: dict):
        with (self.folder / "events.jsonl").open("a") as f:
            f.write(json.dumps(payload, default=_plain) + "\n")
            f.flush()
        if payload.get("event") == "progress":
            self.save("progress.json", {**payload, "updated_at": time.time()})

    def status(self, state: str, **extra):
        path = self.folder / "status.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if "families" in previous and "families" not in extra:
                extra["families"] = previous["families"]
            if "phase" in previous and "phase" not in extra:
                extra["phase"] = previous["phase"]
        progress_path = self.folder / "progress.json"
        if state in ("done", "cancelled", "failed") and progress_path.exists():
            progress = json.loads(progress_path.read_text())
            progress["elapsed"] = round(
                progress.get("elapsed", 0) + max(0, time.time() - progress["updated_at"]), 1
            )
            progress["updated_at"] = time.time()
            self.save("progress.json", progress)
        text = json.dumps({"state": state, "pid": os.getpid(), **extra}, indent=2, default=_plain)
        tmp = self.folder / "status.json.tmp"
        tmp.write_text(text)
        os.replace(tmp, self.folder / "status.json")

    def save(self, name: str, data):
        tmp = self.folder / f"{name}.tmp"
        tmp.write_text(json.dumps(data, indent=2, default=_plain))
        os.replace(tmp, self.folder / name)

    def record_leaderboard(self, generation: int, board: list[dict]):
        self.save("leaderboard.json", board[:10])
        self.save("leaderboard_generation.json", generation)

    def record_best(self, history: dict | None):
        tmp = self.folder / "eval_history.json.tmp"
        tmp.write_text(json.dumps(history, default=_plain, allow_nan=False))
        os.replace(tmp, self.folder / "eval_history.json")

    def record_generation(self, curve: dict):
        generation = curve["generation"]
        if generation in self.recorded_generations:
            return
        text = json.dumps(curve, default=_plain, allow_nan=False)
        with (self.folder / "generation_curves.jsonl").open("a") as stream:
            stream.write(text + "\n")
            stream.flush()
        self.recorded_generations.add(generation)


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def run(folder: Path) -> int:
    if "torch" in sys.modules:
        raise SystemExit("the forecast worker must not share a process with torch")
    folder = Path(folder)
    job = _Run(folder)
    started = time.monotonic()
    request = json.loads((folder / "request.json").read_text())
    job.status("running", started_at=time.time())
    try:
        request = parse_request(request)
        if request.get("version") == 2 and not (folder / "coordinator.json").exists():
            started_at = time.time()
            job.save(
                "coordinator.json",
                {
                    "started_at": started_at,
                    "deadline": started_at + request["time_budget_s"],
                    "phase": "broad",
                    "families": {},
                },
            )
        spec = spec_from_dict(request["spec"])
        ws = load_workspace(spec, Path(request["prepared"]), request["summary"])
        job.save("profile.json", ws.profile)
        budget = Budget(
            **{
                **request.get("budget", {}),
                "nthread": request.get("nthread") or os.cpu_count() or 1,
            }
        )
        warm = _warm_start(request.get("warm_from"))
        if request.get("recipe"):
            # a refit: the saved recipe, no search
            result = {
                "leaderboard": [],
                "best": request["recipe"],
                "population": [],
                "stopped": "refit",
            }
        elif request.get("version") == 2:
            from smolsmort.forecast.launcher import launch_families

            result = launch_families(folder, request, job)
        else:
            result = search(
                ws,
                budget,
                warm=warm,
                on_event=job.event,
                on_best=job.record_best,
                on_generation=job.record_generation,
                on_leaderboard=job.record_leaderboard,
                stop=lambda: job.cancelled,
            )
            job.save("leaderboard.json", result["leaderboard"])
            job.save("population.json", result["population"])
        if job.cancelled or result["stopped"] == "cancelled":
            job.status("cancelled", stopped="cancelled", elapsed=result.get("elapsed", 0))
            job.event({"event": "cancelled", "stopped": "cancelled"})
            return 0
        if result["best"] is None:
            job.status("failed", reason="no candidate could be fitted")
            return 1
        job.event(
            {
                "event": "finishing",
                "stopped": result["stopped"],
                "elapsed": round(time.monotonic() - started, 1),
            }
        )
        if result.get("finalized"):
            final = json.loads((folder / "result.json").read_text())
        else:
            final = finish(ws, genome_from_dict(result["best"]), nthread=budget.nthread)
            _write_result(folder, spec, result, final)
        if request.get("version") == 2 and not result.get("finalized"):
            recipe = json.loads((folder / "recipe.json").read_text())
            job.save(
                "recipe.json",
                {**recipe, "family": result["best"].get("family", "xgboost"), "transform": "none"},
            )
        job.status(
            "done",
            stopped=result["stopped"],
            verdict=final["verdict"],
            elapsed=result.get("elapsed", round(time.monotonic() - started, 1)),
            **({"families": result.get("families", {})} if request.get("version") == 2 else {}),
        )
        job.event({"event": "done", "verdict": final["verdict"]})
        return 0
    except Exception as exc:
        # any failure must reach the ui as a status, never as a run that looks alive
        job.status("failed", reason=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
        return 1


def _warm_start(previous) -> list[dict] | None:
    if not previous:
        return None
    path = Path(previous) / "population.json"
    return json.loads(path.read_text()) if path.exists() else None


def _write_result(folder, spec, result, final):
    recipe = {"genome": result["best"], "spec": asdict(spec), "stopped": result["stopped"]}
    summary = {
        "verdict": final["verdict"],
        "model_error": final["model_error"],
        "baseline_error": final["baseline_error"],
        "band": final.get("band"),
        "bands": final.get("bands"),
    }
    if "season_length" in final:
        recipe["season_length"] = summary["season_length"] = final["season_length"]
    for name, data in (("recipe.json", recipe), ("result.json", summary)):
        tmp = folder / f"{name}.tmp"
        tmp.write_text(json.dumps(data, indent=2, default=_plain))
        os.replace(tmp, folder / name)
    for name in ("forecast", "validation", "test"):
        columns = final.get(name)
        if (
            isinstance(columns, dict)
            and columns
            and all(isinstance(v, np.ndarray) for v in columns.values())
        ):
            write_table(folder / f"{name}.parquet", columns)


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))
