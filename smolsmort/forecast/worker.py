"""the search in its own process: `python -m smolsmort.forecast.worker <run folder>`. torch's
openmp and xgboost's abort each other in one process (U0a, 2026-09-23), so this one never loads
torch and xgboost gets every core"""

from __future__ import annotations

import json
import os
import signal
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from smolsmort.forecast.pipeline import finish, genome_from_dict, load_workspace
from smolsmort.forecast.search import Budget, search
from smolsmort.forecast.spec import spec_from_dict
from smolsmort.forecast.tables import write_table


class _Run:
    def __init__(self, folder: Path):
        self.folder = folder
        self.cancelled = False
        signal.signal(signal.SIGTERM, self._cancel)

    def _cancel(self, *_):
        self.cancelled = True

    def event(self, payload: dict):
        with (self.folder / "events.jsonl").open("a") as f:
            f.write(json.dumps(payload, default=_plain) + "\n")
            f.flush()

    def status(self, state: str, **extra):
        text = json.dumps({"state": state, "pid": os.getpid(), **extra}, indent=2, default=_plain)
        tmp = self.folder / "status.json.tmp"
        tmp.write_text(text)
        os.replace(tmp, self.folder / "status.json")

    def save(self, name: str, data):
        (self.folder / name).write_text(json.dumps(data, indent=2, default=_plain))


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
    request = json.loads((folder / "request.json").read_text())
    job.status("running")
    try:
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
        else:
            result = search(ws, budget, warm=warm, on_event=job.event, stop=lambda: job.cancelled)
            job.save("leaderboard.json", result["leaderboard"])
            job.save("population.json", result["population"])
        if result["best"] is None:
            job.status("failed", reason="no candidate could be fitted")
            return 1
        job.event({"event": "finishing", "stopped": result["stopped"]})
        final = finish(ws, genome_from_dict(result["best"]), nthread=budget.nthread)
        _write_result(folder, spec, result, final)
        job.status("done", stopped=result["stopped"], verdict=final["verdict"])
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
    (folder / "recipe.json").write_text(json.dumps(recipe, indent=2))
    summary = {
        "verdict": final["verdict"],
        "model_error": final["model_error"],
        "baseline_error": final["baseline_error"],
        "band": final.get("band"),
        "bands": final.get("bands"),
    }
    (folder / "result.json").write_text(json.dumps(summary, indent=2, default=_plain))
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
