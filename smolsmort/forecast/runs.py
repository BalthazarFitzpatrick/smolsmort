"""search runs as folders: the server writes a request, starts the worker process, reads its
events and results, and cancels it. every file a run leaves behind is listed in RUN_FILES"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from smolsmort.forecast.prep import Prepared
from smolsmort.forecast.spec import PrepSpec

RUN_FILES = (
    "request.json",
    "status.json",
    "events.jsonl",
    "worker.log",
    "profile.json",
    "leaderboard.json",
    "population.json",
    "recipe.json",
    "result.json",
    "forecast.parquet",
    "validation.parquet",
    "test.parquet",
)


class RunError(ValueError):
    pass


def start_run(
    runs_root: Path,
    spec: PrepSpec,
    prepared: Prepared,
    *,
    budget: dict | None = None,
    warm_from: str | None = None,
    recipe: dict | None = None,
    nthread: int | None = None,
) -> str:
    """write the request and start the worker; returns the run id. `recipe` refits a saved
    winner without a search, `warm_from` seeds a search with an earlier run's population"""
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    folder = Path(runs_root) / run_id
    folder.mkdir(parents=True)
    warm = str(Path(runs_root) / warm_from) if warm_from else None
    request = {
        "spec": asdict(spec),
        "prepared": str(prepared.folder),
        "summary": prepared.summary,
        "budget": budget or {},
        "warm_from": warm,
        "recipe": recipe,
        "nthread": nthread,
    }
    (folder / "request.json").write_text(json.dumps(request, indent=2, default=str))
    (folder / "status.json").write_text(json.dumps({"state": "starting"}))
    env = dict(os.environ)
    if nthread is None:
        # the parent may pin openmp to one thread for torch's sake; the worker has no torch
        env.pop("OMP_NUM_THREADS", None)
    log = (folder / "worker.log").open("w")
    process = subprocess.Popen(
        [sys.executable, "-m", "smolsmort.forecast.worker", str(folder)],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    (folder / "pid").write_text(str(process.pid))
    return run_id


def run_folder(runs_root: Path, run_id: str) -> Path:
    folder = (Path(runs_root) / run_id).resolve()
    if Path(runs_root).resolve() not in folder.parents or not folder.is_dir():
        raise RunError(f"no run called {run_id!r}")
    return folder


def run_state(runs_root: Path, run_id: str) -> dict:
    """status, the latest generation event, and whether the process is still alive"""
    folder = run_folder(runs_root, run_id)
    status = json.loads((folder / "status.json").read_text())
    events = []
    path = folder / "events.jsonl"
    if path.exists():
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    generations = [e for e in events if e.get("event") == "generation"]
    alive = _alive(folder)
    if status["state"] in ("starting", "running") and not alive:
        status = {"state": "failed", "reason": "the worker exited without reporting"}
    return {
        **status,
        "alive": alive,
        "generations": generations,
        "latest": events[-1] if events else None,
    }


def _alive(folder: Path) -> bool:
    try:
        pid = int((folder / "pid").read_text())
    except (FileNotFoundError, ValueError):
        return False
    try:
        finished, _ = os.waitpid(pid, os.WNOHANG)
        return finished == 0
    except ChildProcessError:
        # not our child (the server restarted): fall back to asking the os
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True


def cancel_run(runs_root: Path, run_id: str) -> None:
    """ask the worker to stop after its current generation; it still writes what it found"""
    folder = run_folder(runs_root, run_id)
    with contextlib.suppress(FileNotFoundError, ProcessLookupError, ValueError):
        os.kill(int((folder / "pid").read_text()), signal.SIGTERM)


def wait_run(runs_root: Path, run_id: str, timeout: float = 600) -> dict:
    """block until the run finishes; for tests and scripts, never the server"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = run_state(runs_root, run_id)
        if state["state"] in ("done", "failed") or not state["alive"]:
            return run_state(runs_root, run_id)
        time.sleep(0.2)
    raise RunError(f"run {run_id} still going after {timeout}s")


def list_runs(runs_root: Path) -> list[dict]:
    out = []
    for folder in sorted(Path(runs_root).glob("*"), reverse=True):
        if (folder / "request.json").exists():
            status = json.loads((folder / "status.json").read_text())
            out.append(
                {"id": folder.name, "state": status.get("state"), "verdict": status.get("verdict")}
            )
    return out
