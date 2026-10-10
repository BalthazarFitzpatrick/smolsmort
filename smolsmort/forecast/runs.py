"""search runs as folders: the server writes a request, starts the worker process, reads its
events and results, and cancels it. every file a run leaves behind is listed in RUN_FILES"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from smolsmort.forecast.prep import Prepared
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.spec import PrepSpec

RUN_FILES = (
    "request.json",
    "status.json",
    "events.jsonl",
    "progress.json",
    "eval_history.json",
    "generation_curves.jsonl",
    "worker.log",
    "profile.json",
    "leaderboard.json",
    "leaderboard_generation.json",
    "population.json",
    "recipe.json",
    "result.json",
    "forecast.parquet",
    "validation.parquet",
    "test.parquet",
)

_PROCESSES: dict[int, subprocess.Popen] = {}


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
    version: int | None = None,
    families: list[dict] | None = None,
    time_budget_s: float | None = None,
    transform: str = "none",
    ensemble: dict | None = None,
) -> str:
    """write the request and start the worker; returns the run id. `recipe` refits a saved
    winner without a search, `warm_from` seeds a search with an earlier run's population"""
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    folder = Path(runs_root) / run_id
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
    if version is not None or families is not None:
        request.pop("budget")
        request.update(
            version=version if version is not None else 2,
            families=families,
            time_budget_s=time_budget_s,
            transform=transform,
            ensemble=ensemble or {"enabled": False, "top": 3},
        )
    request = parse_request(request)
    folder.mkdir(parents=True)
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
    log.close()
    _PROCESSES[process.pid] = process
    (folder / "pid").write_text(str(process.pid))
    return run_id


def run_folder(runs_root: Path, run_id: str) -> Path:
    folder = (Path(runs_root) / run_id).resolve()
    if Path(runs_root).resolve() not in folder.parents or not folder.is_dir():
        raise RunError(f"no run called {run_id!r}")
    return folder


def run_state(
    runs_root: Path,
    run_id: str,
    *,
    since_generation: int | None = None,
    since_events: int | None = None,
) -> dict:
    """status, the latest generation event, and whether the process is still alive"""
    folder = run_folder(runs_root, run_id)
    status = json.loads((folder / "status.json").read_text())
    events = []
    path = folder / "events.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # the worker may still be writing the final line
                continue
    generations = [e for e in events if e.get("event") == "generation"]
    curve_path = folder / "generation_curves.jsonl"
    cursor = since_generation if since_generation is not None else -1
    event_cursor = since_events if since_events is not None else cursor
    curves = []
    revision_path = folder / "leaderboard_generation.json"
    revision = (
        json.loads(revision_path.read_text())
        if revision_path.exists()
        else generations[-1]["generation"]
        if generations
        else -1
    )
    if curve_path.exists() and (
        since_generation is None or not revision_path.exists() or cursor < revision
    ):
        for line in curve_path.read_text().splitlines():
            try:
                curve = json.loads(line)
            except json.JSONDecodeError:
                # the worker may still be writing the final line
                continue
            if curve["generation"] > cursor:
                curves.append(curve)
    alive = _alive(folder)
    if status["state"] in ("starting", "running") and not alive:
        status = {"state": "failed", "reason": "the worker exited without reporting"}
    state = {
        **status,
        "alive": alive,
        "generations": [event for event in generations if event["generation"] > event_cursor],
        "generation_curves": curves,
        "latest": events[-1] if events else None,
        **_metadata(folder, status, generations, events),
    }
    if since_generation is None:
        history_path = folder / "eval_history.json"
        state["eval_history"] = (
            json.loads(history_path.read_text()) if history_path.exists() else None
        )
    return state


def _metadata(folder: Path, status: dict, generations: list[dict], events: list[dict]) -> dict:
    request_path = folder / "request.json"
    request = json.loads(request_path.read_text()) if request_path.exists() else {}
    result_path = folder / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    kind = (
        "refit" if request.get("recipe") else "warm start" if request.get("warm_from") else "search"
    )
    latest = generations[-1] if generations else {}
    finishing = next((event for event in reversed(events) if event.get("stopped")), {})
    revision_path = folder / "leaderboard_generation.json"
    revision = (
        json.loads(revision_path.read_text())
        if revision_path.exists()
        else latest.get("generation", -1)
    )
    progress_path = folder / "progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else next((event for event in reversed(events) if event.get("event") == "progress"), {})
    )
    progress = {key: value for key, value in progress.items() if key != "event"}
    if latest.get("generation") == progress.get("generation"):
        progress["flat_generations"] = latest.get(
            "flat_generations", progress.get("flat_generations", 0)
        )
    if status["state"] in ("starting", "running"):
        updated_at = progress.get("updated_at", status.get("started_at"))
        if updated_at is not None:
            progress["elapsed"] = round(
                progress.get("elapsed", 0) + max(0, time.time() - updated_at), 1
            )
    progress.pop("updated_at", None)
    return {
        "kind": kind,
        "task": request.get("spec", {}).get("task"),
        "budget": request.get("budget", {}),
        "generation_count": len(generations),
        "generation": latest.get("generation"),
        "best_error": latest.get("best"),
        "best_wape": _validation_wape(request, latest, kind),
        "test_error": result.get("model_error"),
        "stopped": status.get("stopped") or finishing.get("stopped"),
        "elapsed": status.get("elapsed") or finishing.get("elapsed") or latest.get("elapsed"),
        "leaderboard_generation": revision,
        "progress": progress,
    }


def _validation_wape(request: dict, generation: dict, kind: str) -> float | None:
    if kind == "refit":
        return None
    metrics = generation.get("metrics") or {}
    wape = metrics.get("wape")
    if not isinstance(wape, (int, float)) or isinstance(wape, bool) or not math.isfinite(wape):
        # older regression events recorded their validation wape as best
        wape = (
            generation.get("best") if request.get("spec", {}).get("task") == "regression" else None
        )
    if isinstance(wape, (int, float)) and not isinstance(wape, bool) and math.isfinite(wape):
        return wape
    return None


def _alive(folder: Path) -> bool:
    try:
        pid = int((folder / "pid").read_text())
    except (FileNotFoundError, ValueError):
        return False
    process = _PROCESSES.get(pid)
    if process is not None:
        return process.poll() is None
    if sys.platform == "win32":
        return _windows_alive(pid)
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


def _windows_alive(pid: int) -> bool:
    """query a reattached worker without windows' terminating os.kill semantics"""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def cancel_run(runs_root: Path, run_id: str) -> None:
    """ask the worker to stop after its current candidate; it still writes what it found"""
    folder = run_folder(runs_root, run_id)
    (folder / "cancel").touch()


def wait_run(runs_root: Path, run_id: str, timeout: float = 600) -> dict:
    """block until the run finishes; for tests and scripts, never the server"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = run_state(runs_root, run_id)
        if state["state"] in ("done", "failed", "cancelled") or not state["alive"]:
            return run_state(runs_root, run_id)
        time.sleep(0.2)
    raise RunError(f"run {run_id} still going after {timeout}s")


def list_runs(runs_root: Path) -> list[dict]:
    out = []
    for folder in sorted(Path(runs_root).glob("*"), reverse=True):
        if (folder / "request.json").exists():
            state = run_state(runs_root, folder.name, since_generation=2**63 - 1)
            out.append(
                {
                    "id": folder.name,
                    **{
                        key: value
                        for key, value in state.items()
                        if key
                        not in (
                            "generations",
                            "generation_curves",
                            "latest",
                            "trace",
                            "leaderboard_generation",
                        )
                    },
                }
            )
    return out
