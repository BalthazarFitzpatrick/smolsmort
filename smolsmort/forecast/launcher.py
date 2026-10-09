"""minimal parallel launcher; the time allocator and ensemble belong to u7"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from smolsmort.forecast.family_worker import read_json


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def launch_families(folder: Path, request: dict, job) -> dict:
    processes = {}
    names = [item["name"] for item in request["families"]]
    started_at = time.time()
    states = {}
    try:
        for name in names:
            target = folder / "families" / name
            target.mkdir(parents=True, exist_ok=True)
            if read_json(target / "status.json", {}).get("state") == "done":
                continue
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = str(
                max(1, (request.get("nthread") or os.cpu_count() or 1) // len(names))
            )
            with (target / "worker.log").open("a") as log:
                processes[name] = subprocess.Popen(
                    [sys.executable, "-m", "smolsmort.forecast.family_worker", str(folder), name],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
        while True:
            states = {
                name: read_json(folder / "families" / name / "status.json", {"state": "starting"})
                for name in names
            }
            for name, process in processes.items():
                if process.poll() is not None and states[name]["state"] in ("running", "starting"):
                    states[name] = {
                        **states[name],
                        "state": "failed",
                        "reason": "family worker exited without reporting; restart to resume",
                    }
                    job.save(f"families/{name}/status.json", states[name])
            job.status("running", started_at=started_at, families=states)
            if job.cancelled:
                break
            if all(process.poll() is not None for process in processes.values()):
                break
            time.sleep(0.1)
    finally:
        for process in processes.values():
            stop_process(process)
    board = []
    for name in names:
        for entry in read_json(folder / "families" / name / "leaderboard.json", []):
            board.append({**entry, "family": name})
    board.sort(
        key=lambda entry: (
            entry["fitness"] if entry["fitness"] is not None else float("inf"),
            entry["families"],
        )
    )
    job.save("leaderboard.json", board)
    viable = [
        entry
        for entry in board
        if entry["fitness"] is not None and states[entry["family"]]["state"] == "done"
    ]
    if not viable and not job.cancelled:
        reasons = "; ".join(
            f"{name}: {state.get('reason', state['state'])}" for name, state in states.items()
        )
        raise ValueError(f"no family could be fitted; {reasons}")
    if viable:
        winner = viable[0]
        job.save(
            "eval_history.json",
            read_json(folder / "families" / winner["family"] / "eval_history.json"),
        )
    return {
        "leaderboard": board,
        "best": viable[0]["genome"] if viable else None,
        "population": [],
        "families": states,
        "stopped": "cancelled" if job.cancelled else "families complete",
    }
