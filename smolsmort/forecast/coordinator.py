"""one durable wall-clock budget shared by isolated family searches"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

from smolsmort.forecast.family_worker import read_json
from smolsmort.forecast.pipeline import genome_from_dict
from smolsmort.forecast.search import Entry, _significant

GRACE_S = 2.0


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=0.5)


def stop_processes(processes):
    """terminate together so cleanup time does not multiply by family count"""
    active = [process for process in processes if process.poll() is None]
    for process in active:
        process.terminate()
    cleanup_deadline = time.monotonic() + 0.5
    for process in active:
        try:
            process.wait(timeout=max(0, cleanup_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
    cleanup_deadline = time.monotonic() + 0.5
    for process in active:
        process.wait(timeout=max(0, cleanup_deadline - time.monotonic()))


def best_entry(folder, name):
    entries = read_json(folder / "families" / name / "checkpoint.json", {}).get("entries", [])
    viable = [e for e in entries if e["fitness"] is not None and e.get("note") != "sample"]
    return min(viable, key=lambda e: (e["fitness"], e["families"]), default=None)


def paired_entry(data):
    return Entry(
        genome_from_dict(data["genome"]),
        data["fitness"],
        np.asarray(data["contrib"]) if data.get("contrib") is not None else None,
        data["metrics"],
        data["generation"],
    )


def launch_families(folder, request, job):
    names = [item["name"] for item in request["families"]]
    now = time.time()
    state = read_json(folder / "coordinator.json") or {
        "started_at": now,
        "deadline": now + request["time_budget_s"],
        "phase": "broad",
        "families": {},
    }
    deadline = state["deadline"]
    search_deadline = state["started_at"] + request["time_budget_s"] * 0.7
    states = state["families"]
    for name in names:
        states.setdefault(
            name,
            {
                "state": "pending",
                "phase": "broad",
                "slice_s": 0,
                "best_error": None,
                "pruned_reason": None,
            },
        )
        if state["phase"] != "broad" and states[name]["state"] == "running":
            states[name]["state"] = "paused"

    def save():
        job.save("coordinator.json", state)
        job.status(
            "running",
            started_at=state["started_at"],
            phase=state["phase"],
            families=states,
            elapsed=round(max(0, time.time() - state["started_at"]), 1),
        )

    def prune_families():
        viable = {name: entry for name in names if (entry := best_entry(folder, name)) is not None}
        if not viable:
            return
        leader = min(viable, key=lambda name: viable[name]["fitness"])
        for name, entry in viable.items():
            if name != leader and _significant(paired_entry(viable[leader]), paired_entry(entry)):
                states[name].update(
                    state="pruned",
                    pruned_reason=f"paired validation error significantly behind {leader}",
                )

    def run_slices(selected, slice_end):
        processes = {}
        nthread = max(1, (request.get("nthread") or os.cpu_count() or 1) // max(1, len(selected)))
        try:
            for name in selected:
                if job.cancelled or time.time() >= min(slice_end, deadline):
                    break
                target = folder / "families" / name
                target.mkdir(parents=True, exist_ok=True)
                states[name].update(
                    state="running", phase=state["phase"], slice_s=max(0, slice_end - time.time())
                )
                job.save(
                    f"families/{name}/slice.json",
                    {
                        "deadline": min(slice_end, deadline),
                        "phase": state["phase"],
                        "parent_pid": os.getpid(),
                        "nthread": nthread,
                    },
                )
                env = dict(os.environ)
                env["OMP_NUM_THREADS"] = str(nthread)
                with (target / "worker.log").open("a") as log:
                    processes[name] = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "smolsmort.forecast.family_worker",
                            str(folder),
                            name,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                states[name]["pid"] = processes[name].pid
                save()
            while any(p.poll() is None for p in processes.values()):
                for name in processes:
                    status = read_json(folder / "families" / name / "status.json", {})
                    states[name].update(
                        {k: v for k, v in status.items() if k not in ("state", "phase", "pid")}
                    )
                save()
                if job.cancelled or time.time() >= min(slice_end, deadline) + GRACE_S - 1:
                    break
                time.sleep(0.05)
        finally:
            interrupted = {name for name, process in processes.items() if process.poll() is None}
            stop_processes(processes.values())
            for name in processes:
                terminated = name in interrupted
                status = read_json(folder / "families" / name / "status.json", {})
                states[name].update(
                    {k: v for k, v in status.items() if k not in ("state", "phase", "pid")}
                )
                entry = best_entry(folder, name)
                states[name]["best_error"] = entry["fitness"] if entry else None
                states[name]["state"] = (
                    "cancelled"
                    if job.cancelled
                    else (
                        "paused"
                        if terminated or status.get("reason") in ("time cap", "broad defaults")
                        else status.get("state", "failed")
                    )
                )
                if terminated:
                    states[name]["terminated"] = True
                    states[name]["reason"] = (
                        "cancelled"
                        if job.cancelled
                        else "slice deadline exceeded grace; worker terminated"
                    )
                    job.event(
                        {"event": "terminated", "family": name, "reason": states[name]["reason"]}
                    )
            save()

    save()
    if state["phase"] == "broad":
        selected = [name for name in names if states[name]["state"] in ("pending", "running")]
        broad_end = min(search_deadline, state["started_at"] + request["time_budget_s"] * 0.4)
        run_slices(selected, broad_end)
        for name in selected:
            if states[name]["state"] in ("pending", "running"):
                states[name]["state"] = "paused"
        state["phase"] = "halving"
        save()
    while state["phase"] == "halving" and not job.cancelled:
        prune_families()
        selected = [name for name in names if states[name]["state"] == "paused"]
        if (
            len(selected) <= 1
            or time.time() >= state["started_at"] + request["time_budget_s"] * 0.55
        ):
            state["phase"] = "deep"
            save()
            break
        save()
        run_slices(selected, min(search_deadline, time.time() + request["time_budget_s"] * 0.1))
    if state["phase"] == "deep" and not job.cancelled:
        selected = [name for name in names if states[name]["state"] == "paused"]
        run_slices(selected, search_deadline)
        prune_families()
        state["phase"] = "finish"
        save()
    board = []
    winners = {}
    for name in names:
        checkpoint_entries = read_json(folder / "families" / name / "checkpoint.json", {}).get(
            "entries", []
        )
        states[name]["candidates_done"] = len(checkpoint_entries)
        winner = best_entry(folder, name)
        if winner:
            winners[name] = {**winner, "family": name}
            if states[name]["state"] == "paused":
                states[name]["state"] = "done"
            target = folder / "families" / name
            status = read_json(target / "status.json", {})
            if status.get("state") == "running":
                job.save(
                    f"families/{name}/status.json",
                    {
                        **status,
                        "state": "done",
                        "reason": states[name].get("reason", "time cap"),
                        "best_error": winner["fitness"],
                        "candidates_done": states[name].get("candidates_done", 0),
                        "terminated": states[name].get("terminated", False),
                    },
                )
            job.save(f"families/{name}/winner.json", winner["genome"])
        elif states[name]["state"] in ("running", "paused", "pending"):
            states[name].update(
                state="failed", reason="budget expired without a completed candidate"
            )
        family_board = [
            {
                key: value
                for key, value in entry.items()
                if key not in ("contrib", "prediction", "eval_history")
            }
            for entry in checkpoint_entries
            if entry.get("note") != "sample"
        ]
        family_board.sort(
            key=lambda entry: (
                entry["fitness"] if entry["fitness"] is not None else float("inf"),
                entry["families"],
            )
        )
        if family_board:
            job.save(f"families/{name}/leaderboard.json", family_board[:10])
        for entry in family_board[:10]:
            board.append({**entry, "family": name})
    board.sort(
        key=lambda e: (e["fitness"] if e["fitness"] is not None else float("inf"), e["families"])
    )
    job.save("leaderboard.json", board)
    save()
    ranked = sorted(winners.values(), key=lambda e: e["fitness"])
    result = {
        "leaderboard": board,
        "best": ranked[0]["genome"] if ranked else None,
        "population": [],
        "families": states,
        "stopped": "cancelled" if job.cancelled else "families complete",
        "elapsed": round(max(0, time.time() - state["started_at"]), 1),
    }
    if job.cancelled:
        return result
    if not ranked:
        raise ValueError(
            "no family could be fitted; "
            + "; ".join(f"{n}: {s.get('reason', s['state'])}" for n, s in states.items())
        )
    job.save("finalists.json", ranked)
    history = ranked[0].get("eval_history")
    job.save("eval_history.json", {**history, "key": ranked[0]["key"]} if history else None)
    if not (folder / "finish_complete.json").exists():
        if time.time() >= deadline:
            raise ValueError("deadline exhausted before final fitting")
        with (folder / "finish.log").open("a") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "smolsmort.forecast.ensemble",
                    str(folder),
                    str(os.getpid()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        state["finish_pid"] = process.pid
        save()
        try:
            while (
                process.poll() is None
                and not job.cancelled
                and time.time() < deadline + GRACE_S - 1
            ):
                time.sleep(0.05)
        finally:
            terminated = process.poll() is None
            stop_process(process)
        if terminated:
            state["finish_terminated"] = True
            save()
            if not job.cancelled:
                raise ValueError("final fit exceeded deadline grace; worker terminated")
        if job.cancelled:
            result["stopped"] = "cancelled"
            return result
        if process.returncode != 0:
            raise ValueError("final fit failed; see finish.log")
    result["finalized"] = True
    result["elapsed"] = round(max(0, time.time() - state["started_at"]), 1)
    return result
