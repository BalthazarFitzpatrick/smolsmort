"""one durable wall-clock budget shared by isolated family searches"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from collections import deque

import numpy as np

from smolsmort.forecast.family_worker import read_json
from smolsmort.forecast.pipeline import genome_from_dict
from smolsmort.forecast.processes import is_process_alive
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


def calibrate_families(folder, states, calibration, remaining_s):
    for name, status in states.items():
        checkpoint = read_json(folder / "families" / name / "checkpoint.json", {})
        entries = checkpoint.get("candidate_times", checkpoint.get("entries", []))
        measured = [entry for entry in entries if entry.get("elapsed_s", 0) > 0]
        broad = [
            entry
            for entry in measured
            if entry.get("phase") == "broad"
            and entry.get("cost_hint", 0) > 0
            and (entry.get("fitness") is not None or entry.get("note") == "too slow")
        ]
        prior = calibration.get(name, {})
        if broad and (
            "seconds_per_cost" not in prior
            or prior.get("estimate_is_lower_bound")
            and any(entry.get("fitness") is not None for entry in broad)
        ):
            prior.update(
                seconds_per_cost=sum(entry["elapsed_s"] for entry in broad)
                / sum(entry["cost_hint"] for entry in broad),
                broad_baseline=min(
                    (entry["fitness"] for entry in broad if entry.get("fitness") is not None),
                    default=None,
                ),
                broad_cost=sum(entry["cost_hint"] for entry in broad) / len(broad),
                estimate_is_lower_bound=not any(
                    entry.get("fitness") is not None for entry in broad
                ),
            )
        total_s = sum(entry["elapsed_s"] for entry in measured)
        hints = list(checkpoint.get("cost_hints", {}).values())
        hint = sum(hints) / len(hints) if hints else prior.get("broad_cost")
        estimate = (
            prior["seconds_per_cost"] * hint if hint and "seconds_per_cost" in prior else None
        )
        best = best_entry(folder, name)
        baseline = prior.get("broad_baseline")
        gain = max(0, baseline - best["fitness"]) if best and baseline is not None else 0
        rate = gain / total_s if total_s else 0
        prior.update(
            seconds_per_candidate=estimate,
            measured_seconds=total_s,
            improvement_per_second=rate,
            estimate_is_lower_bound=prior.get("estimate_is_lower_bound", False),
        )
        calibration[name] = prior
        status.update(
            seconds_per_candidate=estimate,
            candidates_expected=int(max(0, remaining_s) / estimate) if estimate else 0,
            improvement_per_second=rate,
            estimate_is_lower_bound=prior.get("estimate_is_lower_bound", False),
        )


def rank_families(selected, states):
    return sorted(
        selected,
        key=lambda name: (
            -states[name].get("improvement_per_second", 0),
            states[name].get("seconds_per_candidate") or float("inf"),
            states[name].get("best_error")
            if states[name].get("best_error") is not None
            else float("inf"),
        ),
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
    cap = request.get("max_concurrent_families", 4)
    total_threads = request.get("nthread") or os.cpu_count() or 1
    calibration = read_json(folder / "cost-calibration.json", {})
    orphan_pids = {
        status["pid"]
        for status in states.values()
        if status.get("state") == "running" and status.get("pid")
    }
    if state.get("finish_pid") and not (folder / "finish_complete.json").exists():
        orphan_pids.add(state["finish_pid"])
    until = time.monotonic() + GRACE_S
    while any(is_process_alive(pid) for pid in orphan_pids):
        if time.monotonic() >= until:
            raise ValueError("previous workers still stopping; resume again after cleanup")
        time.sleep(0.02)
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
        active_names = [
            name for name in names if states[name]["state"] in ("paused", "running", "pending")
        ]
        share_s = (
            max(0, search_deadline - time.time())
            * min(cap, len(active_names))
            / max(1, len(active_names))
        )
        calibrate_families(folder, states, calibration, share_s)
        for name in names:
            if states[name]["state"] not in ("paused", "running", "pending"):
                states[name]["candidates_expected"] = 0
        job.save("cost-calibration.json", calibration)
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
        phase_end = min(slice_end, deadline)
        waves = max(1, math.ceil(len(selected) / cap))
        slice_s = max(0, phase_end - time.time()) / waves
        queue = deque(selected)
        active = {}
        allocations = {}

        def complete(name, terminated=False):
            status = read_json(folder / "families" / name / "status.json", {})
            states[name].update(
                {k: v for k, v in status.items() if k not in ("state", "phase", "pid")}
            )
            entry = best_entry(folder, name)
            states[name]["best_error"] = entry["fitness"] if entry else None
            states[name]["state"] = (
                "cancelled"
                if job.cancelled
                else "paused"
                if terminated or status.get("reason") in ("time cap", "broad defaults")
                else status.get("state", "failed")
            )
            if terminated:
                states[name].update(
                    terminated=True,
                    reason="cancelled"
                    if job.cancelled
                    else "slice deadline exceeded grace; worker terminated",
                )
                job.event({"event": "terminated", "family": name, "reason": states[name]["reason"]})

        def split_threads():
            nthread = max(1, total_threads // max(1, len(active)))
            for name in active:
                if states[name].get("nthread") != nthread:
                    job.save(f"families/{name}/thread.json", {"nthread": nthread})
                    states[name]["nthread"] = nthread

        try:
            while queue or active:
                if job.cancelled:
                    break
                expired = []
                for name, process in list(active.items()):
                    if process.poll() is not None:
                        complete(name)
                        del active[name]
                    elif time.time() >= allocations[name] + GRACE_S - 1:
                        expired.append(name)
                    else:
                        status = read_json(folder / "families" / name / "status.json", {})
                        states[name].update(
                            {
                                k: v
                                for k, v in status.items()
                                if k not in ("state", "phase", "pid", "nthread")
                            }
                        )
                if expired:
                    stop_processes(active[name] for name in expired)
                    for name in expired:
                        complete(name, terminated=True)
                        del active[name]
                while queue and len(active) < cap and time.time() < phase_end:
                    name = queue.popleft()
                    target = folder / "families" / name
                    target.mkdir(parents=True, exist_ok=True)
                    started_at = time.time()
                    family_end = min(phase_end, started_at + slice_s)
                    allocation = {
                        "phase": state["phase"],
                        "started_at": started_at,
                        "deadline": family_end,
                        "slice_s": max(0, family_end - started_at),
                    }
                    states[name].setdefault("slice_history", []).append(allocation)
                    states[name].update(
                        state="running", phase=state["phase"], slice_s=allocation["slice_s"]
                    )
                    nthread = max(1, total_threads // min(cap, len(active) + len(queue) + 1))
                    job.save(
                        f"families/{name}/slice.json",
                        {**allocation, "parent_pid": os.getpid(), "nthread": nthread},
                    )
                    job.save(f"families/{name}/thread.json", {"nthread": nthread})
                    states[name]["nthread"] = nthread
                    env = dict(os.environ)
                    env["OMP_NUM_THREADS"] = str(nthread)
                    with (target / "worker.log").open("a") as log:
                        active[name] = subprocess.Popen(
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
                    allocations[name] = family_end
                    states[name]["pid"] = active[name].pid
                split_threads()
                save()
                if not active and (not queue or time.time() >= phase_end):
                    break
                time.sleep(0.05)
        finally:
            interrupted = {name for name, process in active.items() if process.poll() is None}
            stop_processes(active.values())
            for name in active:
                complete(name, terminated=name in interrupted)
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
        selected = rank_families(
            [name for name in names if states[name]["state"] == "paused"], states
        )
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
        selected = rank_families(
            [name for name in names if states[name]["state"] == "paused"], states
        )
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
