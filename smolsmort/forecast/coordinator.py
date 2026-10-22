"""one durable wall-clock budget shared by isolated family searches"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from collections import deque

import numpy as np

from smolsmort.forecast.families import get_family
from smolsmort.forecast.family_worker import read_json
from smolsmort.forecast.machine import describe_machine
from smolsmort.forecast.machine_config import (
    forecast_config_path,
    load_machine_config,
    resolve_parallelism,
)
from smolsmort.forecast.pipeline import genome_from_dict
from smolsmort.forecast.processes import is_process_alive
from smolsmort.forecast.search import Entry, _significant
from smolsmort.forecast.spec import normalize_mode

GRACE_S = 2.0
SHORT_BUDGET_S = 1800


def initialize_state(folder, request, *, started_at=None):
    now = time.time() if started_at is None else started_at
    state = read_json(folder / "coordinator.json") or {
        "started_at": now,
        "deadline": now + request["time_budget_s"],
        "phase": "broad",
        "families": {},
    }
    if "nthread" not in state:
        state["nthread"] = request.get("nthread") or os.cpu_count() or 1
    if "config_values" not in state.get("machine", {}):
        path = forecast_config_path()
        config = load_machine_config(path)
        # cores only matter when the config pins threads per family
        cores = os.cpu_count() or 1 if config["threads_per_family"] is not None else 1
        values = resolve_parallelism(request, config, state["nthread"], cores)
        if "machine" not in state:
            state["machine"] = describe_machine(state["nthread"], values["max_concurrent_families"])
        state["machine"].update(
            max_concurrent_families=values["max_concurrent_families"],
            config_path=str(path),
            config_values=values,
        )
    state.setdefault(
        "phase_history", [{"phase": state["phase"], "started_at": state["started_at"]}]
    )
    return state


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
            measured_threads = {entry.get("nthread") for entry in broad}
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
                nthread=next(iter(measured_threads)) if len(measured_threads) == 1 else None,
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


def estimate_finish_seconds(folder, names, calibration, request):
    costs = []
    for name in names:
        winner = best_entry(folder, name)
        if winner is None:
            continue
        measured = winner.get("elapsed_s")
        prior = calibration.get(name, {})
        estimates = [measured] if measured and math.isfinite(measured) and measured > 0 else []
        if not prior.get("estimate_is_lower_bound"):
            per_cost, hint = prior.get("seconds_per_cost"), winner.get("cost_hint")
            calibrated = (
                per_cost * hint if per_cost and hint else prior.get("seconds_per_candidate")
            )
            if calibrated and math.isfinite(calibrated) and calibrated > 0:
                estimates.append(calibrated)
        if not estimates:
            return None
        costs.append(max(estimates))
    # test and forecast fits cover each winner, including family importance fits
    # entity finishing also refits validation; the reserve adds a further safety factor
    mode = normalize_mode(request.get("spec", {}).get("mode") or "timeseries")
    fits = 3 if mode == "entity" else 2
    return fits * sum(costs) if costs else None


def launch_families(folder, request, job):
    names = [item["name"] for item in request["families"]]
    tunable = {name for name in names if get_family(name).space()}
    state = initialize_state(folder, request)
    deadline = state["deadline"]
    search_deadline = min(deadline, state["started_at"] + request["time_budget_s"] * 0.7)
    search_span = search_deadline - state["started_at"]
    states = state["families"]
    config_values = state["machine"]["config_values"]
    cap = config_values["max_concurrent_families"]
    configured_threads = config_values["threads_per_family"]
    total_threads = state["nthread"]
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

    def close_slice(name, status, *, recovering=False):
        history = states[name].get("slice_history", [])
        if history and "ended_at" not in history[-1]:
            ended_at = status.get("ended_at")
            if status.get("started_at", 0) < history[-1]["started_at"]:
                ended_at = None
            if ended_at is None and not recovering:
                ended_at = time.time()
            history[-1].update(
                ended_at=ended_at,
                wall_s=max(0, ended_at - history[-1]["started_at"])
                if ended_at is not None
                else None,
            )

    def complete(name, terminated=False):
        target = folder / "families" / name
        status = read_json(target / "status.json", {})
        close_slice(name, status)
        states[name].update(
            {k: v for k, v in status.items() if k not in ("state", "phase", "pid", "nthread")}
        )
        checkpoint = read_json(target / "checkpoint.json", {})
        states[name].update(
            generation=checkpoint.get("generation", 0),
            candidates_done=len(checkpoint.get("entries", [])),
        )
        entry = best_entry(folder, name)
        states[name]["best_error"] = entry["fitness"] if entry else None
        if job.cancelled:
            states[name]["state"] = "cancelled"
        elif status.get("state") in ("failed", "cancelled"):
            states[name]["state"] = status["state"]
        elif name not in tunable and entry:
            states[name]["state"] = "done"
        elif terminated or status.get("reason") in ("time cap", "broad defaults"):
            states[name]["state"] = "paused"
        else:
            states[name]["state"] = status.get("state", "failed")
        if terminated:
            states[name].update(
                terminated=True,
                reason="cancelled"
                if job.cancelled
                else "slice deadline exceeded grace; worker terminated",
            )
            job.event({"event": "terminated", "family": name, "reason": states[name]["reason"]})

    for name in names:
        checkpoint = read_json(folder / "families" / name / "checkpoint.json", {})
        states[name].update(
            generation=checkpoint.get("generation", 0),
            candidates_done=len(checkpoint.get("entries", [])),
        )
        if states[name]["state"] in ("running", "paused", "pruned"):
            # worker completion can precede the coordinator's last durable state
            status = read_json(folder / "families" / name / "status.json", {})
            if states[name]["state"] == "running":
                close_slice(name, status, recovering=True)
            if status.get("state") in ("failed", "cancelled") or (
                status.get("state") == "done"
                and (
                    states[name]["state"] == "running"
                    or status.get("reason") not in ("time cap", "broad defaults")
                )
            ):
                complete(name)
                states[name]["pruned_reason"] = None
            elif states[name]["state"] == "running" and state["phase"] != "broad":
                states[name]["state"] = "paused"
        if (
            name not in tunable
            and best_entry(folder, name)
            and states[name]["state"] not in ("failed", "cancelled")
        ):
            states[name].update(state="done", pruned_reason=None)

    def save():
        nonlocal search_deadline, search_span
        active_names = [
            name for name in names if states[name]["state"] in ("paused", "running", "pending")
        ]
        share_s = (
            max(0, search_deadline - time.time())
            * min(cap, len(active_names))
            / max(1, len(active_names))
        )
        calibrate_families(folder, states, calibration, share_s)
        estimate = estimate_finish_seconds(folder, names, calibration, request)
        reserve = request["time_budget_s"] * 0.3
        if request["time_budget_s"] > SHORT_BUDGET_S and estimate is not None:
            reserve = min(reserve, max(estimate * 2, 300))
        search_deadline = deadline - reserve
        search_span = max(0, search_deadline - state["started_at"])
        state.update(
            finish_estimate_s=estimate,
            finish_reserve_s=reserve,
            search_deadline=search_deadline,
        )
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

    def change_phase(phase):
        if state["phase"] != phase:
            state["phase"] = phase
            state["phase_history"].append({"phase": phase, "started_at": time.time()})

    def prune_families():
        viable = {name: entry for name in names if (entry := best_entry(folder, name)) is not None}
        if not viable:
            return
        leader = min(viable, key=lambda name: viable[name]["fitness"])
        for name, entry in viable.items():
            # defaults and partial first generations cannot eliminate a tunable family
            if (
                name in tunable
                and states[name]["state"] == "paused"
                and states[name].get("phase") in ("halving", "deep")
                and states[name].get("generation", 0) >= 1
                and states[name].get("candidates_done", 0) > 1
                and name != leader
                and _significant(
                    paired_entry(viable[leader]),
                    paired_entry(entry),
                    sigma=request.get("prune_sigma", 1.0),
                )
            ):
                states[name].update(
                    state="pruned",
                    pruned_reason=f"paired validation error significantly behind {leader}",
                )

    def run_slices(selected, slice_end, *, leftover=False):
        phase_end = min(slice_end, search_deadline)
        waves = max(1, math.ceil(len(selected) / cap))
        slice_s = max(0, phase_end - time.time()) / waves
        queue = deque(selected)
        active = {}
        allocations = {}

        def split_threads():
            nthread = configured_threads or max(1, total_threads // max(1, len(active)))
            for name in active:
                if states[name].get("nthread") != nthread:
                    job.save(f"families/{name}/thread.json", {"nthread": nthread})
                    states[name]["nthread"] = nthread

        try:
            while queue or active:
                if job.cancelled:
                    break
                phase_end = min(slice_end, search_deadline)
                expired = []
                for name, process in list(active.items()):
                    if process.poll() is not None:
                        complete(name)
                        del active[name]
                    elif time.time() >= min(allocations[name], phase_end) + GRACE_S - 1:
                        expired.append(name)
                    else:
                        status = read_json(folder / "families" / name / "status.json", {})
                        states[name].update(
                            {
                                k: v
                                for k, v in status.items()
                                if k not in ("state", "phase", "pid", "nthread")
                                and (k != "reason" or not leftover)
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
                    if leftover:
                        allocation["reason"] = "resumed with leftover time"
                        states[name].update(reason=allocation["reason"], pruned_reason=None)
                    states[name].setdefault("slice_history", []).append(allocation)
                    states[name].update(
                        state="running", phase=state["phase"], slice_s=allocation["slice_s"]
                    )
                    nthread = configured_threads or max(
                        1, total_threads // min(cap, len(active) + len(queue) + 1)
                    )
                    allocation["nthread"] = nthread
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
        broad_end = min(search_deadline, state["started_at"] + search_span * 4 / 7)
        run_slices(selected, broad_end)
        for name in selected:
            if states[name]["state"] in ("pending", "running"):
                states[name]["state"] = "paused"
        change_phase("halving")
        save()
    while state["phase"] == "halving" and not job.cancelled:
        prune_families()
        selected = rank_families(
            [name for name in names if name in tunable and states[name]["state"] == "paused"],
            states,
        )
        if len(selected) <= 1 or time.time() >= state["started_at"] + search_span * 5.5 / 7:
            change_phase("deep")
            save()
            break
        save()
        run_slices(selected, min(search_deadline, time.time() + search_span / 7))
    if state["phase"] == "deep" and not job.cancelled:
        selected = rank_families(
            [name for name in names if name in tunable and states[name]["state"] == "paused"],
            states,
        )
        run_slices(selected, search_deadline)
        prune_families()
        while not job.cancelled and time.time() < search_deadline:
            selected = rank_families(
                [
                    name
                    for name in names
                    if name in tunable and states[name]["state"] in ("paused", "pruned")
                ],
                states,
            )[:cap]
            if not selected:
                break
            run_slices(selected, search_deadline, leftover=True)
        change_phase("finish")
        save()
    board = []
    winners = {}
    for name in names:
        checkpoint = read_json(folder / "families" / name / "checkpoint.json", {})
        checkpoint_entries = checkpoint.get("entries", [])
        slice_times = [item.get("wall_s") for item in states[name].get("slice_history", [])]
        states[name].update(
            wall_s=sum(slice_times) if all(value is not None for value in slice_times) else None,
            candidate_s=sum(
                entry.get("elapsed_s", 0)
                for entry in checkpoint.get("candidate_times", checkpoint_entries)
            ),
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
        "machine": state["machine"],
        "phase_history": state["phase_history"],
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
        state["phase_history"].append({"phase": "finish_start", "started_at": time.time()})
        save()
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(total_threads)
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
                env=env,
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
            state["phase_history"].append({"phase": "finish_end", "started_at": time.time()})
            save()
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
    result["elapsed"] = max(0, time.time() - state["started_at"])
    final = read_json(folder / "result.json")
    if final:
        final.update(
            machine=state["machine"],
            phase_history=state["phase_history"],
            elapsed=result["elapsed"],
        )
        # retain the final fit's metrics while exposing durable search progress
        for name, status in states.items():
            summary = final.setdefault("families", {}).setdefault(name, {})
            summary.update(
                candidates=status["candidates_done"],
                generation=status["generation"],
                reason=status.get("reason"),
                wall_s=status["wall_s"],
                candidate_s=status["candidate_s"],
                nthread=status.get("nthread"),
            )
        job.save("result.json", final)
    result["finalized"] = True
    return result
