"""bounded family queues, per-candidate limits and measured search costs"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast.pipeline import make_genome
from smolsmort.forecast.requests import parse_request


def make_request(**changes):
    return {
        "version": 2,
        "time_budget_s": 90,
        "families": [{"name": "xgboost"}],
        **changes,
    }


def read_json(path):
    return json.loads(path.read_text())


def test_v2_control_defaults():
    request = parse_request(make_request())
    assert request["max_concurrent_families"] == 4
    assert request["candidate_limit_fraction"] == 0.25


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, 1.5, math.nan, math.inf, "2", None])
def test_invalid_family_concurrency_is_rejected(value):
    with pytest.raises(ValueError, match="max_concurrent_families"):
        parse_request(make_request(max_concurrent_families=value))


@pytest.mark.parametrize("value", [0, -0.1, 1.01, math.nan, math.inf, -math.inf, True, "0.5", None])
def test_invalid_candidate_limit_fraction_is_rejected(value):
    with pytest.raises(ValueError, match="candidate_limit_fraction"):
        parse_request(make_request(candidate_limit_fraction=value))


@pytest.mark.parametrize("concurrency,fraction", [(1, 1), (2, 0.01), (8, 0.5)])
def test_valid_coordinator_controls_are_retained(concurrency, fraction):
    request = parse_request(
        make_request(max_concurrent_families=concurrency, candidate_limit_fraction=fraction)
    )
    assert request["max_concurrent_families"] == concurrency
    assert request["candidate_limit_fraction"] == fraction


def test_v1_controls_are_untouched():
    request = {
        "budget": {"population": 3},
        "max_concurrent_families": 0,
        "candidate_limit_fraction": math.nan,
    }
    assert parse_request(request) is request


def test_cost_calibration_scales_current_population_and_retains_broad_baseline(tmp_path):
    from smolsmort.forecast.coordinator import calibrate_families

    target = tmp_path / "families" / "xgboost"
    target.mkdir(parents=True)
    entries = [{"fitness": 0.5, "families": 1, "phase": "broad", "elapsed_s": 10, "cost_hint": 20}]
    checkpoint = {"entries": entries, "cost_hints": {"a": 100, "b": 200}}
    (target / "checkpoint.json").write_text(json.dumps(checkpoint))
    states = {"xgboost": {"state": "paused"}}
    calibration = {}
    calibrate_families(tmp_path, states, calibration, remaining_s=90)
    assert calibration["xgboost"]["seconds_per_cost"] == 0.5
    assert states["xgboost"]["seconds_per_candidate"] == 75
    assert states["xgboost"]["candidates_expected"] == 1
    assert states["xgboost"]["improvement_per_second"] == 0
    entries.extend(
        [
            {"fitness": 0.2, "families": 1, "phase": "deep", "elapsed_s": 5, "cost_hint": 10},
            {
                "fitness": None,
                "families": 1,
                "phase": "deep",
                "elapsed_s": 4,
                "cost_hint": 12,
                "note": "too slow",
            },
        ]
    )
    checkpoint["cost_hints"] = {"a": 10, "b": 20}
    (target / "checkpoint.json").write_text(json.dumps(checkpoint))
    calibrate_families(tmp_path, states, calibration, remaining_s=90)
    assert calibration["xgboost"]["broad_baseline"] == 0.5
    assert calibration["xgboost"]["measured_seconds"] == 19
    assert states["xgboost"]["seconds_per_candidate"] == 7.5
    assert states["xgboost"]["candidates_expected"] == 12
    assert states["xgboost"]["improvement_per_second"] == pytest.approx(0.3 / 19)


def test_unmeasured_family_has_no_speed_estimate(tmp_path):
    from smolsmort.forecast.coordinator import calibrate_families

    states = {"xgboost": {"state": "pending"}}
    calibration = {}
    calibrate_families(tmp_path, states, calibration, remaining_s=90)
    assert states["xgboost"]["seconds_per_candidate"] is None
    assert states["xgboost"]["candidates_expected"] == 0
    assert states["xgboost"]["improvement_per_second"] == 0


def test_timeout_only_calibration_reports_a_cost_lower_bound(tmp_path):
    from smolsmort.forecast.coordinator import calibrate_families

    target = tmp_path / "families" / "ets"
    target.mkdir(parents=True)
    checkpoint = {
        "entries": [
            {
                "fitness": None,
                "families": 0,
                "phase": "broad",
                "elapsed_s": 4,
                "cost_hint": 20,
                "note": "too slow",
            }
        ],
        "cost_hints": {"default": 20},
    }
    (target / "checkpoint.json").write_text(json.dumps(checkpoint))
    states = {"ets": {"state": "failed"}}
    calibration = {}
    calibrate_families(tmp_path, states, calibration, remaining_s=90)
    assert calibration["ets"]["estimate_is_lower_bound"] is True
    assert calibration["ets"]["broad_baseline"] is None
    assert calibration["ets"]["measured_seconds"] == 4
    assert states["ets"]["seconds_per_candidate"] == 4
    assert states["ets"]["improvement_per_second"] == 0


def test_speed_ranking_prefers_gain_rate_then_candidate_speed():
    from smolsmort.forecast.coordinator import rank_families

    states = {
        "higher_gain": {
            "improvement_per_second": 0.02,
            "seconds_per_candidate": 20,
            "best_error": 0.4,
        },
        "faster_tie": {
            "improvement_per_second": 0.01,
            "seconds_per_candidate": 1,
            "best_error": 0.3,
        },
        "slower_tie": {
            "improvement_per_second": 0.01,
            "seconds_per_candidate": 5,
            "best_error": 0.1,
        },
        "unmeasured": {
            "improvement_per_second": 0,
            "seconds_per_candidate": None,
            "best_error": None,
        },
    }
    assert rank_families(["unmeasured", "slower_tie", "faster_tie", "higher_gain"], states) == [
        "higher_gain",
        "faster_tie",
        "slower_tie",
        "unmeasured",
    ]


def test_family_queue_fills_freed_slots_and_records_wave_slices(tmp_path, monkeypatch):
    from smolsmort.forecast import coordinator

    clock = [100.0]
    names = ["snaive", "xgboost", "lightgbm", "ets"]
    durations = {"snaive": 1, "xgboost": 5, "lightgbm": 2, "ets": 1}
    processes, launched, statuses = [], [], []

    def save_json(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    class QueuedProcess:
        def __init__(self, command, **kwargs):
            self.name = command[-1]
            target = tmp_path / "families" / self.name
            self.allocation = read_json(target / "slice.json")
            self.started = clock[0]
            self.pid = 1000 + len(processes)
            self.returncode = None
            self.observed_threads = set()
            active = sum(process.poll() is None for process in processes)
            assert active < 2
            processes.append(self)
            launched.append((self.name, self.allocation["phase"], self.started))

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            target = tmp_path / "families" / self.name
            threads_path = target / "thread.json"
            threads = (
                read_json(threads_path)["nthread"]
                if threads_path.exists()
                else self.allocation["nthread"]
            )
            self.observed_threads.add(threads)
            if clock[0] < self.started + durations[self.name]:
                return None
            self.returncode = 0
            genome = make_genome(["lag"], "squared", {}, self.name)
            save_json(
                f"families/{self.name}/checkpoint.json",
                {
                    "entries": [
                        {
                            "key": genome.key(),
                            "genome": genome.to_dict(),
                            "families": 1,
                            "fitness": 0.2,
                            "contrib": [0.2] * 4,
                            "generation": 0,
                            "metrics": {},
                            "note": "",
                            "elapsed_s": durations[self.name],
                            "nthread": threads,
                            "cost_hint": 1,
                            "phase": self.allocation["phase"],
                        }
                    ]
                },
            )
            save_json(
                f"families/{self.name}/status.json",
                {
                    "state": "done",
                    "reason": "single candidate"
                    if self.name in ("snaive", "ets")
                    else "broad defaults"
                    if self.allocation["phase"] == "broad"
                    else "grid complete",
                },
            )
            return 0

        def terminate(self):
            self.returncode = -1

        def kill(self):
            self.returncode = -1

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(coordinator.time, "time", lambda: clock[0])
    monkeypatch.setattr(coordinator.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        coordinator.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration)
    )
    monkeypatch.setattr(coordinator.subprocess, "Popen", QueuedProcess)
    save_json("finish_complete.json", {})
    job = SimpleNamespace(
        cancelled=False,
        save=save_json,
        status=lambda *args, **kwargs: statuses.append(deepcopy(kwargs)),
        event=lambda payload: None,
    )
    result = coordinator.launch_families(
        tmp_path,
        make_request(
            families=[{"name": name} for name in names], max_concurrent_families=2, nthread=4
        ),
        job,
    )
    broad = [(name, started) for name, phase, started in launched if phase == "broad"]
    assert {name for name, started in broad} == set(names)
    start_times = dict(broad)
    assert start_times["lightgbm"] >= start_times["snaive"] + 1
    assert start_times["lightgbm"] < start_times["xgboost"] + 5
    assert (
        max(
            sum(
                other.started <= process.started < other.started + durations[other.name]
                for other in processes
            )
            for process in processes
        )
        <= 2
    )
    for name, state in result["families"].items():
        history = state["slice_history"]
        assert history
        for allocation in history:
            assert {"phase", "started_at", "deadline", "slice_s"} <= allocation.keys()
            assert allocation["deadline"] <= 190
            assert allocation["slice_s"] == pytest.approx(
                allocation["deadline"] - allocation["started_at"]
            )
        assert (tmp_path / "families" / name / "thread.json").exists()
        checkpoint = read_json(tmp_path / "families" / name / "checkpoint.json")
        assert all(1 <= entry["nthread"] <= 4 for entry in checkpoint["entries"])
    assert any(len(process.observed_threads) > 1 for process in processes)
    surviving = [
        process
        for process in processes
        if process.name == "xgboost" and process.allocation["phase"] == "broad"
    ]
    assert surviving[0].observed_threads == {2, 4}
    assert any(
        any(state.get("phase") == "broad" for state in status.get("families", {}).values())
        for status in statuses
    )
    calibration = read_json(tmp_path / "cost-calibration.json")
    assert set(calibration) == set(names)
    for name, state in result["families"].items():
        assert {
            "seconds_per_candidate",
            "candidates_expected",
            "improvement_per_second",
        } <= state.keys()
        assert state["seconds_per_candidate"] > 0
        assert state["candidates_expected"] >= 0
        assert state["improvement_per_second"] >= 0
        measured = calibration[name]
        assert measured["seconds_per_cost"] == pytest.approx(durations[name])
        assert measured["broad_baseline"] == pytest.approx(0.2)
        assert measured["measured_seconds"] > 0


def test_immediate_resume_waits_for_orphan_before_reusing_worker_slot(tmp_path, monkeypatch):
    from smolsmort.forecast import coordinator

    clock = [100.0]
    old_pid = 9991
    old_exits_at = 100.4
    launched = []

    def save_json(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def is_process_alive(pid):
        return pid == old_pid and clock[0] < old_exits_at

    class ResumedProcess:
        pid = 9992

        def __init__(self, command, **kwargs):
            assert not is_process_alive(old_pid), (
                "resumed worker exceeded cap while orphan was still alive"
            )
            launched.append(clock[0])
            genome = make_genome(["lag"], "squared", {})
            save_json("families/xgboost/status.json", {"state": "done", "reason": "plateau"})
            save_json(
                "families/xgboost/checkpoint.json",
                {
                    "entries": [
                        {
                            "key": genome.key(),
                            "genome": genome.to_dict(),
                            "families": 1,
                            "fitness": 0.1,
                            "generation": 0,
                            "note": "",
                        }
                    ]
                },
            )

        def poll(self):
            return 0

    save_json(
        "coordinator.json",
        {
            "started_at": 95,
            "deadline": 185,
            "phase": "halving",
            "families": {"xgboost": {"state": "running", "pid": old_pid}},
        },
    )
    save_json("finish_complete.json", {})
    monkeypatch.setattr(coordinator, "is_process_alive", is_process_alive, raising=False)
    monkeypatch.setattr(coordinator.time, "time", lambda: clock[0])
    monkeypatch.setattr(coordinator.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        coordinator.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(coordinator.subprocess, "Popen", ResumedProcess)
    job = SimpleNamespace(
        cancelled=False,
        save=save_json,
        status=lambda *args, **kwargs: None,
        event=lambda payload: None,
    )
    result = coordinator.launch_families(tmp_path, make_request(max_concurrent_families=1), job)
    assert result["finalized"]
    assert len(launched) == 1
    assert old_exits_at <= launched[0] <= 100 + coordinator.GRACE_S
    assert read_json(tmp_path / "coordinator.json")["deadline"] == 185


def score_slow_candidate(ws, genome, *, nthread=1, sample=None):
    """inject a blocked fit into the same spawn boundary used by real scoring"""
    from smolsmort.forecast.pipeline import Scored

    alpha = dict(genome.params)["alpha"]
    if alpha == 1:
        time.sleep(4)
    return Scored(float(alpha), np.array([alpha], dtype=float), {}, None)


def test_first_broad_default_exceeds_fractional_limit_but_second_candidate_cannot(tmp_path):
    from test_forecast_coordinator import process_alive

    from smolsmort.forecast.candidate_worker import CandidateScorer, CandidateTimeout

    (tmp_path / "slice.json").write_text(json.dumps({"phase": "broad"}))
    default = make_genome(["lag"], "squared", {"alpha": 1})
    with CandidateScorer(
        None,
        tmp_path,
        deadline=time.time() + 12,
        slice_s=6,
        fraction=0.05,
        score_fn=score_slow_candidate,
    ) as scorer:
        scored = scorer.score(default)
        assert scored.fitness == 1 and scored.elapsed_s >= 4
        with pytest.raises(CandidateTimeout):
            scorer.score(default)
        assert not process_alive(read_json(tmp_path / "candidate_pid.json")["pid"])


def test_resumed_broad_default_keeps_candidate_limit(tmp_path):
    from smolsmort.forecast.candidate_worker import CandidateScorer, CandidateTimeout

    (tmp_path / "slice.json").write_text(json.dumps({"phase": "broad"}))
    (tmp_path / "checkpoint.json").write_text(json.dumps({"entries": [{"fitness": 0.4}]}))
    default = make_genome(["lag"], "squared", {"alpha": 1})
    with (
        CandidateScorer(
            None,
            tmp_path,
            deadline=time.time() + 12,
            slice_s=6,
            fraction=0.05,
            score_fn=score_slow_candidate,
        ) as scorer,
        pytest.raises(CandidateTimeout),
    ):
        scorer.score(default)


def test_first_broad_default_still_obeys_slice_deadline_and_grace(tmp_path):
    from test_forecast_coordinator import process_alive

    from smolsmort.forecast.candidate_worker import CandidateScorer, CandidateTimeout

    (tmp_path / "slice.json").write_text(json.dumps({"phase": "broad"}))
    default = make_genome(["lag"], "squared", {"alpha": 1})
    started = time.monotonic()
    with (
        CandidateScorer(
            None,
            tmp_path,
            deadline=time.time() + 1,
            slice_s=1,
            fraction=0.01,
            score_fn=score_slow_candidate,
        ) as scorer,
        pytest.raises(CandidateTimeout),
    ):
        scorer.score(default)
    assert time.monotonic() - started < 4
    assert not process_alive(read_json(tmp_path / "candidate_pid.json")["pid"])


def score_with_slow_family_fit(ws, genome, *, nthread=1, sample=None):
    from smolsmort.forecast.families import get_family
    from smolsmort.forecast.pipeline import score

    family = get_family(genome.family)
    original_fit = family.fit

    def fit_candidate(*args, **kwargs):
        if dict(genome.params).get("max_depth") == 99:
            time.sleep(8)
        return original_fit(*args, **kwargs)

    family.fit = fit_candidate
    try:
        return score(ws, genome, nthread=nthread, sample=sample)
    finally:
        family.fit = original_fit


def sleep_candidate_descendant():
    time.sleep(30)


def score_with_nested_child(ws, genome, *, nthread=1, sample=None):
    import multiprocessing
    from pathlib import Path

    child = multiprocessing.get_context("spawn").Process(target=sleep_candidate_descendant)
    child.start()
    Path(ws).write_text(str(child.pid))
    time.sleep(30)


def test_candidate_timeout_reaps_nested_training_child(tmp_path):
    from test_forecast_coordinator import process_alive, wait_for

    from smolsmort.forecast.candidate_worker import CandidateScorer, CandidateTimeout

    pid_path = tmp_path / "nested-child.pid"
    genome = make_genome(["lag"], "squared", {"alpha": 1})
    with CandidateScorer(
        str(pid_path),
        tmp_path,
        deadline=time.time() + 15,
        slice_s=4,
        fraction=0.25,
        score_fn=score_with_nested_child,
    ) as scorer:
        with pytest.raises(CandidateTimeout):
            scorer.score(genome)
        assert pid_path.exists()
        child_pid = int(pid_path.read_text())
        candidate_pid = read_json(tmp_path / "candidate_pid.json")["pid"]
        wait_for(lambda: not process_alive(child_pid), timeout=2)
        assert not process_alive(candidate_pid)
        assert not process_alive(child_pid)


def test_slow_real_family_fit_times_out_and_fast_fit_continues(tmp_path):
    from test_forecast_coordinator import make_trending_panel, process_alive

    pytest.importorskip("xgboost")
    from smolsmort.forecast.candidate_worker import CandidateScorer, CandidateTimeout
    from smolsmort.forecast.pipeline import load_workspace

    spec, prepared = make_trending_panel(tmp_path)
    ws = load_workspace(spec, prepared.folder, prepared.summary)
    slow = make_genome(ws.families, "squared", {"max_depth": 99})
    fast = make_genome(ws.families, "squared", {"max_depth": 3})
    target = tmp_path / "candidate"
    target.mkdir()
    pids = set()
    with CandidateScorer(
        ws,
        target,
        deadline=time.time() + 30,
        slice_s=12,
        fraction=0.25,
        score_fn=score_with_slow_family_fit,
    ) as scorer:
        with pytest.raises(CandidateTimeout, match="too slow"):
            scorer.score(slow, nthread=1)
        pids.add(read_json(target / "candidate_pid.json")["pid"])
        assert not any(process_alive(pid) for pid in pids)
        scored = scorer.score(fast, nthread=2)
        pids.add(read_json(target / "candidate_pid.json")["pid"])
        assert math.isfinite(scored.fitness)
        assert scored.prediction is not None
    assert not any(process_alive(pid) for pid in pids)


def test_timed_out_candidate_is_reaped_and_next_candidate_scores(tmp_path):
    from test_forecast_coordinator import process_alive

    from smolsmort.forecast.candidate_worker import CandidateScorer, CandidateTimeout

    slow = make_genome(["lag"], "squared", {"alpha": 1})
    fast = make_genome(["lag"], "squared", {"alpha": 2})
    pids = set()
    started = time.monotonic()
    with CandidateScorer(
        None,
        tmp_path,
        deadline=time.time() + 15,
        slice_s=4,
        fraction=0.25,
        score_fn=score_slow_candidate,
    ) as scorer:
        with pytest.raises(CandidateTimeout, match="too slow") as caught:
            scorer.score(slow, nthread=2)
        assert caught.value.elapsed_s > 0
        assert time.monotonic() - started < 3
        pids.add(read_json(tmp_path / "candidate_pid.json")["pid"])
        assert not any(process_alive(pid) for pid in pids)
        scored = scorer.score(fast, nthread=4)
        pids.add(read_json(tmp_path / "candidate_pid.json")["pid"])
        assert scored.fitness == 2
    assert not any(process_alive(pid) for pid in pids)


def test_search_records_timeout_threads_and_continues(tmp_path):
    import importlib

    from smolsmort.forecast.candidate_worker import CandidateScorer
    from smolsmort.forecast.families import Param

    module = importlib.import_module("smolsmort.forecast.search")
    family = SimpleNamespace(
        name="lightgbm",
        needs="lag_features",
        space=lambda: [Param("alpha", "int", 1, 3, default=1)],
        objectives=lambda task: ("squared",),
    )
    ws = SimpleNamespace(
        mode="row",
        spec=SimpleNamespace(task="regression"),
        masks={"train": np.ones(20, dtype=bool)},
        families=["lag"],
        profile={"genes": {"objectives": ["squared"]}},
    )
    checkpoints = []
    threads = [2]

    def read_threads():
        current = threads[0]
        threads[0] = 4
        return current

    with CandidateScorer(
        ws,
        tmp_path,
        deadline=time.time() + 15,
        slice_s=4,
        fraction=0.25,
        score_fn=score_slow_candidate,
    ) as scorer:
        result = module.search(
            ws,
            module.Budget(population=3),
            family=family,
            method="grid",
            space={"alpha": [1, 2, 3]},
            evaluate_score=scorer.score,
            read_threads=read_threads,
            cost_hint=lambda genome: float(dict(genome.params)["alpha"]),
            candidate_phase="deep",
            on_checkpoint=checkpoints.append,
        )
    failed = next(
        entry for entry in result["leaderboard"] if entry["genome"]["params"]["alpha"] == 1
    )
    assert failed["fitness"] is None
    assert "too slow" in failed["note"]
    assert result["best"]["params"]["alpha"] == 2
    entries = checkpoints[-1]["entries"]
    assert entries[0]["nthread"] == 2
    assert {entry["nthread"] for entry in entries[1:]} == {4}
    assert all(entry["elapsed_s"] > 0 and entry["cost_hint"] > 0 for entry in entries)
    assert all(entry["phase"] == "deep" for entry in entries)


def test_candidate_timing_retains_sample_and_full_attempts(monkeypatch):
    import importlib

    from smolsmort.forecast.families import Param
    from smolsmort.forecast.pipeline import Scored

    module = importlib.import_module("smolsmort.forecast.search")
    family = SimpleNamespace(
        name="lightgbm",
        needs="lag_features",
        space=lambda: [Param("alpha", "int", 1, 1, default=1)],
        objectives=lambda task: ("squared",),
    )
    ws = SimpleNamespace(
        mode="row",
        spec=SimpleNamespace(task="regression"),
        masks={"train": np.ones(20, dtype=bool)},
        families=["lag"],
        profile={"genes": {"objectives": ["squared"]}},
    )
    checkpoints = []
    calls = []

    def score_candidate(genome, *, nthread=1, sample=None):
        sampled = sample is not None
        calls.append(sampled)
        result = Scored(0.4 if sampled else 0.2, np.array([0.2]), {}, None)
        result.elapsed_s = 2 if sampled else 3
        return result

    monkeypatch.setattr(module, "_sample", lambda ws, rng: np.ones(20, dtype=bool))
    module.search(
        ws,
        module.Budget(population=1),
        family=family,
        method="grid",
        evaluate_score=score_candidate,
        cost_hint=lambda genome: 10,
        candidate_phase="deep",
        on_checkpoint=lambda state: checkpoints.append(deepcopy(state)),
    )
    state = checkpoints[-1]
    assert calls == [True, False]
    assert len(state["entries"]) == 1
    assert len(state["candidate_times"]) == 2
    assert sum(attempt["elapsed_s"] for attempt in state["candidate_times"]) == 5
    calls.clear()
    module.search(
        ws,
        module.Budget(population=1),
        family=family,
        method="grid",
        evaluate_score=score_candidate,
        cost_hint=lambda genome: 10,
        candidate_phase="deep",
        on_checkpoint=lambda state: checkpoints.append(deepcopy(state)),
        resume=state,
    )
    assert not calls
    assert len(checkpoints[-1]["candidate_times"]) == 2


def test_cancel_reaps_inflight_candidate_child(tmp_path):
    from test_forecast_coordinator import process_alive

    from smolsmort.forecast.candidate_worker import CandidateScorer, CandidateTimeout

    cancelled = threading.Event()
    fast = make_genome(["lag"], "squared", {"alpha": 2})
    slow = make_genome(["lag"], "squared", {"alpha": 1})
    with CandidateScorer(
        None,
        tmp_path,
        deadline=time.time() + 15,
        slice_s=10,
        fraction=1,
        stop=cancelled.is_set,
        score_fn=score_slow_candidate,
    ) as scorer:
        scorer.score(fast)
        pid = read_json(tmp_path / "candidate_pid.json")["pid"]
        timer = threading.Timer(0.1, cancelled.set)
        timer.start()
        started = time.monotonic()
        try:
            with pytest.raises(CandidateTimeout):
                scorer.score(slow)
        finally:
            timer.join(timeout=1)
        assert time.monotonic() - started < 1
        assert not process_alive(pid)


def test_absolute_slice_deadline_clips_long_candidate_limit(tmp_path):
    from test_forecast_coordinator import process_alive

    from smolsmort.forecast.candidate_worker import CandidateScorer, CandidateTimeout

    fast = make_genome(["lag"], "squared", {"alpha": 2})
    slow = make_genome(["lag"], "squared", {"alpha": 1})
    with CandidateScorer(
        None,
        tmp_path,
        deadline=time.time() + 15,
        slice_s=10,
        fraction=1,
        score_fn=score_slow_candidate,
    ) as scorer:
        scorer.score(fast)
        pid = read_json(tmp_path / "candidate_pid.json")["pid"]
        scorer.deadline = time.time() + 0.1
        started = time.monotonic()
        with pytest.raises(CandidateTimeout) as caught:
            scorer.score(slow)
        assert 0 < caught.value.elapsed_s < 0.7
        assert time.monotonic() - started < 1
        assert not process_alive(pid)


def test_real_all_family_run_respects_two_worker_cap(tmp_path, monkeypatch):
    from test_forecast_coordinator import assert_final_artifacts, make_trending_panel

    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    pytest.importorskip("statsforecast")
    from smolsmort.forecast import coordinator
    from smolsmort.forecast.families import available_families
    from smolsmort.forecast.worker import _Run

    spec, prepared = make_trending_panel(tmp_path)
    names = [family.name for family in available_families()]
    assert {"xgboost", "lightgbm", "ets", "theta", "arima", "snaive"} <= set(names)
    run = tmp_path / "run"
    run.mkdir()
    request = parse_request(
        make_request(
            spec=asdict(spec),
            prepared=str(prepared.folder),
            summary=prepared.summary,
            families=[{"name": name} for name in names],
            max_concurrent_families=2,
            nthread=4,
            ensemble={"enabled": True, "top": 3},
        )
    )
    (run / "request.json").write_text(json.dumps(request))
    processes, counts, samples = [], [], []
    original_spawn = coordinator.subprocess.Popen
    job = _Run(run)
    original_status = job.status
    started = time.monotonic()

    def record_processes():
        active = [process.pid for process in processes if process.poll() is None]
        counts.append(len(active))
        samples.append({"elapsed_s": time.monotonic() - started, "worker_pids": active})
        assert len(active) <= 2

    def spawn_worker(command, **kwargs):
        process = original_spawn(command, **kwargs)
        if command[2] == "smolsmort.forecast.family_worker":
            processes.append(process)
            record_processes()
        return process

    def record_status(state, **kwargs):
        record_processes()
        original_status(state, **kwargs)

    monkeypatch.setattr(coordinator.subprocess, "Popen", spawn_worker)
    monkeypatch.setattr(job, "status", record_status)
    try:
        coordinated = coordinator.launch_families(run, request, job)
    finally:
        (tmp_path / "family-process-samples.json").write_text(json.dumps(samples))
    assert coordinated["finalized"]
    job.status("done", phase="finish", families=coordinated["families"])
    assert time.monotonic() - started <= 94
    assert max(counts) == 2
    result = assert_final_artifacts(run, names)
    assert set(result["families"]) == set(names)
    for name in names:
        checkpoint = read_json(run / "families" / name / "checkpoint.json")
        assert checkpoint["entries"], name
        assert all(1 <= entry["nthread"] <= 4 for entry in checkpoint["entries"])


def test_real_worker_uses_released_threads_on_its_next_candidate(tmp_path, monkeypatch):
    from test_forecast_coordinator import make_trending_panel

    pytest.importorskip("xgboost")
    from smolsmort.forecast import coordinator
    from smolsmort.forecast.worker import _Run

    spec, prepared = make_trending_panel(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    runner = tmp_path / "delayed_snaive_worker.py"
    runner.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "import time\n"
        "from smolsmort.forecast import family_worker\n"
        "from smolsmort.forecast.families import get_family\n"
        "family = get_family('snaive')\n"
        "original_fit = family.fit\n"
        "def fit_snaive(*args, **kwargs):\n"
        "    time.sleep(.5)\n"
        "    return original_fit(*args, **kwargs)\n"
        "family.fit = fit_snaive\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(family_worker.run(Path(sys.argv[1]), sys.argv[2]))\n"
    )
    request = parse_request(
        make_request(
            spec=asdict(spec),
            prepared=str(prepared.folder),
            summary=prepared.summary,
            families=[{"name": "xgboost"}, {"name": "snaive"}],
            time_budget_s=20,
            max_concurrent_families=2,
            nthread=4,
            ensemble={"enabled": True, "top": 3},
        )
    )
    (run / "request.json").write_text(json.dumps(request))
    started_at = time.time() - 8
    job = _Run(run)
    job.save(
        "coordinator.json",
        {
            "started_at": started_at,
            "deadline": started_at + 20,
            "phase": "deep",
            "families": {
                name: {
                    "state": "paused",
                    "phase": "deep",
                    "slice_s": 0,
                    "best_error": None,
                    "pruned_reason": None,
                }
                for name in ("xgboost", "snaive")
            },
        },
    )
    original_spawn = coordinator.subprocess.Popen
    original_status = job.status
    snaive_finished = []

    def spawn_worker(command, **kwargs):
        if command[2] == "smolsmort.forecast.family_worker":
            command = [sys.executable, str(runner), command[-2], command[-1]]
        return original_spawn(command, **kwargs)

    def record_status(state, **kwargs):
        if kwargs.get("families", {}).get("snaive", {}).get("state") == "done":
            snaive_finished.append(time.time())
        original_status(state, **kwargs)

    monkeypatch.setattr(coordinator.subprocess, "Popen", spawn_worker)
    monkeypatch.setattr(job, "status", record_status)
    coordinated = coordinator.launch_families(run, request, job)
    assert coordinated["finalized"]
    assert snaive_finished
    candidates = read_json(run / "families" / "xgboost" / "checkpoint.json")["candidate_times"]
    thread_counts = [candidate["nthread"] for candidate in candidates]
    assert 2 in thread_counts and 4 in thread_counts, thread_counts
    assert thread_counts.index(2) < thread_counts.index(4)
    assert read_json(run / "families" / "xgboost" / "thread.json")["nthread"] == 4
    assert read_json(run / "families" / "snaive" / "status.json")["reason"] == "single candidate"
