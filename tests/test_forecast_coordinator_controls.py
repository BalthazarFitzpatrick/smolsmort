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


@pytest.fixture(autouse=True)
def stub_machine_probe(monkeypatch):
    from smolsmort.forecast import coordinator

    monkeypatch.setattr(
        coordinator,
        "describe_machine",
        lambda threads, cap: {"nthread": threads, "max_concurrent_families": cap},
    )


@pytest.fixture
def tunable_race(tmp_path, monkeypatch):
    from smolsmort.forecast import coordinator
    from smolsmort.forecast.families import get_family

    clock = [100.0]
    launches, snapshots, processes = [], [], []
    errors = {"theta": 0.2, "snaive": 0.3, "ets": 0.4, "arima": 0.7}
    errors.update(ridge=0.6, lightgbm=0.8, xgboost=0.9)
    plateaus = {}
    partial = set()
    contributions = {}
    candidate_seconds = dict.fromkeys(errors, 0.1)

    def save_json(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    class FamilyProcess:
        def __init__(self, command, **kwargs):
            self.name = command[-1]
            self.target = tmp_path / "families" / self.name
            self.allocation = read_json(self.target / "slice.json")
            self.previous = coordinator.read_json(self.target / "checkpoint.json", {})
            self.pid = 1000 + len(processes)
            self.returncode = None
            self.started = clock[0]
            processes.append(self)
            launches.append(self)

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            broad = self.allocation["phase"] == "broad"
            generation = self.previous.get("generation", 0) + (not broad)
            plateau = generation >= plateaus.get(self.name, float("inf"))
            end = self.started + 0.1 if broad or plateau else self.allocation["deadline"]
            if clock[0] < end:
                return None
            self.returncode = 0
            if self.name in partial and not broad:
                partial.remove(self.name)
                generation = 0
            entries = list(self.previous.get("entries", []))
            for _ in range(1 if broad else 2):
                genome = make_genome(["lag"], "squared", {"candidate": len(entries)}, self.name)
                entries.append(
                    {
                        "key": genome.key(),
                        "genome": genome.to_dict(),
                        "families": 1,
                        "fitness": errors[self.name],
                        "contrib": contributions.get(self.name, [errors[self.name]] * 8),
                        "metrics": {},
                        "generation": generation,
                        "phase": self.allocation["phase"],
                        "elapsed_s": candidate_seconds[self.name],
                        "cost_hint": 1,
                    }
                )
            save_json(
                f"families/{self.name}/checkpoint.json",
                {
                    "entries": entries,
                    "generation": generation,
                    "population": [entries[-1]["genome"]],
                },
            )
            reason = (
                "single candidate"
                if not get_family(self.name).space()
                else "broad defaults"
                if broad
                else "plateau"
                if plateau
                else "time cap"
            )
            save_json(
                f"families/{self.name}/status.json",
                {
                    "state": "done",
                    "reason": reason,
                    "generation": generation,
                    "candidates_done": len(entries),
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
        coordinator.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(coordinator.subprocess, "Popen", FamilyProcess)
    job = SimpleNamespace(
        cancelled=False,
        save=save_json,
        status=lambda *args, **kwargs: snapshots.append(deepcopy(kwargs)),
        event=lambda payload: None,
    )

    def run(names=None, cap=2, **changes):
        names = names or list(errors)
        save_json("finish_complete.json", {})
        save_json("result.json", {"families": {name: {"fitness": errors[name]} for name in names}})
        return coordinator.launch_families(
            tmp_path,
            make_request(
                families=[{"name": name} for name in names],
                max_concurrent_families=cap,
                **{"time_budget_s": 60, **changes},
            ),
            job,
        )

    return SimpleNamespace(
        run=run,
        clock=clock,
        launches=launches,
        snapshots=snapshots,
        errors=errors,
        plateaus=plateaus,
        partial=partial,
        contributions=contributions,
        candidate_seconds=candidate_seconds,
        save=save_json,
        job=job,
    )


def advance_race_clock(race, monkeypatch):
    from smolsmort.forecast import coordinator

    def sleep(seconds):
        ends = [
            process.started + 0.1
            if process.allocation["phase"] == "broad"
            else process.allocation["deadline"]
            for process in race.launches
            if process.returncode is None
        ]
        race.clock[0] = max(race.clock[0] + seconds, min(ends, default=race.clock[0]))

    monkeypatch.setattr(coordinator.time, "sleep", sleep)


def test_measured_day_budget_searches_for_at_least_23_hours(tunable_race, tmp_path, monkeypatch):
    race = tunable_race
    race.errors["ridge"] = race.errors["lightgbm"] = 0.2
    advance_race_clock(race, monkeypatch)
    race.run(["ridge", "lightgbm"], time_budget_s=86400)
    state = read_json(tmp_path / "coordinator.json")
    assert state["deadline"] == 100 + 86400
    assert 23 * 3600 <= race.clock[0] - 100 <= 86400 - 300 + 1
    assert state["finish_reserve_s"] == 300
    assert state["search_deadline"] == 100 + 86400 - 300
    assert {p.allocation["phase"] for p in race.launches} == {"broad", "halving", "deep"}
    halving = next(p for p in race.launches if p.allocation["phase"] == "halving")
    assert halving.allocation["slice_s"] == pytest.approx((86400 - 300) / 7)
    assert max(p.allocation["deadline"] for p in race.launches) == state["search_deadline"]


def test_broad_resume_uses_measured_search_span(tunable_race, tmp_path, monkeypatch):
    race = tunable_race
    genome = make_genome(["lag"], "squared", {}, "ridge")
    race.save(
        "families/ridge/checkpoint.json",
        {
            "entries": [
                {
                    "genome": genome.to_dict(),
                    "key": genome.key(),
                    "fitness": race.errors["ridge"],
                    "families": 1,
                    "contrib": [race.errors["ridge"]] * 8,
                    "metrics": {},
                    "generation": 0,
                    "phase": "broad",
                    "elapsed_s": 0.1,
                    "cost_hint": 1,
                }
            ],
            "generation": 0,
        },
    )
    race.save(
        "coordinator.json",
        {
            "started_at": 100,
            "deadline": 100 + 86400,
            "phase": "broad",
            "families": {"ridge": {"state": "running"}},
        },
    )
    race.clock[0] = 200
    advance_race_clock(race, monkeypatch)
    race.run(["ridge"], time_budget_s=86400)
    assert race.launches[0].allocation["deadline"] == 100 + (86400 - 300) * 4 / 7
    assert read_json(tmp_path / "coordinator.json")["deadline"] == 100 + 86400


@pytest.mark.parametrize("duration", [90, 300, 1000, 1200, 1800])
def test_short_budget_keeps_original_phase_timing(tunable_race, monkeypatch, duration):
    race = tunable_race
    race.errors["ridge"] = race.errors["lightgbm"] = 0.2
    advance_race_clock(race, monkeypatch)
    race.run(["ridge", "lightgbm"], time_budget_s=duration)
    broad = next(p for p in race.launches if p.allocation["phase"] == "broad")
    halving = next(p for p in race.launches if p.allocation["phase"] == "halving")
    assert broad.allocation["deadline"] == pytest.approx(100 + 0.4 * duration)
    assert halving.allocation["slice_s"] == pytest.approx(0.1 * duration)
    assert race.clock[0] == pytest.approx(100 + 0.7 * duration, abs=1)


def test_unmeasured_day_budget_retains_original_reserve(tunable_race, tmp_path, monkeypatch):
    race = tunable_race
    race.candidate_seconds["ridge"] = 0
    advance_race_clock(race, monkeypatch)
    race.run(["ridge"], time_budget_s=86400)
    state = read_json(tmp_path / "coordinator.json")
    assert race.clock[0] == pytest.approx(100 + 0.7 * 86400, abs=1)
    assert state["finish_reserve_s"] == pytest.approx(0.3 * 86400)


@pytest.mark.parametrize("sigma,pruned", [(1, True), (2, False)])
def test_pruning_respects_requested_sigma(tunable_race, sigma, pruned):
    race = tunable_race
    diff = (np.array([-1, 1] * 4) + 1.5 / np.sqrt(7)) * 0.02
    assert diff.sum() / (diff.std(ddof=1) * np.sqrt(len(diff))) == pytest.approx(1.5)
    race.contributions["ridge"] = [0.1] * 8
    race.contributions["lightgbm"] = (0.1 + diff).tolist()
    race.run(["ridge", "lightgbm"], prune_sigma=sigma)
    assert any(s["families"]["lightgbm"]["state"] == "pruned" for s in race.snapshots) == pruned


@pytest.mark.parametrize("candidate_s,reserve_s", [(100, 800), (10000, 25920)])
def test_finish_reserve_scales_winners_and_never_exceeds_thirty_percent(
    tunable_race, tmp_path, monkeypatch, candidate_s, reserve_s
):
    race = tunable_race
    race.candidate_seconds.update(ridge=candidate_s, lightgbm=candidate_s)
    advance_race_clock(race, monkeypatch)
    race.run(["ridge", "lightgbm"], time_budget_s=86400)
    state = read_json(tmp_path / "coordinator.json")
    assert state["finish_reserve_s"] == reserve_s
    assert race.clock[0] == pytest.approx(100 + 86400 - reserve_s, abs=1)


@pytest.mark.parametrize("mode,fits", [("timeseries", 2), ("entity", 3)])
@pytest.mark.parametrize("measured", [None, 10, 100])
def test_finish_estimate_uses_winner_cost_and_completed_measurements(
    tmp_path, mode, fits, measured
):
    from smolsmort.forecast.coordinator import estimate_finish_seconds

    target = tmp_path / "families" / "ridge"
    target.mkdir(parents=True)
    winner = {"fitness": 0.1, "families": 1, "cost_hint": 20, "elapsed_s": measured}
    (target / "checkpoint.json").write_text(json.dumps({"entries": [winner]}))
    calibration = {"ridge": {"seconds_per_cost": 2, "seconds_per_candidate": 1000}}
    estimate = estimate_finish_seconds(tmp_path, ["ridge"], calibration, {"spec": {"mode": mode}})
    assert estimate == fits * max(measured or 0, 40)


def test_finish_estimate_rejects_timeout_only_calibration(tmp_path):
    from smolsmort.forecast.coordinator import estimate_finish_seconds

    target = tmp_path / "families" / "ridge"
    target.mkdir(parents=True)
    winner = {"fitness": 0.1, "families": 1}
    (target / "checkpoint.json").write_text(json.dumps({"entries": [winner]}))
    calibration = {"ridge": {"seconds_per_candidate": 10, "estimate_is_lower_bound": True}}
    assert estimate_finish_seconds(tmp_path, ["ridge"], calibration, {}) is None
    calibration["ridge"]["estimate_is_lower_bound"] = False
    assert estimate_finish_seconds(tmp_path, ["ridge"], calibration, {}) == 20


def test_zero_param_leader_allows_every_tunable_to_breed(tunable_race, tmp_path):
    race = tunable_race
    result = race.run(cap=2)
    for name in ("ridge", "lightgbm", "xgboost"):
        state = result["families"][name]
        assert state["candidates_done"] > 1
        assert state["generation"] >= 1
        for snapshot in race.snapshots:
            status = snapshot["families"][name]
            if status["state"] == "pruned":
                assert status["generation"] >= 1
                assert status["candidates_done"] > 1
        saved = read_json(tmp_path / "result.json")["families"][name]
        assert saved["generation"] >= 1
        assert saved["candidates"] > 1
        assert saved["fitness"] == race.errors[name]
    for name in ("theta", "snaive", "ets", "arima"):
        assert result["families"][name]["state"] == "done"
        assert result["families"][name]["candidates_done"] == 1
        assert [p.allocation["phase"] for p in race.launches if p.name == name] == ["broad"]
    assert result["best"]["family"] == "theta"
    assert 142 <= race.clock[0] < 142.2
    assert all(p.allocation["deadline"] <= 142 for p in race.launches)
    assert (
        max(
            sum(
                p.started <= at < p.allocation["deadline"]
                for p in race.launches
                if p.allocation["phase"] != "broad"
            )
            for at in [p.started for p in race.launches]
        )
        <= 2
    )


def test_leftover_time_resumes_best_pruned_checkpoint_after_survivor_plateaus(tunable_race):
    race = tunable_race
    race.errors["ridge"] = race.errors["theta"]
    race.plateaus.update(ridge=2, lightgbm=2, xgboost=2)
    result = race.run(["theta", "ridge", "lightgbm", "xgboost"], cap=1)
    pruned = [s for s in race.snapshots if s["families"]["lightgbm"]["state"] == "pruned"]
    assert pruned
    resumed = [p for p in race.launches if p.previous.get("generation", 0) >= 1]
    assert [p.name for p in resumed] == ["ridge", "lightgbm", "xgboost"]
    for process in resumed[1:]:
        assert process.allocation["phase"] == "deep"
        assert len(process.previous["entries"]) == 3
        assert process.previous["population"]
        state = result["families"][process.name]
        assert state["generation"] == 2
        assert state["candidates_done"] == 5
        assert state["state"] == "done"
        assert state["pruned_reason"] is None
        assert any(
            snapshot["families"][process.name].get("reason") == "resumed with leftover time"
            for snapshot in race.snapshots
        )
    assert race.clock[0] < 142


def test_partial_generation_is_not_pruned(tunable_race):
    race = tunable_race
    race.partial.add("ridge")
    race.run(["theta", "ridge", "lightgbm"])
    partial = [
        s["families"]["ridge"]
        for s in race.snapshots
        if s["families"]["ridge"].get("candidates_done", 0) > 1
        and s["families"]["ridge"].get("generation") == 0
    ]
    assert partial
    assert all(state["state"] != "pruned" for state in partial)
    assert any(
        p.name == "ridge"
        and p.previous.get("generation") == 0
        and len(p.previous.get("entries", [])) > 1
        for p in race.launches
    )


def test_only_zero_param_families_finish_after_broad(tunable_race):
    race = tunable_race
    result = race.run(["theta", "snaive", "ets", "arima"])
    assert len(race.launches) == 4
    assert all(p.allocation["phase"] == "broad" for p in race.launches)
    assert all(
        s["candidates_done"] == 1 and s["state"] == "done" for s in result["families"].values()
    )
    assert race.clock[0] < 101


@pytest.mark.parametrize("phase", ["halving", "deep"])
@pytest.mark.parametrize("terminal", ["plateau", "grid complete", "failed", "cancelled"])
def test_recovery_resumes_pruned_checkpoint_and_keeps_terminal_workers_stopped(
    tunable_race, tmp_path, phase, terminal
):
    race = tunable_race
    names = ["theta", "ridge", "lightgbm"]
    race.plateaus.update(ridge=1, lightgbm=1)
    race.run(names)
    saved = read_json(tmp_path / "coordinator.json")
    checkpoint = read_json(tmp_path / "families/ridge/checkpoint.json")
    saved["phase"] = phase
    for name in names:
        saved["families"][name].update(state="pruned", pruned_reason="behind theta")
    terminal_state = terminal if terminal in ("failed", "cancelled") else "done"
    saved["families"]["lightgbm"]["reason"] = terminal
    race.save("families/lightgbm/status.json", {"state": terminal_state, "reason": terminal})
    race.save("families/ridge/status.json", {"state": "done", "reason": "time cap"})
    race.save("coordinator.json", saved)
    race.clock[0] = 120
    race.plateaus["ridge"] = 2
    race.launches.clear()
    result = race.run(names)
    assert [p.name for p in race.launches] == ["ridge"]
    assert race.launches[0].previous == checkpoint
    assert result["families"]["ridge"]["generation"] == 2
    assert result["families"]["ridge"]["candidates_done"] == 5
    assert result["families"]["lightgbm"]["state"] == terminal_state
    assert result["families"]["theta"]["state"] == "done"
    assert read_json(tmp_path / "coordinator.json")["deadline"] == 160
    assert race.clock[0] < 142


@pytest.mark.parametrize("worker_reason", ["time cap", "plateau"])
def test_recovery_reconciles_worker_exit_before_coordinator_save(
    tunable_race, tmp_path, worker_reason
):
    race = tunable_race
    names = ["theta", "ridge"]
    race.plateaus["ridge"] = 1
    race.run(names)
    saved = read_json(tmp_path / "coordinator.json")
    saved["phase"] = "deep"
    saved["families"]["ridge"].update(state="running", pid=None)
    race.save("coordinator.json", saved)
    race.save("families/ridge/status.json", {"state": "done", "reason": worker_reason})
    race.clock[0] = 120
    race.plateaus["ridge"] = 2
    race.launches.clear()
    result = race.run(names)
    assert len(race.launches) == (worker_reason == "time cap")
    assert result["families"]["ridge"]["state"] == "done"
    assert result["families"]["ridge"]["generation"] == (2 if worker_reason == "time cap" else 1)


@pytest.mark.parametrize("recorded_end", [100.25, None])
def test_recovery_never_counts_downtime_as_family_wall_time(tunable_race, tmp_path, recorded_end):
    race = tunable_race
    race.run(["theta"], cap=1)
    saved = read_json(tmp_path / "coordinator.json")
    family = saved["families"]["theta"]
    family.update(state="running", pid=None)
    family["slice_history"] = [{"phase": "broad", "started_at": 100}]
    race.save("coordinator.json", saved)
    race.save(
        "families/theta/status.json",
        {
            "state": "done",
            "reason": "single candidate",
            "started_at": 100.01,
            "ended_at": recorded_end,
        },
    )
    race.clock[0] = 120
    result = race.run(["theta"], cap=1)
    expected_wall = 0.25 if recorded_end is not None else None
    assert result["families"]["theta"]["wall_s"] == expected_wall
    assert result["families"]["theta"]["slice_history"][-1]["ended_at"] == recorded_end
    assert read_json(tmp_path / "result.json")["families"]["theta"]["wall_s"] == expected_wall


def test_family_candidate_time_includes_sample_and_timeout_attempts(tunable_race, tmp_path):
    race = tunable_race
    race.run(["theta"], cap=1)
    checkpoint = read_json(tmp_path / "families/theta/checkpoint.json")
    checkpoint["candidate_times"] = [
        {"elapsed_s": 0.01, "note": "sample"},
        {"elapsed_s": 0.02, "note": "too slow"},
        {"elapsed_s": 0.03},
    ]
    race.save("families/theta/checkpoint.json", checkpoint)
    race.run(["theta"], cap=1)
    summary = read_json(tmp_path / "result.json")["families"]["theta"]
    assert summary["candidate_s"] == pytest.approx(0.06)


def test_cancellation_stops_leftover_workers(tunable_race):
    race = tunable_race
    record_status = race.job.status

    def cancel_leftover(*args, **kwargs):
        record_status(*args, **kwargs)
        if any(
            s.get("reason") == "resumed with leftover time" for s in kwargs["families"].values()
        ):
            race.job.cancelled = True

    race.job.status = cancel_leftover
    result = race.run(["theta", "ridge", "lightgbm"])
    assert result["stopped"] == "cancelled"
    assert all(s["state"] == "cancelled" for n, s in result["families"].items() if n != "theta")
    assert all(p.poll() is not None for p in race.launches)
    assert race.clock[0] < 142


def test_v2_control_defaults():
    request = parse_request(make_request())
    assert request["max_concurrent_families"] == 1
    assert request["candidate_limit_fraction"] == 0.25


def test_plateau_controls_default_and_validate():
    request = parse_request(make_request())
    assert (request["plateau"], request["min_generations"]) == (5, 0)
    request = parse_request(make_request(plateau=10, min_generations=3))
    assert (request["plateau"], request["min_generations"]) == (10, 3)
    for changes, field in (
        ({"plateau": 0}, "plateau"),
        ({"plateau": True}, "plateau"),
        ({"min_generations": -1}, "min_generations"),
        ({"min_generations": 1.5}, "min_generations"),
    ):
        with pytest.raises(ValueError, match=field):
            parse_request(make_request(**changes))


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, 1.5, math.nan, math.inf, "2", None])
def test_invalid_family_concurrency_is_rejected(value):
    with pytest.raises(ValueError, match="max_concurrent_families"):
        parse_request(make_request(max_concurrent_families=value))


@pytest.mark.parametrize("value", [0, -0.1, 1.01, math.nan, math.inf, -math.inf, True, "0.5", None])
def test_invalid_candidate_limit_fraction_is_rejected(value):
    with pytest.raises(ValueError, match="candidate_limit_fraction"):
        parse_request(make_request(candidate_limit_fraction=value))


@pytest.mark.parametrize("concurrency,fraction", [(1, 1), (2, 0.01), (4, 0.25), (8, 0.5)])
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
    entries = [
        {
            "fitness": 0.5,
            "families": 1,
            "phase": "broad",
            "elapsed_s": 10,
            "cost_hint": 20,
            "nthread": 8,
        }
    ]
    checkpoint = {"entries": entries, "cost_hints": {"a": 100, "b": 200}}
    (target / "checkpoint.json").write_text(json.dumps(checkpoint))
    states = {"xgboost": {"state": "paused"}}
    calibration = {}
    calibrate_families(tmp_path, states, calibration, remaining_s=90)
    assert calibration["xgboost"]["seconds_per_cost"] == 0.5
    assert calibration["xgboost"]["nthread"] == 8
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
    assert calibration["xgboost"]["nthread"] == 8
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


@pytest.mark.parametrize("cap", [1, 2])
@pytest.mark.parametrize("requested_threads", [None, 4])
def test_family_queue_fills_freed_slots_and_records_wave_slices(
    tmp_path, monkeypatch, cap, requested_threads
):
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
            if command[2] == "smolsmort.forecast.ensemble":
                self.name = "finish"
                self.pid = 2000
                self.returncode = 0
                state = read_json(tmp_path / "coordinator.json")
                assert state["nthread"] == 4
                assert kwargs["env"]["OMP_NUM_THREADS"] == "4"
                assert state["phase_history"][-1]["phase"] == "finish_start"
                clock[0] += 0.25
                save_json("finish_complete.json", {})
                save_json("result.json", {"families": {name: {"fitness": 0.2} for name in names}})
                return
            self.name = command[-1]
            target = tmp_path / "families" / self.name
            self.allocation = read_json(target / "slice.json")
            self.started = clock[0]
            self.pid = 1000 + len(processes)
            self.returncode = None
            self.observed_threads = set()
            active = sum(process.poll() is None for process in processes)
            assert active < cap
            if cap == 1:
                assert self.allocation["nthread"] == 4
                assert kwargs["env"]["OMP_NUM_THREADS"] == "4"
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
    monkeypatch.setattr(coordinator.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(coordinator.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        coordinator.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration)
    )
    monkeypatch.setattr(coordinator.subprocess, "Popen", QueuedProcess)
    job = SimpleNamespace(
        cancelled=False,
        save=save_json,
        status=lambda *args, **kwargs: statuses.append(deepcopy(kwargs)),
        event=lambda payload: None,
    )
    result = coordinator.launch_families(
        tmp_path,
        make_request(
            families=[{"name": name} for name in names],
            nthread=requested_threads,
            **({"max_concurrent_families": cap} if cap != 1 else {}),
        ),
        job,
    )
    broad = [(name, started) for name, phase, started in launched if phase == "broad"]
    assert {name for name, started in broad} == set(names)
    start_times = dict(broad)
    assert start_times["lightgbm"] >= start_times["snaive"] + 1
    if cap == 2:
        assert start_times["lightgbm"] < start_times["xgboost"] + 5
    else:
        assert start_times["lightgbm"] >= start_times["xgboost"] + 5
    assert (
        max(
            sum(
                other.started <= process.started < other.started + durations[other.name]
                for other in processes
            )
            for process in processes
        )
        <= cap
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
            assert allocation["wall_s"] == pytest.approx(
                allocation["ended_at"] - allocation["started_at"]
            )
            assert allocation["wall_s"] >= durations[name]
        assert (tmp_path / "families" / name / "thread.json").exists()
        checkpoint = read_json(tmp_path / "families" / name / "checkpoint.json")
        assert all(1 <= entry["nthread"] <= 4 for entry in checkpoint["entries"])
    assert any(len(process.observed_threads) > 1 for process in processes) == (cap == 2)
    surviving = [
        process
        for process in processes
        if process.name == "xgboost" and process.allocation["phase"] == "broad"
    ]
    assert surviving[0].observed_threads == ({2, 4} if cap == 2 else {4})
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
        assert 1 <= measured["nthread"] <= 4
    state = read_json(tmp_path / "coordinator.json")
    final = read_json(tmp_path / "result.json")
    assert state["nthread"] == 4
    assert (
        final["machine"]
        == state["machine"]
        == {
            "nthread": 4,
            "max_concurrent_families": cap,
        }
    )
    assert final["phase_history"] == state["phase_history"]
    assert [item["phase"] for item in state["phase_history"]] == [
        "broad",
        "halving",
        "deep",
        "finish",
        "finish_start",
        "finish_end",
    ]
    assert state["phase_history"][-1]["started_at"] == clock[0]
    assert final["elapsed"] == result["elapsed"] == clock[0] - 100
    for name, summary in final["families"].items():
        assert summary["fitness"] == 0.2
        assert summary["wall_s"] == sum(
            item["wall_s"] for item in state["families"][name]["slice_history"]
        )
        assert summary["candidate_s"] == durations[name]
        if cap == 1:
            assert summary["nthread"] == 4
    if cap == 1:
        assert sum(summary["wall_s"] for summary in final["families"].values()) <= final["elapsed"]


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
        mode="entity",
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
        mode="entity",
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
            # queued optional families need room to complete their first candidate
            time_budget_s=180,
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
    assert time.monotonic() - started <= 184
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
    started_at = time.time()
    job = _Run(run)
    job.save(
        "coordinator.json",
        {
            "started_at": started_at,
            "deadline": started_at + 20,
            "phase": "broad",
            "families": {
                name: {
                    "state": "pending",
                    "phase": "broad",
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
