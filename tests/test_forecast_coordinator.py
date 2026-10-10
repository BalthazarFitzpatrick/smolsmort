"""shared deadlines, durable coordination and validation-only family comparisons"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from forecast_fixtures import write_panel

from smolsmort.forecast.families import available_families
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.runs import cancel_run, start_run, wait_run
from smolsmort.forecast.spec import Column, PrepSpec


def test_allocator_prunes_paired_noise_and_bounds_overrunning_workers(tmp_path, monkeypatch):
    from smolsmort.forecast import coordinator

    clock = [100.0]
    processes = []
    events = []

    def save_json(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    class HangingProcess:
        def __init__(self, command, **kwargs):
            self.name = command[-1]
            self.started = clock[0]
            self.pid = 1000 + len(processes)
            self.returncode = None
            self.terminated = False
            processes.append(self)
            fitness, contrib = {
                "xgboost": (0.1, [0.1] * 8),
                "lightgbm": (0.11, [0.22, 0] * 4),
                "snaive": (0.9, [0.9] * 8),
            }[self.name]
            save_json(
                f"families/{self.name}/checkpoint.json",
                {
                    "entries": [
                        {
                            "genome": {
                                "families": ["lag"],
                                "objective": "squared",
                                "params": {},
                                "family": self.name,
                            },
                            "families": "lag",
                            "fitness": fitness,
                            "contrib": contrib,
                            "metrics": {},
                            "generation": 0,
                        }
                    ]
                },
            )

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -1

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(coordinator.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        coordinator.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(coordinator.subprocess, "Popen", HangingProcess)
    save_json("finish_complete.json", {})
    job = SimpleNamespace(
        cancelled=False, save=save_json, status=lambda *args, **kwargs: None, event=events.append
    )
    result = coordinator.launch_families(
        tmp_path,
        {
            "families": [{"name": name} for name in ("xgboost", "lightgbm", "snaive")],
            "time_budget_s": 60,
            "nthread": 3,
        },
        job,
    )
    states = result["families"]
    assert states["snaive"]["state"] == "pruned"
    assert "paired" in states["snaive"]["pruned_reason"]
    assert states["lightgbm"]["pruned_reason"] is None
    assert states["xgboost"]["pruned_reason"] is None
    assert all(process.terminated for process in processes)
    assert all(process.started < 160 for process in processes)
    assert clock[0] <= 160 + coordinator.GRACE_S
    assert any(event["event"] == "terminated" for event in events)
    assert sum(process.name == "snaive" for process in processes) == 1
    count = len(processes)
    clock[0] = 165
    coordinator.launch_families(
        tmp_path,
        {"families": [{"name": name} for name in states], "time_budget_s": 60},
        job,
    )
    assert len(processes) == count
    assert read_json(tmp_path / "coordinator.json")["deadline"] == 160


def test_ensemble_band_uses_stored_validation_average(monkeypatch):
    from smolsmort.forecast import evaluate, pipeline

    masks = {
        "train": np.array([True, False, False, False, False, False, False, False]),
        "val": np.array([False, True, True, True, True, False, False, False]),
        "test": np.array([False, False, False, False, False, True, True, False]),
        "future": np.array([False, False, False, False, False, False, False, True]),
    }
    truth = np.array([8, 10, 14, 18, 22, 26, 30, np.nan], dtype=float)
    frame = SimpleNamespace(y=truth, step=np.arange(8), series=np.zeros(8, dtype=int))
    ws = SimpleNamespace(
        mode="series",
        profile={"genes": {}},
        frames={(1, 1): frame},
        series_masks={(1, 1): masks},
        test_steps=2,
    )
    members = [
        pipeline.make_genome(["lag"], "squared", {}, name) for name in ("xgboost", "lightgbm")
    ]
    stored = np.mean([[9, 12, 17, 19], [11, 16, 17, 21]], axis=0)
    calls = []

    def predict_member(ws, member, frame, fit_rows, eval_rows, bucket, nthread):
        assert not np.array_equal(eval_rows, masks["val"])
        calls.append(member.family)
        return np.full(eval_rows.sum(), 25 if member.family == "xgboost" else 27)

    monkeypatch.setattr(pipeline, "_series_predict", predict_member)
    monkeypatch.setattr(
        pipeline, "_series_baseline", lambda frame, mask, bucket, period: np.full(mask.sum(), 20)
    )
    result = pipeline.finish(ws, members[0], members=members, validation=stored)
    expected = evaluate.fit_band(truth[masks["val"]], stored)
    assert result["bands"]["1-1"] == expected.__dict__
    np.testing.assert_array_equal(result["validation"]["prediction"], stored)
    np.testing.assert_array_equal(result["test"]["prediction"], [26, 26])
    assert calls == ["xgboost", "lightgbm", "xgboost", "lightgbm"]


def test_late_broad_resume_uses_remaining_deep_slice(tmp_path, monkeypatch):
    from smolsmort.forecast import coordinator, pipeline

    started = []

    def save_json(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    save_json(
        "coordinator.json",
        {
            "started_at": 100,
            "deadline": 160,
            "phase": "broad",
            "families": {"xgboost": {"state": "running"}},
        },
    )
    save_json("finish_complete.json", {})
    save_json(
        "families/xgboost/checkpoint.json",
        {
            "entries": [
                {
                    "genome": pipeline.make_genome(["lag"], "squared", {}).to_dict(),
                    "families": "lag",
                    "fitness": 0.1,
                    "generation": 0,
                }
            ]
        },
    )

    class CompletedProcess:
        pid = 123

        def __init__(self, command, **kwargs):
            started.append(read_json(tmp_path / "families" / command[-1] / "slice.json"))
            save_json(f"families/{command[-1]}/status.json", {"state": "done", "reason": "plateau"})

        def poll(self):
            return 0

    monkeypatch.setattr(coordinator.time, "time", lambda: 120)
    monkeypatch.setattr(coordinator.subprocess, "Popen", CompletedProcess)
    job = SimpleNamespace(cancelled=False, save=save_json, status=lambda *args, **kwargs: None)
    result = coordinator.launch_families(
        tmp_path, {"families": [{"name": "xgboost"}], "time_budget_s": 60}, job
    )
    assert len(started) == 1
    assert started[0]["phase"] == "deep"
    assert started[0]["deadline"] == 142
    assert result["families"]["xgboost"]["state"] == "done"
    assert result["families"]["xgboost"]["best_error"] == 0.1
    assert read_json(tmp_path / "coordinator.json")["deadline"] == 160


@pytest.mark.parametrize("top", [None, 1])
def test_recipe_selection_uses_validation_even_when_test_error_disagrees(
    tmp_path, monkeypatch, top
):
    from smolsmort.forecast import ensemble, pipeline

    spec = PrepSpec(
        "unused.csv",
        "series",
        "regression",
        (Column("day", "time"), Column("units", "target", aggregation="sum")),
    )
    request = {
        "spec": asdict(spec),
        "prepared": str(tmp_path),
        "summary": {},
        "ensemble": {"enabled": True},
    }
    if top is not None:
        request["ensemble"]["top"] = top
    ranked = [
        {
            "family": name,
            "fitness": fitness,
            "prediction": prediction,
            "genome": pipeline.make_genome(["lag"], "squared", {}, name).to_dict(),
        }
        for name, fitness, prediction in (("xgboost", 0.2, [9, 13]), ("lightgbm", 0.3, [11, 9]))
    ]
    for name, value in (
        ("request.json", request),
        ("finalists.json", ranked),
        ("coordinator.json", {"families": {}}),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    frame = SimpleNamespace(y=np.array([10, 10]), step=np.array([1, 2]))
    ws = SimpleNamespace(
        mode="series",
        frames={(1, 1): frame},
        series_masks={(1, 1): {"val": np.array([True, True])}},
    )
    monkeypatch.setattr(ensemble, "watch_parent", lambda pid: None)
    monkeypatch.setattr(ensemble, "load_workspace", lambda *args: ws)
    validations = []

    def finish_candidate(ws, genome, **kwargs):
        validations.append(np.asarray(kwargs["validation"]))
        return {"verdict": {"trusted": True}, "model_error": 0.9 if kwargs.get("members") else 0.01}

    def write_result(folder, spec, result, final):
        (folder / "recipe.json").write_text(json.dumps({"genome": result["best"]}))
        (folder / "result.json").write_text(json.dumps({"model_error": final["model_error"]}))

    def score_validation(ws, prediction, truth, periods):
        np.testing.assert_array_equal(prediction, [9, 13] if top == 1 else [10, 11])
        return SimpleNamespace(fitness=0.2 if top == 1 else 0.1)

    monkeypatch.setattr(ensemble, "finish", finish_candidate)
    monkeypatch.setattr(ensemble, "_write_result", write_result)
    monkeypatch.setattr(ensemble, "_fitness", score_validation)
    ensemble.run(tmp_path, 123)
    recipe = read_json(tmp_path / "recipe.json")
    assert recipe["family"] == ("xgboost" if top == 1 else "ensemble")
    if top != 1:
        assert recipe["members"] == ["xgboost", "lightgbm"]
    result = read_json(tmp_path / "result.json")
    assert result["model_error"] == (0.01 if top == 1 else 0.9)
    assert result["winner"]["model_error"] == 0.01
    assert result["ensemble"]["members"] == (["xgboost"] if top == 1 else ["xgboost", "lightgbm"])
    np.testing.assert_array_equal(validations[1], [9, 13] if top == 1 else [10, 11])


def read_json(path):
    return json.loads(path.read_text())


def wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    pytest.fail("timed out waiting for coordinator checkpoint")


def make_trending_panel(folder):
    pytest.importorskip("duckdb")
    truth = write_panel(folder)
    with truth.path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    days = sorted({row["day"] for row in rows})
    indices = {day: index // 2 for index, day in enumerate(days)}
    with truth.path.open("w", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=rows[0])
        writer.writeheader()
        for row in rows:
            if row["project"] != "A" or row["product"] not in ("p1", "p2"):
                continue
            week = indices[row["day"]]
            level = 40 if row["product"] == "p1" else 60
            row["units"] = str((level + 4 * week + 2 * np.sin(2 * np.pi * week / 52)) / 2)
            row["orders"] = str((level + 4 * (week + 6)) / 2)
            writer.writerow(row)
    spec = PrepSpec(
        str(truth.path),
        "series",
        "regression",
        (
            Column("day", "time"),
            Column("project", "dimension"),
            Column("product", "dimension"),
            Column("orders", "measure", aggregation="sum"),
            Column("units", "target", aggregation="sum"),
        ),
        horizon=4,
    )
    return spec, prepare(spec, folder / "cache")


def assert_final_artifacts(run, names):
    status = read_json(run / "status.json")
    assert status["state"] == "done", status
    assert status["phase"] == "finish"
    assert set(status["families"]) == set(names)
    for state in status["families"].values():
        assert {"state", "phase", "slice_s", "best_error", "pruned_reason"} <= state.keys()
    result = read_json(run / "result.json")
    assert set(result["families"]) == set(names)
    for name, family in result["families"].items():
        assert "verdict" not in family, name
        assert "test" not in family, name
        assert {"fitness", "candidates", "state", "pruned_reason"} <= family.keys()
    assert result["winner"]["family"] in names
    assert result["winner"]["verdict"]
    ensemble = result["ensemble"]
    assert ensemble["verdict"]
    assert len(ensemble["members"]) == min(3, len(names))
    assert len(set(ensemble["members"])) == len(ensemble["members"])
    assert set(ensemble["members"]) <= set(names)
    recipe = read_json(run / "recipe.json")
    assert recipe["family"] in (*names, "ensemble")
    if recipe["family"] == "ensemble":
        assert recipe["members"] == ensemble["members"]
    assert (run / "test.parquet").exists()
    return result


def test_four_family_budget_prunes_and_finishes_ensemble(tmp_path):
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    pytest.importorskip("statsforecast")
    spec, prepared = make_trending_panel(tmp_path)
    names = ["xgboost", "lightgbm", "ets", "snaive"]
    root = tmp_path / "runs"
    started = time.monotonic()
    run_id = start_run(
        root,
        spec,
        prepared,
        families=[{"name": name} for name in names],
        time_budget_s=60,
        nthread=4,
        ensemble={"enabled": True, "top": 3},
    )
    status = wait_run(root, run_id, timeout=70)
    assert status["state"] == "done", status
    assert time.monotonic() - started <= 64
    result = assert_final_artifacts(root / run_id, names)
    snaive = result["families"]["snaive"]
    assert snaive["state"] == "pruned", result["families"]
    assert snaive["pruned_reason"]
    assert snaive["fitness"] > result["winner"]["fitness"]


def test_every_locally_available_family_shares_one_run(tmp_path):
    spec, prepared = make_trending_panel(tmp_path)
    names = [family.name for family in available_families()]
    assert "snaive" in names
    root = tmp_path / "runs"
    run_id = start_run(
        root,
        spec,
        prepared,
        families=[{"name": name} for name in names],
        time_budget_s=90,
        nthread=max(2, len(names)),
        ensemble={"enabled": True, "top": 3},
    )
    status = wait_run(root, run_id, timeout=100)
    assert status["state"] == "done", status
    result = assert_final_artifacts(root / run_id, names)
    for name, state in result["families"].items():
        assert state["fitness"] is not None, (name, state)
        assert state["state"] != "failed", (name, state)


def process_alive(pid):
    if sys.platform == "win32":
        from smolsmort.forecast.runs import _windows_alive

        return _windows_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_cancel_stops_coordinator_and_every_family_child(tmp_path):
    pytest.importorskip("xgboost")
    spec, prepared = make_trending_panel(tmp_path)
    root = tmp_path / "runs"
    run_id = start_run(
        root,
        spec,
        prepared,
        families=[{"name": "xgboost"}, {"name": "snaive"}],
        time_budget_s=120,
        nthread=2,
    )
    run = root / run_id
    try:

        def read_worker_states():
            states = [
                read_json(path)
                for path in run.glob("families/*/status.json")
                if read_json(path).get("pid")
            ]
            return states if len(states) == 2 else None

        states = wait_for(read_worker_states)
        pids = {state["pid"] for state in states}
        coordinator_pid = int((run / "pid").read_text())
        cancel_run(root, run_id)
        status = wait_run(root, run_id, timeout=10)
        assert status["state"] == "cancelled", status
        wait_for(lambda: not any(process_alive(pid) for pid in pids), timeout=5)
        from smolsmort.forecast.runs import _PROCESSES

        _PROCESSES[coordinator_pid].wait(timeout=5)
        assert not process_alive(coordinator_pid)
        assert not any(process_alive(pid) for pid in pids)
    finally:
        cancel_run(root, run_id)


def test_killed_coordinator_resumes_checkpoint_with_original_deadline(tmp_path):
    pytest.importorskip("xgboost")
    spec, prepared = make_trending_panel(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    request = {
        "version": 2,
        "spec": asdict(spec),
        "prepared": str(prepared.folder),
        "summary": prepared.summary,
        "families": [{"name": "xgboost"}, {"name": "snaive"}],
        "time_budget_s": 30,
        "nthread": 2,
        "ensemble": {"enabled": True, "top": 3},
        "transform": "none",
    }
    (run / "request.json").write_text(json.dumps(request))
    command = [sys.executable, "-m", "smolsmort.forecast.worker", str(run)]
    checkpoint_path = run / "families" / "xgboost" / "checkpoint.json"
    with (run / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for(lambda: checkpoint_path.exists(), timeout=25)
            before = read_json(checkpoint_path)
            state_before = read_json(run / "coordinator.json")
            assert before["entries"]
            process.kill()
            process.wait(timeout=5)
            time.sleep(0.3)
            restarted = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=40)
            assert restarted.returncode == 0, (run / "worker.log").read_text()
        finally:
            (run / "cancel").touch()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    state_after = read_json(run / "coordinator.json")
    assert state_after["deadline"] == state_before["deadline"]
    assert time.time() <= state_before["deadline"] + 4
    after = read_json(checkpoint_path)
    prior = {entry["key"]: entry for entry in before["entries"]}
    current = {entry["key"]: entry for entry in after["entries"]}
    assert prior.keys() <= current.keys()
    assert all(current[key]["fitness"] == entry["fitness"] for key, entry in prior.items())
    assert_final_artifacts(run, ["xgboost", "snaive"])
