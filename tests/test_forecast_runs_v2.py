"""v2 requests, process isolation and durable candidate checkpoints"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from forecast_fixtures import write_bike_daily, write_panel

from smolsmort.forecast import family_worker
from smolsmort.forecast.families import Param, get_family
from smolsmort.forecast.pipeline import Scored, load_workspace
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.runs import start_run, wait_run
from smolsmort.forecast.search import Budget, search
from smolsmort.forecast.spec import Column, PrepSpec


def make_request(folder, families, duration=15):
    pytest.importorskip("duckdb")
    truth = write_panel(folder)
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
    )
    prepared = prepare(spec, folder / "cache")
    request = {
        "version": 2,
        "spec": asdict(spec),
        "prepared": str(prepared.folder),
        "summary": prepared.summary,
        "nthread": 2,
        "transform": "none",
        "families": families,
        "time_budget_s": duration,
        "ensemble": {"enabled": False, "top": 3},
        "warm_from": None,
        "recipe": None,
    }
    return request, spec, prepared


def test_v1_parsing_is_unchanged():
    request = {"budget": {"population": 3}, "recipe": None}
    assert parse_request(request) is request
    assert Budget(**request["budget"]).population == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 3},
        {"transform": "logdiff"},
        {"time_budget_s": 0},
        {"time_budget_s": float("nan")},
        {"families": []},
        {"families": [{"name": "../escape"}]},
        {"families": [{"name": "snaive"}, {"name": "snaive"}]},
        {"families": [{"name": "snaive", "method": "random"}]},
        {"nthread": 0},
    ],
)
def test_invalid_v2_request_is_rejected(changes):
    with pytest.raises(ValueError):
        parse_request(
            dict(version=2, transform="none", time_budget_s=2, families=[{"name": "snaive"}], **{})
            | changes
        )


@pytest.mark.parametrize("name", ["snaive", "ets", "theta", "arima"])
def test_empty_spaces_score_exactly_once(name, monkeypatch):
    module = importlib.import_module("smolsmort.forecast.search")
    ws = SimpleNamespace(
        mode="row",
        spec=SimpleNamespace(task="regression"),
        masks={"train": np.ones(20, dtype=bool)},
        families=["lag"],
        profile={"genes": {"objectives": ["squared"]}},
    )
    calls, events = [], []

    def score_candidate(_ws, genome, **_kwargs):
        calls.append(genome)
        return Scored(0.2, np.array([0.2]), {}, None)

    monkeypatch.setattr(module, "score", score_candidate)
    monkeypatch.setattr(
        module, "_sample", lambda *_args: pytest.fail("empty spaces must score once on full data")
    )
    result = search(ws, Budget(), family=get_family(name), on_event=events.append)
    assert len(calls) == 1 and calls[0].family == name and calls[0].params == ()
    assert result["generations"] == 1 and result["stopped"] == "single candidate"
    assert len([event for event in events if event["event"] == "progress"]) == 1


def test_v2_parallel_workers_finish_and_merge(tmp_path):
    pytest.importorskip("xgboost")
    selections = [
        {"name": "xgboost", "method": "grid", "space": {"max_depth": [3], "eta": [0.1]}},
        {"name": "snaive"},
    ]
    request, spec, prepared = make_request(tmp_path, selections, 30)
    run_id = start_run(
        tmp_path / "runs", spec, prepared, families=selections, time_budget_s=30, nthread=2
    )
    state = wait_run(tmp_path / "runs", run_id, timeout=100)
    run = tmp_path / "runs" / run_id
    assert state["state"] == "done", state
    assert set(state["families"]) == {"xgboost", "snaive"}
    statuses = []
    for name in ("xgboost", "snaive"):
        target = run / "families" / name
        for filename in (
            "status.json",
            "events.jsonl",
            "leaderboard.json",
            "generation_curves.jsonl",
            "eval_history.json",
            "winner.json",
            "population.json",
        ):
            assert (target / filename).exists()
        status = json.loads((target / "status.json").read_text())
        statuses.append(status)
        assert status["state"] == "done"
        assert {"generation", "best_error", "candidates_done"} <= status.keys()
        assert any(
            json.loads(line)["event"] == "progress"
            for line in (target / "events.jsonl").read_text().splitlines()
        )
    assert len({state["pid"], *(status["pid"] for status in statuses)}) == 3
    events = [
        json.loads(line)
        for line in (run / "families" / "snaive" / "events.jsonl").read_text().splitlines()
    ]
    assert statuses[0]["started_at"] < statuses[1]["started_at"] + events[-2]["elapsed"] + 1
    board = json.loads((run / "leaderboard.json").read_text())
    assert {entry["family"] for entry in board} == {"xgboost", "snaive"}
    assert board[0]["fitness"] == min(entry["fitness"] for entry in board)
    recipe = json.loads((run / "recipe.json").read_text())
    assert recipe["family"] == board[0]["family"] and recipe["transform"] == "none"
    assert (run / "result.json").exists() and (run / "forecast.parquet").exists()


def test_missing_library_fails_only_its_family(tmp_path, monkeypatch):
    request, spec, prepared = make_request(tmp_path, [{"name": "ets"}, {"name": "snaive"}])
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(
        "import sys\nclass MissingStats:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'statsforecast':\n"
        "            raise ImportError('simulated missing statsforecast')\n"
        "sys.meta_path.insert(0, MissingStats())\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(hooks) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run_id = start_run(
        tmp_path / "runs", spec, prepared, families=request["families"], time_budget_s=15, nthread=2
    )
    state = wait_run(tmp_path / "runs", run_id, timeout=60)
    assert state["state"] == "done", state
    assert state["families"]["ets"]["state"] == "failed"
    assert "simulated missing statsforecast" in state["families"]["ets"]["reason"]
    assert state["families"]["snaive"]["state"] == "done"


def test_killed_worker_resumes_candidate_cache_and_remaining_budget(tmp_path):
    pytest.importorskip("xgboost")
    request, _, _ = make_request(tmp_path, [{"name": "xgboost"}], 12)
    (tmp_path / "request.json").write_text(json.dumps(request))
    command = [sys.executable, "-m", "smolsmort.forecast.family_worker", str(tmp_path), "xgboost"]
    checkpoint = tmp_path / "families" / "xgboost" / "checkpoint.json"
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while (
                not checkpoint.exists() and process.poll() is None and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert checkpoint.exists()
            process.kill()
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        before = json.loads(checkpoint.read_text())
        assert before["entries"] and before["elapsed"] > 0
        restarted = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        assert restarted.returncode == 0, (tmp_path / "worker.log").read_text()
    after = json.loads(checkpoint.read_text())
    prior = {entry["key"]: entry for entry in before["entries"]}
    current = {entry["key"]: entry for entry in after["entries"]}
    for key, entry in prior.items():
        assert current[key]["fitness"] == entry["fitness"]
        assert current[key]["generation"] == entry["generation"]
    assert after["elapsed"] >= before["elapsed"]
    target = checkpoint.parent
    assert json.loads((target / "status.json").read_text())["state"] == "done"
    before_done = {path.name: path.read_bytes() for path in target.glob("*.json")}
    assert subprocess.run(command, stdout=subprocess.DEVNULL, timeout=10).returncode == 0
    assert before_done == {path.name: path.read_bytes() for path in target.glob("*.json")}


def test_family_worker_refuses_torch(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(SystemExit, match="must not share a process with torch"):
        family_worker.run(tmp_path, "snaive")


def test_checkpoint_resume_skips_completed_scores_and_reproduces_search(monkeypatch):
    module = importlib.import_module("smolsmort.forecast.search")
    ws = SimpleNamespace(
        mode="row",
        masks={"train": np.ones(20, dtype=bool)},
        families=["measure:x"],
        profile={"genes": {"objectives": ["squared"]}},
    )
    calls = []

    def score_candidate(_ws, genome, **_kwargs):
        calls.append(genome.key())
        fitness = float(int(genome.key(), 16))
        return Scored(fitness, np.array([fitness]), {}, None)

    monkeypatch.setattr(module, "score", score_candidate)
    budget = Budget(population=6, max_generations=3, plateau=10)
    expected = search(ws, budget)["leaderboard"]
    calls.clear()
    saved = []

    def crash_after_checkpoint(state):
        saved.append(deepcopy(state))
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        search(ws, budget, family="xgboost", on_checkpoint=crash_after_checkpoint)
    assert len(calls) == 1
    completed_key = calls[0]
    calls.clear()
    result = search(ws, budget, family="xgboost", resume=saved[0])
    assert completed_key not in calls
    assert json.dumps(result["leaderboard"]) == json.dumps(expected)


def test_search_uses_provider_space_objectives_and_grid_overrides(monkeypatch):
    module = importlib.import_module("smolsmort.forecast.search")
    family = SimpleNamespace(
        name="snaive",
        needs="lag_features",
        space=lambda: [
            Param("alpha", "log", 0.1, 10, default=1),
            Param("order", "choice", choices=(1, 2), default=1),
        ],
        objectives=lambda task: ("custom",),
    )
    ws = SimpleNamespace(
        mode="row",
        spec=SimpleNamespace(task="regression"),
        masks={"train": np.ones(20, dtype=bool)},
        families=["lag"],
        profile={"genes": {"objectives": ["squared"]}},
    )
    seen = []

    def score_candidate(_ws, genome, **_kwargs):
        seen.append(genome)
        return Scored(float(dict(genome.params)["alpha"]), np.array([1.0]), {}, None)

    monkeypatch.setattr(module, "score", score_candidate)
    result = search(
        ws, Budget(), family=family, method="grid", space={"alpha": [2, 4], "order": [1, 2]}
    )
    assert len(seen) == 4
    assert all(genome.family == "snaive" and genome.objective == "custom" for genome in seen)
    assert all(set(dict(genome.params)) == {"alpha", "order"} for genome in seen)
    assert result["stopped"] == "grid complete"
    seen.clear()
    search(ws, Budget(population=4, max_generations=2), family=family, space={"alpha": [2, 4]})
    assert all(dict(genome.params)["alpha"] in (2, 4) for genome in seen)


def test_grid_deadline_keeps_best_and_reports_time_cap(monkeypatch):
    module = importlib.import_module("smolsmort.forecast.search")
    family = SimpleNamespace(
        name="snaive",
        needs="lag_features",
        space=lambda: [Param("alpha", "int", 1, 4, default=1)],
        objectives=lambda task: ("squared",),
    )
    ws = SimpleNamespace(
        mode="row",
        spec=SimpleNamespace(task="regression"),
        masks={"train": np.ones(20, dtype=bool)},
        families=["lag"],
        profile={"genes": {"objectives": ["squared"]}},
    )
    elapsed = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: elapsed[0])
    calls = []

    def score_candidate(_ws, genome, **_kwargs):
        calls.append(genome)
        elapsed[0] += 2
        return Scored(0.2, np.array([0.2]), {}, None)

    monkeypatch.setattr(module, "score", score_candidate)
    result = search(
        ws, Budget(time_cap=3), family=family, method="grid", space={"alpha": [1, 2, 3, 4]}
    )
    assert len(calls) == 2 and result["best"] is not None
    assert result["stopped"] == "time cap"


@pytest.mark.parametrize("name", ["ets", "theta", "arima"])
def test_statistical_search_fits_one_real_candidate(tmp_path, monkeypatch, name):
    pytest.importorskip("duckdb")
    pytest.importorskip("statsforecast")
    source = write_bike_daily(tmp_path, days=140)
    spec = PrepSpec(
        str(source),
        "series",
        "regression",
        (Column("dteday", "time"), Column("cnt", "target", aggregation="sum")),
        step="day",
        horizon=4,
    )
    prepared = prepare(spec, tmp_path / "cache")
    ws = load_workspace(spec, prepared.folder, prepared.summary)
    module = importlib.import_module("smolsmort.forecast.search")
    score_candidate = module.score
    calls = []

    def count_candidate(*args, **kwargs):
        calls.append(args[1].key())
        return score_candidate(*args, **kwargs)

    monkeypatch.setattr(module, "score", count_candidate)
    result = search(ws, Budget(), family=name)
    assert len(calls) == 1 and result["stopped"] == "single candidate"
    assert result["leaderboard"][0]["fitness"] is not None, result


def test_expired_checkpoint_keeps_cached_winner_and_progress(tmp_path):
    request, _, _ = make_request(tmp_path, [{"name": "snaive"}])
    (tmp_path / "request.json").write_text(json.dumps(request))
    command = [sys.executable, "-m", "smolsmort.forecast.family_worker", str(tmp_path), "snaive"]
    first = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert first.returncode == 0, first.stderr
    target = tmp_path / "families" / "snaive"
    checkpoint = json.loads((target / "checkpoint.json").read_text())
    checkpoint["elapsed"] = request["time_budget_s"]
    (target / "checkpoint.json").write_text(json.dumps(checkpoint))
    (target / "status.json").write_text(json.dumps({"state": "running"}))
    winner = (target / "winner.json").read_bytes()
    cached = (target / "checkpoint.json").read_bytes()
    restarted = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert restarted.returncode == 0, restarted.stderr
    status = json.loads((target / "status.json").read_text())
    assert status["state"] == "done" and status["reason"] == "time cap"
    assert status["candidates_done"] == 1 and status["best_error"] is not None
    assert (target / "winner.json").read_bytes() == winner
    assert (target / "checkpoint.json").read_bytes() == cached


def test_v2_lightgbm_genetic_search_runs_alongside_xgboost(tmp_path):
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    selections = [
        {"name": "xgboost", "method": "grid", "space": {"max_depth": [3], "eta": [0.1]}},
        {"name": "lightgbm", "method": "genetic"},
    ]
    request, spec, prepared = make_request(tmp_path, selections, 30)
    run_id = start_run(
        tmp_path / "runs", spec, prepared, families=selections, time_budget_s=30, nthread=2
    )
    state = wait_run(tmp_path / "runs", run_id, timeout=120)
    run = tmp_path / "runs" / run_id
    assert state["state"] == "done", state
    lightgbm = json.loads((run / "families" / "lightgbm" / "status.json").read_text())
    assert lightgbm["state"] == "done" and lightgbm["candidates_done"] >= 1
    board = json.loads((run / "leaderboard.json").read_text())
    assert {"xgboost", "lightgbm"} <= {entry["family"] for entry in board}
    params = {
        key for entry in board if entry["family"] == "lightgbm" for key in entry["genome"]["params"]
    }
    assert params & {"num_leaves", "learning_rate", "min_child_samples"}
