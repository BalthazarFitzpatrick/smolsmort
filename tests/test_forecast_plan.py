"""search plan persistence, incremental sets and bounded passes"""

import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import plan, plan_sets, runs, views
from smolsmort.forecast.spec import Column, PrepSpec


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def read_json(path):
    return json.loads(path.read_text())


@pytest.fixture
def config():
    spec = PrepSpec(
        "source.csv",
        "timeseries",
        "regression",
        (
            Column("day", "time"),
            Column("value", "target", aggregation="sum"),
        ),
    )
    return {
        "hours": 1,
        "sets": [{"label": "clients", "horizon": 8, "spec": asdict(spec)}],
        "families": [{"name": "ridge", "budget_s": 600}, {"name": "lightgbm", "budget_s": 600}],
    }


@pytest.mark.parametrize(
    "change, message",
    [
        ({"hours": 0}, "hours"),
        ({"hours": True}, "hours"),
        ({"deadline": "tomorrow"}, "exactly one"),
        ({"families": []}, "families"),
        ({"sets": []}, "sets"),
        ({"passes": {"keep": 0}}, "passes.keep"),
        ({"warm": {"missing": {}}}, "warm"),
        ({"transform": "bad"}, "transform"),
    ],
)
def test_plan_validation(tmp_path, config, change, message):
    config.update(change)
    with pytest.raises(ValueError, match=message):
        plan.save_plan(tmp_path, config)
    write_json(tmp_path / "forecast-plan.json", config)
    with pytest.raises(ValueError, match=message):
        plan.load_plan(tmp_path)


def test_saved_spec_and_horizon(tmp_path, config):
    write_json(tmp_path / "request.json", {"spec": config["sets"][0]["spec"]})
    config["sets"][0].update(spec="request.json", horizon=16)
    assert plan.save_plan(tmp_path, config) == plan.load_plan(tmp_path)
    assert plan._spec(tmp_path, config["sets"][0]).horizon == 16


def make_grid(config):
    source = json.loads(json.dumps(config.pop("sets")[0]["spec"]))
    config["grid"] = {"horizon": [8, 12], "grouping": {"client groups": source, "total": source}}
    return config


def test_grid_expansion_and_saved_sources(tmp_path, config):
    make_grid(config)
    write_json(tmp_path / "request.json", {"spec": config["grid"]["grouping"]["total"]})
    config["grid"]["grouping"]["total"] = "request.json"
    assert plan.save_plan(tmp_path, config) == config
    assert read_json(tmp_path / "forecast-plan.json") == config
    items = plan.expand_sets(plan.load_plan(tmp_path))
    assert [item["label"] for item in items] == [
        "8 weeks - client groups",
        "8 weeks - total",
        "12 weeks - client groups",
        "12 weeks - total",
    ]
    assert [plan._spec(tmp_path, item).horizon for item in items] == [8, 8, 12, 12]
    config["grid"]["selected"] = {"horizon": [12], "grouping": ["total"]}
    plan.save_plan(tmp_path, config)
    assert plan.expand_sets(plan.load_plan(tmp_path)) == [items[-1]]
    assert len(plan.load_plan(tmp_path)["grid"]["grouping"]) == 2


@pytest.mark.parametrize(
    "grid, message",
    [
        ({"horizon": [], "grouping": {}}, "horizon"),
        ({"horizon": [True], "grouping": {}}, "positive integer"),
        ({"horizon": [1.5], "grouping": {}}, "positive integer"),
        ({"horizon": [0], "grouping": {}}, "positive integer"),
        ({"horizon": [8], "grouping": []}, "object"),
        ({"horizon": [8], "grouping": {}}, "non-empty"),
        ({"horizon": [8], "grouping": {" ": {}}}, "names"),
        ({"horizon": [8], "grouping": {"total": 3}}, "spec"),
    ],
)
def test_invalid_grid(tmp_path, config, grid, message):
    config.pop("sets")
    config["grid"] = grid
    with pytest.raises(ValueError, match=message):
        plan.save_plan(tmp_path, config)


def test_grid_and_sets_rejected(tmp_path, config):
    config["grid"] = {}
    with pytest.raises(ValueError, match="not both"):
        plan.save_plan(tmp_path, config)


def test_grid_validates_unselected_sources(tmp_path, config):
    make_grid(config)["grid"]["selected"] = {"horizon": [8], "grouping": ["total"]}
    config["grid"]["grouping"]["client groups"] = {"source": "bad.csv"}
    with pytest.raises(ValueError, match="invalid spec source for grouping client groups"):
        plan.save_plan(tmp_path, config)


def test_grid_rejects_saved_request_without_object(tmp_path, config):
    make_grid(config)["grid"]["grouping"]["total"] = "bad.json"
    write_json(tmp_path / "bad.json", [])
    with pytest.raises(ValueError, match="invalid spec source for grouping total"):
        plan.save_plan(tmp_path, config)


@pytest.mark.parametrize(
    "selected",
    [
        None,
        {},
        {"horizon": [], "grouping": ["total"]},
        {"horizon": [8], "grouping": ["unknown"]},
        {"horizon": [True], "grouping": ["total"]},
    ],
)
def test_invalid_grid_selection(tmp_path, config, selected):
    make_grid(config)["grid"]["selected"] = selected
    with pytest.raises(ValueError, match="selected"):
        plan.save_plan(tmp_path, config)


def test_sets_only_unchanged(tmp_path, config):
    config = json.loads(json.dumps(config))
    assert plan.save_plan(tmp_path, config) == config
    assert plan.load_plan(tmp_path) == config
    assert plan.expand_sets(config) == config["sets"]


def test_grid_cli_status(tmp_path, config, capsys):
    from smolsmort.cli import main

    plan.save_plan(tmp_path, make_grid(config))
    assert main(["forecast", "plan", "status", "--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["permutations"] == [
        item["label"] for item in plan.expand_sets(config)
    ]


def fake_runner(monkeypatch, root, *, on_start=None, on_wait=None):
    calls = []
    monkeypatch.setattr(plan, "prepare", lambda spec, cache: SimpleNamespace(folder=cache))

    def start(runs_root, spec, prepared, **options):
        if on_start:
            on_start(options)
        family = options["families"][0]["name"]
        run_id = f"part-{len(calls)}"
        folder = runs_root / run_id
        write_json(folder / "request.json", {**options, "spec": asdict(spec)})
        write_json(
            folder / "status.json",
            {"state": "done", "families": {family: {"state": "done", "best_error": 0.2}}},
        )
        write_json(folder / "result.json", {"fitness": 0.2, "families": {family: {"fitness": 0.2}}})
        population = [{"family": family, "value": value} for value in (3, 1, 2)]
        write_json(folder / "families" / family / "population.json", population)
        write_json(
            folder / "families" / family / "checkpoint.json",
            {"entries": [{"genome": genome, "fitness": genome["value"]} for genome in population]},
        )
        calls.append(options)
        return run_id

    def wait(runs_root, run_id, **kwargs):
        if on_wait:
            return on_wait(run_id)
        return {"state": "done"}

    return calls, start, wait


def test_resume_scale_and_incremental_tiles(tmp_path, monkeypatch, config):
    config["hours"] = 1000 / 3600
    plan.save_plan(tmp_path, config)
    monkeypatch.setattr(plan.time, "time", lambda: 1000)
    snapshots = []

    def inspect(options):
        sets = [item for item in runs.list_runs(tmp_path / "runs") if item["kind"] == "set"]
        assert len(sets) == 1
        snapshots.append({name: data["state"] for name, data in sets[0]["families"].items()})
        assert sets[0]["state"] == "running"
        assert options["part_of"] == sets[0]["id"]

    calls, start, wait = fake_runner(monkeypatch, tmp_path, on_start=inspect)
    result = plan.run_plan(
        tmp_path, tmp_path / "runs", tmp_path / "cache", start_run=start, wait_run=wait
    )
    assert result["jobs_done"] == 2
    assert snapshots == [
        {"ridge": "pending", "lightgbm": "pending"},
        {"ridge": "done", "lightgbm": "pending"},
    ]
    assert calls[0]["time_budget_s"] == 400
    state = read_json(tmp_path / "forecast-plan-state.json")
    assert state["finished"]["1:0:ridge"]["scale"] == pytest.approx(2 / 3)
    assert calls[1]["time_budget_s"] == 600
    assert (
        plan.run_plan(
            tmp_path, tmp_path / "runs", tmp_path / "cache", start_run=start, wait_run=wait
        )["state"]
        == "done"
    )
    assert len(calls) == 2
    assert all(
        value["state"] == "done"
        for value in runs.list_runs(tmp_path / "runs")[0]["families"].values()
    )


def test_failure_continues(tmp_path, monkeypatch, config):
    plan.save_plan(tmp_path, config)
    calls, start, wait = fake_runner(
        monkeypatch,
        tmp_path,
        on_wait=lambda run_id: {"state": "failed" if run_id == "part-0" else "done"},
    )
    result = plan.run_plan(
        tmp_path, tmp_path / "runs", tmp_path / "cache", start_run=start, wait_run=wait
    )
    assert len(calls) == 2
    assert result["jobs_done"] == 2


def test_stop_cancels_current_and_preserves_resume(tmp_path, monkeypatch, config):
    plan.save_plan(tmp_path, config)
    cancelled = []
    monkeypatch.setattr(runs, "cancel_run", lambda root, run_id: cancelled.append(run_id))

    def stop(run_id):
        plan.stop_plan(tmp_path)
        return {"state": "cancelled", "stopped": "cancelled"}

    calls, start, wait = fake_runner(monkeypatch, tmp_path, on_wait=stop)
    result = plan.run_plan(
        tmp_path, tmp_path / "runs", tmp_path / "cache", start_run=start, wait_run=wait
    )
    assert result["state"] == "stopped"
    assert cancelled == ["part-0"]
    assert len(calls) == 1
    assert read_json(tmp_path / "forecast-plan-state.json")["finished"] == {}
    assert (
        next(item for item in runs.list_runs(tmp_path / "runs") if item["kind"] == "set")["state"]
        == "cancelled"
    )


def test_second_pass_keeps_best_seeds(tmp_path, monkeypatch, config):
    config["families"] = config["families"][:1]
    config["passes"] = {
        "second_pass": True,
        "families": ["ridge"],
        "keep": 2,
        "min_generations": 8,
        "patience": 6,
    }
    plan.save_plan(tmp_path, config)
    clock = [1000]
    monkeypatch.setattr(plan.time, "time", lambda: clock[0])
    seeds = []

    def inspect(options):
        if options["warm_from"]:
            seeds.extend(
                read_json(
                    tmp_path
                    / "runs"
                    / options["warm_from"]
                    / "families"
                    / "ridge"
                    / "population.json"
                )
            )

    def wait(run_id):
        if run_id == "part-1":
            clock[0] += 3500
        return {"state": "done"}

    calls, start, wait = fake_runner(monkeypatch, tmp_path, on_start=inspect, on_wait=wait)
    plan.run_plan(tmp_path, tmp_path / "runs", tmp_path / "cache", start_run=start, wait_run=wait)
    assert len(calls) == 2
    assert [entry["value"] for entry in seeds] == [1, 2]
    assert calls[1]["min_generations"] == 8
    assert calls[1]["plateau"] == 6


def test_longer_horizon_uses_shorter_finished_population(tmp_path, monkeypatch, config):
    config["families"] = config["families"][:1]
    longer = deepcopy(config["sets"][0])
    longer.update(label="clients long", horizon=16)
    config["sets"].append(longer)
    plan.save_plan(tmp_path, config)
    calls, start, wait = fake_runner(monkeypatch, tmp_path)
    plan.run_plan(tmp_path, tmp_path / "runs", tmp_path / "cache", start_run=start, wait_run=wait)
    assert calls[0]["warm_from"] is None
    assert calls[1]["warm_from"] == "warm-clients-long-ridge"
    assert read_json(
        tmp_path / "runs" / calls[1]["warm_from"] / "families" / "ridge" / "population.json"
    ) == read_json(tmp_path / "runs" / "part-0" / "families" / "ridge" / "population.json")


def test_cancelled_dead_worker_reports_cancelled(tmp_path, monkeypatch, config):
    folder = tmp_path / "part"
    write_json(
        folder / "request.json",
        {"spec": config["sets"][0]["spec"], "part_of": "set", "label": "clients - ridge"},
    )
    write_json(folder / "status.json", {"state": "running"})
    monkeypatch.setattr(
        "smolsmort.forecast.processes.stop_run_processes", lambda *args, **kwargs: None
    )
    runs.cancel_run(tmp_path, "part")
    state = runs.run_state(tmp_path, "part")
    assert state["state"] == "cancelled"
    assert state["part_of"] == "set"
    assert state["label"] == "clients - ridge"


def test_ensemble_recomputed_over_incremental_members(tmp_path, monkeypatch, config):
    tables = {}

    def write_table(path, table):
        path.touch()
        tables[str(path)] = deepcopy(table)

    monkeypatch.setattr(plan_sets, "write_table", write_table)
    monkeypatch.setattr(plan_sets, "read_table", lambda path: deepcopy(tables[str(path)]))
    folder = plan_sets.create_set(
        tmp_path, "set", "clients", config["sets"][0]["spec"], ["ridge", "lightgbm"]
    )
    for name, prediction in (("ridge", 8), ("lightgbm", 12)):
        part = tmp_path / name
        write_json(part / "result.json", {"fitness": 0.2})
        write_json(part / "status.json", {"state": "done"})
        for kind in ("validation", "test", "forecast"):
            write_table(
                part / f"winner_{kind}.parquet",
                {
                    "series": np.array(["a"]),
                    "step": np.array([1]),
                    "actual": np.array([10.0]),
                    "prediction": np.array([prediction]),
                },
            )
    plan_sets.update_set(tmp_path, "set", {"ridge": "ridge"})
    assert read_json(folder / "result.json")["ensemble"]["members"] == ["ridge"]
    plan_sets.update_set(tmp_path, "set", {"ridge": "ridge", "lightgbm": "lightgbm"})
    result = read_json(folder / "result.json")["ensemble"]
    assert result["members"] == ["ridge", "lightgbm"]
    assert result["fitness"] == 0
    assert tables[str(folder / "ensemble_validation.parquet")]["prediction"].tolist() == [10]


def test_set_kept_test_scores_for_both_families(tmp_path, config):
    folder = plan_sets.create_set(
        tmp_path, "set", "clients", config["sets"][0]["spec"], ["ridge", "lightgbm"]
    )
    for name, ratio in (("ridge", 0.8), ("lightgbm", 1.2)):
        write_json(tmp_path / name / "request.json", {})
        write_json(tmp_path / name / "status.json", {"state": "done"})
        write_json(
            tmp_path / name / "result.json",
            {
                "winner": {"family": name},
                "families": {name: {"state": "done", "fitness": ratio, "kept": True}},
                "verdict": {"accuracy": {"ratio": ratio, "val_error": ratio}},
                "model_error": ratio,
                "baseline_error": 1.0,
                "val_error": ratio,
            },
        )
    plan_sets.update_set(tmp_path, "set", {name: name for name in ("ridge", "lightgbm")})
    result = read_json(folder / "result.json")
    tiles = views.family_tiles(
        folder, read_json(folder / "request.json"), read_json(folder / "status.json")
    )
    for name, ratio in (("ridge", 0.8), ("lightgbm", 1.2)):
        assert np.isfinite(tiles[name]["test_ratio"])
        assert tiles[name]["test_ratio"] == ratio
        summary = result["families"][name]
        assert summary["kept"] is True
        assert summary["model_error"] == summary["val_error"] == ratio
        assert summary["baseline_error"] == 1.0


def test_start_launches_detached_and_reuses_live_pid(tmp_path, monkeypatch, config):
    plan.save_plan(tmp_path, config)
    launched = []
    process = SimpleNamespace(pid=91234, poll=lambda: None)
    monkeypatch.setattr(plan, "_PROCESSES", {})

    def launch(command, **options):
        launched.append((command, options))
        return process

    monkeypatch.setattr(plan.subprocess, "Popen", launch)
    stop = tmp_path / "forecast-plan.stop"
    stop.touch()
    assert plan.start_plan(tmp_path, tmp_path / "runs", tmp_path / "cache") == 91234
    assert not stop.exists()
    assert (tmp_path / "forecast-plan.pid").read_text() == "91234"
    command, options = launched[0]
    assert command[1:3] == ["-m", "smolsmort.forecast.plan"]
    assert command[3:] == [str(tmp_path), str(tmp_path / "runs"), str(tmp_path / "cache")]
    assert options["start_new_session"] is True
    assert plan.start_plan(tmp_path, tmp_path / "runs", tmp_path / "cache") == 91234
    assert len(launched) == 1
    monkeypatch.setattr(plan, "_PROCESSES", {})
    monkeypatch.setattr(plan, "is_process_alive", lambda pid: pid == 91234)
    assert plan.start_plan(tmp_path, tmp_path / "runs", tmp_path / "cache") == 91234
    assert len(launched) == 1


@pytest.mark.parametrize("grid", [False, True])
def test_resume_reattaches_persisted_current_run(tmp_path, monkeypatch, config, grid):
    config["families"] = config["families"][:1]
    if grid:
        make_grid(config)["grid"]["selected"] = {"horizon": [8], "grouping": ["total"]}
    plan.save_plan(tmp_path, config)

    def interrupt(run_id):
        raise KeyboardInterrupt

    calls, start, wait = fake_runner(monkeypatch, tmp_path, on_wait=interrupt)
    with pytest.raises(KeyboardInterrupt):
        plan.run_plan(
            tmp_path, tmp_path / "runs", tmp_path / "cache", start_run=start, wait_run=wait
        )
    state = read_json(tmp_path / "forecast-plan-state.json")
    assert state["current"]["run_id"] == "part-0"
    waited = []

    def finish(runs_root, run_id, **kwargs):
        waited.append(run_id)
        return {"state": "done"}

    def forbidden_start(*args, **kwargs):
        pytest.fail("resumed current run must not start a replacement")

    result = plan.run_plan(
        tmp_path, tmp_path / "runs", tmp_path / "cache", start_run=forbidden_start, wait_run=finish
    )
    assert result["jobs_done"] == 1
    assert waited == ["part-0"]
    assert len(calls) == 1


def test_resume_restarts_stopped_unfinished_job(tmp_path, monkeypatch, config):
    config["families"] = config["families"][:1]
    plan.save_plan(tmp_path, config)
    monkeypatch.setattr(runs, "cancel_run", lambda *args: None)
    interrupted = [False]

    def finish(run_id):
        if not interrupted[0]:
            interrupted[0] = True
            plan.stop_plan(tmp_path)
            return {"state": "cancelled", "stopped": "cancelled"}
        return {"state": "done"}

    calls, start, wait = fake_runner(monkeypatch, tmp_path, on_wait=finish)
    assert (
        plan.run_plan(
            tmp_path, tmp_path / "runs", tmp_path / "cache", start_run=start, wait_run=wait
        )["state"]
        == "stopped"
    )
    (tmp_path / "forecast-plan.stop").unlink()
    result = plan.run_plan(
        tmp_path, tmp_path / "runs", tmp_path / "cache", start_run=start, wait_run=wait
    )
    assert result["jobs_done"] == 1
    assert len(calls) == 2
    assert (
        read_json(tmp_path / "forecast-plan-state.json")["finished"]["1:0:ridge"]["run_id"]
        == "part-1"
    )
