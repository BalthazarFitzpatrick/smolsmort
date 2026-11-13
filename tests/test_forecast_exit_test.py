"""exit scoring is isolated from search and shares finish's exact prediction cache"""

import inspect
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import coordinator, ensemble, exit_test, pipeline, search
from smolsmort.forecast.spec import Column, PrepSpec
from smolsmort.forecast.tables import read_table


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def exit_workspace(monkeypatch):
    spec = PrepSpec(
        "unused.csv",
        "timeseries",
        "regression",
        (Column("day", "time"), Column("units", "target", aggregation="sum")),
        horizon=1,
    )
    masks = {
        name: np.array([part == name for part in ("train", "val", "val", "test", "test", "future")])
        for name in ("train", "val", "test", "future")
    }
    frame = SimpleNamespace(
        y=np.array([8, 10, 14, 26, 30, np.nan]),
        step=np.arange(6),
        series=np.array(["a"] * 6),
        weeks_ahead=None,
    )
    ws = SimpleNamespace(
        mode="timeseries",
        horizon_mode="mixed",
        spec=spec,
        profile={"genes": {}},
        frames={(1, 1): frame},
        series_masks={(1, 1): masks},
        test_steps=2,
        excluded_series=[],
    )
    calls = []

    def predict_member(ws, genome, frame, fit_rows, eval_rows, bucket, nthread, **kwargs):
        calls.append((genome.key(), nthread))
        if "importance" in kwargs:
            kwargs["importance"].append({"lag_1": 1.0})
        return np.full(eval_rows.sum(), 25 + dict(genome.params).get("alpha", 0))

    monkeypatch.setattr(pipeline, "_timeseries_predict", predict_member)
    monkeypatch.setattr(
        pipeline, "_timeseries_baseline", lambda frame, mask, *args: np.full(mask.sum(), 20)
    )
    monkeypatch.setattr(exit_test, "load_workspace", lambda *args, **kwargs: ws)
    monkeypatch.setattr(ensemble, "load_workspace", lambda *args, **kwargs: ws)
    return ws, calls


def test_exit_stores_key_and_finish_reuses_only_matching_recipe(
    tmp_path, monkeypatch, exit_workspace
):
    ws, calls = exit_workspace
    genome = pipeline.make_genome(["lag"], "squared", {"alpha": 1}, "ridge")
    entry = {"family": "ridge", "genome": genome.to_dict(), "fitness": 0.2, "prediction": [9, 12]}
    save_json(
        tmp_path / "request.json",
        {
            "version": 2,
            "families": [{"name": "ridge"}],
            "spec": asdict(ws.spec),
            "prepared": str(tmp_path),
            "summary": {},
            "ensemble": {"enabled": False},
        },
    )
    exit_test.score_exit(tmp_path, "ridge", entry)
    target = tmp_path / "families" / "ridge"
    marker = json.loads((target / "exit_test.json").read_text())
    assert marker["recipe_key"] == genome.key()
    assert marker["model_error"] == pytest.approx(4 / 56)
    assert marker["verdict"]["accuracy"]["ratio"] == pytest.approx((4 / 56) / (16 / 56))
    np.testing.assert_array_equal(read_table(target / "test.parquet")["prediction"], [26, 26])
    assert len(calls) == 2
    assert all(threads == 1 for _, threads in calls)
    save_json(target / "winner.json", genome.to_dict())
    save_json(target / "leaderboard.json", [entry])
    from smolsmort.forecast.tiles import read_tiles

    tile = read_tiles(tmp_path, include_ensembles=False)[0]
    assert tile["recipe_key"] == genome.key() and tile["scored_at_exit"]
    assert tile["error_metrics"]["test_ratio"]["value"] == marker["verdict"]["accuracy"]["ratio"]
    assert tile["metrics"]["wape"]["test"] == marker["model_error"]

    monkeypatch.setattr(ensemble, "watch_parent", lambda *args: None)
    monkeypatch.setattr(
        ensemble,
        "fit_importance",
        lambda *args, **kwargs: pytest.fail("cached importance refitted"),
    )
    save_json(
        tmp_path / "coordinator.json", {"families": {"ridge": {"state": "done"}}, "nthread": 1}
    )
    save_json(tmp_path / "finalists.json", [entry])
    ensemble.run(tmp_path, 123)
    assert len(calls) == 2
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["winner"]["verdict"] == marker["verdict"]

    different = pipeline.make_genome(["lag"], "squared", {"alpha": 2}, "ridge")
    save_json(tmp_path / "finalists.json", [{**entry, "genome": different.to_dict()}])
    ensemble.run(tmp_path, 123)
    assert len(calls) == 4
    assert calls[-1][0] == different.key()


def test_exit_cache_reproduces_uncached_member_and_ensemble_outputs(tmp_path, exit_workspace):
    ws, calls = exit_workspace
    genomes = [
        pipeline.make_genome(["lag"], "squared", {"alpha": alpha}, "ridge") for alpha in (1, 2)
    ]
    save_json(
        tmp_path / "request.json",
        {"spec": asdict(ws.spec), "prepared": str(tmp_path), "summary": {}},
    )
    predictions, importance = {}, {}
    for index, genome in enumerate(genomes):
        name = f"ridge{index}"
        exit_test.score_exit(tmp_path, name, {"genome": genome.to_dict(), "prediction": [9, 12]})
        assert exit_test.restore_exit_cache(
            tmp_path / "families" / name, genome, predictions, importance
        )
    calls.clear()
    cached = pipeline.finish(
        ws,
        genomes[0],
        members=genomes,
        validation=[9, 12],
        prediction_cache=predictions,
        importance_cache=importance,
    )
    assert calls == []
    uncached = pipeline.finish(ws, genomes[0], members=genomes, validation=[9, 12])
    assert len(calls) == 4
    assert cached["verdict"] == uncached["verdict"]
    for split in ("validation", "test", "forecast"):
        for column in cached[split]:
            np.testing.assert_array_equal(cached[split][column], uncached[split][column])


def test_repeated_exit_keeps_previous_publication_readable(tmp_path, exit_workspace):
    ws, calls = exit_workspace
    genome = pipeline.make_genome(["lag"], "squared", {"alpha": 1}, "ridge")
    save_json(
        tmp_path / "request.json",
        {"spec": asdict(ws.spec), "prepared": str(tmp_path), "summary": {}},
    )
    entry = {"genome": genome.to_dict(), "prediction": [9, 12]}
    target = tmp_path / "families" / "ridge"
    exit_test.score_exit(tmp_path, "ridge", entry)
    previous = json.loads((target / "exit_test.json").read_text())
    previous_table = target / previous["tables_path"] / "test.parquet"
    previous_bytes = previous_table.read_bytes()
    exit_test.score_exit(tmp_path, "ridge", entry)
    current = json.loads((target / "exit_test.json").read_text())
    assert current["recipe_key"] == previous["recipe_key"]
    assert current["tables_path"] != previous["tables_path"]
    assert previous_table.read_bytes() == previous_bytes
    assert len(calls) == 4


def test_exit_queue_is_serial_nonblocking_and_bounded(tmp_path, monkeypatch):
    clock = [100.0]
    spawned = []

    class Process:
        pid = 123
        returncode = None

        def __init__(self, command, **kwargs):
            self.command, self.env = command, kwargs["env"]
            spawned.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(exit_test.time, "time", lambda: clock[0])
    monkeypatch.setattr(exit_test.subprocess, "Popen", Process)
    scorer = exit_test.ExitScorer(tmp_path, enabled=True)
    entry = {"genome": pipeline.make_genome(["lag"], "squared", {}, "ridge").to_dict()}
    for name in ("ridge", "snaive"):
        scorer.enqueue(name, entry, 1)
        scorer.enqueue(name, entry, 1)
    scorer.poll()
    for _ in range(20):
        scorer.poll()
    assert len(spawned) == 1 and len(scorer.pending) == 1
    assert spawned[0].env["OMP_NUM_THREADS"] == "1"
    clock[0] += exit_test.EXIT_TIMEOUT_S + 1
    scorer.poll()
    assert len(spawned) == 2 and spawned[0].returncode == -1
    assert not scorer.completed
    spawned[1].returncode = 0
    scorer.poll()
    assert len(scorer.completed) == 1
    scorer.enqueue("ridge", entry, 2)
    assert "ridge" in scorer.pending
    scorer.poll()
    scorer.close()
    assert spawned[-1].returncode == -1 and not scorer.pending


@pytest.mark.parametrize("exit_hangs", [False, True])
def test_one_shot_exit_starts_during_broad_while_other_family_runs(
    tmp_path, monkeypatch, exit_hangs
):
    clock = [100.0]
    commands, states_at_exit = [], []

    def save(name, data):
        save_json(tmp_path / name, data)

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 1000 + len(commands)
            self.returncode = None
            self.module = command[2]
            commands.append(command)
            if self.module == "smolsmort.forecast.family_worker":
                name = command[-1]
                genome = pipeline.make_genome(["lag"], "squared", {}, name)
                save(
                    f"families/{name}/checkpoint.json",
                    {
                        "entries": [
                            {
                                "genome": genome.to_dict(),
                                "fitness": 0.2,
                                "families": "lag",
                                "metrics": {},
                                "generation": 0,
                            }
                        ]
                    },
                )
                save(
                    f"families/{name}/status.json",
                    {
                        "state": "done",
                        "reason": "grid complete" if name == "snaive" else "broad defaults",
                    },
                )
                if name == "snaive":
                    self.returncode = 0
            elif self.module == "smolsmort.forecast.exit_test":
                states_at_exit.append(
                    json.loads((tmp_path / "coordinator.json").read_text())["families"]
                )
                self.returncode = None if exit_hangs else 0
            else:
                self.returncode = 0
                save("finish_complete.json", {})

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(coordinator, "describe_machine", lambda *args: {})
    monkeypatch.setattr(coordinator.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        coordinator.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(coordinator.subprocess, "Popen", Process)
    original_read = coordinator.read_json

    def guard_read(path, *args):
        assert "exit_test" not in str(path) and "exit_scores" not in str(path)
        return original_read(path, *args)

    monkeypatch.setattr(coordinator, "read_json", guard_read)
    job = SimpleNamespace(
        cancelled=False, save=save, status=lambda *args, **kwargs: None, event=lambda *args: None
    )
    result = coordinator.launch_families(
        tmp_path,
        {
            "version": 2,
            "families": [{"name": "snaive"}, {"name": "ridge"}],
            "time_budget_s": 60,
            "nthread": 2,
            "max_concurrent_families": 2,
        },
        job,
    )
    exits = [command for command in commands if command[2] == "smolsmort.forecast.exit_test"]
    assert exits[0][-2] == "snaive"
    assert sum(command[-2] == "snaive" for command in exits) == 1
    assert states_at_exit[0]["ridge"]["state"] == "running"
    if exit_hangs:
        exit_index = commands.index(exits[0])
        assert any(
            command[2] == "smolsmort.forecast.family_worker"
            for command in commands[exit_index + 1 :]
        )
    assert result["finalized"]
    assert clock[0] < 160


def test_search_prune_and_rank_have_no_exit_artifact_reads():
    # keep held-out artifacts out of validation selection, even as scheduling evolves
    assert "exit_test" not in inspect.getsource(search)
    assert "exit_scores" not in inspect.getsource(search)
    source = inspect.getsource(coordinator._launch_families)
    prune = source.split("    def prune_families():", 1)[1].split("    def run_slices", 1)[0]
    for text in (
        prune,
        inspect.getsource(coordinator.rank_families),
        inspect.getsource(coordinator.best_entry),
    ):
        assert "exit_test" not in text and "exit_scores" not in text


@pytest.mark.parametrize(
    ("version", "mode", "enabled"),
    [(2, "entity", False), (1, "timeseries", False), (2, "timeseries", True)],
)
def test_only_v2_timeseries_schedule_exit_subprocess(tmp_path, monkeypatch, version, mode, enabled):
    def launch(folder, request, job, scorer):
        assert scorer.enabled is enabled
        return "complete"

    monkeypatch.setattr(coordinator, "_launch_families", launch)
    assert (
        coordinator.launch_families(tmp_path, {"version": version, "spec": {"mode": mode}}, None)
        == "complete"
    )


@pytest.mark.parametrize("terminal", ["done", "pruned"])
def test_terminal_exit_runs_once_and_completed_event_survives_resume(
    tmp_path, monkeypatch, terminal
):
    entry = {
        "genome": pipeline.make_genome(["lag"], "squared", {}, "ridge").to_dict(),
        "fitness": 0.2,
        "families": "lag",
        "generation": 1,
    }
    save_json(
        tmp_path / "coordinator.json",
        {
            "started_at": 100,
            "deadline": 160,
            "phase": "finish",
            "families": {"ridge": {"state": terminal}},
        },
    )
    save_json(tmp_path / "families" / "ridge" / "checkpoint.json", {"entries": [entry]})
    save_json(tmp_path / "families" / "ridge" / "leaderboard.json", [{**entry, "fitness": 0.4}])
    spawned = []

    class Process:
        pid = 123
        returncode = 0

        def __init__(self, command, **kwargs):
            spawned.append(command[2])
            if command[2] == "smolsmort.forecast.exit_test":
                board = json.loads(
                    (tmp_path / "families" / "ridge" / "leaderboard.json").read_text()
                )
                assert board[0]["fitness"] == entry["fitness"]
            if command[2] == "smolsmort.forecast.ensemble":
                save_json(tmp_path / "finish_complete.json", {})

        def poll(self):
            return self.returncode

    monkeypatch.setattr(coordinator, "describe_machine", lambda *args: {})
    monkeypatch.setattr(coordinator.time, "time", lambda: 100)
    monkeypatch.setattr(coordinator.time, "sleep", lambda *args: None)
    monkeypatch.setattr(coordinator.subprocess, "Popen", Process)
    job = SimpleNamespace(
        cancelled=False,
        save=lambda name, value: save_json(tmp_path / name, value),
        status=lambda *args, **kwargs: None,
    )
    request = {"version": 2, "families": [{"name": "ridge"}], "time_budget_s": 60}
    coordinator.launch_families(tmp_path, request, job)
    assert spawned == ["smolsmort.forecast.exit_test", "smolsmort.forecast.ensemble"]
    (tmp_path / "finish_complete.json").unlink()
    coordinator.launch_families(tmp_path, request, job)
    assert spawned.count("smolsmort.forecast.exit_test") == 1
