"""the search: reproducible, stops on a plateau, warm-starts, and runs end to end in its own
process with a verdict at the end"""

from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from forecast_fixtures import write_panel, write_rows

from smolsmort.forecast import pipeline, worker
from smolsmort.forecast import runs as run_store
from smolsmort.forecast.pipeline import Scored, load_workspace, make_genome, score
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.runs import cancel_run, run_state, start_run, wait_run
from smolsmort.forecast.search import Budget, search
from smolsmort.forecast.spec import Censor, Column, PrepSpec
from smolsmort.forecast.tables import read_table

pytest.importorskip("xgboost")
pytest.importorskip("duckdb")

TINY = {"population": 6, "plateau": 2, "max_generations": 3, "time_cap": 120, "seed": 0}


def _series(folder):
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
    return spec, prepare(spec, folder / "cache"), truth


def _rows(folder, n=3000):
    truth = write_rows(folder, n=n)
    spec = PrepSpec(
        str(truth.path),
        "row",
        "regression",
        (
            Column("created", "anchor"),
            Column("branch", "dimension"),
            Column("size", "measure"),
            Column("planned_offset", "measure"),
            Column("leak_plus1", "measure"),
            Column("line", "ignore"),
            Column("lead_weeks", "target"),
        ),
        censor=Censor("created"),
    )
    return spec, prepare(spec, folder / "cache"), truth


@pytest.fixture(scope="module")
def series_ws(tmp_path_factory):
    spec, prepared, truth = _series(tmp_path_factory.mktemp("series"))
    return load_workspace(spec, prepared.folder, prepared.summary), truth


@pytest.fixture(scope="module")
def row_ws(tmp_path_factory):
    spec, prepared, truth = _rows(tmp_path_factory.mktemp("rows"))
    return load_workspace(spec, prepared.folder, prepared.summary), truth


def test_a_genome_scores_in_both_modes_and_the_leak_is_never_offered(
    series_ws, row_ws, monkeypatch
):
    row_fit, series_fit = pipeline._row_fit, pipeline._series_fit

    def check_row_fit(ws, genome, rows, phase, nthread):
        assert phase == "val" and np.array_equal(rows, ws.masks["train"])
        assert not np.any(rows & ws.masks["test"])
        return row_fit(ws, genome, rows, phase, nthread)

    def check_series_fit(ws, genome, frame, rows, nthread):
        bucket = next(key for key, value in ws.frames.items() if value is frame)
        assert np.array_equal(rows, ws.series_masks[bucket]["train"])
        assert not np.any(rows & ws.series_masks[bucket]["test"])
        return series_fit(ws, genome, frame, rows, nthread)

    monkeypatch.setattr(pipeline, "_row_fit", check_row_fit)
    monkeypatch.setattr(pipeline, "_series_fit", check_series_fit)
    for ws, _ in (series_ws, row_ws):
        genome = make_genome(ws.families, "squared", {"max_depth": 4, "eta": 0.1})
        result = score(ws, genome)
        assert np.isfinite(result.fitness) and result.fitness == pytest.approx(result.contrib.sum())
        assert result.eval_history["metric"] == "rmse"
        assert len(result.eval_history["training"]) == len(result.eval_history["validation"])
        if ws.mode == "series":
            assert result.eval_history["bucket"] == list(next(iter(ws.frames)))
    ws, truth = row_ws
    assert not [f for f in ws.families if truth.leak in f]
    assert "aft" in ws.profile["genes"]["objectives"]


def test_a_fixed_seed_gives_the_same_leaderboard(series_ws):
    ws, _ = series_ws
    first = search(ws, Budget(**TINY))
    second = search(ws, Budget(**TINY))
    assert [e["key"] for e in first["leaderboard"]] == [e["key"] for e in second["leaderboard"]]
    assert [e["fitness"] for e in first["leaderboard"]] == [
        e["fitness"] for e in second["leaderboard"]
    ]


def test_the_search_stops_on_a_plateau_before_the_cap(series_ws):
    ws, _ = series_ws
    result = search(ws, Budget(**{**TINY, "max_generations": 30, "plateau": 1}))
    assert result["stopped"] == "plateau" and result["generations"] < 30


def test_a_warm_start_seeds_generation_zero(series_ws):
    ws, _ = series_ws
    earlier = make_genome(ws.families[:3], "absolute", {"max_depth": 3, "eta": 0.2})
    result = search(ws, Budget(**{**TINY, "max_generations": 1}), warm=[earlier.to_dict()])
    keys = {e["key"]: e for e in result["leaderboard"]}
    assert earlier.key() in keys and keys[earlier.key()]["generation"] == 0


def test_a_run_ends_with_a_forecast_and_a_verdict(tmp_path):
    spec, prepared, truth = _series(tmp_path)
    run_id = start_run(tmp_path / "runs", spec, prepared, budget=TINY, nthread=1)
    state = wait_run(tmp_path / "runs", run_id, timeout=300)
    assert state["state"] == "done", state
    folder = tmp_path / "runs" / run_id
    history = state["eval_history"]
    board = json.loads((folder / "leaderboard.json").read_text())
    assert history["key"] == board[0]["key"]
    assert history == json.loads((folder / "eval_history.json").read_text())
    assert len(history["training"]) == len(history["validation"]) > 0
    assert not any("eval_history" in entry for entry in board + state["generations"])
    curves = state["generation_curves"]
    assert [curve["generation"] for curve in curves] == [
        event["generation"] for event in state["generations"]
    ]
    assert all(curve["validation"] and "training" not in curve for curve in curves)
    assert run_state(tmp_path / "runs", run_id)["generation_curves"] == curves
    result = json.loads((folder / "result.json").read_text())
    assert set(result["verdict"]) == {"trusted", "summary", "checks"}
    forecast = read_table(folder / "forecast.parquet")
    assert len(forecast["prediction"]) == truth.series * spec.horizon
    assert np.all(forecast["lower"] <= forecast["upper"])
    assert json.loads((folder / "recipe.json").read_text())["genome"]["families"]


def test_a_row_run_predicts_every_open_row(tmp_path):
    spec, prepared, truth = _rows(tmp_path, n=2000)
    run_id = start_run(tmp_path / "runs", spec, prepared, budget=TINY, nthread=1)
    state = wait_run(tmp_path / "runs", run_id, timeout=300)
    assert state["state"] == "done", state
    assert state["eval_history"]["training"] and state["eval_history"]["validation"]
    forecast = read_table(tmp_path / "runs" / run_id / "forecast.parquet")
    assert len(forecast["prediction"]) == prepared.summary["predict_rows"]
    assert "predicted_date" in forecast


def test_a_cancelled_run_still_leaves_a_readable_leaderboard(tmp_path):
    spec, prepared, _ = _series(tmp_path)
    budget = {**TINY, "max_generations": 200, "plateau": 200, "population": 8}
    runs = tmp_path / "runs"
    run_id = start_run(runs, spec, prepared, budget=budget, nthread=1)
    for _ in range(600):
        live_state = run_state(runs, run_id)
        if live_state["generations"]:
            assert live_state["eval_history"]["training"]
            live_board = json.loads((runs / run_id / "leaderboard.json").read_text())
            assert 0 < len(live_board) <= 10
            assert live_board[0]["fitness"] <= live_state["best_error"]
            break
        time.sleep(0.1)
    cancel_run(runs, run_id)
    state = wait_run(runs, run_id, timeout=300)
    assert state["state"] == "cancelled" and state["stopped"] == "cancelled", state
    assert state["progress"]["candidate"] > 0
    assert not (runs / run_id / "forecast.parquet").exists()
    board = json.loads((runs / run_id / "leaderboard.json").read_text())
    assert board and board[0]["fitness"] is not None


def test_the_worker_refuses_to_share_a_process_with_torch(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(SystemExit, match="must not share a process with torch"):
        worker.run(tmp_path)


def test_a_saved_recipe_refits_on_newer_history_without_a_search(tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    spec, prepared, truth = _series(tmp_path / "old")
    runs = tmp_path / "runs"
    first = start_run(runs, spec, prepared, budget=TINY, nthread=1)
    assert wait_run(runs, first, timeout=300)["state"] == "done"
    recipe = json.loads((runs / first / "recipe.json").read_text())["genome"]
    newer = write_panel(tmp_path / "new", weeks=truth.weeks + 8)
    spec_new = PrepSpec(newer.path.as_posix(), spec.mode, spec.task, spec.columns)
    prepared_new = prepare(spec_new, tmp_path / "new" / "cache")
    started = time.monotonic()
    refit = start_run(runs, spec_new, prepared_new, recipe=recipe, nthread=1)
    state = wait_run(runs, refit, timeout=300)
    assert state["state"] == "done" and state["stopped"] == "refit", state
    assert not state["generations"] and not (runs / refit / "leaderboard.json").exists()
    assert state["eval_history"] is None
    assert state["generation_curves"] == []
    assert time.monotonic() - started < 60
    forecast = read_table(runs / refit / "forecast.parquet")
    last_old = read_table(runs / first / "forecast.parquet")["step"].max()
    assert forecast["step"].min() > last_old, "the forecast moved forward with the new history"


def test_best_history_replaced_before_generation_events_and_cleared_when_unusable(
    tmp_path, monkeypatch
):
    import importlib

    search_module = importlib.import_module("smolsmort.forecast.search")
    ws = SimpleNamespace(
        mode="row",
        masks={"train": np.ones(20, dtype=bool)},
        families=["measure:x"],
        profile={"genes": {"objectives": ["squared"]}},
    )
    genomes = [
        make_genome(ws.families, "squared", {"max_depth": depth}) for depth in (6, 7, 8, 9, 4, 2, 1)
    ]
    children = iter(genomes[4:])
    monkeypatch.setattr(search_module, "_seed_population", lambda *_: genomes[:4])
    monkeypatch.setattr(search_module._Variation, "mutate", lambda *_: next(children))

    def fake_score(_ws, genome, **_kwargs):
        depth = dict(genome.params)["max_depth"]
        history = {"metric": "rmse", "training": [float(depth)], "validation": [depth + 0.5]}
        return Scored(float(depth), np.array([float(depth)]), {}, history if depth > 2 else None)

    monkeypatch.setattr(search_module, "score", fake_score)
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    job = worker._Run(tmp_path)
    job.status("done")
    snapshots = []

    def record_event(event):
        job.event(event)
        if event["event"] == "generation":
            snapshots.append(run_state(tmp_path.parent, tmp_path.name)["eval_history"])

    search(
        ws, Budget(population=4, max_generations=3), on_best=job.record_best, on_event=record_event
    )
    assert [snapshot["key"] if snapshot else None for snapshot in snapshots] == [
        genomes[0].key(),
        genomes[4].key(),
        None,
    ]
    assert snapshots[0]["training"] == [6.0]
    assert snapshots[1]["training"] == [4.0]
    assert json.loads((tmp_path / "eval_history.json").read_text()) is None
    assert not (tmp_path / "eval_history.json.tmp").exists()


def test_generation_curves_use_the_generation_winner_even_without_a_new_best(monkeypatch):
    import importlib

    search_module = importlib.import_module("smolsmort.forecast.search")
    ws = SimpleNamespace(
        mode="row",
        masks={"train": np.ones(20, dtype=bool)},
        families=["measure:x"],
        profile={"genes": {"objectives": ["squared"]}},
    )
    genomes = [make_genome(ws.families, "squared", {"max_depth": depth}) for depth in range(6, 14)]
    children = iter(genomes[4:] * 2)
    monkeypatch.setattr(search_module, "ELITE", 0)
    monkeypatch.setattr(search_module, "_seed_population", lambda *_: genomes[:4])
    monkeypatch.setattr(search_module._Variation, "mutate", lambda *_: next(children))

    def fake_score(_ws, genome, **_kwargs):
        depth = dict(genome.params)["max_depth"]
        history = {
            "metric": "rmse",
            "bucket": [1, 3],
            "training": [float(depth)],
            "validation": [depth + 0.5],
        }
        return Scored(float(depth), np.array([float(depth)]), {}, history)

    monkeypatch.setattr(search_module, "score", fake_score)
    curves, events = [], []
    search(
        ws,
        Budget(population=4, max_generations=3, plateau=2),
        on_generation=curves.append,
        on_event=events.append,
    )
    assert [event["best"] for event in events if event["event"] == "generation"] == [6.0] * 3
    assert curves == [
        {
            "generation": generation,
            "key": genomes[index].key(),
            "metric": "rmse",
            "bucket": [1, 3],
            "validation": [value],
        }
        for generation, index, value in [(0, 0, 6.5), (1, 4, 10.5), (2, 4, 10.5)]
    ]


def test_cached_generation_winners_keep_their_validation_curves(series_ws):
    ws, _ = series_ws
    curves = []
    search(ws, Budget(**TINY), on_generation=curves.append)
    assert len(curves) >= 2
    assert all(curve["validation"] for curve in curves)
    assert len({curve["generation"] for curve in curves}) == len(curves)


def test_worker_records_generation_curves_once_and_polling_returns_only_unseen(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    job = worker._Run(tmp_path)
    job.status("done")
    job.record_best({"training": [2.0], "validation": [3.0]})
    curves = [
        {
            "generation": generation,
            "key": str(generation),
            "metric": "rmse",
            "bucket": None,
            "validation": [float(generation)],
        }
        for generation in range(3)
    ]
    for curve in curves:
        job.record_generation(curve)
        job.record_generation(curve)
    job.save("leaderboard_generation.json", 2)
    path = tmp_path / "generation_curves.jsonl"
    assert len(path.read_text().splitlines()) == 3
    state = run_state(tmp_path.parent, tmp_path.name, since_generation=0)
    assert state["state"] == "done"
    assert state["generation_curves"] == curves[1:]
    assert "eval_history" not in state
    assert run_state(tmp_path.parent, tmp_path.name, since_generation=2)["generation_curves"] == []
    with path.open("a") as stream:
        stream.write('{"generation":3')
    assert (
        run_state(tmp_path.parent, tmp_path.name, since_generation=0)["generation_curves"]
        == curves[1:]
    )


def test_search_does_not_record_generations_without_validation_history(series_ws, monkeypatch):
    import importlib

    search_module = importlib.import_module("smolsmort.forecast.search")
    monkeypatch.setattr(
        search_module, "score", lambda *_args, **_kwargs: Scored(1.0, np.array([1.0]), {}, None)
    )
    curves = []
    search(series_ws[0], Budget(**TINY), on_generation=curves.append)
    assert curves == []


def test_worker_publishes_bounded_atomic_leaderboards_before_generation_events(
    tmp_path, monkeypatch
):
    import importlib

    search_module = importlib.import_module("smolsmort.forecast.search")
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    job = worker._Run(tmp_path)
    job.status("done")
    ws = SimpleNamespace(
        mode="row",
        masks={"train": np.ones(20, dtype=bool)},
        families=["measure:x"],
        profile={"genes": {"objectives": ["squared"]}},
    )
    genomes = [make_genome(ws.families, "squared", {"max_depth": n}) for n in range(12, 0, -1)]
    monkeypatch.setattr(search_module, "_seed_population", lambda *_: genomes)
    monkeypatch.setattr(search_module._Variation, "crossover", lambda _self, a, _b: a)
    monkeypatch.setattr(search_module._Variation, "mutate", lambda _self, genome: genome)
    monkeypatch.setattr(
        search_module,
        "score",
        lambda _ws, genome, **_kwargs: Scored(
            float(dict(genome.params)["max_depth"]), np.array([1.0]), {}, None
        ),
    )
    snapshots = []

    def record_event(event):
        if event["event"] != "generation":
            job.event(event)
            return
        board = json.loads((tmp_path / "leaderboard.json").read_text())
        revision = json.loads((tmp_path / "leaderboard_generation.json").read_text())
        assert revision == event["generation"]
        assert len(board) == 10
        assert [entry["fitness"] for entry in board] == list(range(1, 11))
        assert not any("eval_history" in entry for entry in board)
        assert not list(tmp_path.glob("*.tmp"))
        snapshots.append(board)
        job.event(event)

    result = search(
        ws,
        Budget(population=12, max_generations=2, plateau=10),
        on_leaderboard=job.record_leaderboard,
        on_event=record_event,
    )
    assert len(snapshots) == 2
    assert snapshots[-1] == result["leaderboard"]


def test_run_summaries_expose_kind_progress_stop_reason_and_test_error(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "_alive", lambda folder: folder.name == "20260103")
    requests = [
        ("20260101", {"recipe": {"objective": "squared"}}, "refit", "done"),
        ("20260102", {"warm_from": "/previous"}, "warm start", "done"),
        ("20260103", {}, "search", "running"),
        ("20260104", {}, "search", "failed"),
    ]
    for run_id, extra, _kind, state in requests:
        folder = tmp_path / run_id
        folder.mkdir()
        request = {"spec": {"task": "regression"}, "budget": {"plateau": 5}, **extra}
        (folder / "request.json").write_text(json.dumps(request))
        (folder / "status.json").write_text(
            json.dumps({"state": state, "reason": "bad fit" if state == "failed" else None})
        )
        events = [
            {"event": "generation", "generation": 0, "best": 0.4, "elapsed": 4.0},
            {"event": "finishing", "stopped": "plateau", "elapsed": 5.0},
        ]
        (folder / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
        if state == "done":
            (folder / "result.json").write_text(json.dumps({"model_error": 0.3}))
    summaries = run_store.list_runs(tmp_path)
    assert [summary["id"] for summary in summaries] == [
        "20260104",
        "20260103",
        "20260102",
        "20260101",
    ]
    by_id = {summary["id"]: summary for summary in summaries}
    for run_id, _extra, kind, _state in requests:
        summary = by_id[run_id]
        assert summary["kind"] == kind
        assert summary["generation_count"] == 1 and summary["generation"] == 0
        assert summary["best_error"] == 0.4
        assert summary["budget"] == {"plateau": 5}
        assert summary["task"] == "regression"
        assert summary["stopped"] == "plateau" and summary["elapsed"] == 5.0
        assert "generation_curves" not in summary and "generations" not in summary
    assert by_id["20260101"]["test_error"] == 0.3
    assert by_id["20260103"]["test_error"] is None
    assert by_id["20260104"]["reason"] == "bad fit"


@pytest.mark.parametrize(
    ("task", "recipe", "events", "expected"),
    [
        ("regression", None, [{"best": 0.3}], 0.3),
        ("regression", None, [{"best": 0.3}, {"best": 0.2, "metrics": {"wape": 0.15}}], 0.15),
        ("regression", None, [{"best": 0.3}, {"best": 0.0, "metrics": {"wape": 0.0}}], 0.0),
        ("regression", None, [], None),
        ("regression", {"objective": "squared"}, [{"best": 0.3}], None),
        ("classification", None, [{"best": 0.3, "metrics": {"log_loss": 0.3}}], None),
        ("classification", None, [{"best": 0.3, "metrics": {"wape": 0.2}}], 0.2),
        ("regression", None, [{"best": None, "metrics": {"wape": None}}], None),
    ],
)
def test_run_summaries_read_latest_validation_wape_from_events(
    tmp_path, monkeypatch, task, recipe, events, expected
):
    monkeypatch.setattr(run_store, "_alive", lambda folder: True)
    folder = tmp_path / "old-run"
    folder.mkdir()
    (folder / "request.json").write_text(json.dumps({"spec": {"task": task}, "recipe": recipe}))
    (folder / "events.jsonl").write_text(
        "\n".join(
            json.dumps({"event": "generation", "generation": generation, **event})
            for generation, event in enumerate(events)
        )
    )
    for status in ("running", "done"):
        (folder / "status.json").write_text(json.dumps({"state": status}))
        state = run_store.run_state(tmp_path, folder.name)
        assert state["best_wape"] == expected
        assert state["test_error"] is None
        summary = run_store.list_runs(tmp_path)[0]
        assert summary["best_wape"] == expected and summary["state"] == status


def test_progress_events_follow_each_candidate_and_stop_at_cancellation(monkeypatch):
    import importlib

    search_module = importlib.import_module("smolsmort.forecast.search")
    ws = SimpleNamespace(
        mode="row",
        masks={"train": np.ones(20, dtype=bool)},
        families=["measure:x"],
        profile={"genes": {"objectives": ["squared"]}},
    )
    genomes = [make_genome(ws.families, "squared", {"max_depth": n}) for n in range(1, 5)]
    monkeypatch.setattr(search_module, "_seed_population", lambda *_: genomes)
    scored = []

    def score_candidate(_ws, genome, **_kwargs):
        scored.append(genome.key())
        if len(scored) == 2:
            raise search_module.ModelError("bad candidate")
        return Scored(1.0, np.array([1.0]), {}, None)

    monkeypatch.setattr(search_module, "score", score_candidate)
    events = []
    result = search(
        ws,
        Budget(population=4, max_generations=2),
        on_event=events.append,
        stop=lambda: len(scored) >= 3,
    )
    progress = [event for event in events if event["event"] == "progress"]
    assert [event["candidate"] for event in progress] == [1, 2, 3]
    assert all(event["generation"] == 0 and event["population"] == 4 for event in progress)
    assert all(event["flat_generations"] == 0 for event in progress)
    assert all(
        set(event)
        == {"event", "generation", "candidate", "population", "flat_generations", "elapsed"}
        for event in progress
    )
    assert result["stopped"] == "cancelled" and result["best"] is not None
    assert events[-1]["flat_generations"] == 0


def test_progress_polls_advance_elapsed_and_terminal_states_freeze_it(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    monkeypatch.setattr(run_store, "_alive", lambda _folder: True)
    monkeypatch.setattr(worker.time, "time", lambda: 100.0)
    job = worker._Run(tmp_path)
    job.status("running", started_at=90.0)
    event = {
        "event": "progress",
        "generation": 2,
        "candidate": 7,
        "population": 10,
        "flat_generations": 2,
        "elapsed": 54.0,
    }
    job.event(event)
    assert json.loads((tmp_path / "events.jsonl").read_text()) == event
    assert json.loads((tmp_path / "progress.json").read_text())["updated_at"] == 100.0
    monkeypatch.setattr(run_store.time, "time", lambda: 103.0)
    state = run_state(tmp_path.parent, tmp_path.name, since_generation=2)
    assert state["progress"] == {key: value for key, value in event.items() if key != "event"} | {
        "elapsed": 57.0,
    }
    for terminal in ("done", "cancelled", "failed"):
        job.status(terminal)
        assert run_state(tmp_path.parent, tmp_path.name)["progress"]["elapsed"] == 57.0


@pytest.mark.parametrize("sampled", [False, True])
def test_progress_includes_cached_candidates_and_full_sample_rescores(monkeypatch, sampled):
    import importlib

    search_module = importlib.import_module("smolsmort.forecast.search")
    ws = SimpleNamespace(
        mode="row",
        masks={"train": np.ones(20, dtype=bool)},
        families=["measure:x"],
        profile={"genes": {"objectives": ["squared"]}},
    )
    genomes = [make_genome(ws.families, "squared", {"max_depth": n}) for n in range(1, 5)]
    monkeypatch.setattr(search_module, "ELITE", 4)
    monkeypatch.setattr(search_module, "_seed_population", lambda *_: genomes)
    monkeypatch.setattr(search_module, "_sample", lambda *_: ws.masks["train"] if sampled else None)
    scores = []

    def score_candidate(_ws, genome, **kwargs):
        scores.append((genome.key(), kwargs["sample"] is not None))
        return Scored(1.0, np.array([1.0]), {}, None)

    monkeypatch.setattr(search_module, "score", score_candidate)
    events = []
    search(ws, Budget(population=4, max_generations=2), on_event=events.append)
    progress = [event for event in events if event["event"] == "progress"]
    candidates = [1, 2, 3, 4, 4] if sampled else [1, 2, 3, 4]
    assert [event["candidate"] for event in progress] == candidates * 2
    assert [event["generation"] for event in progress] == [0] * len(candidates) + [1] * len(
        candidates
    )
    assert len(scores) == (5 if sampled else 4)


def test_unchanged_poll_does_not_read_generation_curves(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    job = worker._Run(tmp_path)
    job.status("done")
    job.save("leaderboard_generation.json", 2)
    job.record_generation({"generation": 2, "validation": [0.3], "metric": "rmse"})
    from pathlib import Path

    read_text = Path.read_text

    def read_without_curves(path, *args, **kwargs):
        assert path.name != "generation_curves.jsonl"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_without_curves)
    state = run_state(tmp_path.parent, tmp_path.name, since_generation=2)
    assert state["generation_curves"] == []
