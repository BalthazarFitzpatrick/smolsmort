"""the search: reproducible, stops on a plateau, warm-starts, and runs end to end in its own
process with a verdict at the end"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pytest
from forecast_fixtures import write_panel, write_rows

from smolsmort.forecast import worker
from smolsmort.forecast.pipeline import load_workspace, make_genome, score
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


def test_a_genome_scores_in_both_modes_and_the_leak_is_never_offered(series_ws, row_ws):
    for ws, _ in (series_ws, row_ws):
        genome = make_genome(ws.families, "squared", {"max_depth": 4, "eta": 0.1})
        result = score(ws, genome)
        assert np.isfinite(result.fitness) and result.fitness == pytest.approx(result.contrib.sum())
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
    forecast = read_table(tmp_path / "runs" / run_id / "forecast.parquet")
    assert len(forecast["prediction"]) == prepared.summary["predict_rows"]
    assert "predicted_date" in forecast


def test_a_cancelled_run_still_leaves_a_readable_leaderboard(tmp_path):
    spec, prepared, _ = _series(tmp_path)
    budget = {**TINY, "max_generations": 200, "plateau": 200, "population": 8}
    runs = tmp_path / "runs"
    run_id = start_run(runs, spec, prepared, budget=budget, nthread=1)
    for _ in range(600):
        if run_state(runs, run_id)["generations"]:
            break
        time.sleep(0.1)
    cancel_run(runs, run_id)
    state = wait_run(runs, run_id, timeout=300)
    assert state["state"] == "done" and state["stopped"] == "cancelled", state
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
    assert time.monotonic() - started < 60
    forecast = read_table(runs / refit / "forecast.parquet")
    last_old = read_table(runs / first / "forecast.parquet")["step"].max()
    assert forecast["step"].min() > last_old, "the forecast moved forward with the new history"
