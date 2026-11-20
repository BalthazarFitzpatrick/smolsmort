"""saved-run builders shared by the export tests; registered as fixtures through conftest"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from smolsmort.forecast import tiles
from smolsmort.forecast.tables import write_table


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def saved_run(tmp_path):
    pytest.importorskip("duckdb")
    folder = tmp_path / "20261115-155627-6f006a"
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    request = {
        "version": 2,
        "prepared": str(prepared),
        "spec": {
            "source": str(tmp_path / "demand.csv"),
            "mode": "timeseries",
            "unit": "week",
            "horizon": 2,
            "columns": [
                {"name": "week", "role": "time"},
                {"name": "units", "role": "target"},
                {"name": "region", "role": "dimension"},
            ],
        },
        "families": [{"name": "ridge"}, {"name": "snaive"}],
        "seed": 17,
        "holdout": {"test": 2, "validation": 2},
    }
    write_json(folder / "request.json", request)
    write_table(prepared / "panel.parquet", {"__series": ["north"], "region": ["north"]})
    finalists = []
    for family, prediction in (("ridge", 8.0), ("snaive", 6.0)):
        genome = {
            "family": family,
            "objective": "squared",
            "params": {"alpha": 0.75} if family == "ridge" else {},
            "transform": "log1p",
            "families": [
                "target@time:lag:1",
                "target@time:roll:4:mean",
                "time:cal:month",
                "promo@time:lead",
                "price:measure",
                "region:dim",
            ],
        }
        key = tiles.recipe_key(genome)
        finalists.append({"family": family, "key": key, "genome": genome, "seed": 17})
        target = folder / "families" / family
        write_json(target / "winner.json", genome)
        write_json(target / "leaderboard.json", [{"fitness": 0.2, "genome": genome}])
        write_json(target / "exit_test.json", {"recipe_key": key, "fold_count": 2})
        for phase in ("validation", "test", "forecast"):
            columns = {
                "series": ["north", "north"],
                "origin": np.array(["2026-01-05", "2026-01-12"], dtype="datetime64[D]"),
                "step": np.array(["2026-01-12", "2026-01-19"], dtype="datetime64[D]"),
                "weeks_ahead": [1, 1],
                "prediction": [prediction, prediction],
                "actual": [10.0, 10.0] if phase != "forecast" else [np.nan, np.nan],
                "lower": [prediction - 1, prediction - 1],
                "upper": [prediction + 1, prediction + 1],
            }
            write_table(target / f"{phase}.parquet", columns)
        write_table(target / "lead_errors.parquet", {"weeks_ahead": [1], "error": [2.0]})
    write_json(folder / "finalists.json", finalists)
    write_json(folder / "recipe.json", {"family": "ridge", "genome": finalists[0]["genome"]})
    write_json(folder / "finish_complete.json", {})
    write_json(
        folder / "result.json",
        {
            "winner": {"family": "ridge", "fitness": 0.2},
            "ensemble": {"members": ["ridge", "snaive"], "fitness": 0.3},
            "holdout": {"validation_start": "2026-01-05", "test_start": "2026-01-12"},
            "families": {"ridge": {"state": "done", "fold_count": 2, "fitness": 0.2}},
        },
    )
    models = tiles.read_tiles(folder, include_ensembles=False)
    ensemble = tiles.ensemble_rows(folder)[0]
    tiles.write_ensemble_views(folder, [ensemble])
    write_json(folder / "ensembles.json", [ensemble])
    return folder, request, models, ensemble


@pytest.fixture
def input_run(saved_run, tmp_path):
    pytest.importorskip("duckdb")
    from smolsmort.forecast.prep import prepare
    from smolsmort.forecast.spec import spec_from_dict

    folder, request, models, ensemble = saved_run
    source = Path(request["spec"]["source"])
    source.write_text(
        "week,region,units,keep\n"
        "2026-01-06,north,2,yes\n"
        "2026-01-07,north,3,yes\n"
        "2026-01-20,north,7,yes\n"
        "2026-01-06,south,11,yes\n"
        "2026-01-20,south,13,yes\n"
        "2026-01-13,north,100,no\n"
        "2026-01-27,north,200,yes\n",
        encoding="utf-8",
    )
    request["spec"].pop("unit")
    request["spec"].update(
        task="regression", step="week", horizon=1, where="keep = 'yes'", as_of="2026-01-28"
    )
    request["spec"]["columns"][1]["aggregation"] = "sum"
    prepared = prepare(spec_from_dict(request["spec"]), tmp_path / "cache")
    request["prepared"] = str(prepared.folder)
    write_json(folder / "request.json", request)
    return folder, request, models, ensemble


def real_run_folder(run_id: str) -> Path:
    """a saved run from a real workspace; opt in by naming its .forecast-runs folder in
    SMOLSMORT_REAL_RUNS, otherwise the test skips with that reason"""
    root = os.environ.get("SMOLSMORT_REAL_RUNS")
    if not root:
        pytest.skip("real-data test; set SMOLSMORT_REAL_RUNS to a .forecast-runs folder")
    folder = Path(root) / run_id
    if not folder.is_dir():
        pytest.skip(f"run {run_id} not found under SMOLSMORT_REAL_RUNS")
    return folder
