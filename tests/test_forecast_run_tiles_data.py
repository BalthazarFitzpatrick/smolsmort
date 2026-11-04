"""raw tile contract and incremental recipe refits on synthetic rows"""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import plan_sets, runs, tiles
from smolsmort.forecast.tables import write_table


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def save_model(folder, family, prediction):
    genome = {"family": family, "families": ["y:lag"], "objective": "squared", "params": {}}
    target = folder / "families" / family
    save_json(target / "winner.json", genome)
    save_json(target / "status.json", {"state": "done", "generation": 2})
    save_json(target / "leaderboard.json", [{"genome": genome, "fitness": 0.2, "generation": 1}])
    table = {
        "series": np.array(["a", "a"]),
        "step": np.array([1, 2]),
        "actual": np.array([10.0, 20.0]),
        "prediction": np.array(prediction, dtype=float),
    }
    for split in ("validation", "test", "forecast"):
        write_table(target / f"{split}.parquet", table)
    return genome


def test_picker_metrics_keep_saved_search_score_and_each_models_verdict(tmp_path):
    from smolsmort.forecast import views

    save_json(
        tmp_path / "request.json",
        {
            "version": 2,
            "spec": {"task": "regression", "mode": "timeseries"},
            "families": [{"name": "ridge"}, {"name": "snaive"}],
        },
    )
    save_json(tmp_path / "status.json", {"state": "done"})
    save_json(
        tmp_path / "result.json",
        {
            "winner": {"family": "ridge", "verdict": {"accuracy": {"ratio": 0.7}}},
            "verdict": {"accuracy": {"ratio": 0.1}},
        },
    )
    save_model(tmp_path, "ridge", [8, 18])
    save_model(tmp_path, "snaive", [10, 20])
    models = {tile["family"]: tile for tile in tiles.read_tiles(tmp_path, include_ensembles=False)}
    ridge = models["ridge"]["error_metrics"]
    assert ridge["validation_error"]["value"] == 0.2
    assert ridge["wape"]["value"] == pytest.approx(4 / 30)
    assert ridge["mae"]["value"] == 2
    assert ridge["mape"]["value"] == pytest.approx(0.15)
    assert ridge["test_ratio"]["value"] == 0.7
    assert models["snaive"]["error_metrics"]["test_ratio"]["value"] is None
    assert models["snaive"]["error_metrics"]["wape"]["value"] == 0
    request = tiles.read_json(tmp_path / "request.json", {})
    for tile in models.values():
        payload = views.run_metric_payload(tmp_path, request, model=tile["tile_id"])
        assert payload["error_metrics"] == tile["error_metrics"]


def test_legacy_tile_picker_metrics_read_root_artifacts(tmp_path):
    save_json(tmp_path / "request.json", {"spec": {"task": "regression"}})
    save_json(tmp_path / "status.json", {"state": "done"})
    genome = {"family": "ridge", "params": {"alpha": 1}}
    save_json(tmp_path / "recipe.json", {"genome": genome})
    save_json(tmp_path / "leaderboard.json", [{"genome": genome, "fitness": 0.42}])
    save_json(tmp_path / "result.json", {"verdict": {"accuracy": {"ratio": 0.8}}})
    write_table(
        tmp_path / "test.parquet",
        {"actual": np.array([0.0, 4.0, -2.0]), "prediction": np.array([1.0, 3.0, -1.0])},
    )
    tile = tiles.read_tiles(tmp_path)[0]
    assert tile["error_metrics"]["validation_error"]["value"] == 0.42
    assert tile["error_metrics"]["test_ratio"]["value"] == 0.8
    assert tile["error_metrics"]["mape"]["value"] == 0.25
    assert tile["error_metrics"]["mape"]["excluded_rows"] == 2


@pytest.mark.parametrize("missing", ["validation", "test", "forecast"])
def test_results_only_offer_tiles_with_every_required_prediction_split(tmp_path, missing):
    from smolsmort.forecast import views

    save_json(
        tmp_path / "request.json",
        {"version": 2, "families": [{"name": "ridge"}, {"name": "snaive"}]},
    )
    save_model(tmp_path, "ridge", [8, 18])
    save_model(tmp_path, "snaive", [10, 20])
    saved = tiles.read_tiles(tmp_path)
    (tmp_path / "families/ridge/forecast.parquet").replace(tmp_path / "winner_forecast.parquet")
    (tmp_path / "families/snaive" / f"{missing}.parquet").replace(
        tmp_path / f"unused_{missing}.parquet"
    )
    marked = views._mark_viewable(tmp_path, saved, "ridge")
    assert all(tile["viewable"] == (tile["family"] == "ridge") for tile in marked)


def test_tile_results_resolved_family_and_aligned_ensemble_predictions(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from smolsmort.forecast import views
    from smolsmort.forecast.tables import read_table

    save_json(
        tmp_path / "request.json",
        {"version": 2, "families": [{"name": "ridge"}, {"name": "snaive"}]},
    )
    save_json(tmp_path / "status.json", {"state": "done"})
    save_json(tmp_path / "result.json", {"winner": {"family": "ridge"}})
    save_model(tmp_path, "ridge", [8, 18])
    save_model(tmp_path, "snaive", [10, 20])
    saved = tiles.read_tiles(tmp_path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        snapshots = list(pool.map(lambda _: tiles.read_tiles(tmp_path), range(24)))
    assert all(
        [item["tile_id"] for item in snapshot] == [item["tile_id"] for item in saved]
        for snapshot in snapshots
    )
    for tile in saved:
        query = {"model": tile["tile_id"]}
        selected = views._select_model(tmp_path, {"winner": {"family": "ridge"}}, query)
        assert selected["family"] == tile["family"]
        for split in ("validation", "test", "forecast"):
            table = read_table(views._prediction_path(tmp_path, query, split))
            expected = (
                [9, 19]
                if tile["kind"] == "ensemble"
                else [8, 18]
                if tile["family"] == "ridge"
                else [10, 20]
            )
            np.testing.assert_allclose(table["prediction"], expected)
        assert views._optional_prediction_path(tmp_path, query, "curve") is None
    with pytest.raises(views.ViewError, match="tile in this run"):
        views._prediction_path(tmp_path, {"model": "model:ridge:unknown"}, "test")


def test_tiles_stable_ids_full_scores_legacy_curves_and_top_five(tmp_path):
    families = ["ridge", "lightgbm", "xgboost", "snaive"]
    save_json(
        tmp_path / "request.json", {"version": 2, "families": [{"name": name} for name in families]}
    )
    save_json(tmp_path / "status.json", {"state": "done"})
    for index, family in enumerate(families):
        save_model(tmp_path, family, [8 + index, 18 + index])
    curve_path = tmp_path / "families/ridge/generation_curves.jsonl"
    curve_path.write_text(
        json.dumps({"generation": 0, "metric": "rmse", "validation": [4, 3]}) + '\n{"generation":',
        encoding="utf-8",
    )
    first = tiles.read_tiles(tmp_path)
    assert first == tiles.read_tiles(tmp_path)
    models = [tile for tile in first if tile["kind"] == "model"]
    ensembles = [tile for tile in first if tile["kind"] == "ensemble"]
    assert len(models) == 4 and len(ensembles) == 5
    assert len(tiles.ensemble_rows(tmp_path)) == 11
    assert first == sorted(first, key=tiles.rank_tile)
    ridge = next(tile for tile in models if tile["family"] == "ridge")
    assert ridge["curves"]["0"]["validation"]["rmse"] == [4, 3]
    assert ridge["curves"]["0"]["train_loss"] == []
    assert ridge["metrics"]["wae"] == {"validation": 4, "test": 4}
    assert all(set(tile["metrics"]) == set(tiles.CURVE_METRICS) for tile in first)
    assert all(set(tile["members"]) <= {model["tile_id"] for model in models} for tile in ensembles)
    json.dumps(first, allow_nan=False)
    # test changes cannot affect ensemble membership or validation ranking
    for family in families:
        target = tmp_path / "families" / family / "test.parquet"
        write_table(
            target,
            {
                "series": np.array(["a", "a"]),
                "step": np.array([1, 2]),
                "actual": np.array([10.0, 20.0]),
                "prediction": np.array([100.0, 200.0]),
            },
        )
    assert [tile["tile_id"] for tile in tiles.read_tiles(tmp_path)] == [
        tile["tile_id"] for tile in first
    ]


def test_tile_route_full_read_and_new_curves(tmp_path):
    from smolsmort.forecast.tab import forecast_tab

    folder = tmp_path / ".forecast-runs" / "saved"
    save_json(folder / "request.json", {"version": 2, "families": [{"name": "ridge"}]})
    save_json(folder / "status.json", {"state": "running"})
    genome = save_model(folder, "ridge", [9, 19])
    record = {
        "generation": 2,
        "key": tiles.recipe_key(genome),
        "train_loss": [3, 2],
        "curves": {name: [2, 1] for name in tiles.CURVE_METRICS},
        "loss": {"name": "rmse", "validation": [3, 2]},
        "candidates": [{"key": tiles.recipe_key(genome), "validation_metrics": {"mae": 1}}],
    }
    path = folder / "families/ridge/generation_curves.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    app = SimpleNamespace(state=SimpleNamespace(bases={"forecast": str(tmp_path)}))
    get = forecast_tab().get["/api/forecast-run-tiles"]
    result = get(app, {"run": "saved", "since_generation": "999"})
    assert result == get(app, {"run": "saved"})
    assert result["tiles"][0]["curves"]["2"]["train_loss"] == [3, 2]
    assert result["tiles"][0]["curves"]["2"]["validation"]["mae"] == [2, 1]
    record["key"] = "another-leader"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    candidate_curve = get(app, {"run": "saved"})["tiles"][0]["curves"]["2"]
    assert candidate_curve["train_loss"] == []
    assert candidate_curve["validation"]["mae"] == [1]
    for family in ("snaive", "xgboost", "lightgbm"):
        save_model(folder, family, [11, 21])
    save_json(
        folder / "request.json",
        {
            "version": 2,
            "families": [{"name": name} for name in ("ridge", "snaive", "xgboost", "lightgbm")],
        },
    )
    full = get(app, {"run": "saved"})
    assert len(full["tiles"]) == 9 and len(full["ensembles"]) == 11


def test_refit_run_real_worker_incremental_no_search(tmp_path, monkeypatch):
    from test_forecast_single import build_workspace, candidate

    from smolsmort.forecast import worker
    from smolsmort.forecast.prep import Prepared

    ws = build_workspace(monkeypatch, weeks=85, series=1, holdout="block")
    genome = candidate(ws).to_dict()
    prepared = Prepared(tmp_path, "", {}, False)
    monkeypatch.setattr(worker, "load_workspace", lambda *args, **kwargs: ws)
    monkeypatch.setattr(worker, "screen_families", lambda *args, **kwargs: {})
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker, "search", lambda *args, **kwargs: pytest.fail("refit searched"))
    snapshots = []

    def start(root, spec, prepared, **options):
        snapshots.append(tiles.read_tiles(root / options["part_of"]))
        part_id = f"part-{len(snapshots)}"
        folder = root / part_id
        save_json(
            folder / "request.json",
            {
                **options,
                "prepared": str(prepared.folder),
                "summary": prepared.summary,
                "spec": asdict(spec),
            },
        )
        assert worker.run(folder) == 0
        return part_id

    run_id = plan_sets.build_refit_run(
        tmp_path,
        ws.spec,
        prepared,
        {"ridge": genome, "snaive": {**genome, "family": "snaive", "params": {}}},
        run_id="set",
        start_run=start,
    )
    assert all(tile["state"] == "running" for tile in snapshots[0])
    assert next(tile for tile in snapshots[1] if tile["family"] == "ridge")["state"] == "done"
    result = tiles.read_tiles(tmp_path / run_id)
    assert result[0]["state"] == "done"
    assert all(tile["best_generation"] == 0 for tile in result if tile["kind"] == "model")
    assert all(
        value["validation"] is not None and value["test"] is not None
        for value in result[0]["metrics"].values()
    )
    ridge = next(tile for tile in result if tile["family"] == "ridge")
    assert (
        ridge["tile_id"]
        == next(tile for tile in snapshots[0] if tile["family"] == "ridge")["tile_id"]
    )
    assert ridge["curves"]["0"]["train_loss"]
    assert ridge["curves"]["0"]["validation"]["wape"] == pytest.approx(
        [ridge["metrics"]["wape"]["validation"]]
    )
    record = json.loads((tmp_path / run_id / "families/ridge/generation_curves.jsonl").read_text())
    assert set(record["validation_metrics"]) == set(tiles.CURVE_METRICS)
    assert record["loss"]["name"] == "penalized_squared_error"
    assert record["loss"]["objective"] == "squared"
    assert len([tile for tile in result if tile["kind"] == "ensemble"]) == 1
    assert json.loads((tmp_path / "part-1/request.json").read_text())["part_of"] == run_id
    assert runs.run_state(tmp_path, run_id)["state"] == "done"


def test_refit_cancelled_state_stays_cancelled(tmp_path):
    from smolsmort.forecast.spec import Column, PrepSpec

    spec = PrepSpec(
        "unused.csv",
        "timeseries",
        "regression",
        (Column("day", "time"), Column("y", "target", "sum")),
    )
    genome = {"family": "ridge", "families": [], "params": {}, "objective": "squared"}

    def start(root, spec, prepared, **options):
        save_json(root / "part/request.json", {"part_of": options["part_of"]})
        return "part"

    run_id = plan_sets.build_refit_run(
        tmp_path,
        spec,
        None,
        {"ridge": {"genome": genome}},
        start_run=start,
        wait_run=lambda *args: {"state": "cancelled"},
    )
    assert json.loads((tmp_path / run_id / "status.json").read_text())["state"] == "cancelled"
    assert tiles.read_tiles(tmp_path / run_id)[0]["state"] == "done"


def test_mark_viewable_flags_only_tiles_with_saved_predictions(tmp_path):
    from smolsmort.forecast.views import _mark_viewable

    (tmp_path / "families" / "xgboost").mkdir(parents=True)
    for kind in ("validation", "test", "forecast"):
        (tmp_path / "families" / "xgboost" / f"{kind}.parquet").write_bytes(b"")
        (tmp_path / f"winner_{kind}.parquet").write_bytes(b"")
    tiles = [
        {"tile_id": "model:xgboost:a", "kind": "model", "family": "xgboost"},
        {"tile_id": "model:ridge:b", "kind": "model", "family": "ridge"},
        {"tile_id": "model:lightgbm:c", "kind": "model", "family": "lightgbm"},
        {"tile_id": "ensemble:d", "kind": "ensemble", "members": ["model:xgboost:a"]},
        {
            "tile_id": "ensemble:e",
            "kind": "ensemble",
            "members": ["model:xgboost:a", "model:ridge:b"],
        },
    ]
    flags = {
        tile["tile_id"]: tile["viewable"] for tile in _mark_viewable(tmp_path, tiles, "lightgbm")
    }
    # complete family or winner predictions open; an ensemble needs all of its members
    assert flags == {
        "model:xgboost:a": True,
        "model:ridge:b": False,
        "model:lightgbm:c": True,
        "ensemble:d": True,
        "ensemble:e": False,
    }
