"""raw tile contract and incremental recipe refits on synthetic rows"""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import plan_sets, runs, tiles
from smolsmort.forecast.tables import write_table


@pytest.mark.parametrize("cache", ["missing", "fresh", "stale"])
def test_ensemble_scores_reused_until_member_changes(tmp_path, monkeypatch, cache):
    save_json(
        tmp_path / "request.json",
        {"version": 2, "families": [{"name": name} for name in ("ridge", "snaive")]},
    )
    for name in ("ridge", "snaive"):
        save_model(tmp_path, name, [9, 19])
    expected = tiles.ensemble_rows(tmp_path)
    path = tmp_path / "ensembles.json"
    if cache != "missing":
        save_json(path, expected)
    if cache == "stale":
        member = tmp_path / "families/ridge/test.parquet"
        os.utime(path, ns=(member.stat().st_mtime_ns - 1, member.stat().st_mtime_ns - 1))
    calls = []
    original = tiles.score_ensembles

    def score(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(tiles, "score_ensembles", score)
    first = tiles.read_tiles(tmp_path)
    assert len(calls) == (0 if cache == "fresh" else 1)
    assert json.loads(path.read_text()) == expected

    def reject(*args):
        pytest.fail("saved ensemble scores must be reused")

    monkeypatch.setattr(tiles, "score_ensembles", reject)
    assert tiles.read_tiles(tmp_path) == first
    from smolsmort.forecast import tab

    monkeypatch.setattr(tab, "_run_dir", lambda *_args: tmp_path)
    assert tab._run_tiles(None, {"run": "saved"}) == {"tiles": first, "ensembles": expected}


def test_legacy_missing_family_predictions_has_no_ensembles(tmp_path):
    save_json(
        tmp_path / "request.json",
        {"version": 2, "families": [{"name": "ridge"}, {"name": "snaive"}]},
    )
    assert all(row["kind"] == "model" for row in tiles.read_tiles(tmp_path))
    assert json.loads((tmp_path / "ensembles.json").read_text()) == []


def test_saved_ensemble_views_have_calibrated_bands(tmp_path, monkeypatch):
    from smolsmort.forecast import views
    from smolsmort.forecast.tables import read_table

    save_json(
        tmp_path / "request.json",
        {"version": 2, "families": [{"name": name} for name in ("ridge", "snaive")]},
    )
    save_model(tmp_path, "ridge", [8, 18])
    save_model(tmp_path, "snaive", [10, 20])
    for family, prediction in (("ridge", 8.0), ("snaive", 10.0)):
        for split in ("validation", "test", "forecast"):
            write_table(
                tmp_path / "families" / family / f"{split}.parquet",
                {
                    "series": np.repeat(["a", "b"], 30),
                    "step": np.tile(np.arange(30), 2),
                    "weeks_ahead": np.ones(60, dtype=int),
                    "actual": np.repeat([10.0, 20.0], 30),
                    "prediction": np.full(60, prediction),
                },
            )
    rows = tiles.ensemble_rows(tmp_path)
    save_json(tmp_path / "ensembles.json", rows)
    tiles.write_ensemble_views(tmp_path, rows)

    def reject(*args):
        pytest.fail("saved ensemble predictions must not be rebuilt")

    monkeypatch.setattr(tiles, "align_tables", reject)
    for split in ("validation", "test", "forecast"):
        path = views._tile_prediction_path(tmp_path, rows[0]["tile_id"], split)
        table = read_table(path)
        np.testing.assert_array_equal(table["prediction"], np.full(60, 9.0))
        np.testing.assert_allclose(table["upper"], np.repeat([10.0, 20.0], 30))
        for name in ("lower", "upper", "lower_50", "upper_50"):
            assert np.all(np.isfinite(table[name]))
        assert np.all(table["lower"] <= table["lower_50"])
        assert np.all(table["upper"] >= table["upper_50"])


def test_real_ensemble_scores_match_original_golden():
    folder = (
        Path.home()
        / "Documents/dev/ai_lab/forecast_workspace/.forecast-runs/20261104-140018-573593"
    )
    if not folder.exists():
        pytest.skip("real run fixture is local")
    golden = json.loads(
        (Path(__file__).parent / "fixtures/ensemble_scores_golden.json").read_text()
    )
    rows = tiles.ensemble_rows(folder)
    assert [row["tile_id"] for row in rows] == [row["tile_id"] for row in golden]
    for row, expected in zip(rows, golden, strict=True):
        for metric in ("wape", "mae", "mape"):
            for split in ("validation", "test"):
                assert row["metrics"][metric][split] == pytest.approx(
                    expected["metrics"][metric][split], abs=1e-12, rel=0
                )


def test_ensemble_join_keeps_subset_rows_shared_keys_and_duplicate_order():
    import itertools

    tables = {}
    for name, steps in (("a", [3, 1, 3, 2]), ("b", [2, 1, 3]), ("c", [1, 2])):
        table = {
            "series": np.full(len(steps), "s"),
            "step": np.asarray(steps),
            "actual": np.asarray(steps, dtype=float),
            "prediction": np.asarray(steps, dtype=float) + ord(name) / 100,
        }
        if name != "c":
            table["weeks_ahead"] = np.ones(len(steps), dtype=int)
        tables[name] = {"validation": table, "test": table}
    rows = tiles.score_ensembles([{"tile_id": name} for name in tables], tables)
    for count in (2, 3):
        for members in itertools.combinations(tables, count):
            parts = [tables[name]["validation"] for name in members]
            keys = [
                key
                for key in ("series", "step", "weeks_ahead")
                if all(key in part for part in parts)
            ]
            indexes = [
                {
                    tuple(str(part[key][i]) for key in keys): i
                    for i in range(len(part["prediction"]))
                }
                for part in parts
            ]
            common = [key for key in indexes[0] if all(key in index for index in indexes)]
            expected = [
                {name: values[[index[key] for key in common]] for name, values in part.items()}
                for part, index in zip(parts, indexes, strict=True)
            ]
            aligned = tiles.align_tables(parts)
            for actual, reference in zip(aligned, expected, strict=True):
                for key in actual:
                    np.testing.assert_array_equal(actual[key], reference[key])
            metrics = tiles.prediction_metrics(
                expected[0]["actual"], np.mean([part["prediction"] for part in expected], axis=0)
            )
            row = next(row for row in rows if row["members"] == list(members))
            assert {key: value["validation"] for key, value in row["metrics"].items()} == metrics


def test_ensemble_scoring_joins_only_once_per_split(tmp_path, monkeypatch):
    names = ("ridge", "snaive", "xgboost", "lightgbm")
    save_json(
        tmp_path / "request.json", {"version": 2, "families": [{"name": name} for name in names]}
    )
    for name in names:
        save_model(tmp_path, name, [9, 19])
    original = tiles._join_indexes
    calls = []

    def join(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(tiles, "_join_indexes", join)
    assert len(tiles.ensemble_rows(tmp_path)) == 11
    assert len(calls) == 2
    tiles.write_ensemble_views(tmp_path, tiles.ensemble_rows(tmp_path))
    assert len(list((tmp_path / "tile_views").glob("*/forecast.parquet"))) == 5


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


def test_selection_metrics_follow_saved_tile_leads_and_blends(tmp_path, monkeypatch):
    from smolsmort.forecast import metrics, views

    request = {
        "version": 2,
        "spec": {"task": "regression", "mode": "timeseries", "horizon": 2},
        "families": [{"name": name} for name in ("ridge", "snaive")],
    }
    save_json(tmp_path / "request.json", request)
    save_json(tmp_path / "status.json", {"state": "done"})
    save_json(tmp_path / "result.json", {"winner": {"family": "ridge"}})
    for family, prediction in (("ridge", [9, 4]), ("snaive", [6, 9])):
        save_model(tmp_path, family, prediction)
        table = {
            "series": np.array(["a", "a"]),
            "step": np.array(["2026-01-19"] * 2, dtype="datetime64[D]"),
            "origin": np.array(["2026-01-12", "2026-01-05"], dtype="datetime64[D]"),
            "weeks_ahead": np.array([1, 2]),
            "actual": np.array([10.0, 10.0]),
            "prediction": np.array(prediction, dtype=float),
            "naive_prediction": np.array([5.0, 5.0]),
        }
        for split in ("validation", "test", "forecast"):
            write_table(tmp_path / "families" / family / f"{split}.parquet", table)
    ensembles = tiles.ensemble_rows(tmp_path)
    save_json(tmp_path / "ensembles.json", ensembles)
    tiles.write_ensemble_views(tmp_path, ensembles)
    saved = tiles.read_tiles(tmp_path)

    def reject(*args, **kwargs):
        pytest.fail("selection metrics must only read saved predictions")

    monkeypatch.setattr(tiles, "align_tables", reject)
    monkeypatch.setattr(tiles, "score_ensembles", reject)
    results = []
    for selection in (
        {"mode": "lead", "n": 1},
        {"mode": "lead", "n": 2},
        {"mode": "blend", "weights": {1: 1, 2: 1}},
    ):
        details = {"run_tiles": views._mark_viewable(tmp_path, saved, "ridge")}
        views._set_selection_metrics(tmp_path, request, {}, details, selection)
        values = {tile["family"]: dict(tile["selection_metrics"]) for tile in saved}
        assert all(
            set(value) == {item["id"] for item in metrics.metric_registry(request["spec"])}
            for value in values.values()
        )
        assert values["ridge"]["validation_error"] == 0.2
        results.append(values)
    for values, expected in zip(
        results, ((0.1, 0.4, 0.25), (0.6, 0.1, 0.35), (0.35, 0.25, 0.3)), strict=True
    ):
        for family, error in zip(("ridge", "snaive", "ensemble"), expected, strict=True):
            assert values[family]["wape"] == pytest.approx(error)
            assert values[family]["mae"] == pytest.approx(error * 10)
            assert values[family]["mape"] == pytest.approx(error)
            assert values[family]["test_ratio"] == pytest.approx(error * 2)
    assert sorted(results[0], key=lambda name: results[0][name]["wape"]) == [
        "ridge",
        "ensemble",
        "snaive",
    ]
    assert sorted(results[1], key=lambda name: results[1][name]["wape"]) == [
        "snaive",
        "ensemble",
        "ridge",
    ]
    ensemble = next(tile for tile in saved if tile["kind"] == "ensemble")
    path = tmp_path / "tile_views" / ensemble["recipe_key"] / "test.parquet"
    path.unlink()
    views._set_selection_metrics(tmp_path, request, {}, details, {"mode": "lead", "n": 1})
    assert ensemble["selection_metrics"]["wape"] is None
    assert not path.exists()


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
    # other test files import torch, and the real worker refuses to share a process with it
    monkeypatch.delitem(sys.modules, "torch", raising=False)
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
    ensemble = next(tile for tile in result if tile["kind"] == "ensemble")
    from smolsmort.forecast.tables import read_table

    for split in ("validation", "test", "forecast"):
        table = read_table(
            tmp_path / run_id / "tile_views" / ensemble["recipe_key"] / f"{split}.parquet"
        )
        assert {"lower", "upper", "lower_50", "upper_50"} <= table.keys()
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
