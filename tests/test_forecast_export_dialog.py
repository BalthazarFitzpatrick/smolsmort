"""run bundles preserve saved recipes, selected predictions and export boundaries"""

import csv
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from smolsmort.forecast import tab, tiles, views
from smolsmort.forecast.export_bundle import build_recipe, export_bundle, render_readme
from smolsmort.forecast.export_wide import export_wide_csv
from smolsmort.forecast.tables import write_table
from smolsmort.review.routes import RequestError


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def saved_run(tmp_path):
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


def test_recipe_and_readme_use_saved_model_fields(saved_run):
    folder, _, models, _ = saved_run
    model = next(model for model in models if model["family"] == "ridge")
    recipe = build_recipe(folder, {"model": model["tile_id"]}, made=datetime(2026, 11, 18, 12))
    assert recipe["made"] == "2026-11-18T12:00:00"
    assert recipe["kind"] == "single"
    assert recipe["scores"] == {"metric": "wape", "validation": 0.2, "test": 0.2, "ratio": 0.5}
    member = recipe["members"][0]
    assert member["genome"]["params"] == {"alpha": 0.75}
    assert member["seed"] == 17
    assert member["exit_test"]["fold_count"] == 2
    readme = render_readme(recipe)
    for text in (
        "## scores",
        "## data and spec",
        "## splits and cut dates",
        "## member: ridge",
        "### params",
        "| alpha | 0.75 |",
        "| transform | log1p |",
        "| seed | 17 |",
        "| dataset | demand.csv |",
        "week",
        "units",
        "region",
        "2026-01-12",
        "| lags |",
        "| rolling |",
        "| calendar |",
        "| leads |",
        "| measures |",
        "| dimensions |",
        "### saved fit facts",
        "### exit test",
    ):
        assert text in readme


def test_readme_groups_legacy_feature_names(saved_run):
    folder, _, models, _ = saved_run
    model = next(model for model in models if model["family"] == "ridge")
    recipe = build_recipe(folder, {"model": model["tile_id"]})
    recipe["members"][0]["genome"]["families"] = [
        "lag:1",
        "roll:4:mean",
        "cal:month",
        "lead:promo",
        "measure:price",
        "dim:region",
    ]
    readme = render_readme(recipe)
    for kind, feature in (
        ("lags", "lag:1"),
        ("rolling", "roll:4:mean"),
        ("calendar", "cal:month"),
        ("leads", "lead:promo"),
        ("measures", "measure:price"),
        ("dimensions", "dim:region"),
    ):
        assert f'| {kind} | ["{feature}"] |' in readme


def test_readme_explains_omitted_member_target_transform(saved_run):
    folder, _, models, _ = saved_run
    model = next(model for model in models if model["family"] == "ridge")
    recipe = build_recipe(folder, {"model": model["tile_id"]})
    del recipe["members"][0]["genome"]["transform"]
    recipe["request"]["transform"] = "log1p"
    readme = render_readme(recipe)
    assert "target transform is omitted in the saved genome; smolsmort defaults to none." in readme
    assert "| transform | none |" not in readme


def test_ensemble_alias_scores_use_requested_metric(saved_run):
    folder, _, _, ensemble = saved_run
    cached = folder / "tile_views" / ensemble["recipe_key"]
    for phase in ("validation", "test", "forecast"):
        (folder / f"ensemble_{phase}.parquet").write_bytes(
            (cached / f"{phase}.parquet").read_bytes()
        )
    recipe = build_recipe(folder, {"model": "ensemble", "metric": "mae"})
    assert recipe["scores"] == {"metric": "mae", "validation": 3.0, "test": 3.0, "ratio": 0.75}


@pytest.mark.parametrize(
    "include,extra",
    [
        ("details only", set()),
        ("details + predictions", {"results.csv", "wide.csv"}),
        (
            "everything",
            {
                "results.csv",
                "wide.csv",
                "validation.csv",
                "test.csv",
                "forecast.csv",
                "lead_errors.csv",
            },
        ),
    ],
)
def test_include_levels_reuse_selected_exports(saved_run, tmp_path, include, extra):
    folder, request, models, _ = saved_run
    model = next(model for model in models if model["family"] == "ridge")["tile_id"]
    query = {"folder": str(tmp_path / "exports"), "include": include, "model": model}
    target = export_bundle(folder, query)
    assert {path.name for path in target.iterdir()} == {"README.md", "recipe.json"} | extra
    assert json.loads((target / "recipe.json").read_text())["model"] == model
    if extra:
        assert (target / "results.csv").read_bytes() == views.export_csv(
            folder, request, {"model": model}
        )
        assert (target / "wide.csv").read_bytes() == export_wide_csv(
            folder, request, {"model": model}
        )
    if include == "everything":
        rows = list(csv.DictReader(io.StringIO((target / "test.csv").read_text())))
        assert [float(row["prediction"]) for row in rows] == [8.0, 8.0]


def test_run_name_local_date_and_collision_suffix(saved_run, tmp_path):
    folder, _, models, _ = saved_run
    query = {"folder": str(tmp_path / "exports"), "model": models[0]["tile_id"]}
    before = datetime.now().astimezone().date().isoformat()
    first = export_bundle(folder, query)
    after = datetime.now().astimezone().date().isoformat()
    assert first.name in {f"{folder.name}_{before}", f"{folder.name}_{after}"}
    second = export_bundle(folder, query)
    assert second.name == f"{first.name}_1"
    assert (first / "README.md").is_file()


@pytest.mark.parametrize("include", ("details only", "details + predictions", "everything"))
def test_ensemble_export_preserves_recipe_without_refit_script(saved_run, tmp_path, include):
    folder, _, _, ensemble = saved_run
    target = export_bundle(
        folder,
        {"folder": str(tmp_path / "exports"), "model": ensemble["tile_id"], "include": include},
    )
    recipe = json.loads((target / "recipe.json").read_text())
    assert recipe["kind"] == "ensemble"
    assert {member["family"] for member in recipe["members"]} == {"ridge", "snaive"}
    assert recipe["weighting"] == ensemble["recipe"]
    for member in recipe["members"]:
        assert member["recipe_key"]
        genome = member["genome"]
        assert genome["params"] == ({"alpha": 0.75} if member["family"] == "ridge" else {})
        assert genome["transform"] == "log1p"
        assert "target@time:lag:1" in genome["families"]
    readme = (target / "README.md").read_text()
    assert "## ensemble" in readme
    assert "recreate" not in readme
    assert "members use equal weights, saved in `ensembles.json`." in readme
    assert readme.count("no model weights are stored") == 1
    assert not (target / "recreate.py").exists()


def test_saved_ensemble_beyond_top_five_can_export(saved_run, tmp_path):
    folder, _, _, ensemble = saved_run
    ensembles = [
        {
            **ensemble,
            "tile_id": f"ensemble:saved-{index}",
            "recipe_key": f"saved-{index}",
            "metrics": {"wape": {"validation": index + 0.1, "test": 0.3}},
        }
        for index in range(7)
    ]
    write_json(folder / "ensembles.json", ensembles)
    target = export_bundle(
        folder,
        {
            "folder": str(tmp_path / "exports"),
            "model": ensembles[-1]["tile_id"],
        },
    )
    recipe = json.loads((target / "recipe.json").read_text())
    assert recipe["model"] == ensembles[-1]["tile_id"]
    assert len(recipe["members"]) == 2


def test_ensemble_missing_lead_errors_does_not_borrow_root_table(saved_run, tmp_path):
    folder, _, _, ensemble = saved_run
    write_table(folder / "lead_errors.parquet", {"error": [999.0]})
    target = export_bundle(
        folder,
        {
            "folder": str(tmp_path / "exports"),
            "model": ensemble["tile_id"],
            "include": "everything",
        },
    )
    assert not (target / "lead_errors.csv").exists()
    assert {"validation.csv", "test.csv", "forecast.csv"} <= {
        path.name for path in target.iterdir()
    }
    recipe = json.loads((target / "recipe.json").read_text())
    assert recipe["missing_tables"] == ["lead_errors"]
    assert "## tables not saved\n\nlead_errors" in (target / "README.md").read_text()


@pytest.mark.parametrize("folder", ["", " ", "exports", "../escape", "nested/../../escape"])
def test_empty_and_relative_paths_refused(saved_run, folder):
    run_dir, _, _, _ = saved_run
    with pytest.raises(views.ViewError, match="absolute path"):
        export_bundle(run_dir, {"folder": folder})


def test_uncreatable_path_refused(saved_run, tmp_path):
    run_dir, _, models, _ = saved_run
    blocker = tmp_path / "file"
    blocker.write_text("keep")
    with pytest.raises(OSError):
        export_bundle(run_dir, {"folder": str(blocker / "exports"), "model": models[0]["tile_id"]})
    assert blocker.read_text() == "keep"


def test_route_roundtrip_and_bad_requests(saved_run, tmp_path, monkeypatch):
    folder, _, models, _ = saved_run
    monkeypatch.setattr(tab, "_runs_root", lambda app: folder.parent)
    route = tab.forecast_tab().post["/api/forecast-export-bundle"]
    payload = {
        "id": folder.name,
        "folder": str(tmp_path / "exports"),
        "model": models[0]["tile_id"],
    }
    result = route(None, payload)
    assert result["path"].startswith(str(tmp_path / "exports"))
    assert (Path(result["path"]) / "README.md").is_file()
    for changes in (
        {"folder": "../escape"},
        {"include": "invalid"},
        {"model": "model:missing:key"},
    ):
        with pytest.raises(RequestError) as error:
            route(None, {**payload, **changes})
        assert error.value.status == 400
    with pytest.raises(RequestError):
        route(None, {**payload, "id": "../outside"})


@pytest.mark.parametrize("family,prediction", [("ridge", 8.0), ("snaive", 6.0), ("ensemble", 7.0)])
def test_route_exports_exactly_the_posted_model(
    saved_run, tmp_path, monkeypatch, family, prediction
):
    folder, _, models, ensemble = saved_run
    monkeypatch.setattr(tab, "_runs_root", lambda app: folder.parent)
    route = tab.forecast_tab().post["/api/forecast-export-bundle"]
    tile = ensemble if family == "ensemble" else next(m for m in models if m["family"] == family)
    result = route(
        None,
        {
            "id": folder.name,
            "folder": str(tmp_path / "exports"),
            "model": tile["tile_id"],
            "include": "everything",
        },
    )
    target = Path(result["path"])
    recipe = json.loads((target / "recipe.json").read_text())
    assert recipe["model"] == tile["tile_id"]
    assert recipe["selected"]["tile_id"] == tile["tile_id"]
    by_id = {model["tile_id"]: model for model in models}
    expected = [by_id[member] for member in tile["members"]] if family == "ensemble" else [tile]
    assert [member["recipe_key"] for member in recipe["members"]] == [
        model["recipe_key"] for model in expected
    ]
    if family != "ensemble":
        assert recipe["members"][0]["genome"]["params"] == tile["recipe"]["params"]
    rows = list(csv.DictReader(io.StringIO((target / "test.csv").read_text())))
    assert [float(row["prediction"]) for row in rows] == [prediction, prediction]


def test_route_refuses_bare_family_name(saved_run, tmp_path, monkeypatch):
    folder, _, _, _ = saved_run
    monkeypatch.setattr(tab, "_runs_root", lambda app: folder.parent)
    route = tab.forecast_tab().post["/api/forecast-export-bundle"]
    with pytest.raises(RequestError) as error:
        route(None, {"id": folder.name, "folder": str(tmp_path / "exports"), "model": "ridge"})
    assert error.value.status == 400
