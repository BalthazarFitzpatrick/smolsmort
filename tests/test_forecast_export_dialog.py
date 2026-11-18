"""run bundles preserve saved recipes, selected predictions and export boundaries"""

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from smolsmort.forecast import tab, tiles, views
from smolsmort.forecast.export_bundle import (
    _feature_kind,
    build_recipe,
    export_bundle,
    render_readme,
)
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
        "## scores and how to read them",
        "## data and spec",
        "## how the forecast is built",
        "### member: ridge",
        "### params: ridge",
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
        assert f"| {kind} | 1 | {feature} |" in readme


def assert_feature_definitions(recipe, readme):
    assert readme.count("## feature definitions\n") == 1
    section = readme.split("## feature definitions\n", 1)[1].split("\n## ", 1)[0]
    for member in recipe["members"]:
        for feature in member["genome"].get("families", []):
            kind = _feature_kind(feature)
            row = next(line for line in section.splitlines() if line.startswith(f"| {kind} |"))
            assert "definition unavailable" not in row, f"missing definition for {feature}"
    return section


def test_fixture_readme_defines_every_member_feature_kind(saved_run):
    folder, _, _, ensemble = saved_run
    recipe = build_recipe(folder, {"model": ensemble["tile_id"]})
    readme = render_readme(recipe)
    section = assert_feature_definitions(recipe, readme)
    assert "| ewm |" not in section
    assert "| mix |" not in section
    assert "| cal:week |" not in section
    assert "| cal:month |" in section
    assert "target transform: log1p: z_t = log(1+y_t)" in readme
    assert "after each member's inverse target transform" in readme
    assert "4..19 pairs" in readme
    assert "results-tab ratio = sum(abs(y-p))/sum(abs(y-naive))" in readme


def test_undocumented_feature_kind_fails_definition_check(saved_run):
    folder, _, models, _ = saved_run
    recipe = build_recipe(folder, {"model": models[0]["tile_id"]})
    recipe["members"][0]["genome"]["families"].append("units@week:new_kind:4")
    readme = render_readme(recipe)
    with pytest.raises(AssertionError, match="missing definition for units@week:new_kind:4"):
        assert_feature_definitions(recipe, readme)


def test_readme_defines_dynamics_and_calendar_variants(saved_run):
    folder, _, models, _ = saved_run
    recipe = build_recipe(folder, {"model": models[0]["tile_id"]})
    recipe["members"][0]["genome"]["families"] += [
        "units@week:lag+0",
        "units@week:lag+1",
        "units@week:diff:13",
        "units@week:ewm:0.1",
        "units@week:momentum:8",
        "units@week:slope:4",
        "units@week:season",
        "units@week:cycle_ratio",
        "units@week:roll:52:zeros",
        "units@week:roll:13:q10",
        "country@week:mix",
        "site@week:spread",
        "week:cal:cyclic",
        "week:cal:holiday",
        "week:cal:week",
        "week:cal:quarter",
        "week:cal:weekday",
        "week:cal:dom",
        "week:cal:trend",
        "weeks_ahead",
        "country:freq",
        "price:log",
    ]
    recipe["members"].append({**recipe["members"][0], "genome": {"transform": "logdiff"}})
    section = assert_feature_definitions(recipe, render_readme(recipe))
    assert "single mode: y_(o-k); bucket mode: y_(t-k)" in section
    assert "y_(o-j), including y_o for lag+0" in section
    assert "zeros = count(y = 0)/n" in section
    assert "mean(y in [o-m+1, o]) - mean(y in [o-2*m+1, o-m])" in section
    assert "missing values retain the previous level" in section
    assert "logdiff: z_t = log(1+y_t)-log(1+y_o)" in render_readme(recipe)


def test_readme_explains_omitted_member_target_transform(saved_run):
    folder, _, models, _ = saved_run
    model = next(model for model in models if model["family"] == "ridge")
    recipe = build_recipe(folder, {"model": model["tile_id"]})
    del recipe["members"][0]["genome"]["transform"]
    recipe["request"]["transform"] = "log1p"
    readme = render_readme(recipe)
    assert "target transform is omitted in the saved recipe; the default is none." in readme
    assert "| transform | none |" not in readme


def test_readme_is_a_report_with_sections_in_reading_order(saved_run):
    folder, _, models, _ = saved_run
    recipe = build_recipe(folder, {"model": models[0]["tile_id"]})
    recipe["members"][0]["genome"]["params"] = {"lambda_l2": 1.578007522103636e-7}
    recipe["request"]["spec"]["columns"][0]["known"] = True
    recipe["holdout"]["fell_back"] = False
    readme = render_readme(recipe)
    assert readme.startswith("# forecast report\n\nsmolsmort forecast `units`")
    assert "test wape was 20%" in readme
    assert re.findall(r"^## (.+)$", readme, re.MULTILINE) == [
        "data and spec",
        "how the forecast is built",
        "the model",
        "feature definitions",
        "scores and how to read them",
        "the files in this export",
        "limitations",
        "how to reproduce the numbers",
    ]
    assert '["' not in readme
    assert not re.search(r"\b(?:True|False)\b", readme)
    assert "| lambda_l2 | 1.578e-07 |" in readme
    assert "| week | time | not applicable | yes |" in readme
    assert "| fallback used | no |" in readme
    for mantissa in re.findall(r"\b(\d+(?:\.\d+)?)[eE][+-]\d+\b", readme):
        assert len(mantissa.replace(".", "").lstrip("0")) <= 4
    assert "smolsmort/" not in readme
    assert str(folder) not in readme
    assert "request.json" not in readme


def test_every_exported_csv_has_its_actual_column_glossary(saved_run, tmp_path):
    folder, _, models, _ = saved_run
    target = export_bundle(
        folder,
        {
            "folder": str(tmp_path / "report"),
            "include": "everything",
            "model": models[0]["tile_id"],
        },
    )
    readme = (target / "README.md").read_text()
    for path in target.glob("*.csv"):
        section = (
            readme.split(f"### {path.name}\n", 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]
        )
        with path.open(newline="", encoding="utf-8") as stream:
            columns = next(csv.reader(stream))
        assert re.findall(r"^\| ([^|]+) \|", section, re.MULTILINE)[2:] == columns
    assert "total_forecast = sum(series forecasts)" in readme
    assert "sum of series forecast values" in readme


def test_report_handles_a_saved_run_without_source_name(saved_run):
    folder, _, models, _ = saved_run
    recipe = build_recipe(folder, {"model": models[0]["tile_id"]})
    del recipe["request"]["spec"]["source"]
    readme = render_readme(recipe)
    assert "| dataset | dataset name not recorded |" in readme
    assert "source dataset: dataset name not recorded." in readme


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
        ("details + predictions", {"demo_dataset.csv", "results.csv", "wide.csv"}),
        (
            "everything",
            {
                "results.csv",
                "demo_dataset.csv",
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
        assert not (target / "results.csv").read_text().startswith("#")
        assert views.export_csv(folder, request, {"model": model}).startswith(b"# verdict:")
        assert (target / "wide.csv").read_bytes() == export_wide_csv(
            folder, request, {"model": model}
        )
    if include == "everything":
        rows = list(csv.DictReader(io.StringIO((target / "test.csv").read_text())))
        assert [float(row["forecast"]) for row in rows] == [8.0, 8.0]


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
    assert "## the model" in readme
    assert "recreate" not in readme
    assert "member models with equal weights" in readme
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
    assert "tables not saved: lead_errors" in (target / "README.md").read_text()


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
    assert [float(row["forecast"]) for row in rows] == [prediction, prediction]


def test_route_refuses_bare_family_name(saved_run, tmp_path, monkeypatch):
    folder, _, _, _ = saved_run
    monkeypatch.setattr(tab, "_runs_root", lambda app: folder.parent)
    route = tab.forecast_tab().post["/api/forecast-export-bundle"]
    with pytest.raises(RequestError) as error:
        route(None, {"id": folder.name, "folder": str(tmp_path / "exports"), "model": "ridge"})
    assert error.value.status == 400


@pytest.fixture
def input_run(saved_run, tmp_path):
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


@pytest.mark.parametrize("include", ["details + predictions", "everything"])
def test_input_history_matches_panel_and_is_model_independent(input_run, tmp_path, include):
    folder, _, models, ensemble = input_run
    histories = []
    for tile in [*models, ensemble]:
        target = export_bundle(
            folder,
            {"folder": str(tmp_path / "exports"), "include": include, "model": tile["tile_id"]},
        )
        histories.append((target / "input.csv").read_bytes())
    assert len(set(histories)) == 1
    rows = list(csv.DictReader(io.StringIO(histories[0].decode())))
    assert list(rows[0]) == ["week", "region", "actual"]
    assert {(row["week"], row["region"]): float(row["actual"]) for row in rows} == {
        ("2026-01-05", "north"): 5,
        ("2026-01-12", "north"): 0,
        ("2026-01-19", "north"): 7,
        ("2026-01-05", "south"): 11,
        ("2026-01-12", "south"): 0,
        ("2026-01-19", "south"): 13,
    }
    readme = (target / "README.md").read_text()
    assert "source size (bytes)" in readme and "source modified date" in readme


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_input_dataset_failure_keeps_export_and_reports_reason(
    input_run, tmp_path, monkeypatch, change
):
    folder, request, models, _ = input_run
    source = Path(request["spec"]["source"])
    if change == "missing":
        source.unlink()
    else:
        source.write_text(source.read_text().replace(",2,yes", ",9,yes"))
    monkeypatch.setattr(tab, "_runs_root", lambda app: folder.parent)
    route = tab.forecast_tab().post["/api/forecast-export-bundle"]
    result = route(
        None,
        {
            "id": folder.name,
            "folder": str(tmp_path / "exports"),
            "include": "everything",
            "model": models[0]["tile_id"],
        },
    )
    target = Path(result["path"])
    assert not (target / "input.csv").exists()
    assert (target / "results.csv").is_file()
    assert result["warning"].startswith("input.csv not written:")
    assert change in result["warning"]
    readme = (target / "README.md").read_text()
    for warning in result["warning"].split("; "):
        assert warning in readme
    assert "demo_dataset.csv written without input slice:" in result["warning"]
    with (target / "demo_dataset.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["dataset_slice"] for row in rows} == {"validation", "test", "prediction"}
    assert all(row["actual"] == "" for row in rows)


def test_input_details_only_does_not_read_dataset(input_run, tmp_path, monkeypatch):
    from smolsmort.forecast import export_bundle as bundle

    folder, _, models, _ = input_run
    monkeypatch.setattr(bundle, "_input_csv", lambda recipe: pytest.fail("read dataset"))
    target = export_bundle(
        folder, {"folder": str(tmp_path / "exports"), "model": models[0]["tile_id"]}
    )
    assert not (target / "input.csv").exists()


def test_every_bundle_csv_has_only_grouping_keys_and_numeric_measures(input_run, tmp_path):
    folder, _, models, _ = input_run
    target = export_bundle(
        folder,
        {
            "folder": str(tmp_path / "exports"),
            "include": "everything",
            "model": models[0]["tile_id"],
        },
    )
    keys = ["week", "region", "forecast_made_week", "horizon"]
    values = ["forecast", "actual", "lower_80", "upper_80"]
    expected = {
        "input.csv": ["week", "region", "actual"],
        "demo_dataset.csv": [
            "week",
            "region",
            "actual",
            "prediction",
            "lower_50",
            "upper_50",
            "lower_80",
            "upper_80",
            "wape",
            "horizon_step",
            "dataset_slice",
        ],
        "results.csv": [*keys, *values],
        "wide.csv": [
            *keys,
            "forecast",
            "actual",
            "lower_50",
            "upper_50",
            "lower_80",
            "upper_80",
            "total_forecast",
            "total_actual",
            "total_lower_50",
            "total_upper_50",
            "total_lower_80",
            "total_upper_80",
        ],
        "validation.csv": [*keys, *values],
        "test.csv": [*keys, *values],
        "forecast.csv": [*keys, "forecast", "lower_80", "upper_80"],
        "lead_errors.csv": ["horizon", "error"],
    }
    assert {path.name for path in target.glob("*.csv")} == set(expected)
    for filename, header in expected.items():
        with (target / filename).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == header, filename
            for row in reader:
                for name, value in row.items():
                    if name not in [*keys, "dataset_slice"] and value:
                        assert np.isfinite(float(value)), (filename, name, value)
                    if name == "horizon" and value:
                        assert str(int(value)) == value
    readme = (target / "README.md").read_text()
    assert readme.index("- `input.csv`") < readme.index("- `results.csv`")


def test_entity_input_skip_is_silent(saved_run, tmp_path, monkeypatch):
    folder, request, models, _ = saved_run
    request["spec"]["mode"] = "entity"
    write_json(folder / "request.json", request)
    monkeypatch.setattr(views, "export_csv", lambda *args: b"target\n1\n")
    monkeypatch.setattr(tab, "_runs_root", lambda app: folder.parent)
    route = tab.forecast_tab().post["/api/forecast-export-bundle"]
    result = route(
        None,
        {
            "id": folder.name,
            "folder": str(tmp_path / "exports"),
            "include": "everything",
            "model": models[0]["tile_id"],
        },
    )
    target = Path(result["path"])
    assert result["warning"] == ""
    assert not (target / "input.csv").exists()
    assert not (target / "demo_dataset.csv").exists()
    assert "input.csv not written: entity run" in (target / "README.md").read_text()
    assert "demo_dataset.csv not written: entity run" in (target / "README.md").read_text()


def test_real_run_input_smoke(tmp_path):
    from smolsmort.forecast.prep import _literal, _quote
    from smolsmort.forecast.tables import connect, read_table

    workspace = Path.home() / "Documents" / "dev" / "ai_lab" / "forecast_workspace"
    folder = workspace / ".forecast-runs" / "20261115-155627-6f006a"
    if not (folder / "request.json").is_file():
        pytest.skip("real run absent")
    request = json.loads((folder / "request.json").read_text())
    source = Path(request["spec"]["source"])
    if not source.is_file() or not (Path(request["prepared"]) / "panel.parquet").is_file():
        pytest.skip("real dataset or panel absent")
    target = export_bundle(
        folder, {"folder": str(tmp_path), "include": "everything", "model": "winner"}
    )
    assert_feature_definitions(
        json.loads((target / "recipe.json").read_text()), (target / "README.md").read_text()
    )
    rows = list(csv.DictReader((target / "input.csv").open()))
    panel = read_table(Path(request["prepared"]) / "panel.parquet")
    assert len(rows) == len(panel["__step"])
    assert len({(row["swo_date"], row["client_group"]) for row in rows}) == len(rows)
    assert {(row["swo_date"], row["client_group"]) for row in rows} == {
        (str(step.astype("datetime64[D]")), group)
        for step, group in zip(panel["__step"], panel["client_group"], strict=True)
    }
    spec = request["spec"]
    with connect() as con:
        expected = con.execute(
            f"SELECT SUM(unit) FROM read_csv({_literal(str(source))}, header=true) "
            f"WHERE ({spec['where']}) AND {_quote('swo_date')} < date_trunc('week', CAST({_literal(spec['as_of'])} AS DATE))"
        ).fetchone()[0]
    assert sum(float(row["actual"]) for row in rows) == pytest.approx(expected)
    history = {(row["swo_date"], row["client_group"]): float(row["actual"]) for row in rows}
    with (target / "wide.csv").open(newline="", encoding="utf-8") as stream:
        wide = list(csv.DictReader(stream))
    observed = [row for row in wide if row["actual"]]
    assert observed
    assert all((row["swo_date"], row["client_group"]) in history for row in observed)
    assert all(
        float(row["actual"]) == history[row["swo_date"], row["client_group"]] for row in observed
    )
    for filename in ("results.csv", "wide.csv", "validation.csv", "test.csv", "forecast.csv"):
        with (target / filename).open(newline="", encoding="utf-8") as stream:
            assert next(csv.reader(stream))[:2] == ["swo_date", "client_group"]
