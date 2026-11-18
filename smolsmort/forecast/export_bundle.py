"""portable run reports and saved prediction bundles"""

from __future__ import annotations

import csv
import io
import json
import shutil
from datetime import datetime
from pathlib import Path

from smolsmort.forecast import views
from smolsmort.forecast.export_wide import _format_cell, export_wide_csv
from smolsmort.forecast.features import split_family_name
from smolsmort.forecast.tables import read_table
from smolsmort.forecast.tiles import read_ensemble_rows, read_json, read_tiles, table_metrics

INCLUDE_LEVELS = ("details only", "details + predictions", "everything")


def default_folder() -> str:
    return str(Path.home() / "smolsmort-exports")


def _read_members(run_dir, selected, models, saved_recipe, result):
    finalists = read_json(run_dir / "finalists.json", [])
    if selected.get("kind") == "ensemble":
        ids = selected.get("members", [])
        chosen = [model for model in models if model["tile_id"] in ids]
        if len(chosen) != len(ids):
            raise views.ViewError("ensemble member recipe is missing")
    else:
        chosen = [selected]
    members = []
    for model in chosen:
        family = model["family"]
        finalist = next(
            (item for item in finalists if item.get("key") == model.get("recipe_key")), {}
        )
        genome = finalist.get("genome") or model.get("recipe")
        if not genome:
            genome = saved_recipe.get("genome")
        if not genome:
            raise views.ViewError(f"no saved recipe for {family}")
        target = run_dir / "families" / family
        exit_test = read_json(target / "exit_test.json", {})
        if exit_test.get("recipe_key") != model.get("recipe_key"):
            exit_test = {}
        summary = result.get("families", {}).get(family, {})
        facts = {
            key: summary[key]
            for key in (
                "state",
                "fitness",
                "model_error",
                "baseline_error",
                "val_error",
                "fold_count",
                "fold_scores",
                "horizon_mode",
                "horizon",
                "generation",
                "reason",
                "nthread",
                "holdout",
                "origin_blocks",
            )
            if key in summary
        }
        members.append(
            {
                "family": family,
                "recipe_key": model.get("recipe_key"),
                "genome": genome,
                "fit_facts": facts,
                "exit_test": exit_test,
                **{key: finalist[key] for key in ("seed", "nthread") if key in finalist},
            }
        )
    return members


def build_recipe(run_dir: Path, query: dict, *, made: datetime | None = None) -> dict:
    result = views._require_done(run_dir)
    request = views.read_request(run_dir)
    saved_recipe = read_json(run_dir / "recipe.json", {})
    selection = views._select_model(run_dir, result, query, read_only=True)
    models = read_tiles(run_dir, include_ensembles=False, read_only=True)
    ensembles = read_ensemble_rows(run_dir, read_only=True)
    model_id = selection["model"]
    selected = next((item for item in models + ensembles if item["tile_id"] == model_id), None)
    if selected is None:
        if model_id == "winner":
            selected = next(
                (item for item in models if item["family"] == selection["family"]), None
            )
        else:
            names = set(selection["members"])
            selected = {
                "tile_id": "ensemble",
                "kind": "ensemble",
                "family": "ensemble",
                "members": [item["tile_id"] for item in models if item["family"] in names],
                "recipe": {"weight": "equal"},
            }
    if selected is None:
        raise views.ViewError("no saved model recipe")
    if selected["kind"] == "ensemble" and not selected.get("members"):
        raise views.ViewError("ensemble has no saved member recipes")
    metric = query.get("metric") or "wape"
    if metric not in ("wape", "mae", "mape", "ape", "ae", "wae"):
        raise views.ViewError("unknown export metric")
    metrics = selected.get("metrics", {}).get(metric, {})
    baseline = next((item for item in models if item["family"] == "snaive"), {})
    naive = baseline.get("metrics", {}).get(metric, {}).get("test")
    test = metrics.get("test")
    ratio = test / naive if test is not None and naive else None
    if ratio is None and metric == "wape":
        ratio = selected.get("error_metrics", {}).get("test_ratio", {}).get("value")
    if model_id in ("winner", "ensemble"):
        metrics = {}
        for phase in ("validation", "test"):
            path = views._optional_prediction_path(run_dir, {"model": model_id}, phase)
            if path is not None:
                metrics[phase] = table_metrics(read_table(path)).get(metric)
        test = metrics.get("test")
        ratio = test / naive if test is not None and naive else None
        if ratio is None and metric == "wape":
            payload = views.run_metric_payload(run_dir, request, result=result, model=model_id)
            ratio = payload["error_metrics"].get("test_ratio", {}).get("value")
    return {
        "run_id": run_dir.name,
        "made": (made or datetime.now().astimezone()).isoformat(),
        "model": model_id,
        "name": "ensemble of " + ", ".join(selection["members"])
        if selected["kind"] == "ensemble"
        else selected["family"],
        "kind": "ensemble" if selected["kind"] == "ensemble" else "single",
        "selected": {
            key: selected[key]
            for key in (
                "tile_id",
                "kind",
                "family",
                "recipe_key",
                "members",
                "recipe",
                "metrics",
                "error_metrics",
            )
            if key in selected
        },
        "scores": {"metric": metric, **metrics, "ratio": ratio},
        "request": request,
        "members": _read_members(run_dir, selected, models, saved_recipe, result),
        "weighting": selected.get("recipe", {}) if selected["kind"] == "ensemble" else None,
        "holdout": result.get("holdout"),
    }


def _cell(value):
    if isinstance(value, (dict, list, bool)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows):
    return [
        "| field | value |",
        "| --- | --- |",
        *(f"| {key} | {_cell(value)} |" for key, value in rows if value is not None),
        "",
    ]


def _feature_groups(features):
    groups = {}
    for feature in features:
        operation = split_family_name(feature)[2]
        kind = operation.partition(":")[0].partition("+")[0]
        if kind == "lag":
            kind = "lags"
        elif kind in ("roll", "ewm"):
            kind = "rolling"
        elif kind in ("cal", "date"):
            kind = "calendar"
        elif kind == "lead":
            kind = "leads"
        elif kind in ("measure", "log"):
            kind = "measures"
        elif kind in ("dim", "freq"):
            kind = "dimensions"
        else:
            kind = "other"
        groups.setdefault(kind, []).append(feature)
    return groups.items()


def render_readme(recipe: dict) -> str:
    request = recipe["request"]
    spec = request["spec"]
    columns = spec.get("columns", [])
    lines = ["# run export", ""]
    lines += _table((key, recipe[key]) for key in ("run_id", "made", "name", "kind"))
    lines += ["## scores", ""] + _table(recipe["scores"].items())
    data = [("dataset", Path(spec["source"]).name)] if spec.get("source") else []
    for label, role in (
        ("date column", "time"),
        ("target", "target"),
        ("scaffold dimensions", "dimension"),
    ):
        data.append((label, [column["name"] for column in columns if column["role"] == role]))
    data += [(key, value) for key, value in spec.items() if key != "columns" and key != "source"]
    data += [
        (key, request[key])
        for key in ("transform", "holdout", "horizon_mode", "aggregate", "seed")
        if key in request
    ]
    lines += ["## data and spec", ""] + _table(data)
    if recipe.get("holdout"):
        lines += ["## splits and cut dates", ""] + _table(recipe["holdout"].items())
    for member in recipe["members"]:
        genome = member["genome"]
        lines += [f"## member: {member['family']}", ""]
        lines += _table(
            [
                (key, member[key])
                for key in ("family", "recipe_key", "seed", "nthread")
                if key in member
            ]
            + [(key, genome[key]) for key in ("objective", "transform", "seed") if key in genome]
        )
        if "transform" not in genome:
            lines += [
                "target transform is omitted in the saved genome; smolsmort defaults to none.",
                "",
            ]
        if not any("seed" in source for source in (member, genome, request)):
            lines += ["no random seed is recorded.", ""]
        lines += ["### params", ""] + _table(genome.get("params", {}).items())
        lines += ["### features", ""] + _table(_feature_groups(genome.get("families", [])))
        lines += ["### saved fit facts", ""]
        facts = member["fit_facts"]
        lines += _table((key, value) for key, value in facts.items() if key != "origin_blocks")
        blocks = facts.get("origin_blocks", {})
        if blocks:
            lines += [
                "| phase | start | end | train rows | rows | origin count |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for phase, windows in blocks.items():
                for window in windows:
                    lines.append(
                        "| "
                        + " | ".join(
                            _cell(value)
                            for value in (
                                phase,
                                *(
                                    window.get(key, "")
                                    for key in (
                                        "start",
                                        "end",
                                        "train_rows",
                                        "rows",
                                        "origin_count",
                                    )
                                ),
                            )
                        )
                        + " |"
                    )
            lines += [""]
        exit_test = member["exit_test"]
        if exit_test:
            lines += ["### exit test", ""] + _table(
                (key, value) for key, value in exit_test.items() if key != "verdict"
            )
    if recipe["kind"] == "ensemble":
        lines += ["## ensemble", ""] + _table(recipe["weighting"].items())
        lines += [
            "members use equal weights, saved in `ensembles.json`.",
            "",
            "no model weights are stored.",
            "",
        ]
    if recipe.get("missing_tables"):
        lines += ["## tables not saved", "", ", ".join(recipe["missing_tables"]), ""]
    return "\n".join(lines)


def _table_csv(columns):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(
        [_format_cell(value) for value in row] for row in zip(*columns.values(), strict=True)
    )
    return buffer.getvalue().encode("utf-8")


def export_bundle(run_dir: Path, query: dict) -> Path:
    include = query.get("include", "details only")
    if include not in INCLUDE_LEVELS:
        raise views.ViewError("include must be details only, details + predictions or everything")
    value = query.get("folder")
    if not isinstance(value, str) or not value.strip():
        raise views.ViewError("folder must be a nonempty absolute path")
    try:
        folder = Path(value.strip()).expanduser()
    except RuntimeError as exc:
        raise views.ViewError("folder home directory could not be resolved") from exc
    if not folder.is_absolute():
        raise views.ViewError("folder must be an absolute path")
    folder = folder.resolve()
    recipe = build_recipe(run_dir, query)
    model_query = {"model": recipe["model"]}
    files = {}
    if include != "details only":
        files["results.csv"] = views.export_csv(run_dir, recipe["request"], model_query)
        if recipe["request"]["spec"].get("mode") == "timeseries":
            files["wide.csv"] = export_wide_csv(run_dir, recipe["request"], model_query)
    if include == "everything":
        missing = []
        for kind in ("validation", "test", "forecast", "lead_errors"):
            if recipe["model"].startswith("ensemble:"):
                path = run_dir / "tile_views" / recipe["selected"]["recipe_key"] / f"{kind}.parquet"
                path = path if path.exists() else None
            else:
                path = views._optional_prediction_path(run_dir, model_query, kind)
            if path is not None:
                files[f"{kind}.csv"] = _table_csv(read_table(path))
            else:
                missing.append(kind)
        recipe["missing_tables"] = missing
    files["README.md"] = render_readme(recipe).encode("utf-8")
    files["recipe.json"] = (json.dumps(recipe, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{run_dir.name}_{recipe['made'][:10]}"
    suffix = 0
    while True:
        target = folder / (name if suffix == 0 else f"{name}_{suffix}")
        try:
            target.mkdir()
            break
        except FileExistsError:
            suffix += 1
    try:
        for filename, data in files.items():
            (target / filename).write_bytes(data)
    except OSError:
        shutil.rmtree(target)
        raise
    return target
