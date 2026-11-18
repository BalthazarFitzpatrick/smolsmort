"""exported recipes refit through the run pipeline without another search"""

import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from smolsmort.forecast.export_recreate import recreate_export, recreate_script, recreation_steps
from smolsmort.forecast.pipeline import finish, genome_from_dict, load_workspace
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.spec import spec_from_dict
from smolsmort.forecast.tables import read_table

REAL_RUN = (
    Path.home()
    / "Documents/dev/ai_lab/forecast_workspace/.forecast-runs"
    / "20261115-155627-6f006a"
)


@pytest.fixture
def export_recipe(tmp_path):
    dataset = tmp_path / "dataset.csv"
    with dataset.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["day", "units"])
        for index in range(160):
            day = dt.date(2024, 1, 1) + dt.timedelta(days=index)
            writer.writerow([day, 50 + index / 10 + 5 * np.sin(index * 2 * np.pi / 7)])
    request = {
        "spec": {
            "source": "original-dataset.csv",
            "mode": "timeseries",
            "task": "regression",
            "step": "day",
            "horizon": 4,
            "columns": [
                {"name": "day", "role": "time"},
                {"name": "units", "role": "target", "aggregation": "sum"},
            ],
        },
        "transform": "log1p",
        "holdout": "block",
        "horizon_mode": "single",
    }
    members = [
        {
            "family": family,
            "genome": {
                "family": family,
                "families": ["units@day:lag+0"] if family == "ridge" else [],
                "objective": "squared",
                "params": {"alpha": 1.0} if family == "ridge" else {},
                "transform": "log1p",
            },
        }
        for family in ("ridge", "snaive")
    ]
    recipe = {"request": request, "members": members, "weighting": {"weight": "equal"}}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    return path, dataset, recipe


def test_script_refits_same_member_recipes_and_equal_forecast(export_recipe, tmp_path):
    path, dataset, recipe = export_recipe
    script = tmp_path / "recreate.py"
    script.write_text(recreate_script(), encoding="utf-8")
    output = tmp_path / "predictions.csv"
    result = subprocess.run(
        [sys.executable, str(script), str(dataset), "--output", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(output) in result.stdout
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    request = recipe["request"]
    spec = spec_from_dict({**request["spec"], "source": str(dataset)})
    prepared = prepare(spec, tmp_path / "expected")
    workspace = load_workspace(
        spec, prepared.folder, prepared.summary, transform="log1p", horizon_mode="single"
    )
    predictions = [
        finish(workspace, genome_from_dict(member["genome"]))["forecast"]["prediction"]
        for member in recipe["members"]
    ]
    np.testing.assert_allclose(
        [float(row["prediction"]) for row in rows], np.mean(predictions, axis=0)
    )
    assert len(recreation_steps()) == 4


def test_recreation_refuses_unsupported_weighting_and_overwrite(export_recipe, tmp_path):
    path, dataset, recipe = export_recipe
    output = tmp_path / "predictions.csv"
    output.write_text("keep me", encoding="utf-8")
    recipe["weighting"] = {"weight": "fitted"}
    path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="equal weighting"):
        recreate_export(path, dataset, output)
    recipe["weighting"] = {"weight": "equal"}
    path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        recreate_export(path, dataset, output)
    assert output.read_text(encoding="utf-8") == "keep me"


def test_real_best_ensemble_recreation(tmp_path):
    if not REAL_RUN.is_dir():
        pytest.skip("real run folder is absent")
    pytest.importorskip("catboost", reason="real ensemble requires the catboost extra")
    request = json.loads((REAL_RUN / "request.json").read_text(encoding="utf-8"))
    source = Path(request["spec"]["source"])
    if not source.is_file():
        pytest.skip("real run dataset is absent")
    ensembles = json.loads((REAL_RUN / "ensembles.json").read_text(encoding="utf-8"))
    ensemble = min(ensembles, key=lambda entry: entry["metrics"]["wape"]["validation"])
    finalists = json.loads((REAL_RUN / "finalists.json").read_text(encoding="utf-8"))
    members = []
    for member_id in ensemble["members"]:
        _, family, key = member_id.split(":")
        member = next(
            entry
            for entry in finalists
            if entry["family"] == family and genome_from_dict(entry["genome"]).key() == key
        )
        members.append(member)
    recipe = {"request": request, "members": members, "weighting": ensemble["recipe"]}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    output = recreate_export(path, source, tmp_path / "predictions.csv")
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    saved = read_table(REAL_RUN / "tile_views" / ensemble["recipe_key"] / "forecast.parquet")
    assert len(rows) == len(saved["prediction"])
    assert all(np.isfinite(float(row["prediction"])) for row in rows)
    np.testing.assert_allclose(
        [float(row["prediction"]) for row in rows], saved["prediction"], rtol=1e-7, atol=1e-7
    )
