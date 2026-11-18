"""refit an exported recipe with the same preparation and finish paths as a run"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from smolsmort.forecast.pipeline import finish, genome_from_dict, load_workspace
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.spec import spec_from_dict


def recreate_export(recipe_path, dataset_path, output_path, *, nthread=1):
    """rebuild saved members and write their combined future forecast without searching"""
    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    request = recipe["request"]
    spec = spec_from_dict({**request["spec"], "source": str(Path(dataset_path).resolve())})
    if spec.mode != "timeseries":
        raise ValueError("recreation requires a time series recipe")
    weighting = recipe.get("weighting", "equal")
    if isinstance(weighting, dict):
        weighting = weighting.get("weight", weighting.get("method"))
    if weighting != "equal":
        raise ValueError("recreation supports saved equal weighting only")
    members = [genome_from_dict(member["genome"], spec=spec) for member in recipe["members"]]
    if not members:
        raise ValueError("recipe has no saved members")
    if nthread < 1:
        raise ValueError("threads must be positive")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    with tempfile.TemporaryDirectory(prefix="smolsmort-recreate-") as temporary:
        prepared = prepare(spec, Path(temporary))
        workspace = load_workspace(
            spec,
            prepared.folder,
            prepared.summary,
            transform=request.get("transform", "none"),
            holdout=request.get("holdout", "block"),
            horizon_mode=request.get("horizon_mode", "mixed"),
        )
        fitted = finish(
            workspace,
            members[0],
            members=members if len(members) > 1 else None,
            nthread=nthread,
        )
    table = fitted["forecast"]
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(table)
        writer.writerows(zip(*table.values(), strict=True))
    return output


def recreate_script():
    """small portable entrypoint shipped beside recipe.json"""
    return '''"""refit this export with an installed smolsmort checkout and its forecast extras"""

import argparse
from pathlib import Path

from smolsmort.forecast.export_recreate import recreate_export

parser = argparse.ArgumentParser(description="refit the saved ensemble and write its forecast")
parser.add_argument("dataset", type=Path)
parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
parser.add_argument("--threads", type=int, default=1)
arguments = parser.parse_args()
output = recreate_export(
    Path(__file__).with_name("recipe.json"),
    arguments.dataset,
    arguments.output,
    nthread=arguments.threads,
)
print(output.resolve())
'''


def recreation_steps():
    return [
        "install this version of smolsmort with the forecast extra and the providers needed "
        "by the member families (lightgbm, catboost, stats, or holidays where used)",
        "copy the original dataset to the other machine; keep the saved columns, filters, "
        "cutoff date, and history unchanged to reproduce the same training windows",
        "from the installed smolsmort checkout, run `uv run --extra forecast "
        "--extra lightgbm --extra catboost --extra stats --extra holidays "
        "<export-folder>/recreate.py <dataset-path> --output <new-predictions.csv>`",
        "the script reads recipe.json, prepares the supplied dataset, refits each saved member "
        "with smolsmort's finish pipeline, and writes the equal-weight future forecast; "
        "validation and test windows are refit too, with final forecasts fitted on all known rows",
    ]
