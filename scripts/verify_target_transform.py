"""compare target transforms with the production search on a read-only daily source"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from smolsmort.forecast.evaluate import wape
from smolsmort.forecast.pipeline import finish, genome_from_dict, load_workspace
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.search import Budget, search
from smolsmort.forecast.spec import Column, PrepSpec


def compare_transform(source, transform, output):
    spec = PrepSpec(
        source=str(source),
        mode="series",
        task="regression",
        columns=(Column("dteday", "time"), Column("cnt", "target", "sum")),
        step="day",
        horizon=30,
    )
    with TemporaryDirectory(prefix="target-transform-") as folder:
        prepared = prepare(spec, Path(folder))
        ws = load_workspace(spec, prepared.folder, prepared.summary, transform=transform)
        budget = Budget(population=8, max_generations=3, plateau=3, seed=0, nthread=1, time_cap=480)
        result = search(ws, budget, family="xgboost")
        if result["best"] is None:
            raise RuntimeError(f"{transform} search found no usable candidate")
        final = finish(ws, genome_from_dict(result["best"]), nthread=1)
        validation = final["validation"]
        report = {
            "transform": transform,
            "validation_wape": wape(validation["actual"], validation["prediction"]),
            "test_wape": final["model_error"],
            "baseline_wape": final["baseline_error"],
            "checks": final["verdict"]["checks"],
            "trusted": final["verdict"]["trusted"],
            "winner": result["best"],
            "generations": result["generations"],
            "stopped": result["stopped"],
            "leaderboard_rows": len(result["leaderboard"]),
            "test_rows": len(final["test"]["actual"]),
            "forecast_median": float(np.median(final["forecast"]["prediction"])),
        }
    if output is not None:
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--transform", choices=("none", "log1p", "logdiff"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compare_transform(args.source, args.transform, args.output)


if __name__ == "__main__":
    main()
