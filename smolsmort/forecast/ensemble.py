"""finish only the single winner and the average of distinct family winners"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from smolsmort.forecast.family_worker import read_json, watch_parent
from smolsmort.forecast.pipeline import _fitness, finish, genome_from_dict, load_workspace
from smolsmort.forecast.spec import spec_from_dict
from smolsmort.forecast.tables import write_table
from smolsmort.forecast.worker import _Run, _write_result


def summarize(final):
    return {
        key: final.get(key) for key in ("verdict", "model_error", "baseline_error", "band", "bands")
    }


def run(folder, parent_pid):
    watch_parent(parent_pid)
    job = _Run(folder)
    request = read_json(folder / "request.json")
    spec = spec_from_dict(request["spec"])
    ws = load_workspace(spec, Path(request["prepared"]), request["summary"])
    ranked = read_json(folder / "finalists.json")
    winner = ranked[0]
    genome = genome_from_dict(winner["genome"])
    nthread = request.get("nthread") or 1
    final = finish(ws, genome, nthread=nthread, validation=winner.get("prediction"))
    summary = {
        "winner": {"family": winner["family"], "fitness": winner["fitness"], **summarize(final)}
    }
    chosen = winner["family"]
    ensemble = None
    config = request.get("ensemble", {})
    members = ranked[: config.get("top", 3)]
    if config.get("enabled", True) and members and ws.mode == "series":
        prediction = np.mean([entry["prediction"] for entry in members], axis=0)
        truth = np.concatenate(
            [frame.y[ws.series_masks[bucket]["val"]] for bucket, frame in ws.frames.items()]
        )
        periods = np.concatenate(
            [
                frame.step[ws.series_masks[bucket]["val"]].astype(int)
                for bucket, frame in ws.frames.items()
            ]
        )
        scored = _fitness(ws, prediction, truth, periods)
        ensemble = finish(
            ws,
            genome,
            nthread=nthread,
            members=[genome_from_dict(entry["genome"]) for entry in members],
            validation=prediction,
        )
        summary["ensemble"] = {
            "members": [entry["family"] for entry in members],
            "fitness": scored.fitness,
            **summarize(ensemble),
        }
        if scored.fitness < winner["fitness"]:
            chosen = "ensemble"
    result = {"best": winner["genome"], "stopped": "families complete"}
    _write_result(folder, spec, result, ensemble if chosen == "ensemble" else final)
    recipe = read_json(folder / "recipe.json")
    recipe.update(family=chosen, transform=request.get("transform", "none"))
    if chosen == "ensemble":
        recipe["members"] = [entry["family"] for entry in members]
        recipe["genomes"] = [entry["genome"] for entry in members]
    job.save("recipe.json", recipe)
    states = read_json(folder / "coordinator.json")["families"]
    summary["families"] = {
        name: {
            "fitness": status.get("best_error"),
            "candidates": status.get("candidates_done", 0),
            "state": status["state"],
            "pruned_reason": status.get("pruned_reason"),
            "terminated": status.get("terminated", False),
        }
        for name, status in states.items()
    }
    for label, fitted in (("winner", final), ("ensemble", ensemble)):
        if fitted:
            for kind in ("validation", "test", "forecast"):
                columns = fitted.get(kind)
                if (
                    isinstance(columns, dict)
                    and columns
                    and all(isinstance(v, np.ndarray) for v in columns.values())
                ):
                    write_table(folder / f"{label}_{kind}.parquet", columns)
    job.save("result.json", {**read_json(folder / "result.json"), **summary})
    job.save("finish_complete.json", {"family": chosen})


if __name__ == "__main__":
    run(Path(sys.argv[1]), int(sys.argv[2]))
