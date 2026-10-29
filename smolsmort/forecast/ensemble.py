"""finish only the single winner and the average of distinct family winners"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from smolsmort.forecast.family_worker import read_json, watch_parent
from smolsmort.forecast.importance import combine_importance
from smolsmort.forecast.pipeline import (
    _fitness,
    finish,
    fit_importance,
    genome_from_dict,
    load_workspace,
)
from smolsmort.forecast.spec import spec_from_dict
from smolsmort.forecast.worker import _Run, _write_result, _write_tables


def summarize(final):
    return {
        key: final.get(key)
        for key in (
            "verdict",
            "model_error",
            "baseline_error",
            "band",
            "bands",
            "importance",
            "fold_scores",
            "fold_count",
            "val_error",
            "val_test_gap",
        )
    }


def run(folder, parent_pid):
    watch_parent(parent_pid)
    job = _Run(folder)
    request = read_json(folder / "request.json")
    coordinator = read_json(folder / "coordinator.json")
    nthread = coordinator.get("nthread") or request.get("nthread") or os.cpu_count() or 1
    spec = spec_from_dict(request["spec"])
    ws = load_workspace(
        spec,
        Path(request["prepared"]),
        request["summary"],
        transform=request.get("transform", "none"),
        holdout=request.get("holdout", "block"),
    )
    ranked = read_json(folder / "finalists.json")
    ranked = [
        {**entry, "genome": genome_from_dict(entry["genome"], spec=spec).to_dict()}
        for entry in ranked
    ]
    winner = ranked[0]
    genome = genome_from_dict(winner["genome"])
    prediction_cache = {}
    importance_cache = {}
    final = finish(
        ws,
        genome,
        nthread=nthread,
        validation=winner.get("prediction"),
        prediction_cache=prediction_cache,
        importance_cache=importance_cache,
        aggregate_weight=request.get("aggregate_weight"),
    )
    summary = {
        "winner": {"family": winner["family"], "fitness": winner["fitness"], **summarize(final)}
    }
    chosen = winner["family"]
    ensemble = None
    config = request.get("ensemble", {})
    members = ranked[: config.get("top", 3)]
    if config.get("enabled", True) and members and ws.mode == "timeseries":
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
            prediction_cache=prediction_cache,
            importance_cache=importance_cache,
        )
        summary["ensemble"] = {
            "members": [entry["family"] for entry in members],
            "fitness": scored.fitness,
            **summarize(ensemble),
        }
        if scored.fitness < winner["fitness"]:
            chosen = "ensemble"
    result = {
        "best": winner["genome"],
        "stopped": "families complete",
        "aggregate": request.get("aggregate", "none"),
    }
    _write_result(folder, spec, result, ensemble if chosen == "ensemble" else final)
    recipe = read_json(folder / "recipe.json")
    recipe.update(family=chosen, transform=request.get("transform", "none"))
    if chosen == "ensemble":
        recipe["members"] = [entry["family"] for entry in members]
        recipe["genomes"] = [entry["genome"] for entry in members]
    job.save("recipe.json", recipe)
    states = coordinator["families"]
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
    for entry in ranked:
        name = entry["family"]
        if name not in states:
            continue
        member = genome_from_dict(entry["genome"])
        if ws.mode == "entity":
            importance = final.get("importance")
        else:
            if member.key() not in importance_cache:
                importance_cache[member.key()] = fit_importance(ws, member, nthread=nthread)
            importance = combine_importance(name, importance_cache[member.key()])
        summary["families"][name]["importance"] = importance
        if importance is not None:
            job.save(f"families/{name}/importance.json", importance)
    for label, fitted in (("winner", final), ("ensemble", ensemble)):
        if fitted:
            _write_tables(folder, spec, fitted, prefix=label, aggregate=result["aggregate"])
    job.save("result.json", {**read_json(folder / "result.json"), **summary})
    job.save("finish_complete.json", {"family": chosen})


if __name__ == "__main__":
    run(Path(sys.argv[1]), int(sys.argv[2]))
