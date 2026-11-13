"""finish family winners and the average of distinct family winners"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from smolsmort.forecast.evaluate import mean_predictions
from smolsmort.forecast.exit_test import restore_exit_cache, score_isolated
from smolsmort.forecast.families import get_family
from smolsmort.forecast.family_worker import read_json, watch_parent
from smolsmort.forecast.importance import combine_importance
from smolsmort.forecast.pipeline import (
    PipelineError,
    _fitness,
    _known_rows,
    _test_windows,
    _validation_masks,
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
            "series_bands",
            "bands_50",
            "series_bands_50",
            "band_summary",
            "series_coverage",
            "importance",
            "fold_scores",
            "fold_count",
            "val_error",
            "val_test_gap",
            "horizon_mode",
            "horizon",
            "origin_blocks",
            "lead_errors",
            "validation_lead_errors",
        )
    }


def has_finish_cache(ws, genome, entry, predictions):
    """a recipe marker alone cannot prove all required windows were published"""
    for bucket in ws.frames:
        masks = ws.series_masks[bucket]
        windows = list(_test_windows(ws, bucket))
        if entry.get("prediction") is None:
            windows.extend(_validation_masks(ws, bucket))
        windows = [(fit, rows) for fit, rows in windows if rows.any()]
        windows.append((_known_rows(masks), masks["future"]))
        for fit_rows, eval_rows in windows:
            key = (genome.key(), bucket, fit_rows.tobytes(), eval_rows.tobytes())
            prediction = predictions.get(key)
            if prediction is None or prediction.shape != (int(eval_rows.sum()),):
                return False
    return True


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
        horizon_mode=request.get("horizon_mode", "mixed"),
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
    isolated = {
        entry["family"]
        for entry in ranked
        if getattr(get_family(entry["family"]), "isolated", False)
    }
    if ws.mode == "timeseries":
        for entry in ranked:
            name = entry["family"]
            member = genome_from_dict(entry["genome"])
            target = folder / "families" / name
            restored = restore_exit_cache(
                target,
                member,
                prediction_cache,
                importance_cache,
            )
            if name in isolated and (
                not restored or not has_finish_cache(ws, member, entry, prediction_cache)
            ):
                score_isolated(folder, name, entry, deadline=coordinator["deadline"])
                if not restore_exit_cache(target, member, prediction_cache, importance_cache):
                    raise PipelineError(f"family {name!r}: isolated final fit cache is missing")
                if not has_finish_cache(ws, member, entry, prediction_cache):
                    raise PipelineError(f"family {name!r}: isolated final fit cache is incomplete")
    final = finish(
        ws,
        genome,
        nthread=nthread,
        validation=winner.get("prediction"),
        prediction_cache=prediction_cache,
        importance_cache=importance_cache,
        aggregate_weight=request.get("aggregate_weight"),
        cache_only_families=isolated,
    )
    summary = {
        "winner": {"family": winner["family"], "fitness": winner["fitness"], **summarize(final)}
    }
    chosen = winner["family"]
    ensemble = None
    config = request.get("ensemble", {})
    members = ranked[: config.get("top", 3)]
    if config.get("enabled", True) and members and ws.mode == "timeseries":
        prediction = mean_predictions([entry["prediction"] for entry in members])
        truth = np.concatenate(
            [frame.y[ws.series_masks[bucket]["val"]] for bucket, frame in ws.frames.items()]
        )
        periods = np.concatenate(
            [
                (frame.origin if getattr(ws, "horizon_mode", "mixed") == "single" else frame.step)[
                    ws.series_masks[bucket]["val"]
                ].astype(int)
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
            cache_only_families=isolated,
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
        if ws.mode == "timeseries":
            family_final = (
                final
                if name == winner["family"]
                else finish(
                    ws,
                    member,
                    nthread=nthread,
                    validation=entry.get("prediction"),
                    prediction_cache=prediction_cache,
                    importance_cache=importance_cache,
                    cache_only_families=isolated,
                )
            )
            _write_tables(folder / "families" / name, spec, family_final)
            summary["families"][name].update(summarize(family_final))
        if ws.mode == "entity":
            importance = final.get("importance")
        else:
            if member.key() not in importance_cache:
                if name in isolated:
                    raise PipelineError(f"family {name!r}: isolated importance cache is missing")
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
