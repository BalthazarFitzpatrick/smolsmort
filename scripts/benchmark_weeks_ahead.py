"""time direct pipeline fits on a synthetic weekly panel; no workers or source data"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from smolsmort.forecast import pipeline
from smolsmort.forecast.families import get_family
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec


def make_panel(series=13, weeks=165, horizon=8, seed=29):
    rng = np.random.default_rng(seed)
    positions = np.arange(weeks)
    steps = np.datetime64("2020-01-06") + positions * np.timedelta64(7, "D")
    keys = [f"s{index:02d}" for index in range(series)]
    targets = []
    for index in range(series):
        noise = rng.normal(0, 3, weeks)
        state = np.zeros(weeks)
        for step in range(1, weeks):
            state[step] = 0.8 * state[step - 1] + noise[step]
        targets.append(
            80 + index * 4 + positions / 20 + 10 * np.sin(positions * 2 * np.pi / 52) + state
        )
    panel = {
        SERIES: np.repeat(keys, weeks),
        STEP: np.tile(steps, series),
        "segment": np.repeat(keys, weeks),
        "y": np.concatenate(targets),
        "market": np.tile(100 + positions / 5, series),
    }
    future = steps[-1] + np.arange(1, horizon + 1) * np.timedelta64(7, "D")
    scaffold = {SERIES: np.repeat(keys, horizon), STEP: np.tile(future, series)}
    spec = PrepSpec(
        "synthetic.csv",
        "timeseries",
        "regression",
        (
            Column("day", "time"),
            Column("segment", "dimension"),
            Column("y", "target", "sum"),
            Column("market", "measure", "mean"),
        ),
        step="week",
        horizon=horizon,
    )
    return spec, {"panel.parquet": panel, "scaffold.parquet": scaffold}


def time_call(action, *args, **kwargs):
    started = time.perf_counter()
    result = action(*args, **kwargs)
    return time.perf_counter() - started, result


def fit_frames(workspace, genome, *, final):
    fitted = []
    for bucket, frame in workspace.frames.items():
        masks = workspace.series_masks[bucket]
        rows = ~frame.future & np.isfinite(frame.y) if final else masks["train"]
        model, _ = pipeline._timeseries_fit(workspace, genome, frame, rows, 1)
        fitted.append(model)
    return fitted


def benchmark(*, repeats=3, families=("ridge", "lightgbm")):
    spec, tables = make_panel()
    report = {
        "series": 13,
        "weeks": 165,
        "horizon": 8,
        "holdout": "seasonal",
        "threads": 1,
        "repeats": repeats,
        "platform": platform.platform(),
        "fit_scope": "one model per mode frame; mixed fits three bucket models",
        "finish_scope": "test refits, final forecast, bands and importance; cached validation",
        "results": [],
    }
    with TemporaryDirectory(prefix="weeks-ahead-benchmark-") as folder:
        with patch.object(
            pipeline, "read_table", side_effect=lambda path, **kwargs: tables[path.name]
        ):
            workspaces = {}
            for mode in ("single", "mixed"):
                elapsed, workspace = time_call(
                    pipeline.load_workspace,
                    spec,
                    Path(folder),
                    {"step": "week"},
                    holdout="seasonal",
                    horizon_mode=mode,
                )
                workspaces[mode] = (workspace, elapsed)
        for family_name in families:
            family = get_family(family_name)
            available, reason = family.available()
            if not available:
                raise RuntimeError(reason)
            params = family.defaults()
            objective = "squared" if family_name == "ridge" else "regression"
            for mode, (workspace, build_seconds) in workspaces.items():
                genome = pipeline.make_genome(workspace.families, objective, params, family_name)
                timings = {
                    name: [] for name in ("train_fit_s", "final_fit_s", "validation_s", "finish_s")
                }
                for _ in range(repeats):
                    elapsed, _ = time_call(fit_frames, workspace, genome, final=False)
                    timings["train_fit_s"].append(elapsed)
                    elapsed, _ = time_call(fit_frames, workspace, genome, final=True)
                    timings["final_fit_s"].append(elapsed)
                    elapsed, scored = time_call(pipeline.score, workspace, genome, nthread=1)
                    timings["validation_s"].append(elapsed)
                    elapsed, final = time_call(
                        pipeline.finish, workspace, genome, nthread=1, validation=scored.prediction
                    )
                    timings["finish_s"].append(elapsed)
                result = {
                    "family": family_name,
                    "mode": mode,
                    "objective": objective,
                    "params": params,
                    "feature_families": len(genome.families),
                    "feature_columns": [
                        pipeline._timeseries_design(genome, frame)[0].shape[1]
                        for frame in workspace.frames.values()
                    ],
                    "models_per_fit": len(workspace.frames),
                    "validation_blocks": len(workspace.series_folds),
                    "train_rows": sum(
                        int(masks["train"].sum()) for masks in workspace.series_masks.values()
                    ),
                    "known_rows": sum(
                        int((~frame.future & np.isfinite(frame.y)).sum())
                        for frame in workspace.frames.values()
                    ),
                    "validation_rows": len(scored.prediction),
                    "forecast_rows": len(final["forecast"]["prediction"]),
                    "workspace_s": build_seconds,
                    "median_seconds": {
                        name: statistics.median(values) for name, values in timings.items()
                    },
                    "samples_seconds": timings,
                }
                if family_name == "lightgbm":
                    result["fit_defaults"] = {"max_rounds": 2000, "patience": 50, "inner": 0.15}
                report["results"].append(result)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    rendered = json.dumps(benchmark(repeats=args.repeats), indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
