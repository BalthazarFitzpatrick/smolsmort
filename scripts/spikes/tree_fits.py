"""compare tree libraries on identical workspace rows and probe category translation"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import xgboost
from common import SOURCE, bike_workspace

from smolsmort.forecast.evaluate import wape
from smolsmort.forecast.model import fit_model, predict


def fit_lightgbm(x, y, types, objective, params):
    categorical = [i for i, kind in enumerate(types) if kind == "c"]
    config = {
        "objective": objective,
        "verbosity": -1,
        "num_threads": 1,
        "seed": 0,
        "deterministic": True,
        "force_col_wise": True,
        **params,
    }
    cut = int(len(y) * 0.85)
    rounds = 2000
    if cut >= 10 and len(y) - cut >= 10:
        head = lgb.Dataset(x[:cut], label=y[:cut], categorical_feature=categorical)
        tail = lgb.Dataset(x[cut:], label=y[cut:], categorical_feature=categorical, reference=head)
        probe = lgb.train(
            config,
            head,
            num_boost_round=2000,
            valid_sets=[tail],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        rounds = probe.best_iteration
    fitted = lgb.train(
        config, lgb.Dataset(x, label=y, categorical_feature=categorical), num_boost_round=rounds
    )
    return fitted, rounds


def probe_categories():
    # bike total has no dimensions; exercise the repo's float32 codes separately
    x = np.column_stack([np.tile([0, 1, 2, np.nan], 25), np.arange(100)]).astype("float32")
    y = np.tile([10, 30, 20, 15], 25).astype(float)
    result = {}
    for library in ("lightgbm", "xgboost"):
        start = perf_counter()
        if library == "lightgbm":
            fitted, rounds = fit_lightgbm(x, y, ["c", "q"], "regression", {"min_data_in_leaf": 5})
            output = fitted.predict(x, num_threads=1)
        else:
            fitted = fit_model(
                x, y=y, objective="squared", feature_types=["c", "q"], max_rounds=2000, patience=50
            )
            rounds = fitted.rounds
            output = predict(fitted, x)
        result[library] = {
            "finite": bool(np.isfinite(output).all()),
            "rounds": rounds,
            "seconds": perf_counter() - start,
            "mae": float(np.abs(y - output).mean()),
        }
    return result


def run_fits(source, unit, horizon):
    results = []
    with bike_workspace(source, unit, horizon) as (ws, data):
        metadata = {
            "unit": unit,
            "horizon": horizon,
            "panel_rows": len(data),
            "profile_period": ws.profile.get("period"),
            "genes_period": ws.profile["genes"].get("period"),
            "families": ws.families,
        }
        for bucket, frame in ws.frames.items():
            masks = ws.series_masks[bucket]
            x, names, types = frame.features.pick(set(frame.features.families))
            train = np.flatnonzero(masks["train"])
            train = train[np.argsort(frame.step[train], kind="stable")]
            val = masks["val"]
            if len(train) < 20:
                record = {
                    "bucket": list(bucket),
                    "library": "xgboost",
                    "train_rows": len(train),
                    "val_rows": int(val.sum()),
                    "error": f"only {len(train)} rows carry a usable target - too few to fit",
                }
                print(json.dumps(record), flush=True)
                results.append(record)
                continue
            configs = [
                ("xgboost", objective, {"max_depth": 3, "eta": 0.05})
                for objective in ("absolute", "squared")
            ]
            if bucket == next(iter(ws.frames)):
                configs += [
                    (
                        "lightgbm",
                        objective,
                        {"num_leaves": leaves, "learning_rate": 0.05, "min_data_in_leaf": 20},
                    )
                    for objective in ("regression_l1", "regression")
                    for leaves in (7, 15, 31)
                ]
            for library, objective, params in configs:
                start = perf_counter()
                if library == "xgboost":
                    fitted = fit_model(
                        x[train],
                        y=frame.y[train],
                        objective=objective,
                        feature_types=types,
                        params=params,
                    )
                    rounds = fitted.rounds
                    fit_seconds = perf_counter() - start
                    output = predict(fitted, x[val])
                else:
                    fitted, rounds = fit_lightgbm(
                        x[train], frame.y[train], types, objective, params
                    )
                    fit_seconds = perf_counter() - start
                    output = fitted.predict(x[val], num_threads=1)
                record = {
                    "bucket": list(bucket),
                    "library": library,
                    "objective": objective,
                    "params": params,
                    "rounds": rounds,
                    "fit_seconds": fit_seconds,
                    "wape": wape(frame.y[val], output),
                    "train_rows": len(train),
                    "val_rows": int(val.sum()),
                    "features": len(names),
                    "nan_cells_train": int(np.isnan(x[train]).sum()),
                    "category_columns": sum(kind == "c" for kind in types),
                }
                print(json.dumps(record), flush=True)
                results.append(record)
        metadata["validation_start"] = str(frame.step[val].min())
        metadata["validation_end"] = str(frame.step[val].max())
    return {"metadata": metadata, "fits": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--unit", choices=["day", "week"], default="day")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/private/tmp")):
        parser.error("output must be inside /private/tmp")
    result = run_fits(args.source, args.unit, args.horizon)
    result["runtime"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "xgboost": xgboost.__version__,
    }
    result["categorical_probe"] = probe_categories()
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
