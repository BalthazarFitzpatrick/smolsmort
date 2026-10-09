"""measure statistical models on the workspace's exact rolling validation rows"""

import argparse
import json
import signal
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from common import bike_workspace
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta


def build_model(name, period):
    if name == "ets":
        return AutoETS(season_length=period)
    if name == "theta":
        return AutoTheta(season_length=period)
    return AutoARIMA(
        season_length=period,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        max_order=4,
        nmodels=20,
        approximation=True,
    )


def compute_wape(truth, predicted):
    return float(np.abs(truth - predicted).sum() / np.abs(truth).sum())


def select_rows(output, expected, alias):
    merged = expected.merge(
        output[["unique_id", "ds", "cutoff", alias]],
        left_on=["unique_id", "ds", "origin"],
        right_on=["unique_id", "ds", "cutoff"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(expected) or merged[alias].isna().any():
        raise ValueError("cross validation did not cover the exact workspace rows")
    return merged[alias].to_numpy()


def compute_baseline(panel, expected, horizon, period):
    lag = int(np.ceil(horizon / period)) * period
    days = 1 if (panel.ds.diff().dropna() == pd.Timedelta(days=1)).all() else 7
    source = panel.rename(columns={"ds": "source", "y": "naive"})
    rows = expected.assign(source=expected.ds - pd.Timedelta(days=lag * days))
    joined = rows.merge(source, on=["unique_id", "source"], validate="many_to_one")
    if len(joined) != len(expected):
        raise ValueError("seasonal naive has insufficient history for matched rows")
    return joined.naive.to_numpy()


def verify_forecasts(panel, expected, predicted, name, period, horizon, freq):
    differences = []
    for index in sorted({0, len(expected) - 1}):
        row = expected.iloc[index]
        history = panel[panel.ds <= row.origin]
        model = StatsForecast(models=[build_model(name, period)], freq=freq, n_jobs=1)
        with warnings.catch_warnings(record=True):
            forecast = model.forecast(df=history, h=horizon)
        value = forecast.loc[forecast.ds == row.ds, str(build_model(name, period))]
        differences.append(abs(float(value.iloc[0]) - float(predicted[index])))
    return float(max(differences))


def verify_updates(panel, expected, predicted, name, period, horizon, freq):
    pivot = expected.origin.iloc[len(expected) // 2]
    changed = panel.copy()
    changed.loc[changed.ds == pivot, "y"] += 1000
    model = StatsForecast(models=[build_model(name, period)], freq=freq, n_jobs=1)
    with warnings.catch_warnings(record=True):
        output = model.cross_validation(
            df=changed,
            h=horizon,
            step_size=1,
            n_windows=int(expected.ds.nunique()),
            refit=False,
        )
    modified = select_rows(output, expected, str(build_model(name, period)))
    difference = np.abs(modified - predicted)
    return {
        "changed_step": str(pivot),
        "added_y": 1000,
        "before_max_abs_diff": float(difference[expected.origin < pivot].max()),
        "after_max_abs_diff": float(difference[expected.origin >= pivot].max()),
    }


def measure_bucket(ws, panel, bucket, name, period, freq, max_windows=None):
    horizon = bucket[1]
    frame = ws.frames[bucket]
    mask = ws.series_masks[bucket]["val"]
    expected = pd.DataFrame(
        {
            "unique_id": frame.series[mask],
            "ds": pd.to_datetime(frame.step[mask]).astype("datetime64[ns]"),
            "origin": pd.to_datetime(frame.origin[mask]).astype("datetime64[ns]"),
            "y": frame.y[mask],
        }
    )
    if max_windows:
        latest = sorted(expected.ds.unique())[-max_windows:]
        expected = expected[expected.ds.isin(latest)].reset_index(drop=True)
    panel = panel[panel.ds <= expected.ds.max()].copy()
    windows = int(expected.ds.nunique())
    result = {
        "bucket": list(bucket),
        "rows": len(expected),
        "windows": windows,
        "first_origin": str(expected.origin.min()),
        "last_target": str(expected.ds.max()),
        "initial_history": int((panel.ds <= expected.origin.min()).sum()),
        "modes": {},
    }
    try:
        result["seasonal_naive_wape"] = compute_wape(
            expected.y.to_numpy(), compute_baseline(panel, expected, horizon, period)
        )
    except ValueError as error:
        result["seasonal_naive_error"] = str(error)
    forecasts = {}
    alias = str(build_model(name, period))
    for refit in (True, False):
        started = time.perf_counter()
        try:
            model = StatsForecast(models=[build_model(name, period)], freq=freq, n_jobs=1)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                output = model.cross_validation(
                    df=panel, h=horizon, step_size=1, n_windows=windows, refit=refit
                )
            seconds = time.perf_counter() - started
            predicted = select_rows(output, expected, alias)
            forecasts[refit] = predicted
            result["modes"][str(refit).lower()] = {
                "seconds": seconds,
                "wape": compute_wape(expected.y.to_numpy(), predicted),
                "columns": list(output.columns),
                "all_cv_rows": len(output),
                "prediction_finite": bool(np.isfinite(predicted).all()),
                "warning_count": len(notices),
                "warning_examples": sorted({str(notice.message) for notice in notices})[:3],
            }
            if refit:
                result["standalone_max_abs_diff"] = verify_forecasts(
                    panel, expected, predicted, name, period, horizon, freq
                )
        except (ValueError, NotImplementedError, ZeroDivisionError, IndexError, KeyError) as error:
            result["modes"][str(refit).lower()] = {
                "seconds": time.perf_counter() - started,
                "error": str(error),
                "error_type": type(error).__name__,
            }
    if len(forecasts) == 2:
        result["refit_max_abs_diff"] = float(np.max(np.abs(forecasts[True] - forecasts[False])))
        result["refit_false_speedup"] = (
            result["modes"]["true"]["seconds"] / result["modes"]["false"]["seconds"]
        )
        if horizon == 1:
            result["rolling_update_proof"] = verify_updates(
                panel, expected, forecasts[False], name, period, horizon, freq
            )
    return result


def run_spike(args):
    period = 7 if args.unit == "day" else 52
    freq = "D" if args.unit == "day" else "W-MON"
    result = {
        "unit": args.unit,
        "model": args.model,
        "season_length": period,
        "horizon": args.horizon,
        "arima_limits": {
            "max_p": 2,
            "max_q": 2,
            "max_P": 1,
            "max_Q": 1,
            "max_order": 4,
            "nmodels": 20,
            "approximation": True,
        }
        if args.model == "arima"
        else None,
        "buckets": [],
    }
    with bike_workspace(args.source, unit=args.unit, horizon=args.horizon) as (ws, panel):
        result["profile"] = ws.profile
        panel["ds"] = pd.to_datetime(panel.ds).astype("datetime64[ns]")
        if args.inspect_initial:
            first = next(iter(ws.frames))
            frame = ws.frames[first]
            origin = frame.origin[ws.series_masks[first]["val"]].min()
            model = build_model(args.model, period)
            with warnings.catch_warnings(record=True):
                model.fit(panel.loc[panel.ds <= pd.Timestamp(origin), "y"].to_numpy())
            result["initial_fit"] = {
                name: model.model_.get(name)
                for name in ["m", "components", "method", "modeltype", "arma"]
                if name in model.model_
            }
            args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
            print(json.dumps(result["initial_fit"], default=str), flush=True)
            return
        for bucket in ws.frames:
            if args.bucket is not None and bucket[1] != args.bucket:
                continue
            try:
                measured = measure_bucket(
                    ws, panel, bucket, args.model, period, freq, args.max_windows
                )
            except (ValueError, NotImplementedError, ZeroDivisionError, TimeoutError) as error:
                measured = {"bucket": list(bucket), "error": str(error)}
            result["buckets"].append(measured)
            print(json.dumps(measured, default=str), flush=True)
            args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
            if "error" in measured and "480 seconds" in measured["error"]:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--unit", choices=["day", "week"], default="day")
    parser.add_argument("--model", choices=["ets", "theta", "arima"], required=True)
    parser.add_argument("--bucket", type=int)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--inspect-initial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/private/tmp")):
        parser.error("output must be inside /private/tmp")

    def stop_run(signum, frame):
        raise TimeoutError("spike exceeded 480 seconds; reduce windows or select one bucket")

    signal.signal(signal.SIGALRM, stop_run)
    signal.alarm(480)
    run_spike(args)


if __name__ == "__main__":
    main()
