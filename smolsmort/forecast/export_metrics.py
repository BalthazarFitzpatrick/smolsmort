"""shared slice scores and compact group, pooled and summed error summaries"""

import math

from smolsmort.forecast.metrics import prediction_metrics

MEASURES = (
    "wape",
    "naive_wape",
    "ratio",
    "mae",
    "bias",
    "actual_sum",
    "prediction_sum",
    "rows",
    "coverage_50",
    "coverage_80",
)
BANDS = ("lower_50", "upper_50", "lower_80", "upper_80")


def table_rows(table):
    return [dict(zip(table, values, strict=True)) for values in zip(*table.values(), strict=True)]


def number(value):
    return float(value) if value is not None and value != "" else float("nan")


def score_rows(rows):
    """use finite pairs for model errors, with independently available naive pairs"""
    actuals = [number(row.get("actual")) for row in rows]
    predictions = [number(row.get("forecast")) for row in rows]
    scores = prediction_metrics(actuals, predictions)
    naive = prediction_metrics(actuals, [number(row.get("naive_prediction")) for row in rows])
    pairs = [
        (row, actual, prediction)
        for row, actual, prediction in zip(rows, actuals, predictions, strict=True)
        if math.isfinite(actual) and math.isfinite(prediction)
    ]
    count = len(pairs)
    output = {
        "wape": scores["wape"],
        "naive_wape": naive["wape"],
        "ratio": scores["wape"] / naive["wape"]
        if scores["wape"] is not None and naive["wape"]
        else None,
        "mae": scores["mae"],
        "bias": sum(prediction - actual for _, actual, prediction in pairs) / count
        if count
        else None,
        "actual_sum": sum(actual for _, actual, _ in pairs) if count else None,
        "prediction_sum": sum(prediction for _, _, prediction in pairs) if count else None,
        "rows": count,
    }
    for level in (50, 80):
        edges = [
            (actual, number(row.get(f"lower_{level}")), number(row.get(f"upper_{level}")))
            for row, actual, _ in pairs
        ]
        output[f"coverage_{level}"] = (
            sum(lower <= actual <= upper for actual, lower, upper in edges) / count
            if count and all(math.isfinite(lo) and math.isfinite(hi) for _, lo, hi in edges)
            else None
        )
    return output


def slice_wapes(table, dims):
    """share the demo's group and pooled horizon scores with saved-table exports"""
    rows = table_rows(table)
    groups, horizons = {}, {}
    for row in rows:
        horizon = row.get("horizon", "")
        key = (*[row.get(dim, "") for dim in dims], horizon)
        groups.setdefault(key, []).append(row)
        horizons.setdefault(horizon, []).append(row)
    return (
        {key: score_rows(values)["wape"] for key, values in groups.items()},
        {key: score_rows(values)["wape"] for key, values in horizons.items()},
    )


def append_wapes(table, dims):
    groups, horizons = slice_wapes(table, dims)
    rows = table_rows(table)
    return {
        **table,
        "wape": [
            groups[(*[row.get(dim, "") for dim in dims], row.get("horizon", ""))] for row in rows
        ],
        "total_wape": [horizons[row.get("horizon", "")] for row in rows],
    }


def metric_columns(tables, dims, date, totals):
    """emit each group and both total levels at every horizon and across all horizons"""
    columns = {name: [] for name in ["dataset_slice", *dims, "horizon_step", *MEASURES]}
    for phase in ("validation", "test"):
        rows = table_rows(tables[phase])
        groups, sums = {}, {}
        for row in rows:
            group = tuple(row.get(dim, "") for dim in dims)
            groups.setdefault(group, []).append(row)
            key = (str(row[date]), str(row.get("forecast_made_week", "")), row["horizon"])
            sums.setdefault(key, []).append(row)
        total_rows = []
        for key, members in sums.items():
            total = {"horizon": key[2]}
            for name in ("actual", "forecast", "naive_prediction"):
                values = [number(row.get(name)) for row in members]
                total[name] = sum(values) if all(math.isfinite(value) for value in values) else None
            calibrated = totals.get(phase, {}).get(key)
            if calibrated is not None:
                total.update(zip(BANDS, calibrated[2:], strict=True))
            total_rows.append(total)
        levels = [
            *groups.items(),
            (("all groups pooled",) * len(dims), rows),
            (("total weekly sum",) * len(dims), total_rows),
        ]
        horizons = sorted({row["horizon"] for row in rows})
        for group, members in levels:
            for horizon in [*horizons, "all"]:
                selected = (
                    members
                    if horizon == "all"
                    else [row for row in members if row["horizon"] == horizon]
                )
                record = {
                    "dataset_slice": phase,
                    **dict(zip(dims, group, strict=True)),
                    "horizon_step": horizon,
                    **score_rows(selected),
                }
                for name in columns:
                    columns[name].append(record[name])
    return columns
