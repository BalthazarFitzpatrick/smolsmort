"""shared error labels and honest values from saved forecast artifacts"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from smolsmort.forecast.tables import connect

_DEFINITIONS = (
    {
        "id": "validation_error",
        "label": "validation error",
        "explanation": "wape on held-out validation windows used by the search to rank candidates; lower is better",
        "direction": "lower",
        "unit": "percent",
        "scope": "validation",
        "computation": "sum absolute errors / sum absolute actuals in each validation window, then mean across windows",
    },
    {
        "id": "wape",
        "label": "wape",
        "explanation": "absolute error as a share of absolute actuals on test rows; recipes show saved validation wape",
        "direction": "lower",
        "unit": "percent",
        "scope": "test",
        "computation": "test: sum absolute errors / sum absolute actuals, unknown actuals excluded; recipes: mean of per-validation-window wape",
    },
    {
        "id": "mae",
        "label": "mae",
        "explanation": "mean absolute error in target units on test rows; recipes show saved validation mae",
        "direction": "lower",
        "unit": "target",
        "scope": "test",
        "computation": "mean absolute error; unknown actuals excluded",
    },
    {
        "id": "mape",
        "label": "mape",
        "explanation": "mean absolute percentage error on test rows with actual > 0 only; zero, negative and undefined rows excluded and counted",
        "direction": "lower",
        "unit": "percent",
        "scope": "test",
        "computation": "mean of absolute error / actual for finite predictions and finite actual > 0",
    },
    {
        "id": "test_ratio",
        "label": "test ratio",
        "explanation": "model error / naive error on test windows; below 1 beats naive; lower is better",
        "direction": "lower",
        "unit": "ratio",
        "scope": "test",
        "computation": "saved verdict accuracy ratio: model error / naive error",
    },
)


def metric_registry(spec: dict) -> list[dict]:
    """limit the picker to scores whose meaning applies to the saved task"""
    definitions = [dict(item) for item in _DEFINITIONS]
    if spec.get("task") == "classification":
        definitions[0].update(
            explanation="log loss on held-out validation rows used by the search to rank candidates; lower is better",
            computation="mean negative log probability assigned to the observed class",
            unit="number",
        )
    elif spec.get("censor"):
        definitions[0].update(
            explanation="search score on held-out validation rows; hazard uses log loss, other regression objectives use wape; lower is better",
            computation="saved search fitness for the candidate objective; hazard and aft test verdicts use integrated brier score",
            unit="number",
        )
    else:
        return definitions
    return [item for item in definitions if item["id"] in {"validation_error", "test_ratio"}]


def _finite_number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _metric_value(value=None, *, scope="test", reason="metric not saved", **details):
    value = _finite_number(value)
    return {"value": value, "reason": reason if value is None else None, "scope": scope, **details}


def error_metrics(spec, *, fitness=None, verdict=None, path=None, candidate=None):
    """read test artifacts for models, or existing validation metrics for recipes"""
    values = {
        "validation_error": _metric_value(
            fitness, scope="validation", reason="validation search score not saved yet"
        ),
        "test_ratio": _metric_value(
            ((verdict or {}).get("accuracy") or {}).get("ratio"),
            reason="test verdict not saved for this model",
        ),
    }
    registry = metric_registry(spec)
    if len(registry) > 2:
        if candidate is not None:
            for name in ("wape", "mae"):
                values[name] = _metric_value(
                    candidate.get(name),
                    scope="validation",
                    reason="validation metric not saved for this candidate",
                    note="saved validation metric; test predictions are not saved per candidate",
                )
            values["mape"] = _metric_value(reason="mape is not saved per candidate")
            values["test_ratio"] = _metric_value(reason="test verdict is not saved per candidate")
        else:
            values.update(read_test_metrics(path))
    return {item["id"]: values[item["id"]] for item in registry}


def _unavailable_test(reason):
    return {name: _metric_value(reason=reason) for name in ("wape", "mae", "mape")}


def read_test_metrics(path: Path | None) -> dict:
    """cache immutable artifact versions so polling does not rescan test rows"""
    if path is None:
        return _unavailable_test("test predictions not saved for this model")
    try:
        stat = path.stat()
    except OSError:
        return _unavailable_test("test predictions not saved for this model")
    values = _read_test_metrics(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    return {name: dict(value) for name, value in values.items()}


@lru_cache(maxsize=256)
def _read_test_metrics(path, modified_ns, size):
    try:
        import duckdb
    except ImportError:
        return _unavailable_test("duckdb is unavailable to read saved test predictions")
    try:
        with connect() as con:
            source = "read_parquet('" + path.replace("'", "''") + "')"
            columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}
            actual = "actual" if "actual" in columns else "truth"
            if not {actual, "prediction"} <= columns:
                return _unavailable_test(
                    "saved test predictions have no actual and prediction pair"
                )
            # use every saved holdout row, including horizon buckets, as the test verdict does
            known = f"isfinite({actual})"
            valid = f"{known} AND isfinite(prediction)"
            positive = f"{valid} AND {actual} > 0"
            total, count, invalid, denominator, absolute, mape, positive_count = con.execute(
                f"""SELECT COUNT(*), COUNT(*) FILTER (WHERE {known}),
                    COUNT(*) FILTER (WHERE {known} AND NOT COALESCE(isfinite(prediction), FALSE)),
                    SUM(ABS({actual})) FILTER (WHERE {known}),
                    SUM(ABS({actual} - prediction)) FILTER (WHERE {valid}),
                    AVG(ABS({actual} - prediction) / {actual}) FILTER (WHERE {positive}),
                    COUNT(*) FILTER (WHERE {positive}) FROM {source}"""
            ).fetchone()
    except (duckdb.Error, OSError, ValueError, TypeError):
        return _unavailable_test("saved test predictions are incomplete or unreadable")
    if not count:
        reason = "test predictions have no known actuals"
    elif invalid:
        reason = "test predictions contain undefined predictions for known actuals"
    else:
        reason = None
    wape = absolute / denominator if reason is None and denominator else None
    mae = absolute / count if reason is None else None
    return {
        "wape": _metric_value(
            wape, reason=reason or "wape is undefined because absolute actuals sum to zero"
        ),
        "mae": _metric_value(mae, reason=reason or "mae is undefined"),
        "mape": _metric_value(
            mape,
            reason="mape has no finite predictions with actual > 0",
            excluded_rows=total - positive_count,
        ),
    }
