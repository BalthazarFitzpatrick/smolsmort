"""one row per saved group, target, origin and lead for tableau"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from smolsmort.forecast import evaluate as ev
from smolsmort.forecast import views
from smolsmort.forecast.spec import SERIES, normalize_mode
from smolsmort.forecast.tables import read_table

PHASES = ("validation", "test", "forecast")
VALUES = ("prediction", "actual", "lower_50", "upper_50", "lower", "upper")
VALUE_COLUMNS = ("forecast", "actual", "lower_50", "upper_50", "lower_80", "upper_80")


def _read_predictions(run_dir, selected, phase, tiles=None):
    """resolve the same saved models as results without creating tile caches"""
    model = selected["model"]
    if tiles is not None:
        tile = next(item for item in tiles if item["tile_id"] == model)
        if tile["kind"] == "model":
            path = views._model_prediction_path(
                run_dir, tile["family"], phase, selected["winner_family"], selected["saved_model"]
            )
        else:
            path = run_dir / "tile_views" / tile["recipe_key"] / f"{phase}.parquet"
            if not path.exists():
                from smolsmort.forecast.tiles import align_tables

                members = [
                    _read_predictions(run_dir, {**selected, "model": member}, phase, tiles)
                    for member in tile["members"]
                ]
                if any(not member for member in members):
                    return {}
                members = align_tables(members)
                columns = dict(members[0])
                for name in ("actual", "truth"):
                    if name in columns and not all(
                        np.array_equal(columns[name], member[name], equal_nan=True)
                        for member in members
                    ):
                        raise views.ViewError("ensemble members disagree on actuals")
                columns["prediction"] = ev.mean_predictions([m["prediction"] for m in members])
                for name in VALUES[2:]:
                    columns[name] = np.full(len(columns["prediction"]), np.nan)
                return columns
    else:
        path = run_dir / f"{model}_{phase}.parquet"
        if not path.exists() and model == selected["saved_model"]:
            path = run_dir / f"{phase}.parquet"
    return read_table(path) if path.exists() else {}


def _select_predictions(run_dir, result, query):
    selected = views._select_model(run_dir, result, query, read_only=True)
    model = selected["model"]
    if not model.startswith(("model:", "ensemble:")):
        return selected, None
    _, tiles = views._select_tile(run_dir, model, read_only=True)
    return selected, tiles


def _read_dimensions(request, dims):
    keymap = {}
    if not dims:
        return keymap
    prepared = Path(request["prepared"])
    for name in ("panel.parquet", "scaffold.parquet"):
        path = prepared / name
        if path.exists():
            columns = read_table(path)
            for index, series in enumerate(columns[SERIES]):
                keymap[str(series)] = [columns[dim][index] for dim in dims]
    return keymap


def _sum_totals(columns):
    totals = {}
    actuals = columns.get("actual", np.full(len(columns.get("prediction", [])), np.nan))
    for index in range(len(columns.get("prediction", []))):
        key = (
            str(columns["step"][index]),
            str(columns["origin"][index]),
            int(columns["weeks_ahead"][index]),
        )
        total = totals.setdefault(key, [0.0, 0.0])
        total[0] += columns["prediction"][index]
        total[1] += actuals[index]
    return totals


def _format_cell(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    if isinstance(value, np.datetime64):
        return "" if np.isnat(value) else str(value.astype("datetime64[D]"))
    return value


def export_wide_csv(run_dir: Path, request: dict, query: dict) -> bytes:
    """export all saved phases of the selected time series model, with calibrated totals"""
    result = views._require_done(run_dir)
    if normalize_mode(request["spec"]["mode"]) != "timeseries":
        raise views.ViewError(
            "wide csv requires a time series run; entity runs have no horizon or origin"
        )
    selected, tiles = _select_predictions(run_dir, result, query)
    tables = {phase: _read_predictions(run_dir, selected, phase, tiles) for phase in PHASES}
    dims = views._spec_dims(request["spec"])
    keymap = _read_dimensions(request, dims)
    totals = {phase: _sum_totals(columns) for phase, columns in tables.items()}
    validation = totals["validation"]
    leads = np.array([key[2] for key in validation], dtype=int)
    predictions = np.array([value[0] for value in validation.values()])
    actuals = np.array([value[1] for value in validation.values()])
    bands = {
        level: ev.fit_lead_bands(actuals, predictions, leads, level=level) for level in (0.5, 0.8)
    }

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "phase",
            "week",
            *dims,
            "forecast_made_week",
            "horizon",
            *VALUE_COLUMNS,
            *(f"total_{name}" for name in VALUE_COLUMNS),
        ]
    )
    for phase, columns in tables.items():
        phase_totals = totals[phase]
        keys = list(phase_totals)
        total_predictions = np.array([phase_totals[key][0] for key in keys])
        total_leads = np.array([key[2] for key in keys], dtype=int)
        edges = [
            ev.apply_lead_bands(bands[level], total_predictions, total_leads)
            for level in (0.5, 0.8)
        ]
        total_rows = {
            key: [*phase_totals[key], *(edge[i] for pair in edges for edge in pair)]
            for i, key in enumerate(keys)
        }
        for index in range(len(columns.get("prediction", []))):
            lead = int(columns["weeks_ahead"][index])
            key = (str(columns["step"][index]), str(columns["origin"][index]), lead)
            series = str(columns["series"][index])
            if dims and series not in keymap:
                raise views.ViewError(f"no prepared dimensions for series {series!r}")
            values = [columns[name][index] if name in columns else None for name in VALUES]
            row = [
                phase,
                columns["step"][index],
                *keymap.get(series, []),
                columns["origin"][index],
                lead,
                *values,
                *total_rows[key],
            ]
            writer.writerow([_format_cell(value) for value in row])
    return buffer.getvalue().encode("utf-8")
