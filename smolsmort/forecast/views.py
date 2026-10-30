"""the forecast tab's aggregation, as plain functions over a run folder and its prepared tables.

nothing here talks http - a route calls one of these and returns what comes back. numpy/duckdb
only: the server that imports this module must never load torch or run xgboost in-process.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from smolsmort.forecast import evaluate as ev
from smolsmort.forecast import metrics as fm
from smolsmort.forecast.families import registered_families
from smolsmort.forecast.horizon_view import (
    MIN_BLEND_RESIDUALS,
    blend_rows,
    build_blend_weights,
    fit_blend_band,
    lead_rows,
)
from smolsmort.forecast.spec import normalize_mode, normalize_saved_modes, spec_from_dict
from smolsmort.forecast.tables import _unmask, connect


def write_json_atomic(path: Path, data) -> None:
    """replace `path` in one step; each writer gets its own temp name, since every reader of a
    run [the ui poll, a script waiting on the run] may write the cursors at the same moment"""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


MAX_LINES = 12
MC_DRAWS = 200
MC_SEED = 0
_POLL_LOCK = threading.Lock()


class ViewError(ValueError):
    pass


# ---------------------------------------------------------------- sql helpers


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _lit(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _filter_clause(filters: dict, dims: list[str]) -> str:
    """dimension filters as sql, restricted to columns that are actually dimensions here"""
    clauses = []
    for dim, values in (filters or {}).items():
        if dim not in dims or not values:
            continue
        clauses.append(f"{_quote(dim)} IN ({', '.join(_lit(v) for v in values)})")
    return " AND ".join(clauses) if clauses else "TRUE"


def _iso(value) -> str:
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[D]"))
    if isinstance(value, dt.datetime):
        value = value.date()
    return value.isoformat()


# ---------------------------------------------------------------- run bookkeeping


def read_request(run_dir: Path) -> dict:
    return normalize_saved_modes(json.loads((run_dir / "request.json").read_text()))


def read_result(run_dir: Path) -> dict | None:
    path = run_dir / "result.json"
    return normalize_saved_modes(json.loads(path.read_text())) if path.exists() else None


def read_leaderboard(run_dir: Path, top: int = 10) -> list[dict]:
    request = read_request(run_dir)
    if request.get("version") == 2:
        board = []
        for selection in request["families"]:
            name = selection["name"]
            for entry in _read_json(run_dir / "families" / name / "leaderboard.json", []):
                board.append({**entry, "family": name})
        if board:
            board.sort(
                key=lambda entry: (
                    entry["fitness"] if entry.get("fitness") is not None else math.inf,
                    entry["family"],
                    str(entry.get("key", "")),
                )
            )
            return _candidate_metrics(board[:top], request)
    path = run_dir / "leaderboard.json"
    if not path.exists():
        return []
    board = [
        {
            **entry,
            "family": entry.get("family") or (entry.get("genome") or {}).get("family") or "xgboost",
        }
        for entry in _read_json(path, [])[:top]
    ]
    return _candidate_metrics(board, request)


def _candidate_metrics(board, request):
    return [
        {
            **entry,
            "error_metrics": fm.error_metrics(
                request.get("spec", {}),
                fitness=entry.get("fitness"),
                candidate=entry.get("metrics") or {},
            ),
        }
        for entry in board
    ]


def _read_json(path: Path, default=None):
    try:
        data = json.loads(path.read_text())
        return normalize_saved_modes(data) if isinstance(data, dict) else data
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_prepared_spec(prepared, spec):
    path = prepared.folder / "spec.json"
    path.write_text(json.dumps(asdict(spec)), encoding="utf-8")


def read_prepared_spec(folder: Path, runs_root: Path):
    data = _read_json(folder / "spec.json")
    if data is None:
        for run in sorted(runs_root.glob("*"), reverse=True):
            request = _read_json(run / "request.json", {})
            if request.get("prepared") and Path(request["prepared"]).resolve() == folder:
                data = request.get("spec")
                break
    if not isinstance(data, dict):
        raise ViewError("prepared folder has no saved spec; provide spec")
    return spec_from_dict(data)


def probe_families() -> dict:
    """native library probes stay out of the host's torch process"""
    code = (
        "import json; from smolsmort.forecast.families import registered_families; "
        "print(json.dumps({family.name: family.available() for family in registered_families()}))"
    )
    try:
        process = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=True
        )
        return json.loads(process.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise ViewError("could not check family libraries in the isolated worker") from exc


def family_catalog(
    runs_root: Path, spec=None, prepared=None, task="regression", horizon_mode="single"
) -> list[dict]:
    """suitability uses the worker's split and features, without fitting any models"""
    from smolsmort.forecast.pipeline import PipelineError, load_workspace
    from smolsmort.forecast.splits import SplitError

    workspace, history_error = None, None
    if spec is not None:
        task = spec.task
        try:
            workspace = load_workspace(
                spec, prepared.folder, prepared.summary, horizon_mode=horizon_mode
            )
        except (PipelineError, SplitError) as exc:
            history_error = str(exc)
    estimates = {}
    for folder in sorted(runs_root.glob("*"), reverse=True):
        request = _read_json(folder / "request.json", {})
        if request.get("spec", {}).get("task", "regression") != task:
            continue
        calibration = _read_json(folder / "cost-calibration.json", {})
        for name, data in calibration.items():
            estimate = data.get("seconds_per_candidate")
            if (
                name not in estimates
                and isinstance(estimate, (int, float))
                and not isinstance(estimate, bool)
                and math.isfinite(estimate)
                and estimate > 0
            ):
                estimates[name] = {
                    "seconds_per_candidate": estimate,
                    "estimate_is_lower_bound": bool(data.get("estimate_is_lower_bound", False)),
                }
    catalog = []
    availability = probe_families()
    for family in registered_families():
        available, install = availability[family.name]
        try:
            objectives = list(family.objectives(task))
        except ValueError:
            objectives = []
        reason = None
        if not available:
            reason = f"library unavailable; {install}"
        elif not objectives:
            reason = f"family {family.name!r} does not support {task}"
        elif spec is not None and spec.mode == "entity" and family.name != "xgboost":
            reason = "entity runs support xgboost only"
        elif history_error:
            reason = history_error
        elif workspace is not None:
            reason = _suitability_reason(family, workspace)
        catalog.append(
            {
                "name": family.name,
                "label": family.label,
                "needs": family.needs,
                "available": available,
                "install_command": f"uv sync --extra {family.pip_extra}" if not available else None,
                "space": [asdict(param) for param in family.space()],
                "objectives": objectives,
                "suitable": False if reason else True if spec is not None else None,
                "reason": reason
                if reason
                else "provide a spec or prepared folder to check training history"
                if spec is None
                else None,
                **estimates.get(family.name, {}),
            }
        )
    return catalog


def _suitability_reason(family, ws):
    if family.needs == "raw_series":
        from smolsmort.forecast.spec import SERIES, STEP
        from smolsmort.forecast.splits import timeseries_cuts

        steps = np.unique(ws.panel[STEP])
        cut = steps[timeseries_cuts(len(steps), ws.spec.horizon).val_start]
        target = ws.spec.named("target")[0]
        counts = [
            int(
                (
                    (ws.panel[SERIES] == key)
                    & (ws.panel[STEP] < cut)
                    & np.isfinite(ws.panel[target])
                ).sum()
            )
            for key in np.unique(ws.panel[SERIES])
        ]
        period = (
            ws.profile.get("period")
            or {"day": 7, "week": 52, "month": 12, "quarter": 4, "year": 1}[ws.unit]
        )
        minimum = min(counts, default=0)
        if minimum < 2 * period:
            return f"needs two seasons ({2 * period} training steps per series); shortest series has {minimum}"
    else:
        if ws.mode == "timeseries":
            counts = [int(masks["train"].sum()) for masks in ws.series_masks.values()]
        else:
            if ws.spec.task == "classification":
                labels = ws.table[ws.spec.named("target")[0]]
                usable = np.array(
                    [value is not None and value == value and value != "" for value in labels]
                )
            else:
                lower = ws.bounds["val"][0]
                usable = np.isfinite(lower)
            counts = [int((ws.masks["train"] & usable).sum())]
        minimum = min(counts, default=0)
        if minimum < 20:
            return (
                f"needs at least 20 usable training rows per bucket; smallest split has {minimum}"
            )
    return None


def winning_genome(run_dir: Path) -> dict:
    recipe = _read_json(run_dir / "recipe.json", {})
    genome = dict(recipe.get("genome") or {})
    if not genome:
        raise ViewError(f"run {run_dir.name!r} has no saved winner")
    result = _read_json(run_dir / "result.json", {})
    family = (
        result.get("winner", {}).get("family")
        or genome.get("family")
        or recipe.get("family", "xgboost")
    )
    genome["family"] = genome.get("family", "xgboost") if family == "ensemble" else family
    return genome


def multi_run_state(run_dir, request, status, *, since_generation=None, since_events=None):
    """durable cursors keep late events from slower families visible across reattachment"""
    with _POLL_LOCK:
        changed = not (run_dir / "route-cursors.json").exists()
        snapshot = _read_json(
            run_dir / "route-cursors.json",
            {
                "counts": {},
                "events": [],
                "curves": [],
                "board": [],
                "board_revision": -1,
            },
        )
        sources = [
            (selection["name"], run_dir / "families" / selection["name"])
            for selection in request["families"]
        ]
        sources.append((None, run_dir))
        for name, target in sources:
            for filename, collection in (
                ("events.jsonl", "events"),
                ("generation_curves.jsonl", "curves"),
            ):
                path = target / filename
                lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
                source = f"{name or 'coordinator'}/{filename}"
                offset = snapshot["counts"].get(source, 0)
                for index, line in enumerate(lines[offset:], offset):
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        if index == len(lines) - 1:
                            break
                        offset += 1
                        changed = True
                        continue
                    item = {**item, "family": name if name is not None else item.get("family")}
                    if "generation" in item:
                        item["family_generation"] = item["generation"]
                    if collection == "curves":
                        item["generation"] = len(snapshot["curves"])
                    else:
                        item["sequence"] = len(snapshot["events"])
                    snapshot[collection].append(item)
                    offset += 1
                    changed = True
                snapshot["counts"][source] = offset
        board = read_leaderboard(run_dir)
        if board != snapshot["board"]:
            snapshot["board"] = board
            snapshot["board_revision"] += 1
            changed = True
        if changed:
            write_json_atomic(run_dir / "route-cursors.json", snapshot)
    families = {}
    family_state = "pending"
    if status["state"] not in ("starting", "queued") and (
        request.get("recipe") or status["state"] in ("cancelled", "failed")
    ):
        family_state = status["state"]
    for selection in request["families"]:
        name = selection["name"]
        coordinator = status.get("families", {}).get(name, {})
        worker = _read_json(run_dir / "families" / name / "status.json", {})
        families[name] = {
            "state": family_state,
            "phase": "refit" if request.get("recipe") else "broad",
            "slice_s": 0,
            "best_error": None,
            "pruned_reason": None,
            "reason": None,
            "seconds_per_candidate": None,
            "candidates_done": 0,
            "candidates_expected": 0,
            "nthread": 0,
            **worker,
            **coordinator,
        }
        families[name].pop("trace", None)
        families[name].pop("slice_history", None)
        families[name]["threads"] = families[name]["nthread"]
    events = snapshot["events"]
    generations = [event for event in events if event.get("event") == "generation"]
    event_cursor = since_events if since_events is not None else -1
    curve_cursor = since_generation if since_generation is not None else -1
    unseen_events = [event for event in events if event["sequence"] > event_cursor][:200]
    delivered_cursor = unseen_events[-1]["sequence"] if unseen_events else event_cursor
    result = _read_json(run_dir / "result.json", {})
    recipe = _read_json(run_dir / "recipe.json", {})
    # list_runs uses both maximum cursors for summaries; charts belong to the selected run
    include_charts = (since_generation, since_events) != (2**63 - 1, 2**63 - 1)
    tiles = family_tiles(
        run_dir,
        request,
        status,
        result=result,
        recipe=recipe,
        include_charts=include_charts,
        search_snapshot=snapshot,
    )
    for name, tile in tiles.items():
        families[name]["error_metrics"] = tile["error_metrics"]
        if include_charts:
            families[name]["tile"] = tile
    kind = (
        "refit" if request.get("recipe") else "warm start" if request.get("warm_from") else "multi"
    )
    winner = recipe.get("family") or result.get("winner", {}).get("family")
    best = min(
        (data["best_error"] for data in families.values() if data.get("best_error") is not None),
        default=None,
    )
    out = {
        **status,
        "families": families,
        "kind": kind,
        "task": request["spec"].get("task"),
        "winner_family": winner,
        "time_budget_s": request["time_budget_s"],
        "budget": {},
        "generation_count": len(generations),
        "generation": len(snapshot["curves"]) - 1,
        "events_cursor": delivered_cursor,
        "leaderboard_generation": snapshot["board_revision"],
        "leaderboard": board,
        "events": unseen_events,
        "generations": [event for event in unseen_events if event.get("event") == "generation"],
        "generation_curves": [
            curve for curve in snapshot["curves"] if curve["generation"] > curve_cursor
        ],
        "latest": events[-1] if events else None,
        "best_error": best,
        "best_wape": best
        if request["spec"].get("task") == "regression" and kind != "refit"
        else None,
        "test_error": result.get("model_error"),
        "excluded_series": result.get("excluded_series", []),
        **run_metric_payload(run_dir, request, {"best_error": best}, result=result, recipe=recipe),
    }
    if since_generation is None:
        out["eval_history"] = _read_json(run_dir / "eval_history.json")
    return out


def _finite_number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _tile_elapsed(progress):
    wall_s = _finite_number(progress.get("wall_s"))
    if wall_s is not None:
        return max(0, wall_s)
    history = progress.get("slice_history") or []
    if history and isinstance(history, list) and all(isinstance(item, dict) for item in history):
        elapsed = 0
        for item in history:
            duration = _finite_number(item.get("wall_s"))
            if duration is None and item.get("started_at") is not None:
                end = item.get("ended_at")
                if end is None and progress.get("state") == "running":
                    end = time.time()
                if end is not None:
                    duration = max(0, end - item["started_at"])
            if duration is None:
                return None
            elapsed += duration
        return elapsed
    elapsed = _finite_number(progress.get("elapsed"))
    if elapsed is not None:
        return max(0, elapsed)
    start = _finite_number(progress.get("started_at"))
    end = _finite_number(progress.get("ended_at"))
    if start is not None and (end is not None or progress.get("state") == "running"):
        return max(0, (end if end is not None else time.time()) - start)
    return None


def _read_tile_search_records(target):
    records = {}
    for filename, collection in (("events.jsonl", "events"), ("generation_curves.jsonl", "curves")):
        try:
            lines = (target / filename).read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        records[collection] = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records[collection].append(record)
    return records


def _tile_search_curve(records, *, fitness=None, metrics=None):
    """use the same saved events and validation loss curves as the search tables"""
    points = {}

    def get_point(record):
        generation = record.get("family_generation", record.get("generation"))
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            return None
        return points.setdefault(
            generation + 1,
            {"generation": generation + 1, "validation_error": None, "wape": None, "mae": None},
        )

    for event in records["events"]:
        if event.get("event") != "generation":
            continue
        point = get_point(event)
        if point is None:
            continue
        point["validation_error"] = _finite_number(event.get("best"))
        values = event.get("metrics")
        if isinstance(values, dict):
            for metric in ("wape", "mae"):
                point[metric] = _finite_number(values.get(metric))
    for curve in records["curves"]:
        metric = curve.get("metric")
        validation = curve.get("validation")
        if (
            not isinstance(metric, str)
            or not metric
            or metric in {"generation", "validation_error", "mape", "test_ratio"}
            or not isinstance(validation, list)
            or not validation
        ):
            continue
        point = get_point(curve)
        if point is not None and point.get(metric) is None:
            point[metric] = _finite_number(validation[-1])
    if points:
        if not any(
            value is not None
            for point in points.values()
            for metric, value in point.items()
            if metric != "generation"
        ):
            return [], False
        metric_names = {
            metric for point in points.values() for metric in point if metric != "generation"
        }
        series = [
            {
                "generation": generation,
                **{
                    metric: points.get(generation, {}).get(metric)
                    for metric in sorted(metric_names)
                },
            }
            for generation in range(1, max(points) + 1)
        ]
        return series, False
    values = {"validation_error": _finite_number(fitness)}
    values.update(
        {metric: _finite_number((metrics or {}).get(metric)) for metric in ("wape", "mae")}
    )
    if any(value is not None for value in values.values()):
        return [{"generation": 1, **values}], True
    return [], False


TILE_ROUND_GENERATIONS = 30


def _tile_round_curves(curves):
    """the newest generations' per-round curves, as the tables loss chart draws them"""
    latest = {}
    for curve in curves:
        generation = curve.get("family_generation", curve.get("generation"))
        validation = curve.get("validation")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or not isinstance(validation, list)
            or not validation
        ):
            continue
        latest[generation] = {
            "generation": generation + 1,
            "key": curve.get("key"),
            "metric": curve.get("metric"),
            "validation": validation,
            "curves": curve.get("curves") or {},
        }
    return [latest[generation] for generation in sorted(latest)[-TILE_ROUND_GENERATIONS:]]


def family_tiles(
    run_dir, request, status, *, result=None, recipe=None, include_charts=True, search_snapshot=None
):
    """expose saved family artifacts without inventing loser test scores or fold baselines"""
    result = result if result is not None else _read_json(run_dir / "result.json", {})
    recipe = recipe if recipe is not None else _read_json(run_dir / "recipe.json", {})
    winner = result.get("winner") or {}
    winner_family = winner.get("family") or (recipe.get("genome") or {}).get("family")
    if not winner_family and recipe.get("family") != "ensemble":
        winner_family = recipe.get("family")
    legacy = request.get("version") != 2
    names = [item["name"] for item in request.get("families", [])]
    if legacy:
        names = [winner_family or "xgboost"]
        winner_family = names[0]
    origins = (result.get("holdout") or {}).get("validation_origins", [])
    tiles = {}
    for name in names:
        target = run_dir if legacy else run_dir / "families" / name
        saved = result.get("families", {}).get(name, {})
        progress = {
            **_read_json(target / "status.json", {}),
            **(status if legacy else status.get("families", {}).get(name, {})),
            **saved,
        }
        board = _read_json(target / "leaderboard.json", [])
        entries = [
            entry
            for entry in board
            if _finite_number(entry.get("fitness")) is not None and entry.get("note") != "sample"
        ]
        best = min(entries, key=lambda entry: entry["fitness"], default={})
        genome = _read_json(target / "winner.json") or best.get("genome") or {}
        if progress.get("state") in {"running", "paused"} and best.get("genome"):
            genome = best["genome"]
        if name == winner_family and not genome:
            genome = recipe.get("genome") or {}
        summary = saved
        if name == winner_family:
            summary = {**result, **winner} if recipe.get("family") != "ensemble" else winner
        path = target / "test.parquet"
        if (legacy or not path.exists()) and name == winner_family:
            path = run_dir / "winner_test.parquet"
            if not path.exists() and recipe.get("family") != "ensemble":
                path = run_dir / "test.parquet"
        accuracy = (summary.get("verdict") or {}).get("accuracy") or {}
        importance = saved.get("importance") or _read_json(target / "importance.json")
        if importance is None and name == winner_family:
            importance = summary.get("importance")
            if importance is None and recipe.get("family") != "ensemble":
                importance = _read_json(run_dir / "importance.json")
        scores = (best.get("metrics") or {}).get("fold_scores") or summary.get("fold_scores") or []
        folds = [
            {
                "label": str(origins[index]) if index < len(origins) else f"fold {index + 1}",
                "error": _finite_number(error),
                "ratio": None,
            }
            for index, error in enumerate(scores)
        ]
        test_metric = accuracy.get("metric") or (
            "wape" if request.get("spec", {}).get("mode") == "timeseries" else None
        )
        test_error = next(
            (
                value
                for value in (
                    summary.get("test_error"),
                    summary.get("model_error"),
                    accuracy.get("model_error"),
                )
                if _finite_number(value) is not None
            ),
            None,
        )
        naive_error = next(
            (
                value
                for value in (accuracy.get("baseline_error"), summary.get("baseline_error"))
                if _finite_number(value) is not None
            ),
            None,
        )
        tiles[name] = {
            "state": progress.get("state", status.get("state", "pending")),
            "genome": genome,
            "importance": importance,
            "validation_error": next(
                (
                    value
                    for value in (
                        saved.get("fitness"),
                        progress.get("best_error"),
                        best.get("fitness"),
                        summary.get("fitness"),
                        summary.get("val_error"),
                    )
                    if _finite_number(value) is not None
                ),
                None,
            ),
            "test_ratio": _finite_number(accuracy.get("ratio")),
            "test_error": test_error,
            "test_error_metric": test_metric,
            "test_naive_wape": naive_error if test_metric == "wape" else None,
            "folds": folds,
            "folds_note": "wape per validation window; mark = naive on test"
            if folds
            else "validation folds not saved",
            "search_curve": [],
            "search_single": False,
            "round_curves": [],
            "elapsed": _tile_elapsed(progress),
            "generation": progress.get("generation", best.get("generation")),
        }
        if include_charts:
            records = (
                {
                    collection: [
                        record
                        for record in search_snapshot[collection]
                        if record.get("family") == name
                    ]
                    for collection in ("events", "curves")
                }
                if search_snapshot is not None
                else _read_tile_search_records(target)
            )
            tiles[name]["search_curve"], tiles[name]["search_single"] = _tile_search_curve(
                records, fitness=tiles[name]["validation_error"], metrics=best.get("metrics")
            )
            tiles[name]["round_curves"] = _tile_round_curves(records["curves"])
        if request.get("horizon_mode") == "single":
            tiles[name]["weeks_ahead_shown"] = request["spec"]["horizon"]
        tiles[name]["error_metrics"] = fm.error_metrics(
            request.get("spec", {}),
            fitness=tiles[name]["validation_error"],
            verdict=summary.get("verdict"),
            path=path,
        )
    return tiles


def run_metric_payload(run_dir, request, status=None, *, result=None, recipe=None, model=None):
    """expose the selected saved model without borrowing another model's predictions"""
    result = result if result is not None else _read_json(run_dir / "result.json", {})
    recipe = recipe if recipe is not None else _read_json(run_dir / "recipe.json", {})
    default_model = "ensemble" if recipe.get("family") == "ensemble" else "winner"
    model = model or default_model
    summary = result.get(model) or {}
    if model == default_model:
        summary = {**result, **summary}
    path = run_dir / f"{model}_test.parquet"
    if not path.exists() and model == default_model:
        path = run_dir / "test.parquet"
    fitness = next(
        (
            value
            for value in (
                summary.get("fitness"),
                summary.get("val_error"),
                (status or {}).get("best_error") if model == "winner" else None,
            )
            if _finite_number(value) is not None
        ),
        None,
    )
    if fitness is None and model == "winner" and (run_dir / "request.json").exists():
        fitness = next(
            (
                entry["fitness"]
                for entry in read_leaderboard(run_dir)
                if _finite_number(entry.get("fitness")) is not None
            ),
            None,
        )
    spec = request.get("spec", {})
    payload = {
        "metric_registry": fm.metric_registry(spec),
        "error_metrics": fm.error_metrics(
            spec, fitness=fitness, verdict=summary.get("verdict"), path=path
        ),
    }
    if spec.get("mode") == "timeseries":
        payload["horizon_mode_label"] = (
            "single model: all weeks ahead, one fit"
            if request.get("horizon_mode") == "single"
            else "mixed: one model per horizon group"
        )
    return payload


def _spec_dims(spec: dict) -> list[str]:
    return [c["name"] for c in spec["columns"] if c["role"] == "dimension"]


def _spec_target(spec: dict) -> str:
    return next(c["name"] for c in spec["columns"] if c["role"] == "target")


def _require_done(run_dir: Path) -> dict:
    result = read_result(run_dir)
    if result is None:
        raise ViewError(f"run {run_dir.name!r} has no result yet")
    return result


def _select_model(run_dir: Path, result: dict, query: dict) -> dict:
    recipe = _read_json(run_dir / "recipe.json", {})
    winner = result.get("winner") or {}
    winner_family = (
        winner.get("family")
        or (recipe.get("genome") or {}).get("family")
        or (result.get("best") or {}).get("family")
        or (recipe.get("family") if recipe.get("family") != "ensemble" else None)
        or "xgboost"
    )
    ensemble = result.get("ensemble")
    default_model = "ensemble" if recipe.get("family") == "ensemble" else "winner"
    model = query.get("model") or default_model
    if model not in ("winner", "ensemble"):
        raise ViewError("model must be winner or ensemble")
    if model == "ensemble" and not ensemble:
        raise ViewError("run has no saved ensemble")
    selected = ensemble if model == "ensemble" else winner
    return {
        "model": model,
        "family": "ensemble" if model == "ensemble" else winner_family,
        "members": ensemble.get("members", []) if model == "ensemble" else [],
        "winner_family": winner_family,
        "verdict": selected.get("verdict") or result.get("verdict"),
        "default_model": default_model,
    }


def _prediction_path(run_dir: Path, query: dict, kind: str) -> Path:
    selected = _select_model(run_dir, _require_done(run_dir), query)
    path = run_dir / f"{selected['model']}_{kind}.parquet"
    if path.exists():
        return path
    if selected["model"] != selected["default_model"]:
        raise ViewError(f"run has no saved {selected['model']} {kind}")
    return run_dir / f"{kind}.parquet"


def _optional_prediction_path(run_dir: Path, query: dict, kind: str) -> Path | None:
    selected = _select_model(run_dir, _require_done(run_dir), query)
    path = run_dir / f"{selected['model']}_{kind}.parquet"
    if not path.exists() and selected["model"] == selected["default_model"]:
        path = run_dir / f"{kind}.parquet"
    return path if path.exists() else None


def result_details(run_dir: Path, request: dict, query: dict) -> dict:
    """comparison uses validation scores; only the winner and ensemble have test verdicts"""
    result = _require_done(run_dir)
    selected = _select_model(run_dir, result, query)
    selected.pop("default_model")
    tiles = family_tiles(run_dir, request, _read_json(run_dir / "status.json", {}), result=result)
    families = {}
    if request.get("version") == 2:
        status = _read_json(run_dir / "status.json", {}).get("families", {})
        saved = result.get("families", {})
        names = dict.fromkeys([item["name"] for item in request["families"]] + list(saved))
        for name in names:
            progress = {
                **_read_json(run_dir / "families" / name / "status.json", {}),
                **status.get(name, {}),
                **saved.get(name, {}),
            }
            row = {
                "family": name,
                "state": progress.get("state", "done"),
                "fitness": progress.get("fitness", progress.get("best_error")),
                "candidates": progress.get("candidates", progress.get("candidates_done", 0)),
                "pruned_reason": progress.get("pruned_reason"),
                "reason": progress.get("reason"),
                "terminated": progress.get("terminated", False),
            }
            if name == selected["winner_family"]:
                row["verdict"] = result.get("winner", {}).get("verdict") or result.get("verdict")
            row["error_metrics"] = tiles.get(name, {}).get("error_metrics") or fm.error_metrics(
                request.get("spec", {}), fitness=row["fitness"], verdict=row.get("verdict")
            )
            families[name] = row
    importance_family = query.get("importance_family") or selected["family"]
    known_families = set(families) | {selected["winner_family"]}
    if importance_family not in known_families | {"ensemble"}:
        raise ViewError("importance_family must name a family in this run")
    importance = None
    if importance_family != "ensemble":
        importance = result.get("families", {}).get(importance_family, {}).get("importance")
        if importance is None:
            importance = _read_json(run_dir / "families" / importance_family / "importance.json")
        if importance is None and importance_family == selected["winner_family"]:
            importance = _read_json(run_dir / "importance.json")
    winner = result.get("winner") or {
        "family": selected["winner_family"],
        "verdict": result.get("verdict"),
    }
    winner = {
        **winner,
        "error_metrics": run_metric_payload(run_dir, request, result=result, model="winner")[
            "error_metrics"
        ],
    }
    ensemble = result.get("ensemble")
    if ensemble is not None:
        ensemble = {
            **ensemble,
            "error_metrics": run_metric_payload(run_dir, request, result=result, model="ensemble")[
                "error_metrics"
            ],
        }
    return {
        **selected,
        "families": families,
        "winner": winner,
        "ensemble": ensemble,
        "importance_family": importance_family,
        "importance": importance,
        "tiles": tiles,
        **run_metric_payload(run_dir, request, result=result, model=selected["model"]),
    }


def _parse_filters(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ViewError(f"filters is not valid json: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _parse_breakdown(raw: str | None, dims: list[str]) -> list[str]:
    if not raw:
        return []
    wanted = [d.strip() for d in raw.split(",") if d.strip()]
    return [d for d in wanted if d in dims][:3]


# ---------------------------------------------------------------- time series mode view


def _read_view_rows(con, source):
    return {
        name: _unmask(values)
        for name, values in con.execute(f"SELECT * FROM {source}").fetchnumpy().items()
    }


def _horizon_selection(request, query, table):
    if "lead" in query and ("blend" in query or "blend_weights" in query):
        raise ValueError("lead and blend are mutually exclusive")
    horizon = int(request["spec"].get("horizon", 1))
    if "blend_weights" in query:
        if any(key in query for key in ("blend", "strength", "weights")):
            raise ValueError("blend_weights and blend, strength, or weights are mutually exclusive")
        weights = _parse_blend_weights(query["blend_weights"], horizon)
        return {"mode": "blend", "weights": weights} if "weeks_ahead" in table else None
    mode = "blend" if "blend" in query else "lead"
    try:
        n = int(query.get(mode, horizon))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{mode} must be an integer from 1 to {horizon}") from exc
    if not 1 <= n <= horizon:
        raise ValueError(f"{mode} must be an integer from 1 to {horizon}")
    if "weights" in query and mode != "blend":
        raise ValueError("weights requires blend")
    if "strength" in query and mode != "blend":
        raise ValueError("strength requires blend")
    raw_strength = query.get("strength", 1)
    if str(raw_strength) not in ("1", "2", "4", "8"):
        raise ValueError("strength must be 1, 2, 4, or 8")
    if "strength" in query and "weights" in query:
        raise ValueError("strength and weights are mutually exclusive")
    if mode == "blend":
        if "weights" in query:
            weights = _parse_blend_weights(query["weights"], horizon)
        else:
            weights = build_blend_weights(horizon, {n: int(raw_strength)})
        selection = {"mode": "blend", "weights": weights}
    else:
        selection = {"mode": "lead", "n": n}
    return selection if "weeks_ahead" in table else None


def _parse_blend_weights(raw, horizon):
    message = f"blend_weights must map weeks from 1 to {horizon} to weights, such as 1:1,2:4"
    try:
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("{"):
                raw = json.loads(raw)
            else:
                pairs = [part.split(":") for part in raw.split(",")]
                if any(len(pair) != 2 for pair in pairs):
                    raise ValueError
                if len({pair[0] for pair in pairs}) != len(pairs):
                    raise ValueError
                raw = {
                    key: int(value) if str(int(value)) == value else value for key, value in pairs
                }
        if not isinstance(raw, dict):
            raise ValueError
        weights = {}
        for key, value in raw.items():
            if (
                isinstance(key, bool)
                or not isinstance(key, (str, int))
                or str(int(key)) != str(key)
            ):
                raise ValueError
            lead = int(key)
            if lead in weights:
                raise ValueError
            weights[lead] = value
    except (ValueError, TypeError) as exc:
        raise ValueError(message) from exc
    return build_blend_weights(horizon, weights)


def _horizon_label(selection):
    if selection["mode"] == "lead":
        return f"lead {selection['n']}"
    return "blend weighted"


def _select_horizon_rows(table, selection):
    if selection["mode"] == "lead":
        return lead_rows(table, selection["n"])
    return blend_rows(table, weights=selection["weights"])


def _horizon_history_source(validation, test):
    return (
        f"(SELECT * FROM {validation} UNION BY NAME SELECT * FROM {test} "
        "ORDER BY series, step, weeks_ahead, origin)"
    )


def _horizon_sources(con, request, query, validation, test, forecast):
    sources = {
        name: f"read_parquet({_lit(path)})"
        for name, path in (("validation", validation), ("test", test), ("forecast", forecast))
    }
    table = _read_view_rows(con, sources["test"])
    selection = _horizon_selection(request, query, table)
    if selection is None:
        return None, sources, None
    band = None
    if selection["mode"] == "blend":
        validation_rows = _read_view_rows(con, sources["validation"])
        band = fit_blend_band(validation_rows, selection["weights"])
        panel = Path(request["prepared"]) / "panel.parquet"
        target = _quote(_spec_target(request["spec"]))
        nonnegative = bool(
            con.execute(f"SELECT MIN({target}) >= 0 FROM read_parquet({_lit(panel)})").fetchone()[0]
        )
        test_targets = set(zip(table["series"], table["step"], strict=True))
    sources["history"] = _horizon_history_source(sources["validation"], sources["test"])
    for name, source in sources.items():
        rows = table if name == "test" else _read_view_rows(con, source)
        if name != "forecast":
            rows = _select_horizon_rows(rows, selection)
        count = len(rows["step"])
        if selection["mode"] == "blend":
            rows["lower"], rows["upper"] = ev.apply_band(band, rows["prediction"], nonnegative)
            if name in ("validation", "history"):
                # only held-out test targets get historical bounds, using the chart's blend
                keep = np.asarray(
                    [
                        name == "history" and target in test_targets
                        for target in zip(rows["series"], rows["step"], strict=True)
                    ],
                    dtype=bool,
                )
                rows["lower"][~keep] = rows["upper"][~keep] = np.nan
        for column in ("lower", "upper"):
            rows.setdefault(column, np.full(count, np.nan))
        registered = {
            column: values.astype("datetime64[ns]")
            if np.issubdtype(values.dtype, np.datetime64)
            else values
            for column, values in rows.items()
        }
        con.register(f"horizon_{name}", registered)
        sources[name] = f"horizon_{name}"
    return selection, sources, band


def _read_lead_errors(run_dir, result, model, horizon):
    rows = (result.get(model) or {}).get("lead_errors")
    path = run_dir / f"{model}_lead_errors.parquet"
    default_model = _select_model(run_dir, result, {})["default_model"]
    if rows is None and not path.exists() and model == default_model:
        rows = result.get("lead_errors")
        path = run_dir / "lead_errors.parquet"
    if rows is None and path.exists():
        with connect() as con:
            columns = _read_view_rows(con, f"read_parquet({_lit(path)})")
        rows = [
            dict(zip(columns, values, strict=True))
            for values in zip(*columns.values(), strict=True)
        ]
    if isinstance(rows, dict):
        rows = [
            {
                "weeks_ahead": int(k),
                "model_wape": value.get("model"),
                "naive_wape": value.get("naive"),
                "rows": value.get("rows"),
                "ratio": value["model"] / value["naive"]
                if _finite_number(value.get("model")) is not None
                and _finite_number(value.get("naive")) not in (None, 0)
                else None,
            }
            for k, value in rows.items()
        ]
    rows = [row for row in (rows or []) if row.get("phase", "test") == "test"]

    def clean(value):
        if isinstance(value, np.generic):
            value = value.item()
        return None if isinstance(value, float) and not math.isfinite(value) else value

    if not rows:
        return []
    by_lead = {int(row["weeks_ahead"]): row for row in rows}
    return [
        {key: clean(value) for key, value in by_lead.get(k, {"weeks_ahead": k}).items()}
        for k in range(1, horizon + 1)
    ]


def _horizon_baseline(con, request, table):
    from smolsmort.forecast.features import step_index

    if "naive_prediction" in table:
        return table
    if "origin" not in table:
        return {**table, "naive_prediction": np.full(len(table["step"]), np.nan)}
    panel = Path(request["prepared"]) / "panel.parquet"
    target = _quote(_spec_target(request["spec"]))
    rows = con.execute(
        f"SELECT __series, __step, {target} FROM read_parquet({_lit(panel)}) ORDER BY __step"
    ).fetchall()
    unit = request["spec"].get("step", "week")
    period = {"week": 52, "day": 365, "month": 12, "quarter": 4, "year": 1}[unit]
    dates = np.asarray([step for _, step, _ in rows], dtype="datetime64[D]")
    histories = {}
    for (series, _, value), position in zip(rows, step_index(dates, unit), strict=True):
        histories.setdefault(series, {})[position] = value
    targets = step_index(table["step"], unit)
    origins = step_index(table["origin"], unit)
    predictions = []
    for series, target, origin in zip(table["series"], targets, origins, strict=True):
        history = histories.get(series, {})
        seasonal = history.get(target - period) if target - period <= origin else None
        value = seasonal if seasonal is not None and np.isfinite(seasonal) else history.get(origin)
        predictions.append(value)
    return {**table, "naive_prediction": np.asarray(predictions, dtype=float)}


def _horizon_metrics(con, run_dir, request, query, path, selection):
    scope = _horizon_label(selection)

    def metric(value=None, reason=None, **extra):
        value = float(value) if value is not None and np.isfinite(value) else None
        return {
            "value": value,
            "scope": "test",
            "note": scope,
            "reason": reason if value is None else None,
            **extra,
        }

    if path is None or not path.exists():
        return {
            name: metric(reason=f"{scope}: test predictions not saved")
            for name in ("wape", "mae", "mape", "test_ratio")
        }
    table = _read_view_rows(con, f"read_parquet({_lit(path)})")
    if "weeks_ahead" not in table:
        return {
            name: metric(reason=f"{scope}: weeks ahead not saved")
            for name in ("wape", "mae", "mape", "test_ratio")
        }
    test_targets = set(zip(table["series"], table["step"], strict=True))
    if selection["mode"] == "blend":
        validation = path.with_name(path.name.removesuffix("test.parquet") + "validation.parquet")
        if validation.exists():
            source = _horizon_history_source(
                f"read_parquet({_lit(validation)})", f"read_parquet({_lit(path)})"
            )
            table = _read_view_rows(con, source)
    table = _horizon_baseline(con, request, table)
    table = _select_horizon_rows(table, selection)
    if selection["mode"] == "blend":
        keep = np.asarray(
            [target in test_targets for target in zip(table["series"], table["step"], strict=True)],
            dtype=bool,
        )
        table = {name: values[keep] for name, values in table.items()}
    filters = _parse_filters(query.get("filters"))
    if filters:
        where = _filter_clause(filters, _spec_dims(request["spec"]))
        panel = Path(request["prepared"]) / "panel.parquet"
        selected = con.execute(
            f"SELECT DISTINCT __series FROM read_parquet({_lit(panel)}) WHERE {where}"
        ).fetchall()
        keep = np.isin(table["series"], [row[0] for row in selected])
        table = {name: values[keep] for name, values in table.items()}
    actual, prediction, naive = (
        np.asarray(table[name], dtype=float)
        for name in ("actual", "prediction", "naive_prediction")
    )
    known = np.isfinite(actual)
    valid = known & np.isfinite(prediction)
    positive = valid & (actual > 0)
    errors = np.abs(actual - prediction)
    denominator = np.abs(actual[known]).sum()
    reason = (
        "no known actuals in this view"
        if not known.any()
        else ("undefined predictions for known actuals" if not np.all(valid[known]) else None)
    )
    absolute = errors[valid].sum()
    naive_valid = known & np.isfinite(naive)
    naive_error = np.abs(actual[known] - naive[known]).sum()
    ratio_reason = reason or (
        "naive predictions unavailable on these target weeks"
        if not np.all(naive_valid[known])
        else None
    )
    return {
        "wape": metric(
            absolute / denominator if denominator and reason is None else None,
            reason or "absolute actuals sum to zero",
        ),
        "mae": metric(absolute / known.sum() if reason is None else None, reason),
        "mape": metric(
            np.mean(errors[positive] / actual[positive]) if positive.any() else None,
            "no finite predictions with actual > 0",
            excluded_rows=int((~positive).sum()),
        ),
        "test_ratio": metric(
            absolute / naive_error if naive_error and ratio_reason is None else None,
            ratio_reason or "naive error is zero on these target weeks",
        ),
    }


def _horizon_result_metrics(run_dir, request, query, details, selection):
    model = details["model"]
    scope = _horizon_label(selection)
    with connect() as con:
        path = _optional_prediction_path(run_dir, {**query, "model": model}, "test")
        selected_metrics = _horizon_metrics(con, run_dir, request, query, path, selection)
    unavailable = {
        name: {
            "value": None,
            "scope": "test",
            "note": scope,
            "reason": "not computed for this view",
        }
        for name in selected_metrics
    }
    for name in ("winner", "ensemble"):
        entry = details.get(name)
        if entry is not None:
            metrics = selected_metrics if name == model else unavailable
            entry["error_metrics"] = {**entry["error_metrics"], **metrics}
    for family, entry in details["families"].items():
        selected = model == "winner" and family == details["winner_family"]
        metrics = selected_metrics if selected else unavailable
        entry["error_metrics"] = {**entry["error_metrics"], **metrics}
    details["error_metrics"] = dict(details[model]["error_metrics"])
    for metric in details["metric_registry"]:
        if metric["scope"] == "test":
            metric["label"] += f" · {scope}"
            metric["explanation"] += f"; {scope}"
            if metric["id"] == "test_ratio":
                metric["computation"] = (
                    "view model wape / naive wape on the same target weeks and with the same weights"
                )


def _series_combos(con, panel_path: Path, breakdown: list[str], where: str) -> list[tuple]:
    if not breakdown:
        return [()]
    cols = ", ".join(_quote(d) for d in breakdown)
    rows = con.execute(
        f"SELECT DISTINCT {cols} FROM read_parquet({_lit(str(panel_path))}) WHERE {where}"
        f" ORDER BY {cols}"
    ).fetchall()
    return rows


def _combo_clause(breakdown: list[str], combo: tuple) -> str:
    if not breakdown:
        return "TRUE"
    return " AND ".join(f"{_quote(d)} = {_lit(v)}" for d, v in zip(breakdown, combo, strict=True))


def _combo_label(combo: tuple) -> str:
    return "|".join(str(v) for v in combo) if combo else ""


def _fetch_series(con, sql: str) -> dict[str, float]:
    """step -> value, from a two-column (step, value) result"""
    return {
        _iso(step): float(value) for step, value in con.execute(sql).fetchall() if value is not None
    }


def _align(values: dict[str, float], x: list[str]) -> list[float | None]:
    return [values.get(day) for day in x]


def timeseries_view(run_dir: Path, request: dict, query: dict) -> dict:
    prepared = Path(request["prepared"])
    spec = request["spec"]
    dims = _spec_dims(spec)
    target = _spec_target(spec)
    result = _require_done(run_dir)
    filters = _parse_filters(query.get("filters"))
    breakdown = _parse_breakdown(query.get("breakdown"), dims)

    panel = prepared / "panel.parquet"
    validation = _prediction_path(run_dir, query, "validation")
    test = _prediction_path(run_dir, query, "test")
    forecast = _prediction_path(run_dir, query, "forecast")

    con = connect()
    selection, sources, blend_band = _horizon_sources(
        con, request, query, validation, test, forecast
    )
    where = _filter_clause(filters, dims)
    combos = _series_combos(con, panel, breakdown, where)
    note = None
    if len(combos) > MAX_LINES:
        combos = combos[:MAX_LINES]
        note = f"capped at {MAX_LINES} of {len(combos)} lines"

    nonnegative = bool(
        con.execute(
            f"SELECT MIN({_quote(target)}) >= 0 FROM read_parquet({_lit(str(panel))})"
        ).fetchone()[0]
    )
    keymap_sql = (
        'SELECT DISTINCT "__series" AS series'
        + (f", {', '.join(_quote(d) for d in dims)}" if dims else "")
        + f" FROM read_parquet({_lit(str(panel))})"
    )

    series_lines, bands, x_set = [], [], set()

    for combo in combos:
        combo_where = f"{where} AND {_combo_clause(breakdown, combo)}"
        label = _combo_label(combo) or "all"

        actual = _fetch_series(
            con,
            f"SELECT __step, SUM({_quote(target)}) FROM read_parquet({_lit(str(panel))}) "
            f"WHERE {combo_where} GROUP BY __step ORDER BY __step",
        )

        history = sources.get("history")
        if history is None:
            history = f"""(
            WITH hist AS (
                SELECT series, step, bucket, actual, prediction
                FROM read_parquet({_lit(str(validation))})
                UNION ALL
                SELECT series, step, bucket, actual, prediction
                FROM read_parquet({_lit(str(test))})
            )
            SELECT series, step, AVG(prediction) AS prediction FROM hist GROUP BY series, step
            )"""
        hist_sql = f"""
            WITH keymap AS ({keymap_sql}),
            joined AS (SELECT p.step, p.prediction, k.* FROM {history} p JOIN keymap k USING (series))
            SELECT step, SUM(prediction) FROM joined WHERE {combo_where} GROUP BY step ORDER BY step
        """
        hist_forecast = _fetch_series(con, hist_sql)

        bucket = "f.weeks_ahead" if selection else "f.bucket"
        future_sql = f"""
            WITH keymap AS ({keymap_sql}),
            joined AS (
                SELECT f.step, {bucket} AS bucket, f.prediction, k.*
                FROM {sources["forecast"]} f JOIN keymap k USING (series)
            )
            SELECT step, bucket, SUM(prediction) FROM joined WHERE {combo_where}
            GROUP BY step, bucket ORDER BY step
        """
        future_rows = con.execute(future_sql).fetchall()
        future_forecast = {_iso(step): float(pred) for step, _bucket, pred in future_rows}

        forecast_series = {**hist_forecast, **future_forecast}
        x_set |= set(actual) | set(forecast_series)

        actual_id = f"actual:{label}" if breakdown else "actual"
        forecast_id = f"forecast:{label}" if breakdown else "forecast"
        actual_label = f"actual: {label}" if breakdown else "actual"
        forecast_label = f"forecast: {label}" if breakdown else "forecast"
        series_lines.append((actual_id, actual_label, actual, False))
        series_lines.append((forecast_id, forecast_label, forecast_series, True))

        if selection is None:
            lo_by_step, hi_by_step = _timeseries_band(
                con, keymap_sql, validation, forecast, combo_where, future_rows, nonnegative
            )
        elif selection["mode"] == "lead":
            lo_by_step, hi_by_step = _horizon_band(con, keymap_sql, sources, combo_where)
            future_lo, future_hi = _timeseries_band(
                con, keymap_sql, validation, forecast, combo_where, future_rows, nonnegative, True
            )
            lo_by_step.update(future_lo)
            hi_by_step.update(future_hi)
        elif blend_band is not None:
            lo_by_step, hi_by_step = _horizon_band(con, keymap_sql, sources, combo_where)
        else:
            continue
        bands.append((f"band:{forecast_id}", forecast_id, lo_by_step, hi_by_step))

    x = sorted(x_set)
    series_payload = [
        {"id": sid, "label": label, "values": _align(values, x), "dashed": dashed}
        for sid, label, values, dashed in series_lines
    ]
    bands_payload = [
        {"id": bid, "series": sid, "lo": _align(lo, x), "hi": _align(hi, x)}
        for bid, sid, lo, hi in bands
    ]
    markers = _timeseries_markers(con, validation, test, forecast)

    out = {
        "x": x,
        "series": series_payload,
        "bands": bands_payload,
        "markers": markers,
        "verdict": result["verdict"],
        "mode": "timeseries",
    }
    if selection:
        model = _select_model(run_dir, result, query)["model"]
        out.update(
            view=selection,
            horizon=int(spec["horizon"]),
            weeks_ahead_shown=selection["n"] if selection["mode"] == "lead" else None,
            lead_errors=_read_lead_errors(run_dir, result, model, int(spec["horizon"])),
        )
        if selection["mode"] == "blend":
            out["blend_band"] = _blend_band_summary(con, keymap_sql, sources, where, blend_band)
    if note:
        out["note"] = note
    con.close()
    return out


def _blend_band_summary(con, keymap_sql, sources, where, band):
    summary = {"target": 0.8, "test_coverage": None, "reason": None}
    if band is None:
        summary["reason"] = f"fewer than {MIN_BLEND_RESIDUALS} finite validation residuals"
        return summary
    rows = con.execute(f"""
        WITH keymap AS ({keymap_sql}), targets AS (
            SELECT DISTINCT series, step FROM {sources["test"]}
        )
        SELECT h.actual, h.lower, h.upper FROM {sources["history"]} h
        JOIN targets USING (series, step) JOIN keymap USING (series) WHERE {where}
    """).fetchall()
    values = np.asarray(rows, dtype=float).reshape(-1, 3)
    summary["test_coverage"] = _finite_number(ev.coverage(*values.T))
    if summary["test_coverage"] is None:
        summary["reason"] = "no test rows with finite actuals and bounds"
    return summary


def _horizon_band(con, keymap_sql, sources, where):
    rows = con.execute(f"""
        WITH keymap AS ({keymap_sql}), predictions AS (
            SELECT series, step, lower, upper FROM {sources["history"]}
            UNION ALL SELECT series, step, lower, upper FROM {sources["forecast"]}
        ), joined AS (SELECT p.step, p.lower, p.upper, k.* FROM predictions p JOIN keymap k USING (series))
        SELECT step,
            CASE WHEN COUNT(lower) = COUNT(*) THEN SUM(lower) END,
            CASE WHEN COUNT(upper) = COUNT(*) THEN SUM(upper) END
        FROM joined WHERE {where} GROUP BY step ORDER BY step
    """).fetchall()
    lo = {_iso(step): float(low) if low is not None else None for step, low, _ in rows}
    hi = {_iso(step): float(high) if high is not None else None for step, _, high in rows}
    return lo, hi


def _timeseries_band(
    con, keymap_sql, validation, forecast, combo_where, future_rows, nonnegative, single=False
):
    """fit aggregate validation bands separately by lead in single mode or bucket in mixed"""
    buckets = sorted({b for _s, b, _p in future_rows})
    lo_by_step: dict[str, float] = {}
    hi_by_step: dict[str, float] = {}
    for bucket in buckets:
        predicate = f"v.weeks_ahead = {int(bucket)}" if single else f"v.bucket = {_lit(bucket)}"
        val_sql = f"""
            WITH keymap AS ({keymap_sql}),
            joined AS (
                SELECT v.step, v.actual, v.prediction, k.*
                FROM read_parquet({_lit(str(validation))}) v JOIN keymap k USING (series)
                WHERE {predicate}
            )
            SELECT step, SUM(actual), SUM(prediction) FROM joined WHERE {combo_where}
            GROUP BY step ORDER BY step
        """
        rows = con.execute(val_sql).fetchall()
        if not rows:
            continue
        actual_arr = np.array([r[1] for r in rows], dtype=float)
        pred_arr = np.array([r[2] for r in rows], dtype=float)
        band = ev.fit_band(actual_arr, pred_arr)
        bucket_future = [(s, p) for s, b, p in future_rows if b == bucket]
        steps = np.array([_iso(s) for s, _p in bucket_future])
        preds = np.array([p for _s, p in bucket_future], dtype=float)
        lo, hi = ev.apply_band(band, preds, nonnegative)
        for day, lv, hv in zip(steps, lo, hi, strict=True):
            lo_by_step[day] = None if not np.isfinite(lv) else float(lv)
            hi_by_step[day] = None if not np.isfinite(hv) else float(hv)
    return lo_by_step, hi_by_step


def _timeseries_markers(con, validation: Path, test: Path, forecast: Path) -> list[dict]:
    markers = []
    for path, label in (
        (validation, "validation start"),
        (test, "test start"),
        (forecast, "forecast origin"),
    ):
        if not path.exists():
            continue
        row = con.execute(f"SELECT MIN(step) FROM read_parquet({_lit(str(path))})").fetchone()
        if row and row[0] is not None:
            markers.append({"x": _iso(row[0]), "label": label})
    return markers


# ---------------------------------------------------------------- entity mode view


def _entity_unit_days(spec: dict) -> float:
    from smolsmort.forecast.spec import UNIT_DAYS

    censor = spec.get("censor")
    unit = censor["unit"] if censor else "week"
    return UNIT_DAYS[unit]


def _entity_keymap_sql(rows_path: Path, dims: list[str]) -> str:
    cols = ", ".join(_quote(d) for d in dims)
    return (
        'SELECT "__row" AS row'
        + (f", {cols}" if dims else "")
        + f" FROM read_parquet({_lit(str(rows_path))})"
    )


def _weekify(days: np.ndarray) -> np.ndarray:
    """the monday of the iso week each date falls in, as an array of date objects"""
    days = np.asarray(days).astype("datetime64[D]")
    weekday = (days.astype("datetime64[D]").view("int64") - 3) % 7  # 1970-01-01 was a thursday
    return (days - weekday.astype("timedelta64[D]")).astype("datetime64[D]")


def _is_hazard_entity(run_dir: Path, query: dict) -> bool:
    if _optional_prediction_path(run_dir, query, "curve") is not None:
        return True
    selected = _select_model(run_dir, _require_done(run_dir), query)
    recipe = _read_json(run_dir / "recipe.json", {})
    genome = recipe.get("genome") or {}
    return selected["model"] == "winner" and genome.get("objective") == "hazard"


def _hazard_entity_view(run_dir: Path, request: dict, query: dict) -> dict:
    from smolsmort.forecast.aggregate import REGISTRY

    result = _require_done(run_dir)
    out = {
        "x": [],
        "series": [],
        "bands": [],
        "markers": [],
        "verdict": result["verdict"],
        "mode": "entity",
        "objective": "hazard",
        "aggregate": None,
    }
    aggregate_path = _optional_prediction_path(run_dir, query, "aggregate")
    if aggregate_path is None:
        return out

    spec = request["spec"]
    dims = _spec_dims(spec)
    filters = _parse_filters(query.get("filters"))
    breakdown = _parse_breakdown(query.get("breakdown"), dims)
    recipe = _read_json(run_dir / "recipe.json", {})
    name = request.get("aggregate", recipe.get("aggregate", "none"))
    label = name.replace("_", " ")
    con = connect()
    if not filters and not breakdown:
        values = _fetch_series(
            con, f"SELECT step, prediction FROM read_parquet({_lit(str(aggregate_path))})"
        )
        lines = [("aggregate", label, values)]
    else:
        curve_path = _optional_prediction_path(run_dir, query, "curve")
        rule = REGISTRY.get(name)
        if curve_path is None or rule is None or name == "none":
            out["note"] = "weekly series unavailable for this selection"
            return out
        rows_path = Path(request["prepared"]) / "rows.parquet"
        where = _filter_clause(filters, dims)
        combos = _series_combos(con, rows_path, breakdown, where)
        if len(combos) > MAX_LINES:
            out["note"] = f"capped at {MAX_LINES} of {len(combos)} lines"
            combos = combos[:MAX_LINES]
        keymap_sql = _entity_keymap_sql(rows_path, dims)
        unit = (spec.get("censor") or {}).get("unit", "week")
        lines = []
        for combo in combos:
            combo_where = f"{where} AND {_combo_clause(breakdown, combo)}"
            curves = con.execute(
                f"""
                WITH keymap AS ({keymap_sql}),
                selected_rows AS (SELECT row FROM keymap WHERE {combo_where})
                SELECT c.* FROM read_parquet({_lit(str(curve_path))}) c
                JOIN selected_rows k ON c.row = k.row
                """
            ).fetchnumpy()
            summary = rule(curves, unit)
            if summary is None:
                continue
            values = {
                _iso(step): float(prediction)
                for step, prediction in zip(summary["step"], summary["prediction"], strict=True)
            }
            combo_label = _combo_label(combo)
            lines.append(
                (
                    f"aggregate:{combo_label}" if breakdown else "aggregate",
                    f"{label}: {combo_label}" if breakdown else label,
                    values,
                )
            )
    x = sorted({step for _sid, _label, values in lines for step in values})
    out.update(
        x=x,
        series=[
            {"id": sid, "label": line_label, "values": _align(values, x), "dashed": True}
            for sid, line_label, values in lines
        ],
        aggregate={"rule": name, "unit": "week"},
    )
    return out


def entity_view(run_dir: Path, request: dict, query: dict) -> dict:
    if _is_hazard_entity(run_dir, query):
        return _hazard_entity_view(run_dir, request, query)
    prepared = Path(request["prepared"])
    spec = request["spec"]
    dims = _spec_dims(spec)
    target = _spec_target(spec)
    result = _require_done(run_dir)
    filters = _parse_filters(query.get("filters"))
    breakdown = _parse_breakdown(query.get("breakdown"), dims)
    unit_days = _entity_unit_days(spec)

    rows_path = prepared / "rows.parquet"
    forecast_path = _prediction_path(run_dir, query, "forecast")
    validation_path = _prediction_path(run_dir, query, "validation")

    con = connect()
    where = _filter_clause(filters, dims)
    combos = _series_combos(con, rows_path, breakdown, where)
    note = None
    if len(combos) > MAX_LINES:
        combos = combos[:MAX_LINES]
        note = f"capped at {MAX_LINES} of {len(combos)} lines"

    keymap_sql = _entity_keymap_sql(rows_path, dims)

    resid_rows = con.execute(
        f"SELECT truth, prediction FROM read_parquet({_lit(str(validation_path))})"
    ).fetchall()
    resid_pool = np.array(
        [(t - p) / p for t, p in resid_rows if p not in (0, None) and t is not None],
        dtype=float,
    )
    resid_pool = resid_pool[np.isfinite(resid_pool)]

    series_lines, bands, x_set = [], [], set()

    for combo in combos:
        combo_where = f"{where} AND {_combo_clause(breakdown, combo)}"
        label = _combo_label(combo) or "all"

        load_sql = f"""
            WITH keymap AS ({keymap_sql})
            SELECT f.predicted_date, f.prediction, k.*
            FROM read_parquet({_lit(str(forecast_path))}) f JOIN keymap k ON f.row = k.row
            WHERE {combo_where}
        """
        load_rows = con.execute(load_sql).fetchall()
        load_by_week: dict[str, int] = {}
        predictions, week_of = [], []
        for predicted_date, prediction, *_rest in load_rows:
            week = _iso(_weekify(np.array([predicted_date]))[0])
            load_by_week[week] = load_by_week.get(week, 0) + 1
            predictions.append(prediction)
            week_of.append(week)

        actual_sql = f"""
            WITH keymap AS ({keymap_sql})
            SELECT r."__anchor", r.{_quote(target)}, k.*
            FROM read_parquet({_lit(str(rows_path))}) r JOIN keymap k ON r."__row" = k.row
            WHERE r.{_quote(target)} IS NOT NULL AND {combo_where}
        """
        actual_by_week: dict[str, int] = {}
        for anchor, value, *_rest in con.execute(actual_sql).fetchall():
            resolved = np.datetime64(anchor, "D") + np.timedelta64(
                int(round(value * unit_days)), "D"
            )
            week = _iso(_weekify(np.array([resolved]))[0])
            actual_by_week[week] = actual_by_week.get(week, 0) + 1

        x_set |= set(load_by_week) | set(actual_by_week)
        load_id = f"load:{label}" if breakdown else "load"
        actual_id = f"actual:{label}" if breakdown else "actual"
        series_lines.append((actual_id, label, actual_by_week, False))
        series_lines.append((load_id, label, load_by_week, True))

        lo, hi = _entity_mc_band(predictions, week_of, resid_pool, unit_days)
        bands.append((f"band:{load_id}", load_id, lo, hi))

    x = sorted(x_set)
    series_payload = [
        {"id": sid, "label": label, "values": _align(values, x), "dashed": dashed}
        for sid, label, values, dashed in series_lines
    ]
    bands_payload = [
        {"id": bid, "series": sid, "lo": _align(lo, x), "hi": _align(hi, x)}
        for bid, sid, lo, hi in bands
    ]

    out = {
        "x": x,
        "series": series_payload,
        "bands": bands_payload,
        "markers": [],
        "verdict": result["verdict"],
        "mode": "entity",
    }
    if note:
        out["note"] = note
    return out


def _entity_mc_band(predictions, week_of, resid_pool, unit_days) -> tuple[dict, dict]:
    """200 draws: each open row's week moves by prediction * (1 + a sampled validation residual);
    the 10/90 percentile of the resulting weekly counts is the band"""
    if not predictions or not len(resid_pool):
        return {}, {}
    rng = np.random.default_rng(MC_SEED)
    predictions = np.asarray(predictions, dtype=float)
    base_weeks_arr = np.array([np.datetime64(w, "D") for w in week_of])
    draws = []
    weeks: set[str] = set(week_of)
    for _draw in range(MC_DRAWS):
        r = rng.choice(resid_pool, size=len(predictions), replace=True)
        shifted_days = (predictions * (1 + r) * unit_days).round().astype("timedelta64[D]")
        draw_weeks = [_iso(w) for w in _weekify(base_weeks_arr + shifted_days)]
        draws.append(draw_weeks)
        weeks.update(draw_weeks)
    ordered_weeks = sorted(weeks)
    week_index = {w: i for i, w in enumerate(ordered_weeks)}
    counts = np.zeros((MC_DRAWS, len(ordered_weeks)), dtype=int)
    for draw, draw_weeks in enumerate(draws):
        for w in draw_weeks:
            counts[draw, week_index[w]] += 1
    lo_vals = np.percentile(counts, 10, axis=0)
    hi_vals = np.percentile(counts, 90, axis=0)
    return (
        {w: float(lo_vals[i]) for i, w in enumerate(ordered_weeks)},
        {w: float(hi_vals[i]) for i, w in enumerate(ordered_weeks)},
    )


def view(run_dir: Path, request: dict, query: dict) -> dict:
    mode = normalize_mode(request["spec"]["mode"])
    details = result_details(run_dir, request, query)
    payload = (
        entity_view(run_dir, request, query)
        if mode == "entity"
        else timeseries_view(run_dir, request, query)
    )
    if payload.get("view"):
        _horizon_result_metrics(run_dir, request, query, details, payload["view"])
    for series in payload["series"]:
        if series.get("dashed"):
            series["family"] = details["family"]
            series["members"] = details["members"]
    return {**payload, **details}


# ---------------------------------------------------------------- table


def _paginate(rows: list, page: int, size: int) -> tuple[list, int]:
    total = len(rows)
    start = page * size
    return rows[start : start + size], total


def _family_column(request: dict) -> str:
    name = "family"
    dims = _spec_dims(request["spec"])
    while name in dims:
        name = f"model_{name}"
    return name


def timeseries_table(run_dir: Path, request: dict, query: dict) -> dict:
    prepared = Path(request["prepared"])
    spec = request["spec"]
    dims = _spec_dims(spec)
    filters = _parse_filters(query.get("filters"))
    where = _filter_clause(filters, dims)
    panel = prepared / "panel.parquet"
    validation = _prediction_path(run_dir, query, "validation")
    test = _prediction_path(run_dir, query, "test")
    forecast = _prediction_path(run_dir, query, "forecast")
    con = connect()
    selection, sources, _ = _horizon_sources(con, request, query, validation, test, forecast)
    single = selection is not None
    hist_leads = ", h.origin, h.weeks_ahead" if single else ""
    future_leads = ", f.origin, f.weeks_ahead" if single else ""

    keymap_sql = (
        'SELECT DISTINCT "__series" AS series'
        + (f", {', '.join(_quote(d) for d in dims)}" if dims else "")
        + f" FROM read_parquet({_lit(str(panel))})"
    )
    dims_sel = ", ".join(f"k.{_quote(d)}" for d in dims)
    hist_sql = f"""
        WITH keymap AS ({keymap_sql})
        SELECT h.step, {dims_sel + "," if dims else ""} h.prediction, NULL AS lower, NULL AS upper,
               h.actual, FALSE AS future{hist_leads}
        FROM read_parquet({_lit(str(validation))}) h
        JOIN keymap k USING (series)
        UNION ALL
        SELECT h.step, {dims_sel + "," if dims else ""} h.prediction, NULL AS lower, NULL AS upper,
               h.actual, FALSE AS future{hist_leads}
        FROM read_parquet({_lit(str(test))}) h
        JOIN keymap k USING (series)
        UNION ALL
        SELECT f.step, {dims_sel + "," if dims else ""} f.prediction, f.lower, f.upper,
               CAST(NULL AS DOUBLE) AS actual, TRUE AS future{future_leads}
        FROM read_parquet({_lit(str(forecast))}) f
        JOIN keymap k USING (series)
    """
    if selection:
        selected_leads = (
            hist_leads if selection["mode"] == "lead" else ", NULL AS origin, NULL AS weeks_ahead"
        )
        hist_sql = f"""
            WITH keymap AS ({keymap_sql})
            SELECT h.step, {dims_sel + "," if dims else ""} h.prediction, h.lower, h.upper,
                   h.actual, FALSE AS future{selected_leads}
            FROM {sources["history"]} h JOIN keymap k USING (series)
            UNION ALL
            SELECT f.step, {dims_sel + "," if dims else ""} f.prediction, f.lower, f.upper,
                   CAST(NULL AS DOUBLE) AS actual, TRUE AS future{future_leads}
            FROM {sources["forecast"]} f JOIN keymap k USING (series)
        """
    full_sql = f"SELECT * FROM ({hist_sql}) t WHERE {where}"
    sort = query.get("sort") or "future"
    if sort == _family_column(request):
        sort = "future"
    order = "DESC" if sort == "future" or query.get("desc") == "1" else "ASC"
    full_sql += f" ORDER BY {_quote(sort) if sort != 'future' else 'future'} {order}, step ASC"
    rows = con.execute(full_sql).fetchall()
    columns = ["step", *dims, "prediction", "lower", "upper", "actual", "future"]
    if single:
        columns += ["origin", "weeks_ahead"]
    page = int(query.get("page", 0))
    size = int(query.get("size", 50))
    page_rows, total = _paginate(rows, page, size)
    out_rows = [_row_to_json(r) for r in page_rows]
    con.close()
    return {"columns": columns, "rows": out_rows, "total": total, "page": page}


def _row_to_json(row: tuple) -> list:
    out = []
    for value in row:
        if isinstance(value, (dt.date, dt.datetime)):
            out.append(_iso(value))
        elif isinstance(value, bool):
            out.append(value)
        elif isinstance(value, float) and not np.isfinite(value):
            out.append(None)
        else:
            out.append(value)
    return out


def entity_table(run_dir: Path, request: dict, query: dict) -> dict:
    prepared = Path(request["prepared"])
    spec = request["spec"]
    dims = _spec_dims(spec)
    filters = _parse_filters(query.get("filters"))
    where = _filter_clause(filters, dims)
    unit_days = _entity_unit_days(spec)
    rows_path = prepared / "rows.parquet"
    forecast_path = _prediction_path(run_dir, query, "forecast")

    keymap_sql = _entity_keymap_sql(rows_path, dims)
    dims_sel = ", ".join(f"k.{_quote(d)}" for d in dims)
    con = connect()
    hazard = _is_hazard_entity(run_dir, query)
    values_sql = (
        f"f.prediction * {unit_days / 7} AS median_weeks, "
        f"f.lower * {unit_days / 7} AS p10_weeks, f.upper * {unit_days / 7} AS p90_weeks"
        if hazard
        else "f.prediction, "
        f"f.anchor + CAST(ROUND(f.lower * {unit_days}) AS INTEGER) * INTERVAL 1 DAY AS lower, "
        f"f.anchor + CAST(ROUND(f.upper * {unit_days}) AS INTEGER) * INTERVAL 1 DAY AS upper"
    )
    sql = f"""
        WITH keymap AS ({keymap_sql})
        SELECT f.row, {dims_sel + "," if dims else ""} f.anchor, f.predicted_date, {values_sql}
        FROM read_parquet({_lit(str(forecast_path))}) f
        JOIN keymap k ON f.row = k.row
    """
    sql = f"SELECT * FROM ({sql}) t WHERE {where}"
    sort = query.get("sort") or "predicted_date"
    if sort == _family_column(request):
        sort = "predicted_date"
    order = "DESC" if query.get("desc") == "1" else "ASC"
    sql += f" ORDER BY {_quote(sort)} {order}"
    rows = con.execute(sql).fetchall()
    value_columns = (
        ["median_weeks", "p10_weeks", "p90_weeks"] if hazard else ["prediction", "lower", "upper"]
    )
    columns = ["row", *dims, "anchor", "predicted_date", *value_columns]

    group = query.get("group")
    groups = None
    if group and group in dims:
        idx = columns.index(group)
        acc: dict[str, list] = {}
        for r in rows:
            acc.setdefault(r[idx], []).append(r[columns.index("predicted_date")])
        groups = {
            str(key): {"earliest": _iso(min(dates)), "latest": _iso(max(dates))}
            for key, dates in acc.items()
        }

    page = int(query.get("page", 0))
    size = int(query.get("size", 50))
    page_rows, total = _paginate(rows, page, size)
    out_rows = [_row_to_json(r) for r in page_rows]
    out = {"columns": columns, "rows": out_rows, "total": total, "page": page}
    if hazard:
        out["objective"] = "hazard"
    if groups is not None:
        out["groups"] = groups
    return out


def table(run_dir: Path, request: dict, query: dict) -> dict:
    mode = normalize_mode(request["spec"]["mode"])
    selected = _select_model(run_dir, _require_done(run_dir), query)
    payload = (
        entity_table(run_dir, request, query)
        if mode == "entity"
        else timeseries_table(run_dir, request, query)
    )
    family_column = _family_column(request)
    payload["columns"].append(family_column)
    for row in payload["rows"]:
        row.append(selected["family"])
    return {
        **payload,
        "model": selected["model"],
        "family": selected["family"],
        "family_column": family_column,
        "members": selected["members"],
    }


# ---------------------------------------------------------------- export


def export_csv(run_dir: Path, request: dict, query: dict | None = None) -> bytes:
    """the full forecast table (page 0, everything) as csv, with the error ratio on every row"""
    result = _require_done(run_dir)
    query = query or {}
    verdict = _select_model(run_dir, result, query)["verdict"]
    payload = table(run_dir, request, {**query, "size": 10**9})
    ratio = verdict["accuracy"]["ratio"]

    buf = io.StringIO()
    buf.write(f"# verdict: {verdict['summary']}\n")
    writer = csv.writer(buf)
    writer.writerow([*payload["columns"], "ratio"])
    for row in payload["rows"]:
        writer.writerow([*row, ratio])
    return buf.getvalue().encode("utf-8")
