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
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np

from smolsmort.forecast import evaluate as ev
from smolsmort.forecast.families import registered_families
from smolsmort.forecast.spec import spec_from_dict
from smolsmort.forecast.tables import connect

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
    return json.loads((run_dir / "request.json").read_text())


def read_result(run_dir: Path) -> dict | None:
    path = run_dir / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def read_leaderboard(run_dir: Path, top: int = 10) -> list[dict]:
    if read_request(run_dir).get("version") == 2:
        board = []
        for selection in read_request(run_dir)["families"]:
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
            return board[:top]
    path = run_dir / "leaderboard.json"
    if not path.exists():
        return []
    return [
        {
            **entry,
            "family": entry.get("family") or (entry.get("genome") or {}).get("family") or "xgboost",
        }
        for entry in json.loads(path.read_text())[:top]
    ]


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
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


def family_catalog(runs_root: Path, spec=None, prepared=None, task="regression") -> list[dict]:
    """suitability uses the worker's split and features, without fitting any models"""
    from smolsmort.forecast.pipeline import PipelineError, load_workspace
    from smolsmort.forecast.splits import SplitError

    workspace, history_error = None, None
    if spec is not None:
        task = spec.task
        try:
            workspace = load_workspace(spec, prepared.folder, prepared.summary)
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
        elif spec is not None and spec.mode == "row" and family.name != "xgboost":
            reason = "per-row runs support xgboost only"
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
        from smolsmort.forecast.splits import series_cuts

        steps = np.unique(ws.panel[STEP])
        cut = steps[series_cuts(len(steps), ws.spec.horizon).val_start]
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
        if ws.mode == "series":
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
            tmp = run_dir / "route-cursors.json.tmp"
            tmp.write_text(json.dumps(snapshot), encoding="utf-8")
            tmp.replace(run_dir / "route-cursors.json")
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
    }
    if since_generation is None:
        out["eval_history"] = _read_json(run_dir / "eval_history.json")
    return out


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


def result_details(run_dir: Path, request: dict, query: dict) -> dict:
    """comparison uses validation scores; only the winner and ensemble have test verdicts"""
    result = _require_done(run_dir)
    selected = _select_model(run_dir, result, query)
    selected.pop("default_model")
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
    return {
        **selected,
        "families": families,
        "winner": result.get("winner")
        or {"family": selected["winner_family"], "verdict": result.get("verdict")},
        "ensemble": result.get("ensemble"),
        "importance_family": importance_family,
        "importance": importance,
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


# ---------------------------------------------------------------- series mode view


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


def series_view(run_dir: Path, request: dict, query: dict) -> dict:
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

        hist_sql = f"""
            WITH keymap AS ({keymap_sql}),
            hist AS (
                SELECT series, step, bucket, actual, prediction
                FROM read_parquet({_lit(str(validation))})
                UNION ALL
                SELECT series, step, bucket, actual, prediction
                FROM read_parquet({_lit(str(test))})
            ),
            per_step AS (
                SELECT series, step, AVG(prediction) AS prediction
                FROM hist GROUP BY series, step
            ),
            joined AS (SELECT p.step, p.prediction, k.* FROM per_step p JOIN keymap k USING (series))
            SELECT step, SUM(prediction) FROM joined WHERE {combo_where} GROUP BY step ORDER BY step
        """
        hist_forecast = _fetch_series(con, hist_sql)

        future_sql = f"""
            WITH keymap AS ({keymap_sql}),
            joined AS (
                SELECT f.step, f.bucket, f.prediction, k.*
                FROM read_parquet({_lit(str(forecast))}) f JOIN keymap k USING (series)
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
        series_lines.append((actual_id, label, actual, False))
        series_lines.append((forecast_id, label, forecast_series, True))

        lo_by_step, hi_by_step = _series_band(
            con, keymap_sql, validation, forecast, combo_where, future_rows, nonnegative
        )
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
    markers = _series_markers(con, validation, test, forecast)

    out = {
        "x": x,
        "series": series_payload,
        "bands": bands_payload,
        "markers": markers,
        "verdict": result["verdict"],
        "mode": "series",
    }
    if note:
        out["note"] = note
    return out


def _series_band(con, keymap_sql, validation, forecast, combo_where, future_rows, nonnegative):
    """per bucket: fit a band on the view's aggregated validation residuals, apply it to that
    bucket's aggregated future prediction - buckets partition the horizon, so every future step
    gets exactly one band"""
    buckets = sorted({b for _s, b, _p in future_rows})
    lo_by_step: dict[str, float] = {}
    hi_by_step: dict[str, float] = {}
    for bucket in buckets:
        val_sql = f"""
            WITH keymap AS ({keymap_sql}),
            joined AS (
                SELECT v.step, v.actual, v.prediction, k.*
                FROM read_parquet({_lit(str(validation))}) v JOIN keymap k USING (series)
                WHERE v.bucket = {_lit(bucket)}
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


def _series_markers(con, validation: Path, test: Path, forecast: Path) -> list[dict]:
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


# ---------------------------------------------------------------- row mode view


def _row_unit_days(spec: dict) -> float:
    from smolsmort.forecast.spec import UNIT_DAYS

    censor = spec.get("censor")
    unit = censor["unit"] if censor else "week"
    return UNIT_DAYS[unit]


def _row_keymap_sql(rows_path: Path, dims: list[str]) -> str:
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


def row_view(run_dir: Path, request: dict, query: dict) -> dict:
    prepared = Path(request["prepared"])
    spec = request["spec"]
    dims = _spec_dims(spec)
    target = _spec_target(spec)
    result = _require_done(run_dir)
    filters = _parse_filters(query.get("filters"))
    breakdown = _parse_breakdown(query.get("breakdown"), dims)
    unit_days = _row_unit_days(spec)

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

    keymap_sql = _row_keymap_sql(rows_path, dims)

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

        lo, hi = _row_mc_band(predictions, week_of, resid_pool, unit_days)
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
        "mode": "row",
    }
    if note:
        out["note"] = note
    return out


def _row_mc_band(predictions, week_of, resid_pool, unit_days) -> tuple[dict, dict]:
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
    mode = request["spec"]["mode"]
    details = result_details(run_dir, request, query)
    payload = (
        row_view(run_dir, request, query) if mode == "row" else series_view(run_dir, request, query)
    )
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


def series_table(run_dir: Path, request: dict, query: dict) -> dict:
    prepared = Path(request["prepared"])
    spec = request["spec"]
    dims = _spec_dims(spec)
    filters = _parse_filters(query.get("filters"))
    where = _filter_clause(filters, dims)
    panel = prepared / "panel.parquet"
    validation = _prediction_path(run_dir, query, "validation")
    test = _prediction_path(run_dir, query, "test")
    forecast = _prediction_path(run_dir, query, "forecast")

    keymap_sql = (
        'SELECT DISTINCT "__series" AS series'
        + (f", {', '.join(_quote(d) for d in dims)}" if dims else "")
        + f" FROM read_parquet({_lit(str(panel))})"
    )
    dims_sel = ", ".join(f"k.{_quote(d)}" for d in dims)
    hist_sql = f"""
        WITH keymap AS ({keymap_sql})
        SELECT h.step, {dims_sel + "," if dims else ""} h.prediction, NULL AS lower, NULL AS upper,
               h.actual, FALSE AS future
        FROM read_parquet({_lit(str(validation))}) h
        JOIN keymap k USING (series)
        UNION ALL
        SELECT h.step, {dims_sel + "," if dims else ""} h.prediction, NULL AS lower, NULL AS upper,
               h.actual, FALSE AS future
        FROM read_parquet({_lit(str(test))}) h
        JOIN keymap k USING (series)
        UNION ALL
        SELECT f.step, {dims_sel + "," if dims else ""} f.prediction, f.lower, f.upper,
               CAST(NULL AS DOUBLE) AS actual, TRUE AS future
        FROM read_parquet({_lit(str(forecast))}) f
        JOIN keymap k USING (series)
    """
    con = connect()
    full_sql = f"SELECT * FROM ({hist_sql}) t WHERE {where}"
    sort = query.get("sort") or "future"
    if sort == _family_column(request):
        sort = "future"
    order = "DESC" if sort == "future" or query.get("desc") == "1" else "ASC"
    full_sql += f" ORDER BY {_quote(sort) if sort != 'future' else 'future'} {order}, step ASC"
    rows = con.execute(full_sql).fetchall()
    columns = ["step", *dims, "prediction", "lower", "upper", "actual", "future"]
    page = int(query.get("page", 0))
    size = int(query.get("size", 50))
    page_rows, total = _paginate(rows, page, size)
    out_rows = [_row_to_json(r) for r in page_rows]
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


def row_table(run_dir: Path, request: dict, query: dict) -> dict:
    prepared = Path(request["prepared"])
    spec = request["spec"]
    dims = _spec_dims(spec)
    filters = _parse_filters(query.get("filters"))
    where = _filter_clause(filters, dims)
    unit_days = _row_unit_days(spec)
    rows_path = prepared / "rows.parquet"
    forecast_path = _prediction_path(run_dir, query, "forecast")

    keymap_sql = _row_keymap_sql(rows_path, dims)
    dims_sel = ", ".join(f"k.{_quote(d)}" for d in dims)
    con = connect()
    sql = f"""
        WITH keymap AS ({keymap_sql})
        SELECT f.row, {dims_sel + "," if dims else ""} f.anchor, f.predicted_date, f.prediction,
               f.anchor + CAST(ROUND(f.lower * {unit_days}) AS INTEGER) * INTERVAL 1 DAY AS lower,
               f.anchor + CAST(ROUND(f.upper * {unit_days}) AS INTEGER) * INTERVAL 1 DAY AS upper
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
    columns = ["row", *dims, "anchor", "predicted_date", "prediction", "lower", "upper"]

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
    if groups is not None:
        out["groups"] = groups
    return out


def table(run_dir: Path, request: dict, query: dict) -> dict:
    mode = request["spec"]["mode"]
    selected = _select_model(run_dir, _require_done(run_dir), query)
    payload = (
        row_table(run_dir, request, query)
        if mode == "row"
        else series_table(run_dir, request, query)
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
