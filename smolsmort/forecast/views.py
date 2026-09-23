"""the forecast tab's aggregation, as plain functions over a run folder and its prepared tables.

nothing here talks http - a route calls one of these and returns what comes back. numpy/duckdb
only: the server that imports this module must never load torch or run xgboost in-process.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
from pathlib import Path

import numpy as np

from smolsmort.forecast import evaluate as ev
from smolsmort.forecast.tables import connect

MAX_LINES = 12
MC_DRAWS = 200
MC_SEED = 0


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
    path = run_dir / "leaderboard.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())[:top]


def _spec_dims(spec: dict) -> list[str]:
    return [c["name"] for c in spec["columns"] if c["role"] == "dimension"]


def _spec_target(spec: dict) -> str:
    return next(c["name"] for c in spec["columns"] if c["role"] == "target")


def _require_done(run_dir: Path) -> dict:
    result = read_result(run_dir)
    if result is None:
        raise ViewError(f"run {run_dir.name!r} has no result yet")
    return result


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
    validation = run_dir / "validation.parquet"
    test = run_dir / "test.parquet"
    forecast = run_dir / "forecast.parquet"

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
    forecast_path = run_dir / "forecast.parquet"
    validation_path = run_dir / "validation.parquet"

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
    return (
        row_view(run_dir, request, query) if mode == "row" else series_view(run_dir, request, query)
    )


# ---------------------------------------------------------------- table


def _paginate(rows: list, page: int, size: int) -> tuple[list, int]:
    total = len(rows)
    start = page * size
    return rows[start : start + size], total


def series_table(run_dir: Path, request: dict, query: dict) -> dict:
    prepared = Path(request["prepared"])
    spec = request["spec"]
    dims = _spec_dims(spec)
    filters = _parse_filters(query.get("filters"))
    where = _filter_clause(filters, dims)
    panel = prepared / "panel.parquet"

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
        FROM read_parquet({_lit(str(run_dir / "validation.parquet"))}) h
        JOIN keymap k USING (series)
        UNION ALL
        SELECT h.step, {dims_sel + "," if dims else ""} h.prediction, NULL AS lower, NULL AS upper,
               h.actual, FALSE AS future
        FROM read_parquet({_lit(str(run_dir / "test.parquet"))}) h
        JOIN keymap k USING (series)
        UNION ALL
        SELECT f.step, {dims_sel + "," if dims else ""} f.prediction, f.lower, f.upper,
               CAST(NULL AS DOUBLE) AS actual, TRUE AS future
        FROM read_parquet({_lit(str(run_dir / "forecast.parquet"))}) f
        JOIN keymap k USING (series)
    """
    con = connect()
    full_sql = f"SELECT * FROM ({hist_sql}) t WHERE {where}"
    sort = query.get("sort") or "future"
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
    forecast_path = run_dir / "forecast.parquet"

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
    return (
        row_table(run_dir, request, query)
        if mode == "row"
        else series_table(run_dir, request, query)
    )


# ---------------------------------------------------------------- export


def export_csv(run_dir: Path, request: dict) -> bytes:
    """the full forecast table (page 0, everything) as csv, the verdict named on every row"""
    result = _require_done(run_dir)
    verdict = result["verdict"]
    payload = table(run_dir, request, {"size": 10**9})
    trust_word = "trusted" if verdict["trusted"] else "not trusted"

    buf = io.StringIO()
    buf.write(f"# verdict: {verdict['summary']}\n")
    writer = csv.writer(buf)
    writer.writerow([*payload["columns"], "verdict"])
    for row in payload["rows"]:
        writer.writerow([*row, trust_word])
    return buf.getvalue().encode("utf-8")
