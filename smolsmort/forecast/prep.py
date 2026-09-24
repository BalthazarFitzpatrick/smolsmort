"""spec + source file -> cached parquet tables, via one duckdb sql script per prep.

the sql text this module builds is executed statement by statement, in the same order, and
written verbatim to prep.sql - what the user sees is what ran, never a reconstruction of it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from smolsmort.forecast.spec import (
    ANCHOR,
    FILES,
    LOWER,
    ROW,
    SERIES,
    STEP,
    UNIT_DAYS,
    UPPER,
    PrepSpec,
)
from smolsmort.forecast.tables import connect, resolve_encoding, type_source

# a working flag only; it never reaches the parquet files
PREDICT = "__predict"

# sql interval unit per step, all accepted as duckdb interval literals
STEP_UNIT = {"day": "DAY", "week": "WEEK", "month": "MONTH", "quarter": "QUARTER", "year": "YEAR"}


@dataclass
class Prepared:
    folder: Path
    sql: str
    summary: dict
    reused: bool


def _quote(name: str) -> str:
    """double-quote a sql identifier, escaping embedded quotes"""
    return '"' + name.replace('"', '""') + '"'


def _literal(value: str) -> str:
    """single-quote a sql string literal, escaping embedded quotes"""
    return "'" + str(value).replace("'", "''") + "'"


def _cols(names: list[str]) -> str:
    return ", ".join(_quote(n) for n in names)


def prepare(spec: PrepSpec, cache_root: Path) -> Prepared:
    """run (or reuse) a prep spec against its source, writing the mode's parquet tables"""
    source_path = Path(spec.source)
    source_bytes = source_path.read_bytes()
    key = f"{spec.digest()}-{hashlib.sha256(source_bytes).hexdigest()[:12]}"
    folder = cache_root / key

    if _is_complete(spec, folder):
        return _load_cached(folder)

    cache_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=cache_root, prefix=".tmp-")).resolve()
    try:
        con = connect()
        # single-threaded: row_number() over the raw scan must match source file order
        con.execute("SET threads=1")
        read_expr = type_source(con, _load_source(spec, source_path, source_bytes, tmp))
        if spec.mode == "row":
            sql, summary = _prep_row(con, spec, read_expr, tmp)
        else:
            sql, summary = _prep_series(con, spec, read_expr, tmp)
        con.close()
        # runs on absolute temp paths, stored relative: the text replays from inside the cache
        # folder, and the process never changes directory (the server serves other threads)
        sql = sql.replace(f"{tmp}{os.sep}", "")
        (tmp / "prep.sql").write_text(sql)
        (tmp / "prep.json").write_text(json.dumps(summary, indent=2, default=str))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if folder.exists():
        # an incomplete leftover from a crash; a complete one was reused above
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    return Prepared(folder, sql, summary, reused=False)


def _is_complete(spec: PrepSpec, folder: Path) -> bool:
    if not (folder / "prep.json").exists():
        return False
    return all((folder / name).exists() for name in FILES[spec.mode])


def _load_cached(folder: Path) -> Prepared:
    sql = (folder / "prep.sql").read_text()
    summary = json.loads((folder / "prep.json").read_text())
    return Prepared(folder, sql, summary, reused=True)


def _load_source(spec: PrepSpec, source_path: Path, source_bytes: bytes, out: Path) -> str:
    """the duckdb table-function call to read the source, decoding to utf-8 first if needed"""
    read_ref = str(source_path.resolve())
    encoding = resolve_encoding(source_bytes, spec.encoding)
    if encoding.lower() not in ("utf-8", "utf8"):
        # python's own codec, not duckdb's narrower non-utf8 csv support - matches cp1252 exactly
        text = source_bytes.decode(encoding)
        copy = out / f"source_utf8{source_path.suffix}"
        copy.write_text(text, encoding="utf-8")
        read_ref = str(copy)
    if source_path.suffix.lower() == ".json":
        return f"read_json_auto({_literal(read_ref)})"
    return f"read_csv({_literal(read_ref)}, header = true)"


# ---------------------------------------------------------------------------
# row mode
# ---------------------------------------------------------------------------


def _prep_row(con, spec: PrepSpec, read_expr: str, out: Path) -> tuple[str, dict]:
    dims = spec.named("dimension")
    measures = spec.named("measure")
    target = spec.named("target")[0]
    anchor = spec.named("anchor")[0]
    where = spec.where or "TRUE"
    predict_where = spec.predict_where or "TRUE"
    target_q, anchor_q, row_q = _quote(target), _quote(ANCHOR), _quote(ROW)

    stmts = []

    def _run(stmt: str) -> None:
        stmts.append(stmt)
        con.execute(stmt)

    _run(
        f"CREATE OR REPLACE TEMP VIEW source AS\n"
        f"SELECT *, row_number() OVER () - 1 AS {row_q}\n"
        f"FROM {read_expr};"
    )
    # the predict filter is written against the source file's own columns, most of which never
    # reach the typed view, so it is evaluated here and carried along as a flag
    _run(
        "CREATE OR REPLACE TEMP VIEW filtered AS\n"
        f"SELECT *, ({predict_where}) AS {_quote(PREDICT)} FROM source WHERE {where};"
    )
    _run(
        f"CREATE OR REPLACE TEMP VIEW anchored AS\n"
        f"SELECT *, CAST({_quote(anchor)} AS DATE) AS {anchor_q} FROM filtered;"
    )
    _run(
        f"CREATE OR REPLACE TEMP VIEW valid AS\nSELECT * FROM anchored WHERE {anchor_q} IS NOT NULL;"
    )

    target_type = "VARCHAR" if spec.task == "classification" else "DOUBLE"
    select_cols = [row_q, anchor_q, _quote(PREDICT)]
    select_cols += [f"CAST({_quote(c)} AS VARCHAR) AS {_quote(c)}" for c in dims]
    select_cols += [f"CAST({_quote(c)} AS DOUBLE) AS {_quote(c)}" for c in measures]
    select_cols.append(f"CAST({target_q} AS {target_type}) AS {target_q}")
    _run(f"CREATE OR REPLACE TEMP VIEW typed AS\nSELECT {', '.join(select_cols)} FROM valid;")

    as_of_value = None
    if spec.task == "classification":
        lower_sql = upper_sql = "CAST(NULL AS DOUBLE)"
    elif spec.censor:
        if spec.censor.as_of:
            as_of_sql = f"CAST({_literal(spec.censor.as_of)} AS DATE)"
            as_of_value = spec.censor.as_of
        else:
            as_of_sql = f"(SELECT MAX({anchor_q}) FROM typed)"
            as_of_value = con.execute(f"SELECT MAX({anchor_q}) FROM typed").fetchone()[0]
        unit_days = UNIT_DAYS[spec.censor.unit]
        lower_sql = (
            f"CASE WHEN {target_q} IS NOT NULL THEN {target_q} "
            f"ELSE date_diff('day', {anchor_q}, {as_of_sql}) / {unit_days} END"
        )
        upper_sql = (
            f"CASE WHEN {target_q} IS NOT NULL THEN {target_q} ELSE CAST('Infinity' AS DOUBLE) END"
        )
    else:
        lower_sql = upper_sql = target_q

    _run(
        "CREATE OR REPLACE TEMP VIEW bounds AS\n"
        f"SELECT typed.*, {lower_sql} AS {_quote(LOWER)}, {upper_sql} AS {_quote(UPPER)}\n"
        "FROM typed;"
    )

    rows_path, predict_path = str(out / "rows.parquet"), str(out / "predict.parquet")
    keep = f"* EXCLUDE ({_quote(PREDICT)})"
    _run(f"COPY (SELECT {keep} FROM bounds) TO {_literal(rows_path)} (FORMAT PARQUET);")
    _run(
        "COPY (\n"
        f"    SELECT {keep} FROM bounds\n"
        f"    WHERE {target_q} IS NULL AND {_quote(PREDICT)}\n"
        f") TO {_literal(predict_path)} (FORMAT PARQUET);"
    )

    source_rows = con.execute("SELECT count(*) FROM source").fetchone()[0]
    filtered_rows = con.execute("SELECT count(*) FROM filtered").fetchone()[0]
    valid_rows = con.execute("SELECT count(*) FROM valid").fetchone()[0]
    open_rows = con.execute(f"SELECT count(*) FROM bounds WHERE {target_q} IS NULL").fetchone()[0]
    predict_rows = con.execute(
        f"SELECT count(*) FROM read_parquet({_literal(predict_path)})"
    ).fetchone()[0]

    summary = {
        "mode": "row",
        "task": spec.task,
        "source_rows": source_rows,
        "rows_after_where": filtered_rows,
        "dropped_null_anchor": filtered_rows - valid_rows,
        "rows": valid_rows,
        "open_rows": open_rows,
        "known_rows": valid_rows - open_rows,
        "predict_rows": predict_rows,
        "as_of": str(as_of_value)[:10] if as_of_value is not None else None,
    }
    return "\n\n".join(stmts), summary


# ---------------------------------------------------------------------------
# series mode
# ---------------------------------------------------------------------------


def _infer_step(con, read_expr: str, time_col: str, where: str) -> str:
    """the step whose interval best matches the median gap between distinct time values"""
    query = f"""
        SELECT median(gap) FROM (
            SELECT date_diff('day', lag(t) OVER (ORDER BY t), t) AS gap
            FROM (
                SELECT DISTINCT CAST({_quote(time_col)} AS DATE) AS t
                FROM {read_expr}
                WHERE {where}
            ) d
        ) g
        WHERE gap IS NOT NULL
    """
    median_gap = con.execute(query).fetchone()[0]
    if median_gap is None or median_gap <= 1.5:
        return "day"
    if median_gap <= 8:
        return "week"
    if median_gap <= 35:
        return "month"
    if median_gap <= 100:
        return "quarter"
    return "year"


def _series_expr(dims: list[str]) -> str:
    if not dims:
        return "'all'"
    return f"concat_ws('|', {_cols(dims)})"


def _agg_expr(col: str, aggregation: str, time_col_q: str) -> str:
    q = _quote(col)
    return {
        "sum": f"SUM({q})",
        "mean": f"AVG({q})",
        "min": f"MIN({q})",
        "max": f"MAX({q})",
        "count": f"COUNT({q})",
        "last": f"arg_max({q}, {time_col_q})",
    }[aggregation]


def _fill_expr(col: str, aggregation: str) -> str:
    q = _quote(col)
    if aggregation in ("sum", "count"):
        return f"COALESCE({q}, 0)"
    if aggregation == "last":
        return (
            f"LAST_VALUE({q} IGNORE NULLS) OVER ("
            f"PARTITION BY {_quote(SERIES)} ORDER BY {_quote(STEP)} "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
        )
    return q  # mean/min/max: a gap stays null, nothing to fill


def _prep_series(con, spec: PrepSpec, read_expr: str, out: Path) -> tuple[str, dict]:
    dims = spec.named("dimension")
    time_col = spec.named("time")[0]
    col_by_name = {c.name: c for c in spec.columns}
    measure_names = spec.named("measure") + spec.named("target")
    time_q = "__time_date"
    where = spec.where or "TRUE"
    step = spec.step or _infer_step(con, read_expr, time_col, where)
    unit = STEP_UNIT[step]
    dims_sel = _cols(dims)
    step_q, series_q = _quote(STEP), _quote(SERIES)
    # the step holding the export date is still running, so history ends before it
    history_sql = (
        f"{time_q} < date_trunc({_literal(step)}, CAST({_literal(spec.as_of)} AS DATE))"
        if spec.as_of
        else "TRUE"
    )

    stmts = [
        f"CREATE OR REPLACE TEMP VIEW filtered AS\nSELECT * FROM {read_expr} WHERE {where};",
        "CREATE OR REPLACE TEMP VIEW timed AS\n"
        f"SELECT *, CAST({_quote(time_col)} AS DATE) AS {time_q} FROM filtered;",
        f"CREATE OR REPLACE TEMP VIEW valid AS\nSELECT * FROM timed WHERE {time_q} IS NOT NULL;",
        f"CREATE OR REPLACE TEMP VIEW history AS\nSELECT * FROM valid WHERE {history_sql};",
        "CREATE OR REPLACE TEMP VIEW stepped AS\n"
        f"SELECT *, CAST(date_trunc({_literal(step)}, {time_q}) AS DATE) AS {step_q} FROM history;",
    ]

    typed_cols = [f"CAST({_quote(c)} AS VARCHAR) AS {_quote(c)}" for c in dims]
    typed_cols += [f"CAST({_quote(c)} AS DOUBLE) AS {_quote(c)}" for c in measure_names]
    stmts.append(
        "CREATE OR REPLACE TEMP VIEW typed AS\n"
        f"SELECT {step_q}, {time_q}, {', '.join(typed_cols)} FROM stepped;"
    )

    agg_cols = [
        f"{_agg_expr(c, col_by_name[c].aggregation, time_q)} AS {_quote(c)}" for c in measure_names
    ]
    group_by = f"{step_q}" + (f", {dims_sel}" if dims else "")
    stmts.append(
        "CREATE OR REPLACE TEMP VIEW grouped AS\n"
        f"SELECT {step_q}" + (f", {dims_sel}" if dims else "") + f", {', '.join(agg_cols)}\n"
        f"FROM typed\nGROUP BY {group_by};"
    )
    stmts.append(
        "CREATE OR REPLACE TEMP VIEW keyed AS\n"
        f"SELECT *, {_series_expr(dims)} AS {series_q} FROM grouped;"
    )

    dims_and_series = f"{series_q}" + (f", {dims_sel}" if dims else "")
    stmts.append(
        "CREATE OR REPLACE TEMP VIEW series_span AS\n"
        f"SELECT {dims_and_series}, MIN({step_q}) AS start_step, MAX({step_q}) AS end_step\n"
        f"FROM keyed\nGROUP BY {dims_and_series};"
    )
    stmts.append(
        "CREATE OR REPLACE TEMP VIEW series_grid AS\n"
        f"SELECT {dims_and_series}, "
        f"UNNEST(generate_series(start_step, end_step, INTERVAL 1 {unit})) AS {step_q}\n"
        "FROM series_span;"
    )

    grid_dims = ", ".join(f"g.{_quote(c)}" for c in dims)
    fill_cols = [
        f"{_fill_expr(c, col_by_name[c].aggregation)} AS {_quote(c)}" for c in measure_names
    ]
    stmts.append(
        "CREATE OR REPLACE TEMP VIEW filled AS\n"
        f"SELECT g.{step_q}, g.{series_q}"
        + (f", {grid_dims}" if dims else "")
        + ", "
        + ", ".join(f"k.{_quote(c)}" for c in measure_names)
        + "\n"
        "FROM series_grid g\n"
        f"LEFT JOIN keyed k USING ({series_q}, {step_q});"
    )
    stmts.append(
        "CREATE OR REPLACE TEMP VIEW panel AS\n"
        f"SELECT {step_q}, {series_q}"
        + (f", {dims_sel}" if dims else "")
        + f", {', '.join(fill_cols)}\n"
        "FROM filled;"
    )

    panel_path = str(out / "panel.parquet")
    stmts.append(f"COPY (SELECT * FROM panel) TO {_literal(panel_path)} (FORMAT PARQUET);")

    stmts.append(
        "CREATE OR REPLACE TEMP VIEW dim_combos AS\n"
        f"SELECT DISTINCT {dims_sel if dims else '1 AS __dummy'} FROM typed;"
    )
    stmts.append(
        "CREATE OR REPLACE TEMP VIEW future_steps AS\n"
        "SELECT UNNEST(generate_series(\n"
        f"    (SELECT MAX({step_q}) FROM panel) + INTERVAL 1 {unit},\n"
        f"    (SELECT MAX({step_q}) FROM panel) + INTERVAL 1 {unit} * {spec.horizon},\n"
        f"    INTERVAL 1 {unit}\n"
        f")) AS {step_q};"
    )
    scaffold_dims = ", ".join(f"d.{_quote(c)}" for c in dims) if dims else ""
    stmts.append(
        "CREATE OR REPLACE TEMP VIEW scaffold AS\n"
        f"SELECT f.{step_q}, {_series_expr(dims)} AS {series_q}"
        + (f", {scaffold_dims}" if dims else "")
        + "\nFROM future_steps f\nCROSS JOIN dim_combos d;"
    )
    scaffold_path = str(out / "scaffold.parquet")
    stmts.append(f"COPY (SELECT * FROM scaffold) TO {_literal(scaffold_path)} (FORMAT PARQUET);")

    for stmt in stmts:
        con.execute(stmt)

    source_rows = con.execute("SELECT count(*) FROM filtered").fetchone()[0]
    valid_rows = con.execute("SELECT count(*) FROM valid").fetchone()[0]
    history_rows = con.execute("SELECT count(*) FROM history").fetchone()[0]
    series_count = con.execute("SELECT count(*) FROM series_span").fetchone()[0]
    panel_rows = con.execute("SELECT count(*) FROM panel").fetchone()[0]
    scaffold_rows = con.execute("SELECT count(*) FROM scaffold").fetchone()[0]
    global_last = con.execute(f"SELECT MAX({step_q}) FROM panel").fetchone()[0]

    summary = {
        "mode": "series",
        "step": step,
        "source_rows": source_rows,
        "dropped_null_time": source_rows - valid_rows,
        "dropped_after_as_of": valid_rows - history_rows,
        "series": series_count,
        "panel_rows": panel_rows,
        "scaffold_rows": scaffold_rows,
        "horizon": spec.horizon,
        "as_of": str(global_last)[:10] if global_last is not None else None,
    }
    return "\n\n".join(stmts), summary
