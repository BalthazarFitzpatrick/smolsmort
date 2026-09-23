"""the forecast tab's http surface. thin on purpose: a handler parses its request, calls one
function and returns what comes back. the aggregation lives in views.py, the search and the fit in
runs.py/prep.py, which spawn the worker process - nothing here imports torch or runs xgboost."""

from __future__ import annotations

import json
from pathlib import Path

from smolsmort.forecast import runs, views
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.runs import RunError
from smolsmort.forecast.spec import SpecError, spec_from_dict
from smolsmort.forecast.tables import connect
from smolsmort.review.routes import RequestError, Tab

SOURCE_SUFFIXES = (".csv", ".json")


def _root(app) -> Path:
    return Path(app.state.bases["forecast"]).resolve()


def _cache_root(app) -> Path:
    return _root(app) / ".forecast-cache"


def _runs_root(app) -> Path:
    return _root(app) / ".forecast-runs"


def _resolve_source(app, source: str) -> Path:
    """a source path from a payload, confined to the forecast root - resolve then check, since
    string-matching on '..' misses a symlink or an absolute path"""
    root = _root(app)
    target = (root / source).resolve()
    if root != target and root not in target.parents:
        raise RequestError(f"source {source!r} is outside the forecast root", 400)
    if not target.is_file():
        raise RequestError(f"no such source: {source}", 404)
    return target


def _run_dir(app, run_id: str) -> Path:
    try:
        return runs.run_folder(_runs_root(app), run_id)
    except RunError as exc:
        raise RequestError(str(exc), 404) from exc


def _spec_from_payload(app, payload: dict, *, resolve_source: bool = True) -> dict:
    """the spec dict, with its source resolved (and checked) to an absolute path"""
    data = dict(payload.get("spec") or {})
    if resolve_source:
        data["source"] = str(_resolve_source(app, data.get("source", "")))
    return data


# ---------------------------------------------------------------- data screen


def _sources(app, query: dict) -> dict:
    root = _root(app)
    files = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            if ".forecast-cache" in path.parts or ".forecast-runs" in path.parts:
                continue
            stat = path.stat()
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
    return {"root": str(root), "files": files}


_KIND_BY_DUCKDB = {
    "DATE": "date",
    "TIMESTAMP": "date",
    "VARCHAR": "text",
    "BOOLEAN": "text",
}


def _column_kind(duckdb_type: str) -> str:
    upper = duckdb_type.upper()
    if upper in _KIND_BY_DUCKDB:
        return _KIND_BY_DUCKDB[upper]
    if any(token in upper for token in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC")):
        return "number"
    return "text"


def _columns(app, payload: dict) -> dict:
    source = _resolve_source(app, payload.get("source", ""))
    encoding = payload.get("encoding") or "utf-8"
    con = connect()
    if encoding.lower() not in ("utf-8", "utf8"):
        text = source.read_bytes().decode(encoding)
        tmp = source.with_suffix(f".utf8{source.suffix}")
        tmp.write_text(text, encoding="utf-8")
        read_expr = (
            f"read_csv('{tmp}', header = true)"
            if source.suffix.lower() != ".json"
            else (f"read_json_auto('{tmp}')")
        )
    else:
        read_expr = (
            f"read_json_auto('{source}')"
            if source.suffix.lower() == ".json"
            else f"read_csv('{source}', header = true)"
        )
    described = con.execute(f"DESCRIBE SELECT * FROM {read_expr}").fetchall()
    mode = payload.get("mode") or "series"
    out = []
    for name, duckdb_type, *_rest in described:
        kind = _column_kind(duckdb_type)
        q = '"' + name.replace('"', '""') + '"'
        distinct, missing = con.execute(
            f"SELECT count(DISTINCT {q}), count(*) - count({q}) FROM {read_expr}"
        ).fetchone()
        sample = [
            row[0]
            for row in con.execute(
                f"SELECT DISTINCT {q} FROM {read_expr} WHERE {q} IS NOT NULL LIMIT 5"
            ).fetchall()
        ]
        if kind == "date":
            suggested = "anchor" if mode == "row" else "time"
        elif kind == "text":
            suggested = "dimension"
        else:
            suggested = "measure"
        out.append(
            {
                "name": name,
                "kind": kind,
                "distinct": distinct,
                "missing": missing,
                "sample": [str(v) for v in sample],
                "suggested_role": suggested,
            }
        )
    return {"columns": out}


def _prep(app, payload: dict) -> dict:
    try:
        spec = spec_from_dict(_spec_from_payload(app, payload))
    except SpecError as exc:
        raise RequestError(str(exc)) from exc
    prepared = prepare(spec, _cache_root(app))
    return {"summary": prepared.summary, "sql": prepared.sql, "reused": prepared.reused}


# ---------------------------------------------------------------- runs


def _start(app, payload: dict) -> dict:
    try:
        spec = spec_from_dict(_spec_from_payload(app, payload))
    except SpecError as exc:
        raise RequestError(str(exc)) from exc
    prepared = prepare(spec, _cache_root(app))
    run_id = runs.start_run(
        _runs_root(app),
        spec,
        prepared,
        budget=payload.get("budget"),
        warm_from=payload.get("warm_from"),
    )
    return {"run_id": run_id}


def _refit(app, payload: dict) -> dict:
    source_run = _run_dir(app, str(payload.get("run_id", "")))
    recipe_path = source_run / "recipe.json"
    if not recipe_path.exists():
        raise RequestError(f"run {payload.get('run_id')!r} has no recipe to refit", 404)
    genome = json.loads(recipe_path.read_text())["genome"]
    try:
        spec = spec_from_dict(_spec_from_payload(app, payload))
    except SpecError as exc:
        raise RequestError(str(exc)) from exc
    prepared = prepare(spec, _cache_root(app))
    run_id = runs.start_run(_runs_root(app), spec, prepared, recipe=genome)
    return {"run_id": run_id}


def _list_runs(app, query: dict) -> dict:
    return {"runs": runs.list_runs(_runs_root(app))}


def _run(app, query: dict) -> dict:
    run_id = query.get("id", "")
    run_dir = _run_dir(app, run_id)
    state = runs.run_state(_runs_root(app), run_id)
    request = views.read_request(run_dir)
    return {
        **state,
        "leaderboard": views.read_leaderboard(run_dir),
        "spec": request.get("spec"),
    }


def _cancel(app, payload: dict) -> dict:
    run_id = str(payload.get("id", ""))
    _run_dir(app, run_id)  # checked containment before touching the process
    runs.cancel_run(_runs_root(app), run_id)
    return {"ok": True}


# ---------------------------------------------------------------- results


def _view(app, query: dict) -> dict:
    run_dir = _run_dir(app, query.get("id", ""))
    request = views.read_request(run_dir)
    try:
        return views.view(run_dir, request, query)
    except views.ViewError as exc:
        raise RequestError(str(exc), 409) from exc


def _table(app, query: dict) -> dict:
    run_dir = _run_dir(app, query.get("id", ""))
    request = views.read_request(run_dir)
    return views.table(run_dir, request, query)


def _export(app, name: str) -> bytes | None:
    if not name.endswith(".csv"):
        return None
    run_id = name[: -len(".csv")]
    try:
        run_dir = runs.run_folder(_runs_root(app), run_id)
    except RunError:
        return None
    request = views.read_request(run_dir)
    try:
        return views.export_csv(run_dir, request)
    except views.ViewError:
        return None


def forecast_tab() -> Tab:
    return Tab(
        name="forecast",
        get={
            "/api/forecast-sources": _sources,
            "/api/forecast-runs": _list_runs,
            "/api/forecast-run": _run,
            "/api/forecast-view": _view,
            "/api/forecast-table": _table,
        },
        post={
            "/api/forecast-columns": _columns,
            "/api/forecast-prep": _prep,
            "/api/forecast-runs": _start,
            "/api/forecast-refit": _refit,
            "/api/forecast-cancel": _cancel,
        },
        images={"/api/forecast-export/": (_export, "text/csv")},
    )
