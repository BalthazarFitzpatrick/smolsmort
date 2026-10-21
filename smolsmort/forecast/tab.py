"""the forecast tab's http surface. thin on purpose: a handler parses its request, calls one
function and returns what comes back. the aggregation lives in views.py, the search and the fit in
runs.py/prep.py, which spawn the worker process - nothing here imports torch or runs xgboost."""

from __future__ import annotations

import json
import math
from pathlib import Path

from smolsmort.forecast import runs, views
from smolsmort.forecast.families import get_family
from smolsmort.forecast.prep import Prepared, prepare
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.runs import RunError
from smolsmort.forecast.spec import SpecError, normalize_mode, normalize_saved_modes, spec_from_dict
from smolsmort.forecast.tables import connect, resolve_encoding, type_source
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
    if not isinstance(payload.get("spec"), dict):
        raise RequestError("spec must be an object")
    data = dict(payload["spec"])
    if resolve_source:
        data["source"] = str(_resolve_source(app, data.get("source", "")))
    return data


def _parse_spec(app, payload):
    try:
        data = _spec_from_payload(app, payload)
        data["mode"] = normalize_mode(data.get("mode"))
        if data.get("task") not in ("regression", "classification"):
            raise SpecError("spec.task must be regression or classification")
        if data["mode"] == "timeseries" and data["task"] != "regression":
            raise SpecError("time series runs require regression")
        if data.get("step") not in (None, "day", "week", "month", "quarter", "year"):
            raise SpecError("spec.step must be day, week, month, quarter or year")
        if type(data.get("horizon", 8)) is not int:
            raise SpecError("spec.horizon must be a positive integer")
        if not isinstance(data.get("columns"), (list, tuple)):
            raise SpecError("spec.columns must be an array")
        for column in data["columns"]:
            if (
                not isinstance(column, dict)
                or not isinstance(column.get("name"), str)
                or not column["name"]
            ):
                raise SpecError("each spec column needs a nonempty string name")
            if column.get("role") not in (
                "time",
                "anchor",
                "dimension",
                "input",
                "measure",
                "target",
                "ignore",
            ):
                raise SpecError(f"unknown column role {column.get('role')!r}")
            if column.get("aggregation") not in (
                None,
                "sum",
                "mean",
                "median",
                "min",
                "max",
                "count",
                "last",
            ):
                raise SpecError(f"unknown aggregation for column {column['name']!r}")
            if "known" in column and type(column["known"]) is not bool:
                raise SpecError("column.known must be a boolean")
        return spec_from_dict(data)
    except (SpecError, TypeError, ValueError) as exc:
        raise RequestError(str(exc)) from exc


def _prepare_spec(app, spec):
    try:
        import duckdb
    except ImportError as exc:
        raise RequestError("preparation needs duckdb; uv sync --extra forecast") from exc
    try:
        return prepare(spec, _cache_root(app))
    except (duckdb.Error, ValueError, OSError) as exc:
        raise RequestError(f"could not prepare spec: {exc}") from exc


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
    data = source.read_bytes()
    encoding = resolve_encoding(data, payload.get("encoding"))
    con = connect()
    if encoding.lower() not in ("utf-8", "utf8"):
        text = data.decode(encoding)
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
    read_expr = type_source(con, read_expr)
    described = con.execute(f"DESCRIBE SELECT * FROM {read_expr}").fetchall()
    row_count = con.execute(f"SELECT count(*) FROM {read_expr}").fetchone()[0]
    mode = normalize_mode(payload.get("mode") or "timeseries")
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
            suggested = "anchor" if mode == "entity" else "time"
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
    return {"columns": out, "encoding": encoding, "row_count": row_count}


def _prep(app, payload: dict) -> dict:
    spec = _parse_spec(app, payload)
    prepared = _prepare_spec(app, spec)
    views.save_prepared_spec(prepared, spec)
    return {"summary": prepared.summary, "sql": prepared.sql, "reused": prepared.reused}


# ---------------------------------------------------------------- runs


def _families(app, query: dict) -> dict:
    spec, prepared = None, None
    if "spec" in query:
        try:
            data = json.loads(query["spec"]) if isinstance(query["spec"], str) else query["spec"]
        except json.JSONDecodeError as exc:
            raise RequestError("spec must be valid json") from exc
        spec = _parse_spec(app, {"spec": data})
        prepared = _prepare_spec(app, spec)
        views.save_prepared_spec(prepared, spec)
    elif "prepared" in query:
        root = _cache_root(app).resolve()
        folder = (root / str(query["prepared"])).resolve()
        if root not in folder.parents:
            raise RequestError("prepared folder is outside the forecast cache")
        try:
            spec = views.read_prepared_spec(folder, _runs_root(app))
            summary = normalize_saved_modes(json.loads((folder / "prep.json").read_text()))
        except (OSError, ValueError, TypeError) as exc:
            raise RequestError("prepared folder has no saved spec; provide spec") from exc
        prepared = Prepared(folder, "", summary, reused=True)
    task = query.get("task", "regression")
    if task not in ("regression", "classification"):
        raise RequestError("task must be regression or classification")
    try:
        return {"families": views.family_catalog(_runs_root(app), spec, prepared, task)}
    except views.ViewError as exc:
        raise RequestError(str(exc), 409) from exc


def _validate_selections(payload, spec):
    for key in ("time_budget_s", "candidate_limit_fraction"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise RequestError(f"{key} must be finite")
    selections = payload.get("families")
    if not isinstance(selections, list) or not selections:
        raise RequestError("select at least one family")
    for selection in selections:
        if not isinstance(selection, dict) or not isinstance(selection.get("name"), str):
            raise RequestError("each family must be an object with a name")
        name = selection["name"]
        try:
            family = get_family(name)
        except KeyError as exc:
            raise RequestError(f"unknown family {name!r}") from exc
        if spec.mode == "entity" and name != "xgboost":
            raise RequestError("entity runs support xgboost only")
        try:
            objectives = family.objectives(spec.task)
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
        if not objectives:
            raise RequestError(f"family {name!r} does not support {spec.task}")
        space = selection.get("space", {})
        if not isinstance(space, dict):
            raise RequestError("family space must be an object")
        params = {param.name: param for param in family.space()}
        for key, values in space.items():
            if key not in params or not isinstance(values, list) or not values:
                raise RequestError(f"invalid space override {name}.{key}; use a nonempty array")
            param = params[key]
            for value in values:
                if param.kind == "choice":
                    valid = value in param.choices
                else:
                    valid = (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and param.lo <= value <= param.hi
                        and math.isfinite(value)
                    )
                    if param.kind == "int":
                        valid = valid and type(value) is int
                if not valid:
                    raise RequestError(f"invalid value {value!r} for {name}.{key}")
    try:
        parsed = parse_request({**payload, "version": payload.get("version", 2)})
    except (ValueError, OverflowError) as exc:
        raise RequestError(str(exc)) from exc
    try:
        availability = views.probe_families()
    except views.ViewError as exc:
        raise RequestError(str(exc), 409) from exc
    for selection in selections:
        name = selection["name"]
        available, install = availability[name]
        if not available:
            raise RequestError(f"family {name!r} is unavailable; {install}")
    return parsed


def _start(app, payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise RequestError("forecast request must be an object")
    multi = "version" in payload or any(
        key in payload
        for key in (
            "families",
            "time_budget_s",
            "transform",
            "max_concurrent_families",
            "candidate_limit_fraction",
            "prune_sigma",
            "ensemble",
        )
    )
    if multi and payload.get("version", 2) != 2:
        raise RequestError("unsupported forecast request version")
    if multi and "budget" in payload:
        raise RequestError("v2 uses time_budget_s; budget belongs to v1")
    warm = payload.get("warm_from")
    if warm is not None and not isinstance(warm, str):
        raise RequestError("warm_from must be a run id")
    if warm and (multi or (_runs_root(app) / str(warm) / "request.json").is_file()):
        source = _run_dir(app, str(warm))
        previous = views.read_request(source)
        if previous.get("version") == 2:
            try:
                winner = views.winning_genome(source)
            except views.ViewError as exc:
                raise RequestError(str(exc)) from exc
        if previous.get("version") == 2 and not multi:
            budget = payload.get("budget") or {}
            if not isinstance(budget, dict):
                raise RequestError("budget must be an object")
            payload = {
                **payload,
                "families": [{"name": winner["family"]}],
                "time_budget_s": budget.get("time_cap", previous["time_budget_s"]),
                "transform": previous.get("transform", "none"),
                "nthread": budget.get("nthread", previous.get("nthread")),
            }
            multi = True
    if multi:
        spec = _parse_spec(app, payload)
        payload = _validate_selections(payload, spec)
    else:
        try:
            payload = parse_request(payload)
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
        spec = None
    try:
        spec = spec or spec_from_dict(_spec_from_payload(app, payload))
    except SpecError as exc:
        raise RequestError(str(exc)) from exc
    prepared = _prepare_spec(app, spec) if multi else prepare(spec, _cache_root(app))
    if multi:
        views.save_prepared_spec(prepared, spec)
        options = {
            key: payload[key]
            for key in (
                "families",
                "time_budget_s",
                "transform",
                "aggregate",
                "max_concurrent_families",
                "candidate_limit_fraction",
                "prune_sigma",
                "ensemble",
                "nthread",
            )
            if key in payload
        }
        try:
            run_id = runs.start_run(
                _runs_root(app), spec, prepared, version=2, warm_from=warm, **options
            )
        except (TypeError, ValueError) as exc:
            raise RequestError(str(exc)) from exc
    else:
        run_id = runs.start_run(
            _runs_root(app),
            spec,
            prepared,
            budget=payload.get("budget"),
            warm_from=payload.get("warm_from"),
            aggregate=payload.get("aggregate", "none"),
        )
    return {
        "run_id": run_id,
        "prepared": {"summary": prepared.summary, "sql": prepared.sql, "reused": prepared.reused},
    }


def _refit(app, payload: dict) -> dict:
    source_run = _run_dir(app, str(payload.get("run_id", "")))
    recipe_path = source_run / "recipe.json"
    if not recipe_path.exists():
        raise RequestError(f"run {payload.get('run_id')!r} has no recipe to refit", 404)
    genome = json.loads(recipe_path.read_text())["genome"]
    request_path = source_run / "request.json"
    previous = views.read_request(source_run) if request_path.exists() else {}
    try:
        spec = (
            _parse_spec(app, payload)
            if previous.get("version") == 2
            else spec_from_dict(_spec_from_payload(app, payload))
        )
    except SpecError as exc:
        raise RequestError(str(exc)) from exc
    prepared = (
        _prepare_spec(app, spec)
        if previous.get("version") == 2
        else prepare(spec, _cache_root(app))
    )
    if previous.get("version") == 2:
        try:
            genome = views.winning_genome(source_run)
        except views.ViewError as exc:
            raise RequestError(str(exc)) from exc
        _validate_selections(
            {
                "families": [{"name": genome["family"]}],
                "time_budget_s": previous["time_budget_s"],
                "transform": previous.get("transform", "none"),
            },
            spec,
        )
        run_id = runs.start_run(
            _runs_root(app),
            spec,
            prepared,
            recipe=genome,
            version=2,
            families=[{"name": genome["family"]}],
            time_budget_s=previous["time_budget_s"],
            transform=previous.get("transform", "none"),
            aggregate=previous.get("aggregate", "none"),
            nthread=previous.get("nthread"),
            ensemble={"enabled": False},
        )
    else:
        run_id = runs.start_run(
            _runs_root(app),
            spec,
            prepared,
            recipe=genome,
            aggregate=previous.get("aggregate", "none"),
        )
    return {
        "run_id": run_id,
        "prepared": {"summary": prepared.summary, "sql": prepared.sql, "reused": prepared.reused},
    }


def _list_runs(app, query: dict) -> dict:
    return {"runs": runs.list_runs(_runs_root(app))}


def _run(app, query: dict) -> dict:
    run_id = query.get("id", "")
    run_dir = _run_dir(app, run_id)
    cursor = None
    if "since_generation" in query:
        try:
            cursor = max(-1, int(query["since_generation"]))
        except (TypeError, ValueError, OverflowError):
            cursor = -1
    event_cursor = None
    if "since_events" in query:
        try:
            event_cursor = max(-1, int(query["since_events"]))
        except (TypeError, ValueError, OverflowError):
            event_cursor = -1
    state = runs.run_state(
        _runs_root(app), run_id, since_generation=cursor, since_events=event_cursor
    )
    request = views.read_request(run_dir)
    board_cursor = -1
    if "since_leaderboard" in query:
        try:
            board_cursor = max(-1, int(query["since_leaderboard"]))
        except (TypeError, ValueError, OverflowError):
            board_cursor = -1
    if request.get("version") == 2:
        if "since_leaderboard" in query and state["leaderboard_generation"] <= board_cursor:
            state.pop("leaderboard", None)
        return {**state, "version": 2, "spec": request.get("spec")}
    for collection in ("generations", "generation_curves"):
        state[collection] = [
            {**entry, "family": entry.get("family") or "xgboost"}
            for entry in state.get(collection, [])
        ]
    if state.get("latest"):
        state["latest"] = {**state["latest"], "family": state["latest"].get("family") or "xgboost"}
    return {
        **state,
        "version": 1,
        "family": "xgboost",
        "winner_family": "xgboost" if state["state"] == "done" else None,
        **(
            {"leaderboard": views.read_leaderboard(run_dir)[:10]}
            if "since_leaderboard" not in query or state["leaderboard_generation"] > board_cursor
            else {}
        ),
        "spec": request.get("spec"),
    }


def _cancel(app, payload: dict) -> dict:
    run_id = str(payload.get("id", ""))
    _run_dir(app, run_id)  # checked containment before touching the process
    try:
        runs.cancel_run(_runs_root(app), run_id)
    except RunError as exc:
        raise RequestError(str(exc), 409) from exc
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
    try:
        return views.table(run_dir, request, query)
    except views.ViewError as exc:
        raise RequestError(str(exc), 409) from exc


def _export(app, name: str) -> bytes | None:
    if not name.endswith(".csv"):
        return None
    run_id = name[: -len(".csv")]
    query = {}
    if "/" in run_id:
        run_id, model = run_id.rsplit("/", 1)
        query["model"] = model
    try:
        run_dir = runs.run_folder(_runs_root(app), run_id)
    except RunError:
        return None
    request = views.read_request(run_dir)
    try:
        return views.export_csv(run_dir, request, query)
    except views.ViewError:
        return None


def forecast_tab() -> Tab:
    return Tab(
        name="forecast",
        get={
            "/api/forecast-families": _families,
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
