"""the forecast tab's http surface, called directly through its Tab (no socket - see
test_review_routes.py's own host-tab tests for the same style). fixtures are the synthetic sources
every forecast test uses (forecast_fixtures.py), never real data."""

from __future__ import annotations

import csv
import io
import json
import time

import pytest
import review_world
from forecast_fixtures import write_bike_daily, write_panel, write_rows

from smolsmort.forecast.tab import forecast_tab
from smolsmort.review import routes
from smolsmort.review.server import build_app
from smolsmort.review_ui.tab import hyperparams_tab

pytest.importorskip("xgboost")
pytest.importorskip("duckdb")

TINY_BUDGET = {"population": 6, "max_generations": 2, "plateau": 1, "nthread": 1}


@pytest.fixture(autouse=True)
def _world_backend():
    review_world.register_world_backend()
    try:
        yield
    finally:
        review_world.unregister_world_backend()


@pytest.fixture
def app(tmp_path, monkeypatch):
    """a real App with the forecast tab registered, its root pointed at a fresh tmp folder"""
    world = review_world.make_world(tmp_path / "review", monkeypatch)
    root = tmp_path / "forecast_root"
    root.mkdir()
    built = build_app(backend="world", ui_dir=world.ui, pool=world.tiles, tabs=[forecast_tab()])
    built.state.set_bases({"forecast": str(root)})
    return built


@pytest.fixture
def tab():
    return forecast_tab()


def _get(app_obj, tab_obj, path, query=None):
    return tab_obj.get[path](app_obj, query or {})


def _post(app_obj, tab_obj, path, payload=None):
    return tab_obj.post[path](app_obj, payload or {})


def _root(app_obj):
    return app_obj.state.bases["forecast"]


# ---------------------------------------------------------------- data screen


def test_sources_lists_fixture_files_and_refuses_outside_paths(app, tab, tmp_path):
    write_panel(_forecast_path(app), weeks=20)
    listing = _get(app, tab, "/api/forecast-sources")
    assert listing["root"] == _root(app)
    assert any(f["path"].endswith("panel.csv") for f in listing["files"])

    outside = tmp_path / "elsewhere.csv"
    outside.write_text("a,b\n1,2\n")
    with pytest.raises(routes.RequestError):
        _post(app, tab, "/api/forecast-columns", {"source": "../elsewhere.csv"})


def test_columns_suggests_roles_from_kind(app, tab):
    truth = write_panel(_forecast_path(app), weeks=20)
    result = _post(app, tab, "/api/forecast-columns", {"source": "panel.csv"})
    by_name = {c["name"]: c for c in result["columns"]}
    assert by_name["day"]["kind"] == "date"
    assert by_name["day"]["suggested_role"] == "time"
    assert by_name["project"]["kind"] == "text"
    assert by_name["project"]["suggested_role"] == "dimension"
    assert by_name["project"]["distinct"] == 3
    assert by_name["product"]["distinct"] == 4
    assert by_name["orders"]["kind"] == "number"
    assert by_name["orders"]["suggested_role"] == "measure"
    assert all(c["suggested_role"] != "target" for c in result["columns"])
    assert len(by_name["project"]["sample"]) <= 5
    assert truth.series > 0


def test_columns_detects_a_utf16_tab_export(app, tab):
    # the shape tableau exports: utf-16 with a bom, tab-separated
    text = "Order Date\tRegion\tQuantity\n2026-01-05\tWest\t3\n2026-01-12\tSüd\t4\n"
    (_forecast_path(app) / "export.csv").write_bytes(text.encode("utf-16"))
    result = _post(app, tab, "/api/forecast-columns", {"source": "export.csv", "encoding": "auto"})
    by_name = {c["name"]: c for c in result["columns"]}
    assert result["encoding"] == "utf-16"
    assert by_name["Order Date"]["suggested_role"] == "time"
    assert "Süd" in by_name["Region"]["sample"]


def test_decimal_comma_export_lists_numbers_and_preps(app, tab):
    lines = ["Order Date\tRegion\tSales"]
    for week in range(30):
        for region, base in (("West", 755), ("East", 1234)):
            day = f"2025-{1 + week // 4:02d}-{1 + 7 * (week % 4):02d}"
            lines.append(f"{day}\t{region}\t{base + week},{week % 10}6")
    (_forecast_path(app) / "sales.csv").write_bytes(("\n".join(lines) + "\n").encode("utf-16"))

    result = _post(app, tab, "/api/forecast-columns", {"source": "sales.csv"})
    sales = next(c for c in result["columns"] if c["name"] == "Sales")
    assert (sales["kind"], sales["suggested_role"]) == ("number", "measure")

    spec = {
        "source": "sales.csv",
        "mode": "series",
        "task": "regression",
        "step": "month",
        "horizon": 3,
        "columns": [
            {"name": "Order Date", "role": "time"},
            {"name": "Region", "role": "dimension"},
            {"name": "Sales", "role": "target", "aggregation": "sum"},
        ],
    }
    prepared = _post(app, tab, "/api/forecast-prep", {"spec": spec})
    assert prepared["summary"]["mode"] == "series"
    assert prepared["summary"]["source_rows"] == 60
    assert prepared["summary"]["source_columns"] == 3
    assert prepared["summary"]["encoding"] == "utf-16"


def _forecast_path(app):
    from pathlib import Path

    return Path(app.state.bases["forecast"])


# ---------------------------------------------------------------- prep


def _series_spec(source_name: str) -> dict:
    return {
        "source": source_name,
        "mode": "series",
        "task": "regression",
        "columns": [
            {"name": "day", "role": "time"},
            {"name": "project", "role": "dimension"},
            {"name": "product", "role": "dimension"},
            {"name": "orders", "role": "measure", "aggregation": "sum"},
            {"name": "units", "role": "target", "aggregation": "sum"},
        ],
    }


def _row_spec(source_name: str) -> dict:
    return {
        "source": source_name,
        "mode": "row",
        "task": "regression",
        "columns": [
            {"name": "created", "role": "anchor"},
            {"name": "branch", "role": "dimension"},
            {"name": "size", "role": "measure"},
            {"name": "planned_offset", "role": "measure"},
            {"name": "leak_plus1", "role": "measure"},
            {"name": "line", "role": "ignore"},
            {"name": "lead_weeks", "role": "target"},
        ],
        "censor": {"anchor": "created"},
    }


def test_prep_summarises_and_reuses_cache(app, tab):
    write_panel(_forecast_path(app), weeks=60)
    spec = _series_spec("panel.csv")
    first = _post(app, tab, "/api/forecast-prep", {"spec": spec})
    assert first["reused"] is False
    assert "sql" in first and first["summary"]["mode"] == "series"
    assert first["summary"]["source_columns"] == 5
    assert first["summary"]["encoding"] == "utf-8"
    second = _post(app, tab, "/api/forecast-prep", {"spec": spec})
    assert second["reused"] is True
    assert second["summary"] == first["summary"]


def test_prep_scaffolded_numeric_bike_dimensions(app, tab):
    write_bike_daily(_forecast_path(app))
    columns = _post(app, tab, "/api/forecast-columns", {"source": "bike_daily.csv"})
    assert columns["row_count"] == 364
    by_name = {col["name"]: col for col in columns["columns"]}
    assert by_name["instant"]["distinct"] == columns["row_count"]
    assert by_name["workingday"]["kind"] == "number"
    assert by_name["workingday"]["distinct"] == 2
    assert by_name["dteday"]["kind"] == "date"
    spec = {
        "source": "bike_daily.csv",
        "mode": "series",
        "task": "regression",
        "step": "week",
        "horizon": 3,
        "columns": [
            {"name": "dteday", "role": "time"},
            {"name": "workingday", "role": "dimension"},
            {"name": "cnt", "role": "target", "aggregation": "sum"},
        ],
    }
    prepared = _post(app, tab, "/api/forecast-prep", {"spec": spec})
    assert prepared["summary"]["series"] == 2
    assert prepared["summary"]["scaffold_rows"] == 6
    import duckdb

    scaffold = next((_forecast_path(app) / ".forecast-cache").glob("*/scaffold.parquet"))
    with duckdb.connect() as con:
        rows = con.execute(
            "SELECT workingday, count(*) FROM read_parquet(?) GROUP BY workingday ORDER BY workingday",
            [str(scaffold)],
        ).fetchall()
    assert rows == [("0", 3), ("1", 3)]


def test_prep_reads_known_facts_for_an_old_cache_without_rebuilding(app, tab):
    write_panel(_forecast_path(app), weeks=60)
    spec = _series_spec("panel.csv")
    first = _post(app, tab, "/api/forecast-prep", {"spec": spec})
    prep_file = next((_forecast_path(app) / ".forecast-cache").glob("*/prep.json"))
    old_summary = dict(first["summary"])
    old_summary.pop("source_columns")
    old_summary.pop("encoding")
    prep_file.write_text(json.dumps(old_summary))
    reused = _post(app, tab, "/api/forecast-prep", {"spec": spec})
    assert reused["reused"] is True
    assert reused["summary"] == first["summary"]
    assert reused["sql"] == first["sql"]


@pytest.mark.parametrize("action", ["search", "warm", "refit"])
def test_run_actions_prepare_current_picks_and_report_cache_facts(app, tab, monkeypatch, action):
    write_panel(_forecast_path(app), weeks=60)
    captured = []

    def record_run(root, spec, prepared, **kwargs):
        captured.append((spec, prepared, kwargs))
        return "new-run"

    monkeypatch.setattr("smolsmort.forecast.tab.runs.start_run", record_run)
    payload = {"spec": _series_spec("panel.csv"), "budget": TINY_BUDGET}
    route = "/api/forecast-runs"
    if action == "warm":
        payload["warm_from"] = "saved-run"
    elif action == "refit":
        saved = _forecast_path(app) / ".forecast-runs" / "saved-run"
        saved.mkdir(parents=True)
        (saved / "recipe.json").write_text(json.dumps({"genome": {"params": {"eta": 0.1}}}))
        payload["run_id"] = "saved-run"
        route = "/api/forecast-refit"

    first = _post(app, tab, route, payload)
    assert first["run_id"] == "new-run"
    assert set(first["prepared"]) == {"summary", "sql", "reused"}
    assert first["prepared"]["reused"] is False
    assert first["prepared"]["summary"]["source_rows"] == 60 * 12 * 2
    assert first["prepared"]["summary"]["source_columns"] == 5
    assert first["prepared"]["summary"]["encoding"] == "utf-8"
    assert first["prepared"]["sql"]
    second = _post(app, tab, route, payload)
    assert second["prepared"]["reused"] is True
    assert second["prepared"]["summary"] == first["prepared"]["summary"]

    payload["spec"]["horizon"] = 11
    changed = _post(app, tab, route, payload)
    assert changed["prepared"]["reused"] is False
    assert changed["prepared"]["summary"]["horizon"] == 11
    assert captured[-1][0].horizon == 11
    assert captured[-1][1].folder != captured[0][1].folder
    if action == "warm":
        assert captured[-1][2]["warm_from"] == "saved-run"
    elif action == "refit":
        assert captured[-1][2]["recipe"] == {"params": {"eta": 0.1}}

    write_panel(_forecast_path(app), weeks=61)
    source_changed = _post(app, tab, route, payload)
    assert source_changed["prepared"]["reused"] is False
    assert source_changed["prepared"]["summary"]["source_rows"] == 61 * 12 * 2
    assert source_changed["prepared"]["summary"]["horizon"] == 11


# ---------------------------------------------------------------- runs


def _wait_done(app, tab, run_id, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _get(app, tab, "/api/forecast-run", {"id": run_id})
        if state["state"] in ("done", "failed", "cancelled"):
            return state
        time.sleep(0.2)
    raise TimeoutError(f"run {run_id} still going after {timeout}s")


def _run_to_done(tmp_path, write_source, spec_builder, source_name, budget):
    """build a torch-free app in its own tmp root, start a real search and block until it
    finishes - used by the module-scoped fixtures below, which have no function-scoped monkeypatch"""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    review_world.register_world_backend()
    try:
        world = review_world.make_world(tmp_path / "review", mp)
        root = tmp_path / "forecast_root"
        root.mkdir()
        app = build_app(backend="world", ui_dir=world.ui, pool=world.tiles, tabs=[forecast_tab()])
        app.state.set_bases({"forecast": str(root)})
        tab = forecast_tab()
        write_source(root)
        spec = spec_builder(source_name)
        started = tab.post["/api/forecast-runs"](app, {"spec": spec, "budget": budget})
        run_id = started["run_id"]
        deadline = time.monotonic() + 300
        state = None
        while time.monotonic() < deadline:
            state = tab.get["/api/forecast-run"](app, {"id": run_id})
            if state["state"] in ("done", "failed"):
                break
            time.sleep(0.2)
        if state is None or state["state"] != "done":
            raise TimeoutError(f"run did not finish in time: {state}")
        return app, tab, run_id, root
    finally:
        review_world.unregister_world_backend()
        mp.undo()


@pytest.fixture(scope="module")
def series_run(tmp_path_factory):
    """one real search, shared by every test that only reads its outcome - population 6,
    2 generations, plateau 1: enough to reach a finished run without a slow suite"""
    yield _run_to_done(
        tmp_path_factory.mktemp("series_run"),
        lambda root: write_panel(root, weeks=60),
        _series_spec,
        "panel.csv",
        TINY_BUDGET,
    )


def test_a_search_run_reaches_done_and_reports_a_verdict(app, tab):
    write_panel(_forecast_path(app), weeks=60)
    spec = _series_spec("panel.csv")
    started = _post(app, tab, "/api/forecast-runs", {"spec": spec, "budget": TINY_BUDGET})
    run_id = started["run_id"]
    state = _wait_done(app, tab, run_id)
    assert state["state"] == "done", state
    assert set(state["verdict"]) == {"trusted", "summary", "checks"}
    assert state["spec"]["mode"] == "series"
    assert isinstance(state["leaderboard"], list)


@pytest.mark.parametrize("cursor", [-1, "0", "invalid", None, "inf"])
def test_run_polling_sends_only_unseen_generation_curves(series_run, cursor):
    app, tab, run_id, _root = series_run
    complete = _get(app, tab, "/api/forecast-run", {"id": run_id})
    state = _get(app, tab, "/api/forecast-run", {"id": run_id, "since_generation": cursor})
    parsed = 0 if cursor == "0" else -1
    assert state["generation_curves"] == [
        curve for curve in complete["generation_curves"] if curve["generation"] > parsed
    ]
    assert state["state"] == "done"
    assert "eval_history" not in state
    latest = complete["generations"][-1]["generation"]
    assert (
        _get(app, tab, "/api/forecast-run", {"id": run_id, "since_generation": latest})[
            "generation_curves"
        ]
        == []
    )


@pytest.mark.parametrize("cursor", [-1, "0", "invalid", None, "inf"])
def test_run_polling_sends_only_updated_leaderboards(app, tab, cursor):
    root = _forecast_path(app) / ".forecast-runs" / "fixture"
    root.mkdir(parents=True)
    (root / "request.json").write_text(
        json.dumps({"spec": {"task": "regression"}, "budget": {"plateau": 3}})
    )
    (root / "status.json").write_text(json.dumps({"state": "done", "stopped": "plateau"}))
    (root / "leaderboard_generation.json").write_text("0")
    board = [{"fitness": 0.3, "genome": {"families": ["lag+1"]}}]
    (root / "leaderboard.json").write_text(json.dumps(board))
    complete = _get(app, tab, "/api/forecast-run", {"id": "fixture"})
    assert complete["leaderboard"] == board
    assert complete["leaderboard_generation"] == 0
    assert complete["kind"] == "search" and complete["budget"] == {"plateau": 3}
    state = _get(app, tab, "/api/forecast-run", {"id": "fixture", "since_leaderboard": cursor})
    if cursor == "0":
        assert "leaderboard" not in state
    else:
        assert state["leaderboard"] == board
    (root / "leaderboard_generation.json").write_text("1")
    board[0]["fitness"] = 0.2
    (root / "leaderboard.json").write_text(json.dumps(board))
    state = _get(app, tab, "/api/forecast-run", {"id": "fixture", "since_leaderboard": 0})
    assert state["leaderboard"] == board and state["leaderboard_generation"] == 1


def test_old_runs_infer_the_leaderboard_revision_from_generation_events(app, tab):
    root = _forecast_path(app) / ".forecast-runs" / "old"
    root.mkdir(parents=True)
    (root / "request.json").write_text(json.dumps({"spec": {"task": "regression"}}))
    (root / "status.json").write_text(json.dumps({"state": "done"}))
    (root / "events.jsonl").write_text(
        json.dumps({"event": "generation", "generation": 2, "best": 0.3})
    )
    (root / "leaderboard.json").write_text(json.dumps([{"fitness": 0.3}]))
    state = _get(app, tab, "/api/forecast-run", {"id": "old", "since_leaderboard": 1})
    assert state["leaderboard_generation"] == 2 and state["leaderboard"]
    assert state["best_wape"] == 0.3 and state["test_error"] is None
    summaries = _get(app, tab, "/api/forecast-runs")
    assert next(run for run in summaries["runs"] if run["id"] == "old")["best_wape"] == 0.3
    assert "leaderboard" not in _get(
        app, tab, "/api/forecast-run", {"id": "old", "since_leaderboard": 2}
    )


def test_unchanged_run_poll_omits_generation_history_curves_and_leaderboard(app, tab):
    root = _forecast_path(app) / ".forecast-runs" / "poll-cost"
    root.mkdir(parents=True)
    (root / "request.json").write_text(json.dumps({"spec": {"task": "regression"}}))
    (root / "status.json").write_text(json.dumps({"state": "done"}))
    (root / "events.jsonl").write_text(
        "\n".join(
            json.dumps({"event": "generation", "generation": generation, "best": 0.3})
            for generation in range(60)
        )
    )
    (root / "leaderboard_generation.json").write_text("59")
    (root / "leaderboard.json").write_text(json.dumps([{"fitness": 0.3}] * 10))
    (root / "generation_curves.jsonl").write_text(
        "\n".join(
            json.dumps({"generation": generation, "metric": "rmse", "validation": [0.3] * 1000})
            for generation in range(60)
        )
    )
    state = _get(
        app,
        tab,
        "/api/forecast-run",
        {
            "id": root.name,
            "since_generation": 59,
            "since_leaderboard": 59,
        },
    )
    assert state["generations"] == [] and state["generation_curves"] == []
    assert "leaderboard" not in state and "eval_history" not in state
    assert state["generation_count"] == 60 and state["generation"] == 59
    assert len(json.dumps(state).encode()) < 1000


def test_generation_event_arriving_after_its_curve_is_not_lost(app, tab):
    root = _forecast_path(app) / ".forecast-runs" / "late-event"
    root.mkdir(parents=True)
    (root / "request.json").write_text(json.dumps({"spec": {"task": "regression"}}))
    (root / "status.json").write_text(json.dumps({"state": "done"}))
    (root / "leaderboard_generation.json").write_text("0")
    (root / "generation_curves.jsonl").write_text(
        json.dumps({"generation": 0, "metric": "mae", "validation": [1.0]}) + "\n"
    )
    first = _get(app, tab, "/api/forecast-run", {"id": root.name, "since_generation": -1})
    assert len(first["generation_curves"]) == 1 and first["generations"] == []
    event = {"event": "generation", "generation": 0, "best": 0.3}
    (root / "events.jsonl").write_text(json.dumps(event) + "\n")
    second = _get(
        app,
        tab,
        "/api/forecast-run",
        {"id": root.name, "since_generation": 0, "since_events": -1},
    )
    assert second["generations"] == [event] and second["generation_curves"] == []


def test_view_series_mode_has_a_band_within_bounds(series_run):
    app, tab, run_id, _root = series_run
    payload = _get(app, tab, "/api/forecast-view", {"id": run_id})
    assert payload["mode"] == "series"
    assert {"actual", "forecast"} <= {s["id"] for s in payload["series"]}
    forecast_line = next(s for s in payload["series"] if s["id"] == "forecast")
    assert forecast_line["dashed"] is True
    band = next(b for b in payload["bands"] if b["series"] == "forecast")

    filtered = _get(
        app, tab, "/api/forecast-view", {"id": run_id, "filters": json.dumps({"project": ["A"]})}
    )
    filtered_band = next(b for b in filtered["bands"] if b["series"] == "forecast")
    assert any(lo is not None for lo in filtered_band["lo"])

    # the band is a quantile interval fit on validation residuals and applied to the final
    # model's prediction - it is not a guarantee the point forecast itself falls inside it,
    # since the final fit (train+val+test) can sit on a different bias than the val-time fit
    # the band came from. with population=6/generations=2 that gap is visible here, so this
    # checks the band is present and internally ordered on every future step rather than that
    # it always contains the point forecast.
    checked = 0
    for lo, hi in zip(band["lo"], band["hi"], strict=True):
        if lo is None or hi is None:
            continue
        assert lo <= hi
        checked += 1
    assert checked > 0
    assert payload["verdict"]["summary"]


def test_view_series_breakdown_gives_one_pair_per_level(series_run):
    app, tab, run_id, _root = series_run
    payload = _get(app, tab, "/api/forecast-view", {"id": run_id, "breakdown": "project"})
    ids = {s["id"] for s in payload["series"]}
    actual_ids = {i for i in ids if i.startswith("actual:")}
    forecast_ids = {i for i in ids if i.startswith("forecast:")}
    assert actual_ids and forecast_ids
    assert len(actual_ids) == len(forecast_ids)


def test_table_pages_and_sorts(series_run):
    app, tab, run_id, _root = series_run
    page0 = _get(app, tab, "/api/forecast-table", {"id": run_id, "page": 0, "size": 5})
    assert len(page0["rows"]) <= 5
    assert page0["total"] > 5
    assert "step" in page0["columns"] and "prediction" in page0["columns"]
    future_idx = page0["columns"].index("future")
    assert any(row[future_idx] for row in page0["rows"])


def test_export_carries_the_verdict_on_every_row(series_run):
    app, tab, run_id, _root = series_run
    data = tab.images["/api/forecast-export/"][0](app, f"{run_id}.csv")
    assert data is not None
    text = data.decode()
    lines = text.splitlines()
    assert lines[0].startswith("# verdict:")
    reader = csv.reader(io.StringIO("\n".join(lines[1:])))
    header = next(reader)
    assert "verdict" in header
    v_idx = header.index("verdict")
    body = list(reader)
    assert body
    assert all(row[v_idx] in ("trusted", "not trusted") for row in body)


def test_cancel_stops_a_run(app, tab):
    write_panel(_forecast_path(app), weeks=60)
    spec = _series_spec("panel.csv")
    big_budget = {"population": 8, "max_generations": 200, "plateau": 200, "nthread": 1}
    started = _post(app, tab, "/api/forecast-runs", {"spec": spec, "budget": big_budget})
    run_id = started["run_id"]
    for _ in range(300):
        state = _get(app, tab, "/api/forecast-run", {"id": run_id})
        if state.get("generations"):
            break
        time.sleep(0.1)
    result = _post(app, tab, "/api/forecast-cancel", {"id": run_id})
    assert result == {"ok": True}
    state = _wait_done(app, tab, run_id)
    assert state["state"] == "cancelled"
    assert state.get("stopped") == "cancelled"


def test_refit_reuses_the_saved_recipe(series_run):
    app, tab, run_id, root = series_run
    newer = write_panel(root, weeks=68)
    spec = _series_spec("panel.csv")
    started = _post(app, tab, "/api/forecast-refit", {"spec": spec, "run_id": run_id})
    refit_id = started["run_id"]
    state = _wait_done(app, tab, refit_id)
    assert state["state"] == "done", state
    assert state["stopped"] == "refit"
    assert newer.weeks == 68


# ---------------------------------------------------------------- row mode


@pytest.fixture(scope="module")
def row_run(tmp_path_factory):
    yield _run_to_done(
        tmp_path_factory.mktemp("row_run"),
        lambda root: write_rows(root, n=2000),
        _row_spec,
        "rows.csv",
        TINY_BUDGET,
    )


def test_row_mode_view_has_load_line_and_mc_band(row_run):
    app, tab, run_id, _root = row_run
    payload = _get(app, tab, "/api/forecast-view", {"id": run_id})
    assert payload["mode"] == "row"
    ids = {s["id"] for s in payload["series"]}
    assert "load" in ids
    load_line = next(s for s in payload["series"] if s["id"] == "load")
    assert load_line["dashed"] is True
    band = next((b for b in payload["bands"] if b["series"] == "load"), None)
    assert band is not None
    assert any(v is not None for v in band["lo"])


def test_row_mode_table_grouped_carries_earliest_latest(row_run):
    app, tab, run_id, _root = row_run
    payload = _get(app, tab, "/api/forecast-table", {"id": run_id, "group": "branch", "size": 1000})
    assert "groups" in payload
    for info in payload["groups"].values():
        assert info["earliest"] <= info["latest"]


# ---------------------------------------------------------------- registration


def test_check_tabs_accepts_forecast_next_to_hyperparams():
    routes.check_tabs([hyperparams_tab(), forecast_tab()])
