"""entity curves survive result writing; optional calendar rules use conditional event mass"""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import aggregate, ensemble, pipeline, runs, views
from smolsmort.forecast.aggregate import build_curve_table, expected_events
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.spec import ANCHOR, ROW, Censor, Column, PrepSpec
from smolsmort.forecast.tables import read_table, write_table
from smolsmort.forecast.worker import _write_result, _write_tables


@pytest.fixture
def curves():
    return {
        "row": np.array([11, 22, 33]),
        "anchor": np.array(["2024-01-02", "2024-01-04", "2024-01-08"], dtype="datetime64[D]"),
        "week": np.arange(3),
        "hazard": np.array([[0.2, 0.25, 0.5], [0.9, 0.5, 0.5], [0.4, 0.5, 1.0]]),
        "survival": np.array([[0.8, 0.6, 0.3], [1.0, 0.5, 0.25], [0.6, 0.3, 0.0]]),
    }


@pytest.fixture
def spec():
    return PrepSpec(
        "unused.csv",
        "entity",
        "regression",
        (Column("created", "anchor"), Column("branch", "dimension"), Column("duration", "target")),
        censor=Censor("created", unit="day", as_of="2024-01-08"),
    )


@pytest.fixture
def final(curves):
    prediction = np.array([14.0, 21.0, 7.0])
    return {
        "verdict": {"accuracy": {"ratio": 0.8}, "summary": "better than baseline"},
        "model_error": 0.2,
        "baseline_error": 0.25,
        "forecast": {
            "row": curves["row"],
            "anchor": curves["anchor"],
            "prediction": prediction,
            "lower": np.array([7.0, 14.0, 0.0]),
            "upper": np.array([28.0, 35.0, 21.0]),
            "predicted_date": curves["anchor"] + prediction.astype("timedelta64[D]"),
        },
        "validation": {"truth": np.array([12.0, 22.0]), "prediction": np.array([14.0, 21.0])},
        "test": {"rows": 2, "prediction": [14.0, 21.0]},
        "curve": build_curve_table(curves),
    }


def make_result(rule="none", objective="hazard"):
    return {
        "best": pipeline.make_genome([], objective, {}).to_dict(),
        "stopped": "complete",
        "aggregate": rule,
    }


def test_curve_table_has_one_row_per_entity_week(curves):
    table = build_curve_table(curves)
    assert list(table) == ["row", "anchor", "week", "date", "hazard", "survival"]
    np.testing.assert_array_equal(table["row"], np.repeat([11, 22, 33], 3))
    np.testing.assert_array_equal(table["week"], np.tile([0, 1, 2], 3))
    np.testing.assert_array_equal(
        table["date"], table["anchor"] + table["week"] * np.timedelta64(7, "D")
    )
    np.testing.assert_array_equal(table["survival"], curves["survival"].ravel())


@pytest.mark.parametrize("unit", ["week", "day", "month"])
def test_three_open_entities_match_the_calendar_hand_sum(curves, unit):
    table = build_curve_table(curves)
    order = [8, 2, 4, 0, 6, 3, 7, 1, 5]
    summary = expected_events({name: values[order] for name, values in table.items()}, unit)
    np.testing.assert_array_equal(
        summary["step"],
        np.array(["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"], dtype="datetime64[D]"),
    )
    # jan 1: .2 + 0; jan 8: .2 + .5 + .4; jan 15: .3 + .25 + .3; jan 22: .3
    # entity 22 has already waited a week: its raw .9 hazard contributes zero in jan 1
    np.testing.assert_allclose(summary["prediction"], [0.2, 1.1, 0.85, 0.3])


def test_empty_curves_have_an_empty_calendar_series(curves):
    table = {name: values[:0] for name, values in build_curve_table(curves).items()}
    summary = expected_events(table, "week")
    assert summary["step"].size == summary["prediction"].size == 0


@pytest.mark.parametrize("rule", ["none", "expected_events"])
def test_result_writes_curves_and_only_the_requested_aggregate(tmp_path, spec, final, rule):
    pytest.importorskip("duckdb")
    _write_result(tmp_path, spec, make_result(rule), final)
    saved = read_table(tmp_path / "curve.parquet", order_by="row, week")
    for name, values in final["curve"].items():
        np.testing.assert_array_equal(saved[name], values)
    assert (tmp_path / "forecast.parquet").exists()
    assert (tmp_path / "validation.parquet").exists()
    assert (tmp_path / "aggregate.parquet").exists() == (rule != "none")
    assert json.loads((tmp_path / "recipe.json").read_text())["aggregate"] == rule
    if rule != "none":
        saved = read_table(tmp_path / "aggregate.parquet", order_by="step")
        np.testing.assert_allclose(saved["prediction"], [0.2, 1.1, 0.85, 0.3])


def test_a_registered_rule_needs_no_dispatch_changes(tmp_path, monkeypatch, spec, final):
    pytest.importorskip("duckdb")
    seen = []

    def summarize_dummy(curve_table, unit):
        seen.append((curve_table, unit))
        return {
            "step": np.array(["2024-01-01"], dtype="datetime64[D]"),
            "prediction": np.array([42.0]),
        }

    monkeypatch.setitem(aggregate.REGISTRY, "dummy", summarize_dummy)
    request = parse_request({"aggregate": "dummy"})
    _write_result(tmp_path, spec, make_result(request["aggregate"]), final)
    assert seen[0][0] is final["curve"]
    assert seen[0][1] == "day"
    np.testing.assert_array_equal(read_table(tmp_path / "aggregate.parquet")["prediction"], [42.0])


@pytest.mark.parametrize("versioned", [False, True])
@pytest.mark.parametrize("name", ["unknown", None, [], {}, 1])
def test_requests_reject_unknown_aggregation_rules(versioned, name):
    request = {"aggregate": name}
    if versioned:
        request.update(version=2, families=[{"name": "xgboost"}], time_budget_s=10)
    with pytest.raises(ValueError, match="aggregate must be one of"):
        parse_request(request)


def test_request_default_is_none_and_legacy_request_is_unchanged():
    request = {"version": 2, "families": [{"name": "xgboost"}], "time_budget_s": 10}
    assert parse_request(request)["aggregate"] == "none"
    assert "aggregate" not in request
    legacy = {"budget": {"population": 3}}
    assert parse_request(legacy) is legacy


@pytest.mark.parametrize("rule", ["none", "expected_events"])
def test_ensemble_finisher_writes_selected_and_winner_curves(
    tmp_path, monkeypatch, spec, final, rule
):
    pytest.importorskip("duckdb")
    request = {"spec": asdict(spec), "prepared": str(tmp_path), "summary": {}, "aggregate": rule}
    ranked = [{"family": "xgboost", "fitness": 0.2, "genome": make_result()["best"]}]
    for name, value in (
        ("request.json", request),
        ("coordinator.json", {"families": {}, "started_at": 0, "nthread": 1}),
        ("finalists.json", ranked),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(ensemble, "watch_parent", lambda pid: None)
    monkeypatch.setattr(
        ensemble, "load_workspace", lambda *args, **kwargs: SimpleNamespace(mode="entity")
    )
    monkeypatch.setattr(ensemble, "finish", lambda *args, **kwargs: final)
    ensemble.run(tmp_path, 123)
    for prefix in ("", "winner_"):
        assert (tmp_path / f"{prefix}curve.parquet").exists()
        assert (tmp_path / f"{prefix}aggregate.parquet").exists() == (rule != "none")
    assert json.loads((tmp_path / "finish_complete.json").read_text()) == {"family": "xgboost"}


@pytest.fixture
def saved_entity(tmp_path, spec, final):
    pytest.importorskip("duckdb")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    write_table(
        prepared / "rows.parquet",
        {
            ROW: final["forecast"]["row"],
            ANCHOR: final["forecast"]["anchor"],
            "branch": np.array(["a", "b", "a"]),
            "duration": np.full(3, np.nan),
        },
    )
    request = {"prepared": str(prepared), "spec": asdict(spec), "aggregate": "expected_events"}
    _write_result(tmp_path, spec, make_result("expected_events"), final)
    return tmp_path, request


def test_hazard_table_uses_weeks_even_for_a_day_target(saved_entity):
    folder, request = saved_entity
    payload = views.table(folder, request, {"sort": "row"})
    assert payload["objective"] == "hazard"
    assert payload["columns"] == [
        "row",
        "branch",
        "anchor",
        "predicted_date",
        "median_weeks",
        "p10_weeks",
        "p90_weeks",
        "family",
    ]
    assert payload["rows"][0] == [11, "a", "2024-01-02", "2024-01-16", 2.0, 1.0, 4.0, "xgboost"]


def test_hazard_view_shows_saved_weekly_series(saved_entity):
    folder, request = saved_entity
    payload = views.entity_view(folder, request, {})
    assert payload["aggregate"] == {"rule": "expected_events", "unit": "week"}
    assert payload["x"] == ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]
    np.testing.assert_allclose(payload["series"][0]["values"], [0.2, 1.1, 0.85, 0.3])
    assert payload["bands"] == []


def test_hazard_view_recomputes_selected_entities(saved_entity):
    folder, request = saved_entity
    payload = views.entity_view(folder, request, {"filters": json.dumps({"branch": ["b"]})})
    assert payload["x"] == ["2024-01-01", "2024-01-08", "2024-01-15"]
    np.testing.assert_allclose(payload["series"][0]["values"], [0.0, 0.5, 0.25])
    payload = views.entity_view(folder, request, {"breakdown": "branch"})
    lines = {line["id"]: line["values"] for line in payload["series"]}
    np.testing.assert_allclose(lines["aggregate:a"], [0.2, 0.6, 0.6, 0.3])
    assert lines["aggregate:b"][-1] is None
    np.testing.assert_allclose(lines["aggregate:b"][:-1], [0.0, 0.5, 0.25])


def test_hazard_view_with_none_needs_no_residual_table(tmp_path, spec, final):
    pytest.importorskip("duckdb")
    _write_result(
        tmp_path,
        spec,
        make_result(),
        {key: value for key, value in final.items() if key != "validation"},
    )
    request = {"prepared": str(tmp_path), "spec": asdict(spec), "aggregate": "none"}
    payload = views.entity_view(tmp_path, request, {})
    assert payload["aggregate"] is None
    assert payload["series"] == payload["x"] == []
    assert (tmp_path / "curve.parquet").exists()


@pytest.mark.parametrize("dimension", ["week", "date", "anchor", "hazard", "survival"])
def test_hazard_filters_resolve_dimensions_that_share_curve_column_names(saved_entity, dimension):
    folder, request = saved_entity
    request["spec"]["columns"] = [
        {**column, "name": dimension if column["role"] == "dimension" else column["name"]}
        for column in request["spec"]["columns"]
    ]
    path = folder / "prepared" / "rows.parquet"
    rows = read_table(path)
    rows[dimension] = rows.pop("branch")
    write_table(path, rows)
    payload = views.entity_view(
        folder, request, {"filters": json.dumps({dimension: ["b"]}), "breakdown": dimension}
    )
    np.testing.assert_allclose(payload["series"][0]["values"], [0.0, 0.5, 0.25])


def test_selected_winner_uses_its_own_aggregate(saved_entity, spec, final):
    folder, request = saved_entity
    curve = final["curve"]
    selected = {name: values[curve["row"] == 22] for name, values in curve.items()}
    _write_tables(folder, spec, {"curve": selected}, prefix="winner", aggregate="expected_events")
    payload = views.entity_view(folder, request, {"model": "winner"})
    np.testing.assert_allclose(payload["series"][0]["values"], [0.0, 0.5, 0.25])


def test_regression_table_and_load_chart_keep_their_existing_shape(saved_entity, spec, final):
    folder, request = saved_entity
    (folder / "curve.parquet").unlink()
    (folder / "aggregate.parquet").unlink()
    _write_result(
        folder,
        spec,
        make_result(objective="aft"),
        {key: value for key, value in final.items() if key != "curve"},
    )
    payload = views.entity_table(folder, request, {"sort": "row"})
    assert "objective" not in payload
    assert payload["columns"] == [
        "row",
        "branch",
        "anchor",
        "predicted_date",
        "prediction",
        "lower",
        "upper",
    ]
    assert payload["rows"][0] == [
        11,
        "a",
        "2024-01-02",
        "2024-01-16",
        14.0,
        "2024-01-09",
        "2024-01-30",
    ]
    chart = views.entity_view(folder, request, {})
    assert {line["id"] for line in chart["series"]} == {"actual", "load"}
    assert "objective" not in chart and "aggregate" not in chart


def test_entity_finish_passes_the_long_curve_to_result_writing(monkeypatch, spec, curves):
    lower = np.array([0.0, 7.0, 0.0])
    upper = np.full(3, np.inf)
    workspace = SimpleNamespace(
        spec=spec,
        table={ROW: curves["row"], ANCHOR: curves["anchor"], "duration": np.full(3, np.nan)},
        features=SimpleNamespace(pick=lambda families: (None, [], None)),
        profile={"columns": {"duration": {"min": 0}}},
        masks={name: np.ones(3, dtype=bool) for name in ("train", "val", "test")},
        predict_rows=np.ones(3, dtype=bool),
        bounds={"test": (lower, upper), "final": (lower, upper)},
    )
    phase = (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.arange(3))
    monkeypatch.setattr(pipeline, "_entity_phase", lambda *args: phase)
    monkeypatch.setattr(pipeline, "_entity_baseline", lambda *args: phase[0])
    # survival scoring has its own tests, this one checks the curve hand-off
    scored = {"prediction": phase[0], "truth": phase[1]}
    verdict = {"accuracy": {"model_error": 0.2, "baseline_error": 0.4}}
    monkeypatch.setattr(
        pipeline, "_entity_survival_verdict", lambda *args: (scored, scored, verdict)
    )
    monkeypatch.setattr(
        pipeline, "_entity_fit", lambda *args: (SimpleNamespace(cap=2), np.zeros((3, 1)))
    )
    monkeypatch.setattr(pipeline, "measure_importance", lambda *args: None)
    monkeypatch.setattr(pipeline, "combine_importance", lambda *args: None)
    monkeypatch.setattr(pipeline.ev, "judge", lambda **kwargs: {})

    def predict_curves(fitted, x, anchors, *, waited):
        np.testing.assert_array_equal(waited, [0, 1, 0])
        return {
            **curves,
            "prediction": np.array([2, 2, 1]),
            "lower": np.array([0, 1, 0]),
            "upper": np.array([4, 5, 3]),
        }

    monkeypatch.setattr(pipeline, "predict_hazard", predict_curves)
    final = pipeline._finish_entities(workspace, pipeline.make_genome([], "hazard", {}), 1)
    assert "curves" in final
    expected = build_curve_table(curves)
    for name, values in expected.items():
        np.testing.assert_array_equal(final["curve"][name], values)
    np.testing.assert_array_equal(final["forecast"]["prediction"], [14, 14, 7])


@pytest.mark.parametrize("versioned", [False, True])
def test_start_run_saves_the_api_aggregation_choice(tmp_path, monkeypatch, spec, versioned):
    process = SimpleNamespace(pid=123)
    monkeypatch.setattr(runs.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runs, "_PROCESSES", {})
    prepared = SimpleNamespace(folder=tmp_path, summary={})
    options = (
        {"version": 2, "families": [{"name": "xgboost"}], "time_budget_s": 10} if versioned else {}
    )
    run_id = runs.start_run(tmp_path, spec, prepared, aggregate="expected_events", **options)
    request = json.loads((tmp_path / run_id / "request.json").read_text())
    assert request["aggregate"] == "expected_events"


@pytest.mark.parametrize("versioned", [False, True])
def test_api_forwards_aggregation_to_run_request(tmp_path, monkeypatch, spec, versioned):
    from smolsmort.forecast import tab

    (tmp_path / spec.source).write_text("created,branch,duration\n")
    app = SimpleNamespace(state=SimpleNamespace(bases={"forecast": str(tmp_path)}))
    prepared = SimpleNamespace(folder=tmp_path, summary={}, sql="", reused=True)
    monkeypatch.setattr(tab, "_parse_spec", lambda *args: spec)
    monkeypatch.setattr(tab, "_prepare_spec", lambda *args: prepared)
    monkeypatch.setattr(tab, "prepare", lambda *args: prepared)
    monkeypatch.setattr(views, "save_prepared_spec", lambda *args: None)
    monkeypatch.setattr(views, "probe_families", lambda: {"xgboost": (True, "")})
    calls = []
    monkeypatch.setattr(runs, "start_run", lambda *args, **kwargs: calls.append(kwargs) or "new")
    payload = {"spec": asdict(spec), "aggregate": "expected_events"}
    if versioned:
        payload.update(version=2, families=[{"name": "xgboost"}], time_budget_s=10)
    tab.forecast_tab().post["/api/forecast-runs"](app, payload)
    assert calls[0]["aggregate"] == "expected_events"
