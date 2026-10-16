"""multi-family routes and legacy request compatibility on synthetic sources"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from forecast_fixtures import write_panel

from smolsmort.forecast import runs, views
from smolsmort.forecast.families import get_family, registered_families
from smolsmort.forecast.tab import forecast_tab
from smolsmort.forecast.tables import read_table, write_table
from smolsmort.review.routes import RequestError


@pytest.fixture
def app(tmp_path):
    return SimpleNamespace(state=SimpleNamespace(bases={"forecast": str(tmp_path)}))


def make_spec():
    return {
        "source": "panel.csv",
        "mode": "series",
        "task": "regression",
        "step": "week",
        "horizon": 4,
        "columns": [
            {"name": "day", "role": "time"},
            {"name": "project", "role": "dimension"},
            {"name": "product", "role": "dimension"},
            {"name": "orders", "role": "measure", "aggregation": "sum"},
            {"name": "units", "role": "target", "aggregation": "sum"},
        ],
    }


def make_payload(**changes):
    return {"spec": make_spec(), "families": [{"name": "ridge"}], "time_budget_s": 30, **changes}


def post(app, route="/api/forecast-runs", payload=None):
    return forecast_tab().post[route](app, payload if payload is not None else make_payload())


def get(app, route="/api/forecast-run", **query):
    return forecast_tab().get[route](app, query)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_saved_run(app, name="saved", **changes):
    root = Path(app.state.bases["forecast"])
    folder = root / ".forecast-runs" / name
    write_json(folder / "request.json", {"version": 2, **make_payload(), **changes})
    write_json(folder / "status.json", {"state": "done", "phase": "finish"})
    return folder


@pytest.fixture
def completed_models(app, tmp_path):
    pytest.importorskip("duckdb")
    prepared = tmp_path / ".forecast-cache" / "synthetic"
    prepared.mkdir(parents=True)
    dates = np.array(["2025-01-06", "2025-01-13", "2025-01-20"], dtype="datetime64[D]")
    write_table(
        prepared / "panel.parquet",
        {
            "__series": np.array(["a"] * 3),
            "__step": dates,
            "project": np.array(["a"] * 3),
            "product": np.array(["one"] * 3),
            "units": np.array([8.0, 9.0, 10.0]),
        },
    )
    run = make_saved_run(
        app,
        families=[{"name": name} for name in ("xgboost", "ridge", "snaive")],
        prepared=str(prepared),
    )
    winner_verdict = {"trusted": True, "summary": "winner beat baseline"}
    ensemble_verdict = {"trusted": False, "summary": "ensemble did not beat baseline"}
    write_json(
        run / "result.json",
        {
            "verdict": winner_verdict,
            "winner": {"family": "xgboost", "fitness": 0.2, "verdict": winner_verdict},
            "ensemble": {
                "members": ["xgboost", "ridge"],
                "fitness": 0.25,
                "verdict": ensemble_verdict,
            },
            "families": {
                "xgboost": {"fitness": 0.2, "candidates": 6, "state": "done"},
                "ridge": {
                    "fitness": 0.3,
                    "candidates": 4,
                    "state": "done",
                    "importance": {
                        "method": "absolute standardised coefficient",
                        "features": [{"feature": "lag_2", "importance": 1.0}],
                    },
                },
                "snaive": {
                    "fitness": 0.5,
                    "candidates": 1,
                    "state": "pruned",
                    "pruned_reason": "paired validation error behind xgboost",
                    "verdict": {"trusted": True, "summary": "must not expose loser test scores"},
                },
            },
        },
    )
    write_json(run / "recipe.json", {"family": "xgboost", "genome": {"family": "xgboost"}})
    write_json(
        run / "families/xgboost/importance.json",
        {"method": "gain", "features": [{"feature": "lag_1", "importance": 1.0}]},
    )
    for model, prediction in (("winner", 10.0), ("ensemble", 20.0)):
        for kind, date, actual in (
            ("validation", dates[1], 9.0),
            ("test", dates[2], 10.0),
            ("forecast", np.datetime64("2025-01-27", "D"), None),
        ):
            columns = {
                "series": np.array(["a"]),
                "step": np.array([date]),
                "bucket": np.array(["h1"]),
                "prediction": np.array([prediction]),
            }
            if actual is None:
                columns.update(lower=np.array([prediction - 1]), upper=np.array([prediction + 1]))
            else:
                columns["actual"] = np.array([actual])
            write_table(run / f"{model}_{kind}.parquet", columns)
            if model == "winner":
                write_table(run / f"{kind}.parquet", columns)
    return run


def test_results_compare_validation_scores_without_loser_test_verdicts(app, completed_models):
    payload = get(app, "/api/forecast-view", id=completed_models.name)
    assert payload["family"] == payload["winner_family"] == "xgboost"
    assert payload["model"] == "winner" and payload["members"] == []
    assert list(payload["families"]) == ["xgboost", "ridge", "snaive"]
    assert [row["fitness"] for row in payload["families"].values()] == [0.2, 0.3, 0.5]
    assert payload["families"]["xgboost"]["verdict"] == payload["winner"]["verdict"]
    assert "verdict" not in payload["families"]["ridge"]
    assert "verdict" not in payload["families"]["snaive"]
    assert (
        payload["families"]["snaive"]["pruned_reason"] == "paired validation error behind xgboost"
    )
    assert payload["ensemble"]["verdict"]["trusted"] is False
    assert payload["importance"] == {
        "method": "gain",
        "features": [{"feature": "lag_1", "importance": 1.0}],
    }
    forecast = next(line for line in payload["series"] if line["id"] == "forecast")
    assert forecast["family"] == "xgboost" and forecast["values"][-1] == 10


@pytest.mark.parametrize(
    "model,prediction,family,trusted",
    [("winner", 10, "xgboost", True), ("ensemble", 20, "ensemble", False)],
)
def test_selected_result_model_matches_chart_table_and_export(
    app, completed_models, model, prediction, family, trusted
):
    view = get(app, "/api/forecast-view", id=completed_models.name, model=model)
    assert view["model"] == model and view["family"] == family
    assert view["verdict"]["trusted"] is trusted
    forecast = next(line for line in view["series"] if line["id"] == "forecast")
    assert forecast["values"][-1] == prediction
    if model == "ensemble":
        assert view["members"] == ["xgboost", "ridge"] and view["importance"] is None
    table = get(app, "/api/forecast-table", id=completed_models.name, model=model, sort="family")
    assert table["family"] == family and table["columns"][-1] == "family"
    assert all(row[-1] == family for row in table["rows"])
    assert {row[table["columns"].index("prediction")] for row in table["rows"]} == {prediction}
    exported = (
        forecast_tab()
        .images["/api/forecast-export/"][0](app, f"{completed_models.name}/{model}.csv")
        .decode()
    )
    assert exported.splitlines()[0] == f"# verdict: {view['verdict']['summary']}"
    assert f",{family},{'trusted' if trusted else 'not trusted'}" in exported


def test_saved_ensemble_is_default_but_winner_can_still_be_selected(app, completed_models):
    write_json(
        completed_models / "recipe.json", {"family": "ensemble", "genome": {"family": "xgboost"}}
    )
    default = get(app, "/api/forecast-view", id=completed_models.name)
    winner = get(app, "/api/forecast-view", id=completed_models.name, model="winner")
    assert default["family"] == "ensemble" and winner["family"] == "xgboost"
    assert default["verdict"] != winner["verdict"]


def test_model_family_column_preserves_a_source_dimension_named_family(app, completed_models):
    request = views.read_request(completed_models)
    request["spec"]["columns"].append({"name": "family", "role": "dimension"})
    write_json(completed_models / "request.json", request)
    panel = Path(request["prepared"]) / "panel.parquet"
    columns = read_table(panel)
    columns["family"] = np.array(["synthetic dimension"] * len(columns["__series"]))
    write_table(panel, columns)
    payload = get(
        app, "/api/forecast-table", id=completed_models.name, model="winner", sort="model_family"
    )
    assert payload["family_column"] == "model_family"
    dimension_index = payload["columns"].index("family")
    model_index = payload["columns"].index("model_family")
    assert all(row[dimension_index] == "synthetic dimension" for row in payload["rows"])
    assert all(row[model_index] == "xgboost" for row in payload["rows"])


@pytest.mark.parametrize(
    "family,method",
    [("xgboost", "gain"), ("ridge", "absolute standardised coefficient"), ("snaive", None)],
)
def test_importance_selection_never_changes_the_shown_forecast(
    app, completed_models, family, method
):
    payload = get(app, "/api/forecast-view", id=completed_models.name, importance_family=family)
    assert payload["family"] == "xgboost" and payload["importance_family"] == family
    if method is None:
        assert payload["importance"] is None
    else:
        assert payload["importance"]["method"] == method


@pytest.mark.parametrize("query", [{"model": "ridge"}, {"importance_family": "../../other"}])
def test_result_queries_reject_unsaved_models_and_unknown_importance(app, completed_models, query):
    with pytest.raises(RequestError):
        get(app, "/api/forecast-view", id=completed_models.name, **query)


def test_legacy_result_defaults_to_xgboost_without_a_family_comparison(app, completed_models):
    for prefix in ("winner", "ensemble"):
        for path in completed_models.glob(f"{prefix}_*.parquet"):
            path.unlink()
    request = views.read_request(completed_models)
    request.pop("version")
    request.pop("families")
    write_json(completed_models / "request.json", request)
    write_json(completed_models / "recipe.json", {"genome": {"families": ["lag"]}})
    write_json(
        completed_models / "result.json",
        {"verdict": {"trusted": True, "summary": "trusted fixture"}},
    )
    write_json(
        completed_models / "leaderboard.json", [{"fitness": 0.2, "genome": {"families": ["lag"]}}]
    )
    state = get(app, id=completed_models.name)
    assert state["version"] == 1 and state["winner_family"] == "xgboost"
    assert state["leaderboard"][0]["family"] == "xgboost"
    payload = get(app, "/api/forecast-view", id=completed_models.name)
    assert payload["family"] == "xgboost" and payload["families"] == {}


def test_family_poll_preserves_failed_and_terminated_reasons(app):
    run = make_saved_run(app, families=[{"name": "ridge"}, {"name": "snaive"}])
    write_json(
        run / "status.json",
        {
            "state": "done",
            "families": {
                "ridge": {"state": "failed", "reason": "synthetic fit failed"},
                "snaive": {
                    "state": "paused",
                    "terminated": True,
                    "reason": "slice deadline exceeded grace; worker terminated",
                },
            },
        },
    )
    payload = get(app, id=run.name)
    assert payload["version"] == 2
    assert payload["families"]["ridge"]["reason"] == "synthetic fit failed"
    assert payload["families"]["snaive"]["terminated"] is True
    assert "worker terminated" in payload["families"]["snaive"]["reason"]


def test_catalog_lists_all_registered_families_and_typed_spaces(app):
    catalog = get(app, "/api/forecast-families")["families"]
    assert [item["name"] for item in catalog] == [family.name for family in registered_families()]
    for item in catalog:
        family = get_family(item["name"])
        assert item["label"] == family.label and item["needs"] == family.needs
        assert item["objectives"] == list(family.objectives("regression"))
        assert item["space"] == [asdict(param) for param in family.space()]
        assert item["available"] == family.available()[0]
        if not item["available"]:
            assert item["install_command"] == f"uv sync --extra {family.pip_extra}"
            assert item["suitable"] is False and item["reason"]
        assert "seconds_per_candidate" not in item


def test_catalog_never_imports_native_model_libraries_into_the_host(tmp_path):
    code = (
        "import sys; from pathlib import Path; "
        "from smolsmort.forecast.views import family_catalog; "
        "family_catalog(Path(sys.argv[1])); "
        "assert not {'torch', 'xgboost', 'lightgbm', 'statsforecast'} & set(sys.modules)"
    )
    process = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=30
    )
    assert process.returncode == 0, process.stderr


def test_catalog_exposes_calibration_from_the_latest_matching_task(app):
    older = make_saved_run(app, "20260101")
    newer = make_saved_run(app, "20260102")
    invalid = make_saved_run(app, "20260103", spec={"task": "classification"})
    write_json(older / "cost-calibration.json", {"ridge": {"seconds_per_candidate": 2}})
    write_json(
        newer / "cost-calibration.json",
        {"ridge": {"seconds_per_candidate": 0.75, "estimate_is_lower_bound": True}},
    )
    write_json(invalid / "cost-calibration.json", {"ridge": {"seconds_per_candidate": 999}})
    ridge = next(
        item for item in get(app, "/api/forecast-families")["families"] if item["name"] == "ridge"
    )
    assert ridge["seconds_per_candidate"] == 0.75
    assert ridge["estimate_is_lower_bound"] is True


def test_unavailable_family_is_never_suitable_even_with_enough_history(app, tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    write_panel(tmp_path)
    monkeypatch.setattr(
        get_family("lightgbm"), "available", lambda: (False, "uv sync --extra lightgbm")
    )
    monkeypatch.setattr(
        views,
        "probe_families",
        lambda: {family.name: family.available() for family in registered_families()},
    )
    catalog = get(app, "/api/forecast-families", spec=json.dumps(make_spec()))["families"]
    lightgbm = next(item for item in catalog if item["name"] == "lightgbm")
    assert lightgbm["suitable"] is False
    assert lightgbm["install_command"] == "uv sync --extra lightgbm"
    assert "unavailable" in lightgbm["reason"]


@pytest.mark.parametrize("weeks,suitable", [(60, False), (156, True)])
def test_stats_suitability_needs_two_training_seasons_and_prepared_query_works(
    app, tmp_path, weeks, suitable
):
    pytest.importorskip("duckdb")
    write_panel(tmp_path, weeks=weeks)
    initial = get(app, "/api/forecast-families", spec=json.dumps(make_spec()))
    prepared = next((tmp_path / ".forecast-cache").iterdir())
    reopened = get(app, "/api/forecast-families", prepared=str(prepared))
    assert reopened == initial
    for item in initial["families"]:
        if item["needs"] == "raw_series" and item["available"]:
            assert item["suitable"] is suitable, item
            if not suitable:
                assert "two seasons" in item["reason"] and "104" in item["reason"]
        elif item["available"]:
            assert item["suitable"] is True


def test_lag_suitability_uses_the_smallest_bucket_training_split():
    masks = {"train": np.ones(19, dtype=bool)}
    ws = SimpleNamespace(mode="series", series_masks={(1, 1): masks})
    reason = views._suitability_reason(get_family("ridge"), ws)
    assert "20 usable training rows" in reason and "19" in reason
    masks["train"] = np.ones(20, dtype=bool)
    assert views._suitability_reason(get_family("ridge"), ws) is None


def test_prepared_query_recovers_a_legacy_spec_from_its_run(app, tmp_path):
    pytest.importorskip("duckdb")
    write_panel(tmp_path)
    expected = get(app, "/api/forecast-families", spec=json.dumps(make_spec()))
    prepared = next((tmp_path / ".forecast-cache").iterdir())
    spec = json.loads((prepared / "spec.json").read_text())
    (prepared / "spec.json").unlink()
    old = make_saved_run(app, "old")
    write_json(old / "request.json", {"spec": spec, "prepared": str(prepared)})
    assert get(app, "/api/forecast-families", prepared=str(prepared)) == expected


def test_row_classification_suitability_counts_known_labels():
    from smolsmort.forecast.spec import Column, PrepSpec

    spec = PrepSpec(
        "rows.csv", "row", "classification", (Column("day", "anchor"), Column("label", "target"))
    )
    ws = SimpleNamespace(
        mode="row",
        spec=spec,
        masks={"train": np.ones(24, dtype=bool)},
        table={"label": np.array(["open", "closed"] * 12)},
        bounds={"val": (np.full(24, np.nan), np.full(24, np.nan))},
    )
    assert views._suitability_reason(get_family("xgboost"), ws) is None


def test_catalog_task_objectives_and_bad_queries(app):
    catalog = get(app, "/api/forecast-families", task="classification")["families"]
    assert next(item for item in catalog if item["name"] == "ridge")["objectives"] == []
    assert next(item for item in catalog if item["name"] == "snaive")["suitable"] is False
    for query in ({"task": "bad"}, {"spec": "{"}, {"prepared": "../outside"}):
        with pytest.raises(RequestError) as caught:
            get(app, "/api/forecast-families", **query)
        assert caught.value.status == 400


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"version": 3}, "version"),
        ({"families": []}, "at least one"),
        ({"families": [None]}, "object with a name"),
        ({"families": [{"name": "unknown"}]}, "unknown family"),
        ({"families": [{"name": "ridge"}, {"name": "ridge"}]}, "unique"),
        ({"families": [{"name": "ridge", "method": "bad"}]}, "method"),
        ({"families": [{"name": "ridge", "space": []}]}, "space"),
        ({"families": [{"name": "ridge", "space": {"missing": [1]}}]}, "space override"),
        ({"families": [{"name": "ridge", "space": {"alpha": []}}]}, "nonempty array"),
        ({"families": [{"name": "ridge", "space": {"alpha": [0]}}]}, "invalid value"),
        ({"families": [{"name": "ridge", "space": {"alpha": [math.nan]}}]}, "invalid value"),
        ({"families": [{"name": "ridge", "space": {"alpha": [True]}}]}, "invalid value"),
        ({"time_budget_s": 0}, "time_budget_s"),
        ({"time_budget_s": math.inf}, "time_budget_s"),
        ({"time_budget_s": 10**400}, "time_budget_s"),
        ({"time_budget_s": True}, "time_budget_s"),
        ({"transform": "bad"}, "transform"),
        ({"max_concurrent_families": 0}, "max_concurrent_families"),
        ({"max_concurrent_families": True}, "max_concurrent_families"),
        ({"candidate_limit_fraction": 0}, "candidate_limit_fraction"),
        ({"candidate_limit_fraction": math.nan}, "candidate_limit_fraction"),
        ({"candidate_limit_fraction": 10**400}, "candidate_limit_fraction"),
        ({"families": [{"name": "ridge", "space": {"alpha": [10**400]}}]}, "invalid value"),
        ({"ensemble": {"top": 0}}, "ensemble.top"),
        ({"ensemble": {"enabled": 1}}, "ensemble.enabled"),
        ({"nthread": False}, "nthread"),
        ({"budget": {}}, "v2 uses time_budget_s"),
        ({"warm_from": []}, "warm_from"),
        ({"spec": []}, "spec must be an object"),
        ({"spec": {**make_spec(), "horizon": "four"}}, "horizon"),
        ({"spec": {**make_spec(), "mode": "bad"}}, "spec.mode"),
        ({"spec": {**make_spec(), "task": "classification"}}, "require regression"),
        (
            {
                "spec": {
                    **make_spec(),
                    "columns": [
                        {"name": "day", "role": "time"},
                        {"name": 7, "role": "target", "aggregation": "sum"},
                    ],
                }
            },
            "string name",
        ),
        (
            {"spec": {**make_spec(), "columns": [{"name": "day", "role": "clock"}]}},
            "unknown column role",
        ),
        (
            {"spec": {**make_spec(), "columns": [{"name": "day", "role": "time", "known": "yes"}]}},
            "column.known",
        ),
    ],
)
def test_invalid_v2_request_returns_clear_400_without_starting_worker(
    app, tmp_path, monkeypatch, changes, message
):
    write_panel(tmp_path, weeks=60)
    monkeypatch.setattr(
        runs, "start_run", lambda *args, **kwargs: pytest.fail("invalid request started a worker")
    )
    with pytest.raises(RequestError, match=message) as caught:
        post(app, payload=make_payload(**changes))
    assert caught.value.status == 400
    assert not (tmp_path / ".forecast-runs").exists()


def test_unavailable_family_is_rejected_before_start(app, tmp_path, monkeypatch):
    write_panel(tmp_path, weeks=60)
    monkeypatch.setattr(
        get_family("lightgbm"), "available", lambda: (False, "uv sync --extra lightgbm")
    )
    monkeypatch.setattr(
        views,
        "probe_families",
        lambda: {family.name: family.available() for family in registered_families()},
    )
    with pytest.raises(RequestError, match="unavailable; uv sync --extra lightgbm") as caught:
        post(app, payload=make_payload(families=[{"name": "lightgbm"}]))
    assert caught.value.status == 400


def test_missing_source_column_is_reported_as_a_clear_400(app, tmp_path):
    pytest.importorskip("duckdb")
    write_panel(tmp_path, weeks=60)
    spec = make_spec()
    spec["columns"][-1]["name"] = "absent_target"
    with pytest.raises(RequestError, match="absent_target") as caught:
        post(app, payload=make_payload(spec=spec))
    assert caught.value.status == 400


def test_v1_route_passes_the_same_budget_and_warm_arguments(app, tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    write_panel(tmp_path, weeks=60)
    calls = []
    monkeypatch.setattr(runs, "start_run", lambda *args, **kwargs: calls.append(kwargs) or "new")
    budget = {"population": 6, "max_generations": 2, "plateau": 1, "nthread": 1}
    started = post(app, payload={"spec": make_spec(), "budget": budget, "warm_from": "old"})
    assert calls == [{"budget": budget, "warm_from": "old"}]
    assert set(started) == {"run_id", "prepared"}


def test_v2_route_passes_all_coordinator_controls(app, tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    write_panel(tmp_path, weeks=60)
    calls = []
    monkeypatch.setattr(runs, "start_run", lambda *args, **kwargs: calls.append(kwargs) or "new")
    payload = make_payload(
        families=[{"name": "ridge", "method": "grid", "space": {"alpha": [0.1, 1]}}],
        time_budget_s=180,
        transform="none",
        max_concurrent_families=2,
        candidate_limit_fraction=0.1,
        ensemble={"enabled": True, "top": 2},
        nthread=4,
    )
    assert post(app, payload=payload)["run_id"] == "new"
    assert calls == [
        {
            "version": 2,
            "warm_from": None,
            **{key: value for key, value in payload.items() if key != "spec"},
        }
    ]


def test_multi_poll_merges_family_status_and_never_loses_slower_family_updates(app):
    run = make_saved_run(app, families=[{"name": "ridge"}, {"name": "snaive"}])
    write_json(
        run / "status.json",
        {
            "state": "done",
            "phase": "finish",
            "families": {
                "ridge": {
                    "state": "pruned",
                    "phase": "halving",
                    "slice_s": 5,
                    "best_error": 0.4,
                    "pruned_reason": "behind snaive",
                    "seconds_per_candidate": 0.2,
                    "candidates_done": 4,
                    "candidates_expected": 9,
                    "nthread": 2,
                }
            },
        },
    )
    write_json(
        run / "families/ridge/leaderboard.json", [{"fitness": 0.4, "genome": {"family": "ridge"}}]
    )
    write_json(
        run / "families/ridge/status.json",
        {"state": "done", "trace": "large", "slice_history": ["large"]},
    )
    write_json(
        run / "families/ridge/events.jsonl", {"event": "generation", "generation": 5, "best": 0.4}
    )
    write_json(
        run / "families/ridge/generation_curves.jsonl", {"generation": 5, "validation": [0.4]}
    )
    first = get(app, id=run.name)
    ridge = first["families"]["ridge"]
    assert ridge["state"] == "pruned" and ridge["threads"] == 2
    assert {
        "state",
        "phase",
        "slice_s",
        "best_error",
        "pruned_reason",
        "seconds_per_candidate",
        "candidates_done",
        "candidates_expected",
        "threads",
    } <= ridge.keys()
    assert "trace" not in ridge and "slice_history" not in ridge
    assert first["events"][0]["family"] == first["generation_curves"][0]["family"] == "ridge"
    assert first["leaderboard"][0]["family"] == "ridge"
    cursors = {
        "id": run.name,
        "since_generation": first["generation"],
        "since_events": first["events_cursor"],
        "since_leaderboard": first["leaderboard_generation"],
    }
    unchanged = get(app, **cursors)
    assert unchanged["events"] == unchanged["generations"] == unchanged["generation_curves"] == []
    assert "leaderboard" not in unchanged and "eval_history" not in unchanged
    write_json(
        run / "families/snaive/leaderboard.json", [{"fitness": 0.3, "genome": {"family": "snaive"}}]
    )
    write_json(
        run / "families/snaive/events.jsonl", {"event": "generation", "generation": 0, "best": 0.3}
    )
    write_json(
        run / "families/snaive/generation_curves.jsonl", {"generation": 0, "validation": [0.3]}
    )
    slower = get(app, **cursors)
    assert [event["family"] for event in slower["events"]] == ["snaive"]
    assert [curve["family"] for curve in slower["generation_curves"]] == ["snaive"]
    assert slower["generation_curves"][0]["family_generation"] == 0
    assert slower["leaderboard"][0]["family"] == "snaive"
    assert slower["leaderboard_generation"] > first["leaderboard_generation"]


def test_multi_poll_keeps_partial_events_and_late_curves(app):
    run = make_saved_run(app)
    events = run / "families/ridge/events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text('{"event":"generation", "generation":0}\n{"event":')
    first = get(app, id=run.name)
    curves = run / "families/ridge/generation_curves.jsonl"
    write_json(curves, {"generation": 0, "validation": [0.3]})
    events.write_text('{"event":"generation", "generation":0}\n{"event":"candidate"}\n')
    second = get(
        app, id=run.name, since_events=first["events_cursor"], since_generation=first["generation"]
    )
    assert [event["event"] for event in second["events"]] == ["candidate"]
    assert second["generation_curves"][0]["family"] == "ridge"


def test_multi_event_cursor_drains_every_event_without_loss(app):
    run = make_saved_run(app)
    path = run / "families/ridge/events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(json.dumps({"event": "candidate", "candidate": index}) for index in range(250))
    )
    first = get(app, id=run.name, since_events=-1)
    second = get(app, id=run.name, since_events=first["events_cursor"])
    assert len(first["events"]) == 200 and len(second["events"]) == 50
    assert [event["candidate"] for event in first["events"] + second["events"]] == list(range(250))
    assert get(app, id=run.name, since_events=second["events_cursor"])["events"] == []
    summary = get(app, "/api/forecast-runs")["runs"][0]
    assert "events" not in summary and "generation_curves" not in summary


def test_multi_poll_includes_coordinator_events_and_skips_crash_damaged_lines(app):
    run = make_saved_run(app)
    events = run / "families/ridge/events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text('{"event": broken}\n{"event":"candidate","generation":0}\n')
    write_json(
        run / "events.jsonl", {"event": "terminated", "family": "ridge", "reason": "slice expired"}
    )
    state = get(app, id=run.name)
    assert [event["event"] for event in state["events"]] == ["candidate", "terminated"]
    assert [event["family"] for event in state["events"]] == ["ridge", "ridge"]


@pytest.mark.parametrize("winner", ["ridge", "ensemble"])
def test_multi_summaries_identify_kind_families_and_winner(app, winner):
    run = make_saved_run(app, families=[{"name": "ridge"}, {"name": "snaive"}])
    write_json(run / "recipe.json", {"family": winner, "genome": {"family": "ridge"}})
    summary = get(app, "/api/forecast-runs")["runs"][0]
    assert summary["kind"] == "multi" and set(summary["families"]) == {"ridge", "snaive"}
    assert summary["winner_family"] == winner


def test_ensemble_recipe_keeps_an_implicit_xgboost_winner_family(app):
    run = make_saved_run(app)
    write_json(
        run / "recipe.json",
        {
            "family": "ensemble",
            "genome": {"families": ["lag"], "objective": "squared", "params": {}},
        },
    )
    assert views.winning_genome(run)["family"] == "xgboost"


@pytest.mark.parametrize("action", ["refit", "warm"])
@pytest.mark.parametrize("winner", ["ridge", "ensemble"])
def test_refit_and_legacy_warm_action_on_v2_use_the_single_winner_family(
    app, tmp_path, monkeypatch, action, winner
):
    pytest.importorskip("duckdb")
    write_panel(tmp_path, weeks=60)
    run = make_saved_run(app, families=[{"name": "ridge"}, {"name": "snaive"}])
    genome = {
        "family": "ridge",
        "params": {"alpha": 1},
        "objective": "squared",
        "families": ["lag"],
    }
    write_json(run / "recipe.json", {"family": winner, "genome": genome})
    write_json(run / "result.json", {"winner": {"family": "ridge"}})
    calls = []
    monkeypatch.setattr(runs, "start_run", lambda *args, **kwargs: calls.append(kwargs) or "new")
    if action == "refit":
        post(app, "/api/forecast-refit", {"spec": make_spec(), "run_id": run.name})
        assert calls[0]["recipe"] == genome
    else:
        post(app, payload={"spec": make_spec(), "budget": {}, "warm_from": run.name})
        assert calls[0]["warm_from"] == run.name
    assert calls[0]["families"] == [{"name": "ridge"}]
    assert calls[0]["version"] == 2 and calls[0]["transform"] == "none"


def test_legacy_warm_on_multi_keeps_the_requested_duration_and_threads(app, tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    write_panel(tmp_path, weeks=60)
    run = make_saved_run(app)
    write_json(run / "recipe.json", {"genome": {"family": "ridge"}})
    calls = []
    monkeypatch.setattr(runs, "start_run", lambda *args, **kwargs: calls.append(kwargs) or "new")
    post(
        app,
        payload={
            "spec": make_spec(),
            "budget": {"time_cap": 240, "nthread": 2},
            "warm_from": run.name,
        },
    )
    assert calls[0]["time_budget_s"] == 240 and calls[0]["nthread"] == 2


@pytest.mark.parametrize("explicit", [False, True])
def test_warm_from_an_incomplete_multi_run_has_no_winner_to_seed(app, tmp_path, explicit):
    write_panel(tmp_path, weeks=60)
    run = make_saved_run(app)
    payload = (
        make_payload(warm_from=run.name)
        if explicit
        else {"spec": make_spec(), "budget": {}, "warm_from": run.name}
    )
    with pytest.raises(RequestError, match="no saved winner") as caught:
        post(app, payload=payload)
    assert caught.value.status == 400


@pytest.mark.parametrize("name,seeded", [("ridge", True), ("snaive", False)])
def test_warm_worker_seeds_only_the_winner_family(tmp_path, monkeypatch, name, seeded):
    from smolsmort.forecast import family_worker

    source = tmp_path / "previous"
    write_json(source / "recipe.json", {"genome": {"family": "ridge"}})
    for family in ("ridge", "snaive"):
        write_json(source / "families" / family / "population.json", [{"family": family}])
    run = tmp_path / "run"
    write_json(
        run / "request.json",
        {
            "version": 2,
            **make_payload(families=[{"name": "ridge"}, {"name": "snaive"}]),
            "warm_from": str(source),
            "prepared": str(tmp_path),
            "summary": {},
        },
    )
    monkeypatch.setattr(family_worker, "load_workspace", lambda *args: SimpleNamespace())
    calls = []

    def record_search(ws, budget, **kwargs):
        calls.append(kwargs["warm"])
        return {
            "leaderboard": [{"fitness": 0.2}],
            "best": {},
            "population": [],
            "elapsed": 0,
            "stopped": "plateau",
        }

    monkeypatch.setattr(family_worker, "search", record_search)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert family_worker.run(run, name) == 0
    expected = [[{"family": "ridge"}]] if seeded else [None]
    assert calls == expected


def wait_for_state(app, run_id, timeout=110):
    until = time.monotonic() + timeout
    samples = []
    event_cursor = -1
    while time.monotonic() < until:
        state = get(app, id=run_id, since_events=event_cursor)
        samples.append(state)
        event_cursor = state["events_cursor"]
        if state["state"] in ("done", "failed", "cancelled") and not state["alive"]:
            assert state["state"] == "done", state
            if not state["events"]:
                return state, samples
            continue
        time.sleep(0.25)
    pytest.fail(f"run {run_id} did not finish within {timeout}s")


@pytest.mark.parametrize("all_available", [False, True])
def test_posted_multi_run_finishes_reports_families_and_lists_as_multi(
    app, tmp_path, all_available
):
    pytest.importorskip("duckdb")
    if all_available:
        names = [family.name for family in registered_families() if family.available()[0]]
    else:
        pytest.importorskip("xgboost")
        pytest.importorskip("statsforecast")
        names = ["xgboost", "ridge", "ets", "snaive"]
    from test_forecast_coordinator import make_trending_panel

    make_trending_panel(tmp_path)
    payload = make_payload(
        families=[{"name": name} for name in names],
        time_budget_s=90 if all_available else 60,
        nthread=4,
    )
    started = post(app, payload=payload)
    try:
        complete, samples = wait_for_state(app, started["run_id"])
        assert set(complete["families"]) == set(names)
        assert any(sample.get("phase") == "broad" for sample in samples)
        for name, status in complete["families"].items():
            assert status["best_error"] is not None, (name, status)
            assert status["candidates_done"] >= 1, (name, status)
        assert {entry["family"] for entry in complete["leaderboard"]} <= set(names)
        summary = get(app, "/api/forecast-runs")["runs"][0]
        assert summary["kind"] == "multi" and set(summary["families"]) == set(names)
        assert summary["winner_family"] in (*names, "ensemble")
        unchanged = get(
            app,
            id=started["run_id"],
            since_generation=complete["generation"],
            since_events=complete["events_cursor"],
            since_leaderboard=complete["leaderboard_generation"],
        )
        assert "leaderboard" not in unchanged
        assert unchanged["generation_curves"] == unchanged["events"] == []
        if not all_available:
            winner = views.winning_genome(tmp_path / ".forecast-runs" / started["run_id"])["family"]
            refit = post(
                app, "/api/forecast-refit", {"spec": make_spec(), "run_id": started["run_id"]}
            )
            refitted, _ = wait_for_state(app, refit["run_id"], timeout=30)
            assert refitted["kind"] == "refit" and refitted["winner_family"] == winner
            assert list(refitted["families"]) == [winner]
            assert refitted["families"][winner]["state"] == "done"
    finally:
        post(app, "/api/forecast-cancel", {"id": started["run_id"]})


def test_cancel_route_leaves_no_coordinator_family_or_candidate_process(app, tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("xgboost")
    from test_forecast_coordinator import make_trending_panel, process_alive, wait_for

    from smolsmort.forecast.processes import read_run_pids

    make_trending_panel(tmp_path)
    started = post(
        app,
        payload=make_payload(
            families=[{"name": "xgboost"}, {"name": "ridge"}], time_budget_s=120, nthread=2
        ),
    )
    run = tmp_path / ".forecast-runs" / started["run_id"]
    try:
        wait_for(lambda: len(list(run.glob("families/*/candidate_pid.json"))) == 2)
        pids = read_run_pids(run) | {int((run / "pid").read_text())}
        assert len(pids) >= 5
        assert post(app, "/api/forecast-cancel", {"id": run.name}) == {"ok": True}
        assert get(app, id=run.name)["state"] == "cancelled"
        assert not any(process_alive(pid) for pid in pids)
    finally:
        post(app, "/api/forecast-cancel", {"id": run.name})


@pytest.mark.parametrize("ignore_term", [False, True])
def test_cancel_fallback_stops_unrecorded_native_refit_descendants(app, tmp_path, ignore_term):
    from test_forecast_coordinator import process_alive, wait_for

    run = make_saved_run(app, recipe={"family": "ridge"})
    script = run / "blocked_refit.py"
    child_code = (
        "import signal, time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + "time.sleep(60)"
    )
    script.write_text(
        "import json, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "def stop_parent(*args):\n"
        "    child.wait(timeout=2)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop_parent)\n"
        "pid_path = root / 'native-child.pid.tmp'\n"
        "pid_path.write_text(str(child.pid))\n"
        "pid_path.replace(root / 'native-child.pid')\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen([sys.executable, str(script), str(run)], start_new_session=True)
    runs._PROCESSES[process.pid] = process
    (run / "pid").write_text(str(process.pid))
    write_json(run / "status.json", {"state": "running", "pid": process.pid})
    try:
        wait_for(lambda: (run / "native-child.pid").exists())
        child_pid = int((run / "native-child.pid").read_text())
        post(app, "/api/forecast-cancel", {"id": run.name})
        process.wait(timeout=2)
        assert not process_alive(process.pid)
        assert not process_alive(child_pid)
    finally:
        if process.poll() is None:
            from smolsmort.forecast.processes import terminate_run_tree

            terminate_run_tree(process.pid, force=True)
            process.wait(timeout=2)


def test_windows_tree_cleanup_uses_taskkill_without_posix_signals(monkeypatch):
    from smolsmort.forecast import processes

    calls = []
    monkeypatch.setattr(processes.sys, "platform", "win32")
    monkeypatch.setattr(
        processes.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    processes.terminate_run_tree(123, force=True)
    assert calls[0][0] == ["taskkill", "/PID", "123", "/T", "/F"]
    assert calls[0][1]["timeout"] == 5
