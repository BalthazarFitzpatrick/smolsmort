"""selected horizon views use synthetic saved rows, never a forecast worker"""

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import views
from smolsmort.forecast.tab import forecast_tab
from smolsmort.forecast.tables import read_table, write_table
from smolsmort.review.routes import RequestError

# parquet fixtures need duckdb, which ci does not install
pytest.importorskip("duckdb")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_predictions(steps, leads, prediction, actual=None, naive=None):
    steps = np.array(steps, dtype="datetime64[D]")
    leads = np.array(leads, dtype=int)
    prediction = np.array(prediction, dtype=float)
    table = {
        "series": np.array(["synthetic"] * len(steps)),
        "step": steps,
        "origin": steps - (7 * leads).astype("timedelta64[D]"),
        "weeks_ahead": leads,
        "bucket": np.array(["h1"] * len(steps)),
        "prediction": prediction,
        "lower": prediction - 2,
        "upper": prediction + 2,
    }
    if actual is not None:
        table["actual"] = np.array(actual, dtype=float)
    if naive is not None:
        table["naive_prediction"] = np.array(naive, dtype=float)
    return table


@pytest.fixture
def horizon_run(tmp_path):
    run = tmp_path / ".forecast-runs" / "synthetic-horizon"
    prepared = tmp_path / ".forecast-cache" / "synthetic"
    prepared.mkdir(parents=True)
    request = {
        "version": 2,
        "prepared": str(prepared),
        "families": [{"name": "ridge"}, {"name": "snaive"}],
        "spec": {
            "mode": "timeseries",
            "task": "regression",
            "step": "week",
            "horizon": 3,
            "horizon_mode": "single",
            "columns": [
                {"name": "day", "role": "time"},
                {"name": "group", "role": "dimension"},
                {"name": "units", "role": "target", "aggregation": "sum"},
            ],
        },
    }
    verdict = {"accuracy": {"ratio": 0.75}, "summary": "synthetic pooled verdict"}
    result = {
        "val_error": 0.25,
        "verdict": verdict,
        "winner": {"family": "ridge", "fitness": 0.25, "verdict": verdict},
        "ensemble": {"members": ["ridge", "snaive"], "fitness": 0.3, "verdict": verdict},
        "families": {"ridge": {"fitness": 0.25}, "snaive": {"fitness": 0.4}},
    }
    write_json(run / "request.json", request)
    write_json(run / "result.json", result)
    write_json(run / "status.json", {"state": "done", "families": {}})
    write_json(run / "recipe.json", {"family": "ridge", "genome": {"family": "ridge"}})
    panel_steps = np.arange(
        np.datetime64("2025-01-06"), np.datetime64("2025-02-18"), np.timedelta64(7, "D")
    )
    write_table(
        prepared / "panel.parquet",
        {
            "__series": np.array(["synthetic"] * len(panel_steps)),
            "__step": panel_steps,
            "group": np.array(["alpha"] * len(panel_steps)),
            "units": np.array([2, 4, 6, 8, 10, 20, 30], dtype=float),
        },
    )
    test = make_predictions(
        ["2025-02-03"] * 3 + ["2025-02-10"] * 2 + ["2025-02-17"],
        [1, 2, 3, 1, 2, 1],
        [12, 16, 20, 24, 28, 36],
        actual=[10, 10, 10, 20, 20, 30],
        naive=[8, 6, 4, 16, 12, 24],
    )
    validation = make_predictions(
        ["2025-01-27"] * 3, [1, 2, 3], [7, 9, 11], actual=[8] * 3, naive=[6, 4, 2]
    )
    future = make_predictions(["2025-02-24", "2025-03-03", "2025-03-10"], [1, 2, 3], [40, 50, 60])
    for kind, table in (("test", test), ("validation", validation), ("forecast", future)):
        write_table(run / f"{kind}.parquet", table)
        write_table(run / f"winner_{kind}.parquet", table)
        ensemble = {**table, "prediction": table["prediction"] + 2}
        write_table(run / f"ensemble_{kind}.parquet", ensemble)
    family_dir = run / "families" / "snaive"
    family_dir.mkdir(parents=True)
    write_table(family_dir / "test.parquet", {**test, "prediction": test["naive_prediction"]})
    app = SimpleNamespace(state=SimpleNamespace(bases={"forecast": str(tmp_path)}))
    return run, request, app


def fetch_view(horizon_run, **query):
    run, request, _ = horizon_run
    return views.view(run, request, query)


def forecast_values(payload):
    line = next(line for line in payload["series"] if line["id"] == "forecast")
    return dict(zip(payload["x"], line["values"], strict=True))


def metric_values(payload):
    return {name: item["value"] for name, item in payload["error_metrics"].items()}


def test_default_is_longest_lead_without_averaging(horizon_run):
    payload = fetch_view(horizon_run)
    assert payload["view"] == {"mode": "lead", "n": 3}
    assert payload["horizon"] == 3
    values = forecast_values(payload)
    assert values["2025-01-27"] == 11
    assert values["2025-02-03"] == 20
    assert values["2025-02-10"] is None
    assert values["2025-02-17"] is None
    metrics = metric_values(payload)
    assert metrics["wape"] == 1
    assert metrics["mae"] == 10
    assert metrics["mape"] == 1
    assert metrics["test_ratio"] == pytest.approx(5 / 3)


@pytest.mark.parametrize(
    "lead,prediction,wape,mae,mape,ratio",
    [(1, 12, 0.2, 4, 0.2, 1), (2, 16, 14 / 30, 7, 0.5, 7 / 6)],
)
def test_selected_lead_drives_chart_and_all_test_metrics(
    horizon_run, lead, prediction, wape, mae, mape, ratio
):
    payload = fetch_view(horizon_run, lead=str(lead))
    assert payload["view"] == {"mode": "lead", "n": lead}
    assert forecast_values(payload)["2025-02-03"] == prediction
    values = metric_values(payload)
    for key, expected in (("wape", wape), ("mae", mae), ("mape", mape), ("test_ratio", ratio)):
        assert values[key] == pytest.approx(expected)


def test_blend_metrics_score_target_weeks_once_and_keep_pooled_verdict(horizon_run):
    payload = fetch_view(horizon_run, blend="2")
    assert payload["view"] == {"mode": "blend", "n": 2, "strength": 1}
    predictions = forecast_values(payload)
    assert [
        predictions[day] for day in ("2025-02-03", "2025-02-10", "2025-02-17")
    ] == pytest.approx([16, 26, 36])
    values = metric_values(payload)
    assert values["wape"] == pytest.approx(18 / 60)
    assert values["mae"] == pytest.approx(6)
    assert values["mape"] == pytest.approx(11 / 30)
    assert values["test_ratio"] == pytest.approx(18 / 16)
    assert values["validation_error"] == 0.25
    assert payload["verdict"]["accuracy"]["ratio"] == 0.75
    assert payload["winner"]["verdict"]["accuracy"]["ratio"] == 0.75
    assert payload["bands"] == []
    json.dumps(payload, allow_nan=False)


def test_blend_centre_does_not_apply_an_unrequested_decay(horizon_run):
    around_one = fetch_view(horizon_run, blend="1")
    around_three = fetch_view(horizon_run, blend="3")
    assert around_one["series"] == around_three["series"]
    assert metric_values(around_one) == metric_values(around_three)


@pytest.mark.parametrize("strength", [1, 2, 4, 8])
def test_blend_strength_selects_weights_and_labels(horizon_run, strength):
    payload = fetch_view(horizon_run, blend="2", strength=str(strength))
    assert payload["view"] == {"mode": "blend", "n": 2, "strength": strength}
    values = forecast_values(payload)
    assert values["2025-02-03"] == pytest.approx(16)
    assert values["2025-02-10"] == pytest.approx((24 + strength * 28) / (1 + strength))
    assert values["2025-02-17"] == 36
    for metric in payload["metric_registry"]:
        if metric["scope"] == "test":
            assert metric["label"].endswith(f"blend week 2 x{strength}")
    assert payload["bands"] == []


@pytest.mark.parametrize("model", ["winner", "ensemble"])
@pytest.mark.parametrize("selection", [{"lead": "1"}, {"blend": "2", "strength": "4"}])
def test_horizon_metrics_belong_only_to_the_selected_model(horizon_run, model, selection):
    run, _, _ = horizon_run
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    result["families"]["ridge"]["fitness"] = 0.27
    write_json(run / "result.json", result)
    payload = fetch_view(horizon_run, model=model, **selection)
    blended = "blend" in selection
    wape = {"winner": 0.32, "ensemble": 0.42} if blended else {"winner": 0.2, "ensemble": 0.3}
    ratio = {"winner": 48 / 43, "ensemble": 63 / 43} if blended else {"winner": 1, "ensemble": 1.5}
    assert metric_values(payload)["wape"] == pytest.approx(wape[model])
    assert metric_values(payload)["test_ratio"] == pytest.approx(ratio[model])
    assert payload["error_metrics"] == payload[model]["error_metrics"]
    for name, entry in {**payload["families"], "ensemble": payload["ensemble"]}.items():
        selected = name == ("ridge" if model == "winner" else "ensemble")
        for metric in ("wape", "mae", "mape", "test_ratio"):
            saved = entry["error_metrics"][metric]
            if selected:
                assert saved == payload["error_metrics"][metric]
            else:
                assert saved["value"] is None
                assert saved["reason"] == "not computed for this view"
        assert (
            entry["error_metrics"]["validation_error"]["value"]
            == {
                "ridge": 0.27,
                "snaive": 0.4,
                "ensemble": 0.3,
            }[name]
        )
    if model == "ensemble":
        assert payload["winner"]["error_metrics"]["wape"]["value"] is None
        assert payload["winner"]["error_metrics"]["wape"]["reason"] == "not computed for this view"
    assert payload["verdict"]["accuracy"]["ratio"] == 0.75
    assert payload["winner"]["verdict"]["accuracy"]["ratio"] == 0.75
    if blended:
        assert payload["error_metrics"]["wape"]["note"] == "blend week 2 x4"


@pytest.mark.parametrize("selection", [{"lead": "1"}, {"blend": "2", "strength": "4"}])
@pytest.mark.parametrize("breakdown", [None, "group"])
def test_chart_labels_identify_actual_and_forecast(horizon_run, selection, breakdown):
    query = {**selection, **({"breakdown": breakdown} if breakdown else {})}
    payload = fetch_view(horizon_run, **query)
    labels = [line["label"] for line in payload["series"]]
    assert labels == (["actual: alpha", "forecast: alpha"] if breakdown else ["actual", "forecast"])


def test_blend_strength_table_and_export_match_chart(horizon_run):
    run, request, _ = horizon_run
    query = {"blend": "2", "strength": "4", "size": "100"}
    payload = views.timeseries_table(run, request, query)
    rows = [dict(zip(payload["columns"], row, strict=True)) for row in payload["rows"]]
    historical = sorted((row for row in rows if not row["future"]), key=lambda row: row["step"])
    assert [row["prediction"] for row in historical] == pytest.approx([9, 16, 27.2, 36])
    assert all(row["lower"] is None and row["upper"] is None for row in rows)
    export = views.export_csv(run, request, query).decode("utf-8")
    exported = csv.DictReader(io.StringIO(export.partition("\n")[2]))
    target = next(row for row in exported if row["step"] == "2025-02-10")
    assert float(target["prediction"]) == pytest.approx(27.2)


def test_explicit_weights_normalize_over_each_target_tail(horizon_run):
    payload = fetch_view(horizon_run, blend="2", weights=json.dumps({1: 1, 2: 2, 3: 3}))
    assert payload["view"] == {"mode": "blend", "n": 2, "weights": {1: 1.0, 2: 2.0, 3: 3.0}}
    values = forecast_values(payload)
    assert values["2025-02-03"] == pytest.approx(52 / 3)
    assert values["2025-02-10"] == pytest.approx(80 / 3)
    assert values["2025-02-17"] == 36
    assert metric_values(payload)["wape"] == pytest.approx(1 / 3)
    assert metric_values(payload)["test_ratio"] == pytest.approx(15 / 13)
    for metric in payload["metric_registry"]:
        if metric["scope"] == "test":
            assert metric["label"].endswith("blend custom weights")
    assert payload["error_metrics"]["wape"]["note"] == "blend custom weights"
    assert payload["bands"] == []
    json.dumps(payload, allow_nan=False)


def test_saved_bounds_follow_only_the_selected_lead(horizon_run):
    payload = fetch_view(horizon_run, lead="2")
    band = next(band for band in payload["bands"] if band["series"] == "forecast")
    lower = dict(zip(payload["x"], band["lo"], strict=True))
    upper = dict(zip(payload["x"], band["hi"], strict=True))
    assert (lower["2025-01-27"], upper["2025-01-27"]) == (7, 11)
    assert (lower["2025-02-03"], upper["2025-02-03"]) == (14, 18)
    assert (lower["2025-02-10"], upper["2025-02-10"]) == (26, 30)
    assert lower["2025-02-17"] is None and upper["2025-02-17"] is None


@pytest.mark.parametrize(
    "mode,n,expected", [("lead", 2, [9, 16, 28]), ("blend", 2, [9, 16, 26, 36])]
)
def test_table_uses_same_selected_historical_rows(horizon_run, mode, n, expected):
    run, request, _ = horizon_run
    payload = views.timeseries_table(run, request, {mode: str(n), "size": "100"})
    rows = [dict(zip(payload["columns"], row, strict=True)) for row in payload["rows"]]
    historical = sorted((row for row in rows if not row["future"]), key=lambda row: row["step"])
    assert [row["prediction"] for row in historical] == pytest.approx(expected)
    if mode == "blend":
        assert all(row["lower"] is None and row["upper"] is None for row in rows)


def test_selected_model_metrics_match_its_chart_predictions(horizon_run):
    winner = fetch_view(horizon_run, lead="1")
    ensemble = fetch_view(horizon_run, model="ensemble", lead="1")
    assert metric_values(winner)["wape"] == pytest.approx(0.2)
    assert metric_values(ensemble)["wape"] == pytest.approx(0.3)
    assert forecast_values(winner)["2025-02-03"] == 12
    assert forecast_values(ensemble)["2025-02-03"] == 14


@pytest.mark.parametrize("route", ["/api/forecast-view", "/api/forecast-table"])
@pytest.mark.parametrize(
    "query",
    [
        {"lead": "0"},
        {"lead": "4"},
        {"lead": "1.5"},
        {"blend": "0"},
        {"blend": "4"},
        {"blend": "no"},
        {"lead": "1", "blend": "2"},
        {"lead": "1", "weights": '{"1": 1}'},
        {"blend": "1", "weights": '{"1": -1, "2": 1}'},
        {"blend": "1", "weights": '{"1": 0, "2": 0, "3": 0}'},
        {"blend": "1", "weights": '{"1": NaN}'},
        {"blend": "1", "weights": '{"1": Infinity}'},
        {"blend": "1", "weights": "[]"},
        {"blend": "1", "weights": "{"},
    ],
)
def test_invalid_selections_return_plain_400(horizon_run, route, query):
    run, _, app = horizon_run
    with pytest.raises(RequestError) as error:
        forecast_tab().get[route](app, {"id": run.name, **query})
    assert error.value.status == 400
    assert str(error.value) and "Traceback" not in str(error.value)


@pytest.mark.parametrize("route", ["/api/forecast-view", "/api/forecast-table"])
@pytest.mark.parametrize("strength", ["0", "-1", "3", "16", "1.0", "1.5", "no", "", True, None])
def test_invalid_strength_returns_plain_400(horizon_run, route, strength):
    run, _, app = horizon_run
    with pytest.raises(RequestError) as error:
        forecast_tab().get[route](app, {"id": run.name, "blend": "2", "strength": strength})
    assert error.value.status == 400
    assert str(error.value) == "strength must be 1, 2, 4, or 8"


@pytest.mark.parametrize("route", ["/api/forecast-view", "/api/forecast-table"])
@pytest.mark.parametrize(
    "query,message",
    [
        ({"lead": "2", "strength": "4"}, "strength requires blend"),
        (
            {"blend": "2", "strength": "4", "weights": '{"1": 1, "2": 4, "3": 1}'},
            "strength and weights are mutually exclusive",
        ),
    ],
)
def test_strength_rejects_ambiguous_selections(horizon_run, route, query, message):
    run, _, app = horizon_run
    with pytest.raises(RequestError) as error:
        forecast_tab().get[route](app, {"id": run.name, **query})
    assert error.value.status == 400
    assert str(error.value) == message


def test_naive_fallback_uses_each_origin_without_future_leakage(horizon_run):
    run, _, _ = horizon_run
    for name in ("test.parquet", "winner_test.parquet"):
        table = read_table(run / name)
        table.pop("naive_prediction")
        write_table(run / name, table)
    payload = fetch_view(horizon_run, lead="1")
    # origins hold 8, 10, 20: baseline errors 2 + 10 + 10 on the same targets
    assert metric_values(payload)["test_ratio"] == pytest.approx(12 / 22)


def test_naive_seasonal_fallback_matches_the_selected_model(horizon_run):
    run, request, _ = horizon_run
    panel_path = Path(request["prepared"]) / "panel.parquet"
    panel = read_table(panel_path)
    previous = {**panel, "__step": panel["__step"] - np.timedelta64(364, "D")}
    previous["units"] = np.array([1, 2, 3, 4, 6, 8, 10], dtype=float)
    write_table(
        panel_path,
        {name: np.concatenate([previous[name], values]) for name, values in panel.items()},
    )
    for name in ("test.parquet", "winner_test.parquet", "families/snaive/test.parquet"):
        table = read_table(run / name)
        table.pop("naive_prediction")
        write_table(run / name, table)
    payload = fetch_view(horizon_run, lead="1")
    # annual lags hold 6, 8, 10: baseline errors 4 + 12 + 20 for the selected model
    assert metric_values(payload)["test_ratio"] == pytest.approx(12 / 36)
    assert payload["families"]["snaive"]["error_metrics"]["test_ratio"]["value"] is None


def test_filtered_metrics_and_chart_share_the_selected_series(horizon_run):
    run, request, _ = horizon_run
    panel_path = Path(request["prepared"]) / "panel.parquet"
    panel = read_table(panel_path)
    other_panel = {
        **panel,
        "__series": np.array(["other"] * len(panel["__series"])),
        "group": np.array(["beta"] * len(panel["__series"])),
        "units": panel["units"] * 10,
    }
    write_table(
        panel_path,
        {name: np.concatenate([values, other_panel[name]]) for name, values in panel.items()},
    )
    for name in ("test.parquet", "winner_test.parquet"):
        table = read_table(run / name)
        other = {
            **table,
            "series": np.array(["other"] * len(table["series"])),
            "actual": table["actual"] * 10,
            "prediction": table["actual"] * 10,
            "naive_prediction": table["actual"] * 5,
        }
        write_table(
            run / name, {key: np.concatenate([values, other[key]]) for key, values in table.items()}
        )
    payload = fetch_view(horizon_run, lead="1", filters=json.dumps({"group": ["alpha"]}))
    assert forecast_values(payload)["2025-02-03"] == 12
    assert metric_values(payload)["wape"] == pytest.approx(0.2)
    assert metric_values(payload)["test_ratio"] == 1
    other = fetch_view(horizon_run, lead="1", filters=json.dumps({"group": ["beta"]}))
    assert forecast_values(other)["2025-02-03"] == 100
    assert metric_values(other)["wape"] == 0


@pytest.mark.parametrize("source", ["json", "parquet"])
def test_saved_lead_errors_are_visible_without_recomputing_them(horizon_run, source):
    run, _, _ = horizon_run
    rows = [
        {"weeks_ahead": k, "model_wape": k / 10, "naive_wape": 0.4, "ratio": k / 4, "rows": 4 - k}
        for k in (1, 2, 3)
    ]
    if source == "json":
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        write_json(run / "result.json", {**result, "lead_errors": rows})
    else:
        write_table(
            run / "lead_errors.parquet",
            {name: np.array([row[name] for row in rows]) for name in rows[0]},
        )
    payload = fetch_view(horizon_run, blend="2")
    assert payload["lead_errors"] == rows


def test_missing_lead_errors_are_optional(horizon_run):
    assert not fetch_view(horizon_run).get("lead_errors")


@pytest.mark.parametrize("source", ["json", "parquet"])
def test_upstream_lead_error_shapes_show_test_phase(horizon_run, source):
    run, _, _ = horizon_run
    if source == "json":
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        result["lead_errors"] = {"1": {"model": 0.1, "naive": 0.2, "rows": 3}}
        write_json(run / "result.json", result)
    else:
        write_table(
            run / "lead_errors.parquet",
            {
                "phase": np.array(["test", "validation"]),
                "weeks_ahead": np.array([1, 1]),
                "model_wape": np.array([0.1, 0.9]),
                "naive_wape": np.array([0.2, 0.3]),
                "ratio": np.array([0.5, 3]),
                "rows": np.array([3, 9]),
            },
        )
    errors = fetch_view(horizon_run)["lead_errors"]
    assert len(errors) == 3
    assert errors[0]["model_wape"] == 0.1
    assert errors[0]["naive_wape"] == 0.2
    assert errors[0]["ratio"] == 0.5
    assert errors[0]["rows"] == 3


def test_selected_family_metrics_do_not_mutate_tile_metrics(horizon_run):
    payload = fetch_view(horizon_run, lead="1")
    assert payload["families"]["snaive"]["error_metrics"]["wape"]["value"] is None
    assert payload["tiles"]["snaive"]["error_metrics"]["wape"]["value"] == 0.3


def test_blended_test_metrics_include_validation_forecasts_of_test_targets(horizon_run):
    run, _, _ = horizon_run
    validation = read_table(run / "winner_validation.parquet")
    overlap = make_predictions(["2025-02-17"], [3], [0], actual=[30], naive=[0])
    combined = {
        name: np.concatenate([values, overlap[name]]) for name, values in validation.items()
    }
    write_table(run / "winner_validation.parquet", combined)
    payload = fetch_view(horizon_run, blend="2")
    assert forecast_values(payload)["2025-02-17"] == 18
    assert metric_values(payload)["wape"] == pytest.approx(24 / 60)
    assert metric_values(payload)["test_ratio"] == pytest.approx(24 / 28)
    assert metric_values(fetch_view(horizon_run, lead="1"))["wape"] == pytest.approx(0.2)


def test_nullable_lead_errors_are_json_safe(horizon_run):
    run, _, _ = horizon_run
    write_table(
        run / "lead_errors.parquet",
        {
            "weeks_ahead": np.array([1]),
            "model_wape": np.array([0.1]),
            "naive_wape": np.array([0.0]),
            "ratio": np.array([np.nan]),
            "rows": np.array([3]),
        },
    )
    payload = fetch_view(horizon_run)
    assert payload["lead_errors"][0]["ratio"] is None
    json.dumps(payload, allow_nan=False)


def test_lead_errors_follow_the_selected_model(horizon_run):
    run, _, _ = horizon_run
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    result["lead_errors"] = [{"weeks_ahead": 1, "model_wape": 0.9}]
    write_json(run / "result.json", result)
    write_json(run / "recipe.json", {"family": "ensemble"})
    assert fetch_view(horizon_run, model="winner")["lead_errors"] == []
    write_table(
        run / "winner_lead_errors.parquet",
        {
            "weeks_ahead": np.array([1]),
            "model_wape": np.array([0.2]),
        },
    )
    assert fetch_view(horizon_run, model="winner")["lead_errors"][0]["model_wape"] == 0.2
    assert fetch_view(horizon_run, model="ensemble")["lead_errors"][0]["model_wape"] == 0.9


def test_legacy_tables_leave_existing_view_and_metrics_unchanged(horizon_run):
    run, request, _ = horizon_run
    request["spec"].pop("horizon_mode")
    for path in run.glob("*.parquet"):
        table = read_table(path)
        table.pop("weeks_ahead", None)
        table.pop("origin", None)
        write_table(path, table)
    payload = views.view(run, request, {})
    assert not payload.get("view")
    assert not payload.get("horizon")
    assert forecast_values(payload)["2025-02-03"] == 16
    assert metric_values(payload)["wape"] == pytest.approx(36 / 100)
    assert metric_values(payload)["test_ratio"] == 0.75
    assert payload["families"]["snaive"]["error_metrics"]["wape"]["value"] == 0.3
    assert payload["ensemble"]["error_metrics"]["wape"]["value"] == pytest.approx(48 / 100)
    assert [line["label"] for line in payload["series"]] == ["actual", "forecast"]
