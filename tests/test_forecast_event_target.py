"""event dates become duration bounds without changing saved duration specs or aft consumers"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, replace
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast.prep import prepare
from smolsmort.forecast.spec import (
    LOWER,
    ROW,
    UPPER,
    Censor,
    Column,
    PrepSpec,
    SpecError,
    spec_from_dict,
)
from smolsmort.forecast.splits import labels_as_of
from smolsmort.forecast.tab import _parse_spec, forecast_tab
from smolsmort.forecast.tables import read_table
from smolsmort.review.routes import RequestError


def make_spec(source, **changes):
    spec = PrepSpec(
        str(source),
        "entity",
        "regression",
        (Column("created", "anchor"), Column("size", "measure"), Column("completed", "target")),
        censor=Censor("created", as_of="2024-01-29", event="completed"),
    )
    return replace(spec, **changes)


@pytest.fixture
def event_spec(tmp_path):
    source = tmp_path / "events.csv"
    source.write_text(
        "created,size,completed\n"
        "2024-01-01,1,2024-01-22\n"
        "2024-01-08,2,2024-01-08\n"
        "2024-01-01,3,\n"
        "2024-01-08,4,2024-02-05\n"
        "2024-01-15,5,2024-01-14\n",
        encoding="utf-8",
    )
    return make_spec(source)


def test_event_dates_become_bounds_and_future_events_stay_open(tmp_path, event_spec):
    pytest.importorskip("duckdb")
    prepared = prepare(event_spec, tmp_path / "cache")
    rows = read_table(prepared.folder / "rows.parquet", order_by=ROW)
    np.testing.assert_array_equal(rows[ROW], [0, 1, 2, 3])
    np.testing.assert_array_equal(rows[LOWER], [3, 0, 4, 3])
    np.testing.assert_array_equal(rows[UPPER], [3, 0, np.inf, np.inf])
    np.testing.assert_array_equal(rows["completed"], [3, 0, np.nan, np.nan])
    predicted = read_table(prepared.folder / "predict.parquet", order_by=ROW)
    np.testing.assert_array_equal(predicted[ROW], [2, 3])
    expected = {
        "source_rows": 5,
        "rows_after_where": 5,
        "dropped_null_anchor": 0,
        "rows": 4,
        "known_rows": 2,
        "open_rows": 2,
        "predict_rows": 2,
        "refused_event_before_start": 1,
        "as_of": "2024-01-29",
    }
    assert {name: prepared.summary[name] for name in expected} == expected


@pytest.mark.parametrize(
    ("unit", "days"),
    [("day", 1), ("week", 7), ("month", 30.4375), ("quarter", 91.3125), ("year", 365.25)],
)
def test_event_units_keep_fractions_and_events_on_as_of_are_closed(tmp_path, unit, days):
    pytest.importorskip("duckdb")
    source = tmp_path / "events.csv"
    source.write_text(
        "created,size,completed\n2024-01-01,1,2024-01-03\n2024-01-01,2,\n2024-01-01,3,2024-01-04\n",
        encoding="utf-8",
    )
    spec = make_spec(source, censor=Censor("created", unit, "2024-01-04", "completed"))
    prepared = prepare(spec, tmp_path / "cache")
    rows = read_table(prepared.folder / "rows.parquet", order_by=ROW)
    np.testing.assert_allclose(rows[LOWER], [2 / days, 3 / days, 3 / days])
    np.testing.assert_allclose(rows[UPPER], [2 / days, np.inf, 3 / days])
    np.testing.assert_allclose(rows["completed"], [2 / days, np.nan, 3 / days])


def test_default_as_of_remains_the_latest_anchor_before_event_refusals(tmp_path):
    pytest.importorskip("duckdb")
    source = tmp_path / "events.csv"
    source.write_text(
        "created,size,completed\n2024-01-01,1,2024-01-08\n2024-01-15,2,\n2024-01-29,3,2024-01-28\n",
        encoding="utf-8",
    )
    spec = make_spec(source, censor=Censor("created", event="completed"))
    prepared = prepare(spec, tmp_path / "cache")
    rows = read_table(prepared.folder / "rows.parquet", order_by=ROW)
    assert prepared.summary["as_of"] == "2024-01-29"
    assert prepared.summary["refused_event_before_start"] == 1
    np.testing.assert_array_equal(rows[LOWER], [1, 2])
    np.testing.assert_array_equal(rows[UPPER], [1, np.inf])


def test_event_refusals_keep_filters_null_anchor_counts_and_source_row_ids(tmp_path, event_spec):
    pytest.importorskip("duckdb")
    source = tmp_path / "events.csv"
    source.write_text(source.read_text(encoding="utf-8") + ",6,\n", encoding="utf-8")
    spec = replace(event_spec, where="size >= 2", predict_where="size = 4")
    prepared = prepare(spec, tmp_path / "cache")
    rows = read_table(prepared.folder / "rows.parquet", order_by=ROW)
    predicted = read_table(prepared.folder / "predict.parquet", order_by=ROW)
    np.testing.assert_array_equal(rows[ROW], [1, 2, 3])
    np.testing.assert_array_equal(predicted[ROW], [3])
    assert prepared.summary["rows_after_where"] == 5
    assert prepared.summary["dropped_null_anchor"] == 1
    assert prepared.summary["refused_event_before_start"] == 1
    assert prepared.summary["rows"] == 3
    assert prepared.summary["known_rows"] == 1
    assert prepared.summary["open_rows"] == 2
    assert prepared.summary["predict_rows"] == 1


@pytest.mark.parametrize("as_of", [None, "2024-01-29"])
def test_default_event_keeps_the_prepared_cache_key_and_duration_bounds(tmp_path, as_of):
    pytest.importorskip("duckdb")
    source = tmp_path / "durations.csv"
    source.write_text(
        "created,size,completed\n2024-01-01,1,3\n2024-01-08,2,0\n2024-01-01,3,\n2024-01-29,4,1.5\n",
        encoding="utf-8",
    )
    spec = make_spec(source, censor=Censor("created", as_of=as_of))
    legacy = asdict(spec)
    legacy["mode"] = "row"
    legacy.pop("holidays_country")
    legacy["censor"].pop("event")
    # pre-flag specs never carried skip_missing or pin
    for column in legacy["columns"]:
        column.pop("skip_missing")
        column.pop("pin")
    digest = hashlib.sha256(json.dumps(legacy, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]
    assert spec.censor.event is None
    assert spec.digest() == digest
    assert spec_from_dict(legacy) == spec
    assert spec_from_dict(legacy).digest() == digest
    assert replace(spec, censor=replace(spec.censor, event="completed")).digest() != digest
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    prepared = prepare(spec, tmp_path / "cache")
    assert prepared.folder.name == f"{digest}-{source_digest}"
    rows = read_table(prepared.folder / "rows.parquet", order_by=ROW)
    np.testing.assert_array_equal(rows[LOWER], [3, 0, 4, 1.5])
    np.testing.assert_array_equal(rows[UPPER], [3, 0, np.inf, 1.5])
    assert "refused_event_before_start" not in prepared.summary
    explicit_none = replace(spec, censor=replace(spec.censor, event=None))
    reused = prepare(explicit_none, tmp_path / "cache")
    assert reused.reused
    assert reused.folder == prepared.folder
    assert reused.summary == prepared.summary


@pytest.mark.parametrize("event", ["size", "missing", "", 42, []])
def test_event_must_name_the_target_column(event_spec, event):
    with pytest.raises(SpecError, match=r"censor\.event must name the target column"):
        replace(event_spec, censor=replace(event_spec.censor, event=event))


@pytest.mark.parametrize("mode", ["timeseries", "series"])
def test_timeseries_cannot_use_an_event_target(event_spec, mode):
    with pytest.raises(SpecError, match=r"censor\.event is only supported in entity mode"):
        replace(
            event_spec,
            mode=mode,
            columns=(Column("created", "time"), Column("completed", "target", "max")),
        )


def test_event_target_requires_regression_and_the_declared_start(event_spec):
    with pytest.raises(SpecError, match=r"censor\.event requires regression"):
        replace(event_spec, task="classification")
    with pytest.raises(SpecError, match=r"censor\.anchor must name the anchor column"):
        replace(event_spec, censor=replace(event_spec.censor, anchor="size"))


def test_event_spec_round_trips_and_prepares_through_the_route(tmp_path, event_spec):
    pytest.importorskip("duckdb")
    app = SimpleNamespace(state=SimpleNamespace(bases={"forecast": str(tmp_path)}))
    payload = {"spec": {**asdict(event_spec), "source": "events.csv"}}
    assert spec_from_dict(json.loads(json.dumps(asdict(event_spec)))) == event_spec
    assert _parse_spec(app, payload) == event_spec
    response = forecast_tab().post["/api/forecast-prep"](app, payload)
    assert response["summary"]["refused_event_before_start"] == 1
    assert response["summary"]["predict_rows"] == 2
    assert not response["reused"]


@pytest.mark.parametrize("route", ["/api/forecast-prep", "/api/forecast-runs"])
@pytest.mark.parametrize("invalid", ["event", "mode", "task", "anchor"])
def test_event_route_validation_errors_name_the_problem(tmp_path, event_spec, route, invalid):
    app = SimpleNamespace(state=SimpleNamespace(bases={"forecast": str(tmp_path)}))
    data = asdict(event_spec)
    message = r"censor\.event must name the target column"
    if invalid == "event":
        data["censor"]["event"] = "size"
    elif invalid == "mode":
        data["mode"] = "timeseries"
        data["columns"] = [
            {"name": "created", "role": "time"},
            {"name": "completed", "role": "target", "aggregation": "max"},
        ]
        message = r"censor\.event is only supported in entity mode"
    elif invalid == "task":
        data["task"] = "classification"
        message = r"censor\.event requires regression"
    else:
        data["censor"]["anchor"] = "size"
        message = r"censor\.anchor must name the anchor column"
    with pytest.raises(RequestError, match=message) as error:
        forecast_tab().post[route](app, {"spec": data})
    assert error.value.status == 400


def test_event_bounds_recensor_at_an_earlier_cut(tmp_path, event_spec):
    pytest.importorskip("duckdb")
    prepared = prepare(event_spec, tmp_path / "cache")
    rows = read_table(prepared.folder / "rows.parquet", order_by=ROW)
    lower, upper = labels_as_of(rows["__anchor"], rows[LOWER], rows[UPPER], dt.date(2024, 1, 15), 7)
    np.testing.assert_array_equal(lower, [2, 0, 2, 1])
    np.testing.assert_array_equal(upper, [np.inf, 0, np.inf, np.inf])


def test_event_target_runs_a_short_aft_fit_through_load_workspace(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    pytest.importorskip("xgboost")
    from smolsmort.forecast import pipeline

    source = tmp_path / "events.csv"
    lines = ["created,size,completed"]
    for index in range(104):
        anchor = dt.date(2024, 1, 1) + dt.timedelta(weeks=index)
        event = anchor + dt.timedelta(weeks=1 + index % 5)
        if index % 7 == 0:
            event = ""
        elif index % 11 == 0:
            event = anchor
        lines.append(f"{anchor},{1 + index % 3},{event}")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    spec = make_spec(source, censor=Censor("created", as_of="2026-01-05", event="completed"))
    prepared = prepare(spec, tmp_path / "cache")
    workspace = pipeline.load_workspace(spec, prepared.folder, prepared.summary)
    assert workspace.predict_rows.sum() == prepared.summary["open_rows"] > 0
    assert all(mask.any() for mask in workspace.masks.values())
    assert np.any(workspace.table[LOWER] == 0)
    monkeypatch.setattr(
        pipeline, "fit_model", partial(pipeline.fit_model, max_rounds=3, patience=0)
    )
    genome = pipeline.make_genome(workspace.families, "aft", {"max_depth": 2, "eta": 0.1})
    result = pipeline.score(workspace, genome, nthread=1)
    assert np.isfinite(result.fitness)
    assert result.prediction.size > 0
    assert np.isfinite(result.prediction).all()
    assert (result.prediction > 0).all()
