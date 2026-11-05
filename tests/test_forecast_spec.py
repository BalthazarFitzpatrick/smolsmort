"""the forecast contract refuses a spec that cannot be prepared, and the fixtures hold their shape"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace

import pytest
from forecast_fixtures import write_panel, write_rows

from smolsmort.forecast.spec import Censor, Column, PrepSpec, SpecError, spec_from_dict


def _entity_spec(**changes):
    base = {
        "source": "rows.csv",
        "mode": "entity",
        "task": "regression",
        "columns": (
            Column("created", "anchor"),
            Column("branch", "dimension"),
            Column("leak_plus1", "measure", known=False),
            Column("lead_weeks", "target"),
        ),
        "censor": Censor("created"),
    }
    return PrepSpec(**{**base, **changes})


def test_a_valid_entity_spec_hashes_stably_and_lists_only_known_inputs():
    spec = _entity_spec()
    assert spec.digest() == _entity_spec().digest()
    assert spec.inputs() == ["branch"]
    assert spec.named("target") == ["lead_weeks"]


def test_changing_one_choice_changes_the_digest():
    assert _entity_spec().digest() != _entity_spec(horizon=9).digest()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"columns": (Column("created", "anchor"),)}, "no target"),
        ({"columns": (Column("lead_weeks", "target"),), "censor": None}, "exactly one anchor"),
        ({"censor": Censor("nope")}, "not a column"),
        ({"horizon": 0}, "at least one step"),
    ],
)
def test_a_broken_spec_is_refused_by_name(changes, message):
    with pytest.raises(SpecError, match=message):
        _entity_spec(**changes)


def test_a_timeseries_measure_needs_an_aggregation():
    with pytest.raises(SpecError, match="needs an aggregation"):
        PrepSpec(
            "panel.csv",
            "timeseries",
            "regression",
            (Column("day", "time"), Column("units", "target")),
        )


def test_the_json_form_round_trips():
    spec = _entity_spec()
    data = {
        "source": "rows.csv",
        "mode": "entity",
        "task": "regression",
        "columns": [c.__dict__ for c in spec.columns],
        "censor": {"anchor": "created", "unit": "week", "as_of": None},
    }
    assert spec_from_dict(data) == spec


def test_the_row_fixture_censors_the_slow_lines(tmp_path):
    truth = write_rows(tmp_path, n=600)
    rows = list(csv.DictReader(truth.path.open()))
    assert len(rows) == 600
    open_rows = [r for r in rows if not r["lead_weeks"]]
    assert len(open_rows) == truth.open_rows and 0 < len(open_rows) < 300
    assert all(not r["leak_plus1"] for r in open_rows)


def test_the_panel_fixture_has_its_gap_and_its_ended_project(tmp_path):
    truth = write_panel(tmp_path)
    rows = list(csv.DictReader(truth.path.open()))
    weeks_a_p2 = {r["day"] for r in rows if (r["project"], r["product"]) == ("A", "p2")}
    assert len(weeks_a_p2) == 2 * (truth.weeks - 3)
    ended = {r["day"] for r in rows if r["project"] == "C"}
    assert len(ended) == 2 * truth.ended_week


def _fill_spec(**changes):
    return PrepSpec(
        "panel.csv",
        "timeseries",
        "regression",
        (Column("day", "time"), Column("units", "target", "sum")),
        **changes,
    )


def test_history_fill_digest_and_saved_request_round_trip(tmp_path):
    spec = _fill_spec()
    legacy = asdict(spec)
    legacy.pop("history_fill")
    legacy.pop("holidays_country")
    legacy["mode"] = "series"
    for column in legacy["columns"]:
        column.pop("skip_missing")
        column.pop("pin")
    digest = hashlib.sha256(json.dumps(legacy, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]
    assert spec.digest() == digest == spec_from_dict(legacy).digest()
    filled = replace(spec, history_fill={"below": 0.5, "median_weeks": 13, "min_level": 10})
    assert filled.digest() != digest
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"spec": asdict(filled)}))
    assert spec_from_dict(json.loads(request.read_text())["spec"]) == filled
    assert replace(filled, horizon=12).history_fill == filled.history_fill


@pytest.mark.parametrize(
    "field,value",
    [
        ("below", 0),
        ("below", 1),
        ("below", float("nan")),
        ("below", "0.5"),
        ("median_weeks", 3),
        ("median_weeks", 4.0),
        ("median_weeks", True),
        ("min_level", -1),
        ("min_level", float("inf")),
        ("min_level", None),
    ],
)
def test_history_fill_rejects_invalid_options(field, value):
    fill = {"below": 0.5, "median_weeks": 13, "min_level": 10, field: value}
    with pytest.raises(SpecError, match=f"history_fill.{field}"):
        _fill_spec(history_fill=fill)


def test_history_fill_requires_timeseries_and_complete_options():
    with pytest.raises(SpecError, match="time series mode"):
        _entity_spec(history_fill={"below": 0.5, "median_weeks": 13, "min_level": 10})
    with pytest.raises(SpecError, match="needs below"):
        _fill_spec(history_fill={"below": 0.5})
