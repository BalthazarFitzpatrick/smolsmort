"""the forecast contract refuses a spec that cannot be prepared, and the fixtures hold their shape"""

from __future__ import annotations

import csv

import pytest
from forecast_fixtures import write_panel, write_rows

from smolsmort.forecast.spec import Censor, Column, PrepSpec, SpecError, spec_from_dict


def _row_spec(**changes):
    base = {
        "source": "rows.csv",
        "mode": "row",
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


def test_a_valid_row_spec_hashes_stably_and_lists_only_known_inputs():
    spec = _row_spec()
    assert spec.digest() == _row_spec().digest()
    assert spec.inputs() == ["branch"]
    assert spec.named("target") == ["lead_weeks"]


def test_changing_one_choice_changes_the_digest():
    assert _row_spec().digest() != _row_spec(horizon=9).digest()


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
        _row_spec(**changes)


def test_a_series_measure_needs_an_aggregation():
    with pytest.raises(SpecError, match="needs an aggregation"):
        PrepSpec(
            "panel.csv",
            "series",
            "regression",
            (Column("day", "time"), Column("units", "target")),
        )


def test_the_json_form_round_trips():
    spec = _row_spec()
    data = {
        "source": "rows.csv",
        "mode": "row",
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
