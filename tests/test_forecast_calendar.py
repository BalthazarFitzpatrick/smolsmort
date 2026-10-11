"""target calendars are known ahead of time and preserve old feature selections"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import features
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec, spec_from_dict

NEW_FAMILIES = {"cal:cyclic", "cal:dom", "cal:holiday"}


def make_spec(unit="day", country=None):
    return PrepSpec(
        "synthetic.csv",
        "series",
        "regression",
        (
            Column("day", "time"),
            Column("group", "dimension"),
            Column("orders", "measure", "sum"),
            Column("units", "target", "sum"),
        ),
        step=unit,
        horizon=8,
        holidays_country=country,
    )


def make_panel(unit):
    index = np.arange(100)
    if unit in {"day", "week"}:
        steps = np.datetime64("2023-11-06") + index * (1 if unit == "day" else 7)
    else:
        months = {"month": 1, "quarter": 3, "year": 12}[unit]
        steps = (np.datetime64("2020-01", "M") + index * months).astype("datetime64[D]")
    panel = {
        STEP: steps,
        SERIES: np.full(100, "one"),
        "group": np.full(100, "a"),
        "units": index.astype(float) + 5,
        "orders": index.astype(float) * 2 + 1,
    }
    scaffold = {STEP: np.array([], dtype="datetime64[D]"), SERIES: np.array([], dtype=str)}
    return panel, scaffold, index < 80


def test_daily_calendar_values_are_hand_computed():
    days = np.array(["2024-01-01", "2024-02-29", "2024-03-01", "2023-12-31"], dtype="datetime64[D]")
    columns = features._series_calendar(days, "day")
    expected = {
        "cal_day_of_month": [1, 29, 1, 31],
        "cal_month_start": [1, 0, 1, 0],
        "cal_month_end": [0, 1, 0, 1],
        "cal_week_of_month": [1, 5, 1, 5],
    }
    for name, values in expected.items():
        assert columns[name][0] == "cal:dom"
        np.testing.assert_array_equal(columns[name][1], values)
    phases = {
        "weekday": np.array([0, 3, 4, 6]) / 7,
        "month": np.array([0, 1, 2, 11]) / 12,
        "day_of_year": np.array([0 / 366, 59 / 366, 60 / 366, 364 / 365]),
    }
    for part, phase in phases.items():
        for suffix, function in (("sin", np.sin), ("cos", np.cos)):
            family, values = columns[f"cal_{part}_{suffix}"]
            assert family == "cal:cyclic"
            np.testing.assert_allclose(values, function(2 * np.pi * phase), atol=1e-7)
    assert {family for family, _ in columns.values()} == {"cal:cyclic", "cal:dom"}


@pytest.mark.parametrize("unit", ["week", "month", "quarter", "year"])
def test_non_daily_calendars_have_no_weekday_or_dom(unit):
    columns = features._series_calendar(np.array(["2024-02-29"], dtype="datetime64[D]"), unit)
    assert set(columns) == {
        "cal_month_sin",
        "cal_month_cos",
        "cal_day_of_year_sin",
        "cal_day_of_year_cos",
    }
    assert {family for family, _ in columns.values()} == {"cal:cyclic"}


def test_weekly_calendar_uses_monday_across_month_and_year_boundary():
    days = np.array(["2024-01-01", "2024-01-07", "2024-02-01", "2023-01-01"], dtype="datetime64[D]")
    mondays = np.array(
        ["2024-01-01", "2024-01-01", "2024-01-29", "2022-12-26"], dtype="datetime64[D]"
    )
    weekly = features._series_calendar(days, "week")
    daily = features._series_calendar(mondays, "day")
    for name, (family, values) in weekly.items():
        assert family == daily[name][0]
        np.testing.assert_array_equal(values, daily[name][1])


@pytest.mark.parametrize("unit", ["day", "week", "month", "quarter", "year"])
def test_old_feature_columns_values_and_family_order_are_unchanged(monkeypatch, unit):
    panel, scaffold, train = make_panel(unit)
    spec = make_spec(unit)
    new = features.series_features(spec, panel, scaffold, (2, 4), unit, train, periods=())
    monkeypatch.setattr(features, "_series_calendar", lambda *args, **kwargs: {})
    legacy = features.series_features(spec, panel, scaffold, (2, 4), unit, train, periods=())
    families = set(legacy.features.families)
    assert not families & NEW_FAMILIES
    values, names, types = new.features.pick(families)
    assert names == legacy.features.names
    assert types == legacy.features.types
    assert [f for f in new.features.families if f not in NEW_FAMILIES] == legacy.features.families
    assert new.features.names[: len(names)] == names
    np.testing.assert_array_equal(values, legacy.features.x)


@pytest.mark.parametrize("country", [None, "US"])
@pytest.mark.parametrize("bucket", [(1, 1), (2, 4), (5, 8)])
def test_calendar_features_depend_only_on_target_step(country, bucket):
    if country:
        pytest.importorskip("holidays")
    panel, scaffold, train = make_panel("day")
    spec = make_spec(country=country)
    before = features.series_features(spec, panel, scaffold, bucket, "day", train, periods=())
    changed = {name: values.copy() for name, values in panel.items()}
    changed["units"] = np.linspace(-1e6, 1e6, 100)
    changed["orders"] = np.full(100, 1e9)
    after = features.series_features(spec, changed, scaffold, bucket, "day", train, periods=())
    families = {f for f in before.features.families if f.startswith("cal:")}
    np.testing.assert_array_equal(
        before.features.pick(families)[0], after.features.pick(families)[0]
    )
    expected = features._series_calendar(before.step, "day", country)
    for name, (family, values) in expected.items():
        assert family in families
        np.testing.assert_allclose(
            before.features.x[:, before.features.names.index(name)], values, atol=1e-7
        )
    assert not np.array_equal(before.features.pick({"lag+0"})[0], after.features.pick({"lag+0"})[0])


def test_holiday_flags_and_distances_match_the_library_across_years():
    holidays = pytest.importorskip("holidays")
    days = np.arange(np.datetime64("2021-12-20"), np.datetime64("2023-01-10"))
    columns = features._series_calendar(days, "day", "US")
    holiday_dates = sorted(holidays.country_holidays("US", years=[2021, 2022, 2023], observed=True))
    expected = {"cal_holiday": [], "cal_days_to_holiday": [], "cal_days_since_holiday": []}
    for day in days.astype(object):
        expected["cal_holiday"].append(int(day in holiday_dates))
        expected["cal_days_to_holiday"].append(
            min([30, *((h - day).days for h in holiday_dates if h >= day)])
        )
        expected["cal_days_since_holiday"].append(
            min([30, *((day - h).days for h in holiday_dates if h <= day)])
        )
    for name, values in expected.items():
        assert columns[name][0] == "cal:holiday"
        np.testing.assert_array_equal(columns[name][1], values)
    observed = np.flatnonzero(days == np.datetime64("2021-12-31"))[0]
    assert columns["cal_holiday"][1][observed] == 1
    assert columns["cal_days_to_holiday"][1][observed] == 0
    assert columns["cal_days_since_holiday"][1][observed] == 0
    assert max(columns["cal_days_to_holiday"][1]) == 30
    assert max(columns["cal_days_since_holiday"][1]) == 30


def test_missing_holiday_library_names_the_install_command(monkeypatch):
    features._holiday_dates.cache_clear()
    monkeypatch.setitem(sys.modules, "holidays", None)
    days = np.array(["2024-01-01"], dtype="datetime64[D]")
    assert not any(
        family == "cal:holiday" for family, _ in features._series_calendar(days, "day").values()
    )
    with pytest.raises(ImportError, match="uv sync --extra holidays"):
        features._series_calendar(days, "day", "US")


def test_country_with_no_holidays_has_capped_distances(monkeypatch):
    features._holiday_dates.cache_clear()
    calendar = SimpleNamespace(country_holidays=lambda *args, **kwargs: {})
    monkeypatch.setitem(sys.modules, "holidays", calendar)
    try:
        days = np.array(["2024-01-01", "2024-12-31"], dtype="datetime64[D]")
        columns = features._series_calendar(days, "day", "empty")
        np.testing.assert_array_equal(columns["cal_holiday"][1], [0, 0])
        np.testing.assert_array_equal(columns["cal_days_to_holiday"][1], [30, 30])
        np.testing.assert_array_equal(columns["cal_days_since_holiday"][1], [30, 30])
    finally:
        features._holiday_dates.cache_clear()


def test_holiday_features_cover_known_future_scaffold_targets():
    holidays = pytest.importorskip("holidays")
    panel, _, train = make_panel("day")
    scaffold = {
        STEP: panel[STEP][-1] + np.arange(1, 9),
        SERIES: np.full(8, "one"),
    }
    frame = features.series_features(
        make_spec(country="US"), panel, scaffold, (5, 8), "day", train, periods=()
    )
    future_dates = frame.step[frame.future].astype(object)
    library = holidays.country_holidays("US", years=[2024])
    flag_column = frame.features.names.index("cal_holiday")
    np.testing.assert_array_equal(
        frame.features.x[frame.future, flag_column], [int(day in library) for day in future_dates]
    )
    holiday_row = np.flatnonzero(frame.future & (frame.step == np.datetime64("2024-02-19")))
    assert len(holiday_row) == 1
    for name in ("cal_days_to_holiday", "cal_days_since_holiday"):
        assert frame.features.x[holiday_row[0], frame.features.names.index(name)] == 0


def test_default_country_digest_matches_the_legacy_spec():
    spec = make_spec()
    legacy = asdict(spec)
    legacy.pop("holidays_country")
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]
    assert spec.holidays_country is None
    assert spec.digest() == expected
    assert spec_from_dict(legacy).digest() == expected
    assert replace(spec, holidays_country="US").digest() != expected
    assert spec_from_dict(asdict(replace(spec, holidays_country="US"))).holidays_country == "US"


def test_default_country_keeps_the_prepared_cache_key(tmp_path):
    pytest.importorskip("duckdb")
    from smolsmort.forecast.prep import prepare

    source = tmp_path / "source.csv"
    source.write_text("day,units\n2024-01-01,10\n2024-01-02,12\n", encoding="utf-8")
    spec = PrepSpec(
        str(source),
        "series",
        "regression",
        (Column("day", "time"), Column("units", "target", "sum")),
        step="day",
    )
    legacy = asdict(spec)
    legacy.pop("holidays_country")
    digest = hashlib.sha256(json.dumps(legacy, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    prepared = prepare(spec, tmp_path / "cache")
    assert prepared.folder.name == f"{digest}-{source_digest}"
    assert prepare(replace(spec, holidays_country=None), tmp_path / "cache").reused
