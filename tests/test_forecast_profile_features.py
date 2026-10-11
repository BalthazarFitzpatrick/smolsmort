"""the profile reads training rows only and finds what the fixtures hide; features never see past
their origin"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from forecast_fixtures import write_panel, write_rows

from smolsmort.forecast.features import (
    ALPHAS,
    WINDOWS,
    buckets_for,
    row_features,
    series_features,
    step_index,
)
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.profile import build_profile
from smolsmort.forecast.spec import ANCHOR, LOWER, SERIES, STEP, UPPER, Censor, Column, PrepSpec
from smolsmort.forecast.splits import labels_as_of, series_cuts
from smolsmort.forecast.tables import read_table

pytest.importorskip("duckdb")


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    folder = tmp_path_factory.mktemp("panel")
    truth = write_panel(folder)
    spec = PrepSpec(
        str(truth.path),
        "series",
        "regression",
        (
            Column("day", "time"),
            Column("project", "dimension"),
            Column("product", "dimension"),
            Column("orders", "measure", aggregation="sum"),
            Column("units", "target", aggregation="sum"),
        ),
    )
    prepared = prepare(spec, folder / "cache")
    table = read_table(prepared.folder / "panel.parquet", order_by=f"{SERIES}, {STEP}")
    scaffold = read_table(prepared.folder / "scaffold.parquet")
    steps = np.unique(table[STEP])
    cuts = series_cuts(len(steps), spec.horizon)
    train = table[STEP] < steps[cuts.val_start]
    return spec, table, scaffold, train, truth


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    folder = tmp_path_factory.mktemp("rows")
    truth = write_rows(folder, n=800)
    spec = PrepSpec(
        str(truth.path),
        "row",
        "regression",
        (
            Column("created", "anchor"),
            Column("branch", "dimension"),
            Column("size", "measure"),
            Column("planned_offset", "measure"),
            Column("leak_plus1", "measure"),
            Column("line", "ignore"),
            Column("lead_weeks", "target"),
        ),
        censor=Censor("created"),
    )
    table = read_table(prepare(spec, folder / "cache").folder / "rows.parquet")
    cut = dt.date(2025, 10, 1)
    train = table[ANCHOR] < np.datetime64(cut)
    # training sees the bounds an export taken at the cut would have shown
    table[LOWER], table[UPPER] = labels_as_of(table[ANCHOR], table[LOWER], table[UPPER], cut, 7)
    return spec, table, train


def test_the_profile_finds_the_season_and_the_leading_measure(panel):
    spec, table, _, train, truth = panel
    profile = build_profile(spec, table, train)
    assert abs(profile["period"] - truth.period) <= 2
    assert profile["leads"]["orders"]["lag"] == truth.lead_lag
    assert profile["zero_share"]["max"] >= 0.3
    assert profile["genes"]["measure_lags"]["orders"] == truth.lead_lag
    assert "tweedie" not in profile["genes"]["objectives"], "only one sparse series of twelve"


def test_the_profile_flags_a_leak_and_offers_aft_for_open_rows(rows):
    spec, table, train = rows
    profile = build_profile(spec, table, train)
    assert "equals the target plus 1" in profile["leaks"]["leak_plus1"]
    assert profile["genes"]["exclude"] == ["leak_plus1"]
    assert profile["censored"] > 0 and "aft" in profile["genes"]["objectives"]
    assert profile["columns"]["branch"]["levels"] == 5


def test_rows_outside_train_cannot_change_the_profile(panel):
    spec, table, _, train, _ = panel
    before = build_profile(spec, table, train)
    changed = {k: v.copy() for k, v in table.items()}
    for name in ("units", "orders"):
        changed[name][~train] = changed[name][~train] * 10 + 7
    assert build_profile(spec, changed, train) == before


def test_row_features_offer_families_and_freeze_levels_on_train(rows):
    spec, table, train = rows
    features = row_features(spec, table, train)
    families = set(features.families)
    assert {"dim:branch", "freq:branch", "measure:size", "log:size", "date:week"} <= families
    assert "date:trend" in families
    changed = {k: v.copy() for k, v in table.items()}
    changed["branch"] = changed["branch"].astype(object)
    changed["branch"][~train] = "brand new"
    codes = row_features(spec, changed, train).pick({"dim:branch"})[0][:, 0]
    assert np.isnan(codes[~train]).all() and np.isfinite(codes[train]).all()


def test_buckets_cut_at_the_horizon():
    assert buckets_for(8) == [(1, 1), (2, 4), (5, 8)]
    assert buckets_for(60)[-1] == (53, 60)


@pytest.mark.parametrize("bucket", [(1, 1), (2, 4), (5, 8)])
def test_series_features_never_see_past_their_origin(panel, bucket):
    spec, table, scaffold, train, _ = panel
    before = series_features(spec, table, scaffold, bucket, "week", train, period=52)
    origin = np.unique(table[STEP])[100]
    changed = {k: v.copy() for k, v in table.items()}
    later = changed[STEP] > origin
    changed["units"][later] = changed["units"][later] * 3 + 11
    changed["orders"][later] = -changed["orders"][later]
    after = series_features(spec, changed, scaffold, bucket, "week", train, period=52)
    safe = before.origin <= origin.astype("datetime64[D]")
    assert safe.sum() > 100 and (~safe).sum() > 100
    assert np.array_equal(before.features.x[safe], after.features.x[safe], equal_nan=True)
    assert not np.array_equal(before.features.x[~safe], after.features.x[~safe], equal_nan=True)


def test_series_features_build_one_future_row_per_series_and_step(panel):
    spec, table, scaffold, train, truth = panel
    for lo, hi in buckets_for(spec.horizon):
        frame = series_features(spec, table, scaffold, (lo, hi), "week", train)
        assert frame.future.sum() == truth.series * (hi - lo + 1)
    ended = series_features(spec, table, scaffold, (1, 1), "week", train)
    c_future = ended.future & (ended.series == "C|p1")
    lag = ended.features.names.index("lag_1")
    assert c_future.sum() == 1
    assert ended.features.x[c_future, lag][0] == 0, "an ended series sold nothing since"


def test_a_lag_column_is_the_target_that_many_steps_back(panel):
    spec, table, scaffold, train, _ = panel
    frame = series_features(spec, table, scaffold, (2, 4), "week", train)
    mine = (frame.series == "A|p1") & ~frame.future
    column = frame.features.x[mine, frame.features.names.index("lag_4")]
    series = table["units"][table[SERIES] == "A|p1"]
    index = (
        step_index(frame.step[mine], "week")
        - step_index(table[STEP][table[SERIES] == "A|p1"], "week")[0]
    )
    assert np.allclose(column, series[index - 4].astype(np.float32))


def make_seasonal_table(unit, count, periods):
    rng = np.random.default_rng(17)
    index = np.arange(count)
    values = 100 + rng.normal(0, 0.5, count)
    for period in periods:
        values += 20 * np.sin(2 * np.pi * index / period)
    if unit in ("day", "week"):
        steps = np.datetime64("2020-01-06") + index * (1 if unit == "day" else 7)
    else:
        months = {"month": 1, "quarter": 3, "year": 12}[unit]
        steps = (np.datetime64("2020-01", "M") + index * months).astype("datetime64[D]")
    spec = PrepSpec(
        "synthetic.csv",
        "series",
        "regression",
        (Column("day", "time"), Column("units", "target", "sum")),
        step=unit,
        horizon=30,
    )
    return spec, {STEP: steps, SERIES: np.full(count, "daily"), "units": values}


@pytest.mark.parametrize(
    "unit,count,periods,expected",
    [
        ("day", 100, (7,), [7]),
        ("day", 729, (365,), []),
        ("day", 730, (365,), [365]),
        ("day", 900, (7, 365), [7, 365]),
        ("day", 900, (7,), [7]),
        ("week", 103, (52,), []),
        ("week", 104, (52,), [52]),
        ("month", 60, (12,), [12]),
        ("quarter", 40, (4,), [4]),
    ],
)
def test_calendar_periods_need_two_training_cycles(unit, count, periods, expected):
    spec, table = make_seasonal_table(unit, count + 40, periods)
    train = np.arange(count + 40) < count
    profile = build_profile(spec, table, train)
    assert profile["periods"] == expected
    assert profile["period"] == (expected[0] if expected else None)
    table["units"][~train] = 1e9
    assert build_profile(spec, table, train) == profile


@pytest.fixture(scope="module")
def daily():
    spec, table = make_seasonal_table("day", 1000, (7, 365))
    scaffold = {STEP: table[STEP][-1] + np.arange(1, 31), SERIES: np.full(30, "daily")}
    train = np.arange(1000) < 910
    return spec, table, scaffold, train


@pytest.mark.parametrize("bucket", buckets_for(60))
def test_each_daily_feature_uses_only_positions_through_origin(daily, bucket):
    spec, table, scaffold, train = daily
    before = series_features(spec, table, scaffold, bucket, "day", train, period=7)
    names, families = before.features.names, before.features.families
    assert {f"lag+{j}" for j in range(4)} <= set(families)
    new = [
        i
        for i, family in enumerate(families)
        if family.startswith(("lag:", "roll:")) or family == "season"
    ]
    assert new
    for row in np.flatnonzero(~before.future):
        target = int((before.step[row] - table[STEP][0]).astype(int))
        origin = target - bucket[1]
        for col in new:
            name, family = names[col], families[col]
            if family.startswith("lag:") or family == "season":
                lag = int(family.split(":")[1]) if family != "season" else 7
                assert lag >= bucket[1]
                expected = table["units"][target - lag] if target >= lag else np.nan
            else:
                window = int(family.split(":")[1])
                seen = table["units"][max(0, origin - window + 1) : origin + 1]
                stat = name.rsplit("_", 1)[1]
                expected = (
                    {
                        "mean": seen.mean(),
                        "std": seen.std(),
                        "max": seen.max(),
                        "zeros": (seen == 0).mean(),
                    }[stat]
                    if len(seen) >= max(window // 2, 1)
                    else np.nan
                )
            np.testing.assert_allclose(before.features.x[row, col], expected, rtol=1e-6, atol=1e-5)
    changed = {name: values.copy() for name, values in table.items()}
    # change a held-out future value; training-derived columns and earlier origins must be fixed
    changed["units"][920] = 1e9
    after = series_features(spec, changed, scaffold, bucket, "day", train, period=7)
    safe = before.origin < table[STEP][920]
    np.testing.assert_array_equal(before.features.x[safe], after.features.x[safe])
    assert not np.array_equal(before.features.x[~safe], after.features.x[~safe], equal_nan=True)


@pytest.mark.parametrize(
    "unit,windows,lags",
    [
        ("day", (7, 14, 28, 91, 182), (7, 14, 28, 364, 365)),
        ("month", (3, 6, 12), (12,)),
        ("quarter", (2, 4, 8), (4,)),
        ("year", (2, 3, 5), (2, 3)),
    ],
)
def test_step_families_and_missing_genome_families(daily, unit, windows, lags):
    spec, table, scaffold, train = daily
    if unit != "day":
        spec, table = make_seasonal_table(unit, 100, ())
        train = np.arange(100) < 80
        scaffold = {STEP: np.array([], dtype="datetime64[D]"), SERIES: np.array([], dtype=str)}
    frame = series_features(spec, table, scaffold, (1, 1), unit, train)
    families = set(frame.features.families)
    assert {f"roll:{w}" for w in windows} == {f for f in families if f.startswith("roll:")}
    assert {f"lag:{k}" for k in lags} == {f for f in families if f.startswith("lag:")}
    assert frame.features.pick({"lag+0", "absent", "roll:999"})[1] == ["lag_1"]


def test_daily_families_are_capped_by_training_history(daily):
    spec, table, scaffold, _ = daily
    train = np.arange(len(table[STEP])) < 20
    frame = series_features(spec, table, scaffold, (2, 4), "day", train)
    families = set(frame.features.families)
    assert {f for f in families if f.startswith("roll:")} == {"roll:7", "roll:14"}
    assert {f for f in families if f.startswith("lag:")} == {"lag:7", "lag:14"}


def test_supplied_periods_become_independent_absolute_families(daily):
    spec, table, scaffold, train = daily
    frame = series_features(spec, table, scaffold, (2, 4), "day", train, period=7, periods=(7, 52))
    families = set(frame.features.families)
    assert {"lag+3", "season", "lag:7", "lag:52"} <= families
    for family in ("lag+3", "season", "lag:7"):
        column = frame.features.pick({family})[0][:, 0]
        np.testing.assert_array_equal(column, frame.features.pick({"lag:7"})[0][:, 0])


def test_weekly_columns_families_and_values_stayed_identical(panel):
    spec, table, scaffold, train, _ = panel
    for bucket in buckets_for(spec.horizon):
        frame = series_features(spec, table, scaffold, bucket, "week", train, period=52)
        assert not any(f.startswith("lag:") for f in frame.features.families)
        assert {f for f in frame.features.families if f.startswith("roll:")} == {
            f"roll:{w}" for w in WINDOWS
        }
        assert {f for f in frame.features.families if f.startswith("ewm:")} == {
            f"ewm:{a}" for a in ALPHAS
        }
        expected = [f"lag_{bucket[1] + j}" for j in range(4)] + ["lag_52"]
        expected += [f"roll{w}_{stat}" for w in WINDOWS for stat in ("mean", "std", "max", "zeros")]
        expected += [f"ewm{a}" for a in ALPHAS]
        expected += [f"orders_lag_{bucket[1] + j}" for j in range(4)]
        expected += ["cal_week", "cal_month", "cal_quarter", "project", "product"]
        old_families = set(frame.features.families) - {"cal:cyclic", "cal:dom", "cal:holiday"}
        assert frame.features.pick(old_families)[1] == expected
