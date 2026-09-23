"""the profile reads training rows only and finds what the fixtures hide; features never see past
their origin"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from forecast_fixtures import write_panel, write_rows

from smolsmort.forecast.features import buckets_for, row_features, series_features, step_index
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
