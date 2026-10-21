"""synthetic candidate pools, causal inputs and saved rolling selections"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pytest

from smolsmort.forecast.features import (
    LEGACY_ROLL_STATS,
    ROLL_STATS,
    STEP_FEATURES,
    Features,
    _rolling,
    _slope,
    buckets_for,
    series_features,
    step_index,
)
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec


def make_candidate_panel(unit="week", count=156, series_count=12, horizon=60):
    index = np.arange(count + horizon)
    if unit in ("day", "week"):
        dates = np.datetime64("2020-01-06") + index * {"day": 1, "week": 7}[unit]
    else:
        months = {"month": 1, "quarter": 3, "year": 12}[unit]
        dates = (np.datetime64("2020-01", "M") + index * months).astype("datetime64[D]")
    keys = np.array([f"series_{i}" for i in range(series_count)])
    position = np.tile(np.arange(count), series_count)
    rng = np.random.default_rng(17)
    values = 100 + position * 0.2 + 20 * np.sin(2 * np.pi * position / 52)
    values += rng.normal(0, 0.5, len(position))
    spec = PrepSpec(
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
        horizon=horizon,
    )
    panel = {
        STEP: np.tile(dates[:count], series_count),
        SERIES: np.repeat(keys, count),
        "group": np.repeat(keys, count),
        "units": values,
        "orders": values * 1.5 + 2,
    }
    scaffold = {
        STEP: np.tile(dates[count:], series_count),
        SERIES: np.repeat(keys, horizon),
    }
    return spec, panel, scaffold, np.ones(len(position), dtype=bool)


def measure_candidate_pool(repeats=11):
    spec, panel, scaffold, train = make_candidate_panel(horizon=52)

    def build_frames():
        return [
            series_features(spec, panel, scaffold, bucket, "week", train, period=52)
            for bucket in buckets_for(52)
        ]

    frames = build_frames()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        build_frames()
        durations.append(time.perf_counter() - start)
    return {
        "series": 12,
        "steps": 156,
        "buckets": buckets_for(52),
        "columns": [len(frame.features.names) for frame in frames],
        "families": [len(set(frame.features.families)) for frame in frames],
        "union_families": len(set().union(*(frame.features.families for frame in frames))),
        "median_seconds": float(np.median(durations)),
        "repeats": repeats,
    }


@pytest.mark.parametrize("unit", STEP_FEATURES)
@pytest.mark.parametrize("bucket", buckets_for(60))
def test_every_candidate_family_ignores_values_after_its_origin(unit, bucket):
    count = 800 if unit == "day" else 156
    spec, panel, scaffold, train = make_candidate_panel(unit, count, series_count=2)
    origin = count - 70
    indices = step_index(panel[STEP], unit) - step_index(panel[STEP], unit).min()
    train = indices <= origin
    spec = replace(spec, columns=(*spec.columns, Column("site", "input")))
    share = 0.2 + (indices % 7) / 10
    inputs = {
        SERIES: np.repeat(panel[SERIES], 2),
        STEP: np.repeat(panel[STEP], 2),
        "__column": np.full(len(indices) * 2, "site", dtype=object),
        "__level": np.tile(np.array(["s1", "s2"], dtype=object), len(indices)),
        "rows": np.ones(len(indices) * 2, dtype=int),
        "target": (panel["units"][:, None] * np.column_stack([share, 1 - share])).ravel(),
    }
    period = STEP_FEATURES[unit][3]
    before = series_features(
        spec, panel, scaffold, bucket, unit, train, period=period, periods=(), inputs=inputs
    )
    changed = {name: values.copy() for name, values in panel.items()}
    for name in ("units", "orders"):
        changed[name][indices > origin] *= -17
    changed_inputs = {name: values.copy() for name, values in inputs.items()}
    late = np.repeat(indices > origin, 2) & (inputs["__level"] == "s1")
    changed_inputs["target"][late] *= 17
    changed_inputs["__level"][late] = "unseen"
    after = series_features(
        spec,
        changed,
        scaffold,
        bucket,
        unit,
        train,
        period=period,
        periods=(),
        inputs=changed_inputs,
    )
    assert before.features.names == after.features.names
    assert before.features.families == after.features.families
    assert {"mix:site", "spread:site"} <= set(before.features.families)
    origins = step_index(before.step, unit) - step_index(panel[STEP], unit).min() - bucket[1]
    safe = origins <= origin
    assert safe.any() and (~safe).any()
    for family in set(before.features.families):
        old = before.features.pick({family})[0]
        new = after.features.pick({family})[0]
        assert np.isfinite(old[origins == origin]).all(), family
        np.testing.assert_array_equal(old[safe], new[safe], err_msg=family)
    assert not np.array_equal(before.features.x[~safe], after.features.x[~safe], equal_nan=True)
    for family in ("mix:site", "spread:site"):
        assert not np.array_equal(
            before.features.pick({family})[0][~safe],
            after.features.pick({family})[0][~safe],
            equal_nan=True,
        )


@pytest.mark.parametrize("unit", STEP_FEATURES)
@pytest.mark.parametrize("bucket", buckets_for(60))
def test_candidates_match_origin_based_formulas(unit, bucket):
    spec, panel, scaffold, train = make_candidate_panel(unit, 800, series_count=1)
    panel["units"][::17] = 0
    panel["units"][9::23] = np.nan
    period = STEP_FEATURES[unit][3]
    frame = series_features(spec, panel, scaffold, bucket, unit, train, period=period, periods=())
    origins = step_index(frame.step, unit) - step_index(panel[STEP], unit).min() - bucket[1]
    row = np.flatnonzero(origins == 600)[0]
    y = panel["units"]
    for col, family in enumerate(frame.features.families):
        parts = family.split(":")
        if parts[0] == "lag":
            expected = y[600 + bucket[1] - int(parts[1])]
        elif parts[0] == "roll":
            window = int(parts[1])
            seen = y[601 - window : 601]
            seen = seen[np.isfinite(seen)]
            expected = {
                "mean": np.mean(seen),
                "std": np.std(seen),
                "max": np.max(seen),
                "zeros": np.mean(seen == 0),
                "min": np.min(seen),
                "median": np.median(seen),
                "q10": np.quantile(seen, 0.1),
                "q90": np.quantile(seen, 0.9),
            }[parts[2]]
        elif parts[0] == "diff":
            expected = y[600] - y[600 - int(parts[1])]
        elif parts[0] == "slope":
            window = int(parts[1])
            seen = y[601 - window : 601]
            finite = np.isfinite(seen)
            expected = np.polyfit(np.arange(window)[finite], seen[finite], 1)[0]
        elif parts[0] == "momentum":
            half = int(parts[1]) // 2
            expected = np.nanmean(y[601 - half : 601]) - np.nanmean(y[601 - 2 * half : 601 - half])
        elif family == "cycle_ratio":
            expected = y[600] / y[600 - period]
        else:
            continue
        np.testing.assert_allclose(
            frame.features.x[row, col], expected, rtol=1e-5, atol=1e-5, err_msg=family
        )


@pytest.mark.parametrize("window", (2, 3, 4, 7, 13, 52))
def test_rolling_statistics_keep_finite_value_and_partial_window_semantics(window):
    values = np.array([np.nan, np.inf, -np.inf, 0, 6, 2, np.nan, 4, 8, 1, 0, 3] * 5)
    actual = _rolling(values, window)
    assert tuple(actual) == ROLL_STATS
    for origin in range(len(values)):
        seen = values[max(0, origin - window + 1) : origin + 1]
        seen = seen[np.isfinite(seen)]
        if len(seen) < max(window // 2, 1):
            assert all(np.isnan(column[origin]) for column in actual.values())
            continue
        expected = [
            seen.mean(),
            seen.std(),
            seen.max(),
            (seen == 0).mean(),
            seen.min(),
            np.median(seen),
            np.quantile(seen, 0.1),
            np.quantile(seen, 0.9),
        ]
        np.testing.assert_allclose([column[origin] for column in actual.values()], expected)


@pytest.mark.parametrize("window", (2, 3, 4, 13, 52))
def test_slope_needs_two_points_and_preserves_missing_step_positions(window):
    values = 2.5 * np.arange(80) + 10
    values[:3] = np.nan
    values[10::7] = np.inf
    actual = _slope(values, window)
    for origin, slope in enumerate(actual):
        start = max(0, origin - window + 1)
        seen = values[start : origin + 1]
        finite = np.isfinite(seen)
        if finite.sum() < max(window // 2, 2):
            assert np.isnan(slope)
        else:
            assert slope == pytest.approx(2.5)


@pytest.mark.parametrize("unit", STEP_FEATURES)
def test_saved_roll_genomes_select_only_original_columns_in_original_order(unit):
    spec, panel, scaffold, train = make_candidate_panel(unit, 400, series_count=1)
    frame = series_features(spec, panel, scaffold, (2, 4), unit, train, periods=())
    windows = STEP_FEATURES[unit][1]
    aliases = {f"roll:{w}" for w in windows}
    old, names, types = frame.features.pick(aliases)
    assert names == [f"roll{w}_{stat}" for w in windows for stat in LEGACY_ROLL_STATS]
    assert types == ["q"] * len(names)
    explicit = {f"roll:{w}:{stat}" for w in windows for stat in LEGACY_ROLL_STATS}
    np.testing.assert_array_equal(old, frame.features.pick(explicit)[0])
    np.testing.assert_array_equal(old, frame.features.pick(aliases | explicit)[0])
    for window in windows:
        for stat in ROLL_STATS:
            _, picked, _ = frame.features.pick({f"roll:{window}:{stat}"})
            assert picked == [f"roll{window}_{stat}"]
    added = frame.features.pick(aliases | {f"roll:{windows[0]}:q10"})[1]
    assert len(added) == len(names) + 1


def test_saved_roll_values_equal_original_statistics():
    spec, panel, scaffold, train = make_candidate_panel(series_count=1)
    frame = series_features(spec, panel, scaffold, (2, 4), "week", train, periods=())
    old, names, _ = frame.features.pick({"roll:4"})
    assert names == ["roll4_mean", "roll4_std", "roll4_max", "roll4_zeros"]
    for row, step in enumerate(frame.step):
        origin = (
            int(step_index(np.array([step]), "week")[0] - step_index(panel[STEP], "week")[0]) - 4
        )
        seen = panel["units"][max(0, origin - 3) : origin + 1]
        expected = (
            [seen.mean(), seen.std(), seen.max(), (seen == 0).mean()]
            if len(seen) >= 2
            else [np.nan] * 4
        )
        np.testing.assert_allclose(old[row], expected, rtol=1e-5, atol=1e-5)


def test_pick_still_accepts_old_feature_tables():
    features = Features(np.ones((2, 1)), ["roll4_mean"], ["q"], ["roll:4"])
    assert features.pick(["roll:4"])[1] == ["roll4_mean"]


@pytest.mark.parametrize("history", (1, 4, 13, 26, 52, 156))
@pytest.mark.parametrize("unit", STEP_FEATURES)
def test_candidate_history_limits_come_only_from_training(unit, history):
    spec, panel, scaffold, _ = make_candidate_panel(unit, 400, series_count=1)
    train = np.arange(400) < history
    frame = series_features(spec, panel, scaffold, (2, 4), unit, train, periods=())
    lags, windows, differences, period = STEP_FEATURES[unit]
    families = set(frame.features.families)
    assert {f for f in families if f.startswith("lag:")} == {
        f"lag:{k}" for k in lags if 4 <= k < history
    }
    assert {f for f in families if f.startswith("diff:")} == {
        f"diff:{k}" for k in differences if k < history
    }
    for kind in ("slope", "momentum"):
        assert {f for f in families if f.startswith(f"{kind}:")} == {
            f"{kind}:{w}" for w in windows if w <= history
        }
    assert ("cycle_ratio" in families) == (period is not None and period < history)
    if unit == "week":
        assert frame.features.pick({"roll:52"})[1] == [
            f"roll52_{stat}" for stat in LEGACY_ROLL_STATS
        ]


@pytest.mark.parametrize("denominator", (0, np.nan, np.inf, -np.inf))
def test_cycle_ratio_is_missing_for_zero_or_missing_season(denominator):
    spec, panel, scaffold, train = make_candidate_panel(series_count=1)
    panel["units"][8] = denominator
    frame = series_features(spec, panel, scaffold, (1, 1), "week", train, periods=())
    ratio = frame.features.pick({"cycle_ratio"})[0][:, 0]
    assert np.isnan(ratio[60])
    assert np.isnan(ratio[:52]).all()
    assert not np.isinf(ratio).any()


def test_cycle_ratio_uses_the_selected_period_even_beyond_the_bucket():
    spec, panel, scaffold, train = make_candidate_panel("day", 800, series_count=1)
    frame = series_features(
        spec, panel, scaffold, (27, 52), "day", train, period=7, periods=(7, 365)
    )
    assert frame.features.pick({"season"})[1] == []
    ratio = frame.features.pick({"cycle_ratio"})[0][:, 0]
    assert ratio[60] == pytest.approx(panel["units"][60] / panel["units"][53])


def test_weekly_season_collision_keeps_saved_genome_columns():
    spec, panel, scaffold, train = make_candidate_panel(series_count=1)
    frame = series_features(spec, panel, scaffold, (27, 52), "week", train, period=52)
    assert frame.features.pick({"season"})[1] == ["lag_52"]
    assert frame.features.pick({"lag+0"})[1] == []
    assert frame.features.pick({"lag+0", "season"})[1] == ["lag_52"]
    for family in ("season", "lag:52"):
        values = frame.features.pick({family})[0][:, 0]
        np.testing.assert_array_equal(values, frame.base.astype(np.float32))


def test_weekly_156_step_candidate_pool_size():
    spec, panel, scaffold, train = make_candidate_panel(horizon=52)
    frames = [
        series_features(spec, panel, scaffold, bucket, "week", train, period=52)
        for bucket in buckets_for(52)
    ]
    assert [len(frame.features.names) for frame in frames] == [90, 87, 83, 78, 77, 75]
    assert [len(set(frame.features.families)) for frame in frames] == [84, 81, 77, 72, 71, 69]
    assert len(set().union(*(frame.features.families for frame in frames))) == 84
    for bucket, frame in zip(buckets_for(52), frames, strict=True):
        assert {f for f in frame.features.families if f.startswith("lag:")} == {
            f"lag:{k}" for k in (*range(1, 14), 26, 52) if k >= bucket[1]
        }
        assert frame.features.x.dtype == np.float32
        assert len(frame.features.names) == len(set(frame.features.names))


if __name__ == "__main__":
    print(json.dumps(measure_candidate_pool(), indent=2))
