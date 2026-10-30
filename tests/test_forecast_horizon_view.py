"""hand-checked lead selections and blends of saved forecast rows"""

import numpy as np
import pytest

from smolsmort.forecast.evaluate import apply_band, coverage
from smolsmort.forecast.horizon_view import (
    blend_rows,
    build_blend_weights,
    fit_blend_band,
    lead_rows,
)


@pytest.fixture
def overlapping_rows():
    return {
        "series": np.array(["a"] * 6),
        "step": np.array(
            ["2026-01-15"] * 3 + ["2026-01-22"] * 2 + ["2026-01-29"], dtype="datetime64[D]"
        ),
        "origin": np.array(
            ["2026-01-08", "2026-01-01", "2025-12-25", "2026-01-08", "2026-01-01", "2026-01-08"],
            dtype="datetime64[D]",
        ),
        "weeks_ahead": np.array([1, 2, 3, 2, 3, 3]),
        "prediction": np.array([10.0, 20.0, 30.0, 40.0, 60.0, 90.0]),
        "naive_prediction": np.array([8.0, 18.0, 28.0, 38.0, 58.0, 88.0]),
        "actual": np.array([25.0, 25.0, 25.0, 55.0, 55.0, 100.0]),
        "bucket": np.array(["near", "mid", "far", "mid", "far", "far"]),
        "lower": np.array([5.0, 15.0, 25.0, 35.0, 55.0, 85.0]),
        "upper": np.array([15.0, 25.0, 35.0, 45.0, 65.0, 95.0]),
    }


def test_lead_selects_only_exact_lead_without_averaging(overlapping_rows):
    selected = lead_rows(overlapping_rows, 3)
    assert set(selected) == set(overlapping_rows)
    np.testing.assert_array_equal(selected["weeks_ahead"], [3, 3, 3])
    np.testing.assert_array_equal(selected["prediction"], [30, 60, 90])
    np.testing.assert_array_equal(selected["lower"], [25, 55, 85])
    np.testing.assert_array_equal(lead_rows(overlapping_rows, 1)["prediction"], [10])


def test_missing_lead_is_empty(overlapping_rows):
    selected = lead_rows(overlapping_rows, 4)
    assert all(len(values) == 0 for values in selected.values())


def test_equal_blend_normalizes_each_target_including_tail(overlapping_rows):
    blended = blend_rows(overlapping_rows)
    np.testing.assert_allclose(blended["prediction"], [20, 50, 90])
    np.testing.assert_allclose(blended["naive_prediction"], [18, 48, 88])
    np.testing.assert_array_equal(blended["actual"], [25, 55, 100])
    np.testing.assert_array_equal(blended["series"], ["a", "a", "a"])
    assert not {"lower", "upper", "origin", "weeks_ahead", "bucket"} & blended.keys()


def test_explicit_weights_normalize_over_present_leads(overlapping_rows):
    blended = blend_rows(overlapping_rows, {1: 1, 2: 3, 3: 2})
    np.testing.assert_allclose(blended["prediction"], [130 / 6, 48, 90])
    np.testing.assert_allclose(blended["naive_prediction"], [118 / 6, 46, 88])


def test_blend_weights_count_each_week_and_normalize_each_tail():
    weights = build_blend_weights(4, {2: 4, 4: 2})
    assert weights == {1: 1, 2: 4, 3: 1, 4: 2}
    np.testing.assert_allclose(np.array(list(weights.values())) / 8, [1 / 8, 4 / 8, 1 / 8, 2 / 8])
    table = {
        "series": np.array(["synthetic"] * 8),
        "step": np.array(
            ["2026-01-15"] * 4 + ["2026-01-22"] * 2 + ["2026-01-29"] * 2,
            dtype="datetime64[D]",
        ),
        "weeks_ahead": np.array([1, 2, 3, 4, 2, 4, 3, 4]),
        "prediction": np.array([10, 20, 30, 40, 50, 90, 100, 140], dtype=float),
        "naive_prediction": np.array([8, 18, 28, 38, 48, 88, 98, 138], dtype=float),
    }
    blended = blend_rows(table, weights=weights)
    np.testing.assert_allclose(blended["prediction"], [200 / 8, 380 / 6, 380 / 3])
    np.testing.assert_allclose(blended["naive_prediction"], [184 / 8, 368 / 6, 374 / 3])


@pytest.mark.parametrize("weight", [1, 2, 4, 8])
def test_blend_weights_include_every_lead(weight):
    weights = build_blend_weights(4, {2: weight})
    assert weights == {1: 1, 2: weight, 3: 1, 4: 1}
    assert build_blend_weights(4) == {1: 1, 2: 1, 3: 1, 4: 1}


@pytest.mark.parametrize("weight", [0, -1, 3, 16, 1.0, "4", True, None, np.inf, np.nan])
def test_invalid_blend_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="blend_weights values must be 1, 2, 4, or 8"):
        build_blend_weights(4, {2: weight})


@pytest.mark.parametrize("lead", [0, -1, 5, 1.5, "2", True, None])
def test_invalid_blend_week_is_rejected(lead):
    with pytest.raises(ValueError, match="blend_weights weeks must be integers from 1 to 4"):
        build_blend_weights(4, {lead: 1})


def test_explicit_zero_weights_do_not_include_nonfinite_predictions(overlapping_rows):
    overlapping_rows["prediction"][0] = np.nan
    blended = blend_rows(overlapping_rows, {2: 1, 3: 1})
    np.testing.assert_allclose(blended["prediction"], [25, 50, 90])


def test_large_finite_weights_do_not_overflow(overlapping_rows):
    blended = blend_rows(overlapping_rows, {1: 1e308, 2: 1e308, 3: 1e308})
    np.testing.assert_allclose(blended["prediction"], [20, 50, 90])


def test_series_are_never_blended_together(overlapping_rows):
    overlapping_rows["series"][0] = "b"
    blended = blend_rows(overlapping_rows)
    np.testing.assert_array_equal(blended["series"], ["b", "a", "a", "a"])
    np.testing.assert_allclose(blended["prediction"], [10, 25, 50, 90])


def test_single_row_keeps_target_values(overlapping_rows):
    single = {name: values[-1:] for name, values in overlapping_rows.items()}
    blended = blend_rows(single)
    for name in blended:
        np.testing.assert_array_equal(blended[name], single[name])


def test_empty_table_stays_empty(overlapping_rows):
    empty = {name: values[:0] for name, values in overlapping_rows.items()}
    assert all(len(values) == 0 for values in blend_rows(empty).values())


def test_legacy_tables_pass_through(overlapping_rows):
    legacy = {name: values for name, values in overlapping_rows.items() if name != "weeks_ahead"}
    assert lead_rows(legacy, 2) is legacy
    assert blend_rows(legacy) is legacy


@pytest.mark.parametrize("n", [0, -1, 1.5, "2", True])
def test_invalid_lead_is_rejected(overlapping_rows, n):
    with pytest.raises(ValueError, match="lead must be a positive integer"):
        lead_rows(overlapping_rows, n)


@pytest.mark.parametrize(
    "weights",
    [{}, {1: 0, 2: 0}, {1: -1}, {1: np.nan}, {1: np.inf}, {1: "1"}, {1: True}],
)
def test_invalid_weight_values_are_rejected(overlapping_rows, weights):
    with pytest.raises(ValueError, match="weights"):
        blend_rows(overlapping_rows, weights)


def test_weight_integer_beyond_float_range_is_rejected(overlapping_rows):
    with pytest.raises(ValueError, match="weights must be finite and nonnegative"):
        blend_rows(overlapping_rows, {1: 10**400, 2: 1, 3: 1})


@pytest.mark.parametrize("weights", [{0: 1}, {-1: 1}, {1.5: 1}, {"1": 1}, {True: 1}])
def test_invalid_weight_leads_are_rejected(overlapping_rows, weights):
    with pytest.raises(ValueError, match="lead"):
        blend_rows(overlapping_rows, weights)


def test_target_without_a_positive_present_weight_is_rejected(overlapping_rows):
    with pytest.raises(ValueError, match="positive value for every target week"):
        blend_rows(overlapping_rows, {1: 1})


def test_helpers_do_not_modify_input(overlapping_rows):
    original = {name: values.copy() for name, values in overlapping_rows.items()}
    lead_rows(overlapping_rows, 2)
    blend_rows(overlapping_rows, {1: 1, 2: 2, 3: 3})
    fit_blend_band(overlapping_rows, {1: 1, 2: 2, 3: 3})
    for name in original:
        np.testing.assert_array_equal(overlapping_rows[name], original[name])


@pytest.fixture
def band_validation():
    return {
        "series": np.array(["synthetic"] * 10),
        "step": np.repeat(np.arange(5), 2),
        "weeks_ahead": np.tile([1, 2], 5),
        "prediction": np.tile([8.0, 12.0], 5),
        "actual": np.repeat([8.0, 10.0, 12.0, 14.0, 16.0], 2),
        "lower": np.full(10, -100.0),
        "upper": np.full(10, 100.0),
    }


def test_blend_band_offsets_and_test_coverage_are_hand_checked(band_validation):
    band = fit_blend_band(band_validation)
    # five blended residuals [-2, 0, 2, 4, 6] put q10/q90 at indices 0.4/3.6
    assert band.n == 5
    assert not band.relative
    assert (band.lo, band.hi) == pytest.approx((-1.2, 5.2))
    lower, upper = apply_band(band, [10.0] * 5)
    np.testing.assert_allclose(lower, [8.8] * 5)
    np.testing.assert_allclose(upper, [15.2] * 5)
    assert coverage([9, 15, 8, 16, 12], lower, upper) == 3 / 5
    assert coverage([lower[0], upper[1]], lower[:2], upper[:2]) == 1


def test_blend_band_refits_offsets_for_selected_weights(band_validation):
    band = fit_blend_band(band_validation, {1: 1, 2: 4})
    # prediction 11.2 gives residuals [-3.2, -1.2, 0.8, 2.8, 4.8]
    assert (band.lo, band.hi) == pytest.approx((-2.4, 4.0))


@pytest.mark.parametrize("nonnegative,expected_lower", [(True, 0), (False, -0.7)])
def test_blend_band_clips_only_when_target_is_nonnegative(
    band_validation, nonnegative, expected_lower
):
    band = fit_blend_band(band_validation)
    lower, upper = apply_band(band, [0.5], nonnegative)
    assert lower[0] == pytest.approx(expected_lower)
    assert upper[0] == pytest.approx(5.7)


@pytest.mark.parametrize("count", [0, 1, 3, 4])
def test_blend_band_counts_unique_finite_validation_targets(band_validation, count):
    rows = {name: values[: count * 2] for name, values in band_validation.items()}
    band = fit_blend_band(rows)
    if count < 4:
        assert band is None
    else:
        assert band.n == 4
        assert (band.lo, band.hi) == pytest.approx((-1.4, 3.4))


def test_blend_band_excludes_nonfinite_actuals_and_predictions(band_validation):
    band_validation["actual"][:2] = np.nan
    band = fit_blend_band(band_validation)
    assert band.n == 4
    assert (band.lo, band.hi) == pytest.approx((0.6, 5.4))
    band_validation["prediction"][2] = np.inf
    assert fit_blend_band(band_validation) is None
