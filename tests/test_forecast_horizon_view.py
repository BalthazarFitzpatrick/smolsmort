"""hand-checked lead selections and blends of saved forecast rows"""

import numpy as np
import pytest

from smolsmort.forecast.horizon_view import blend_rows, lead_rows


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
    blended = blend_rows(overlapping_rows, 2)
    np.testing.assert_allclose(blended["prediction"], [20, 50, 90])
    np.testing.assert_allclose(blended["naive_prediction"], [18, 48, 88])
    np.testing.assert_array_equal(blended["actual"], [25, 55, 100])
    np.testing.assert_array_equal(blended["series"], ["a", "a", "a"])
    assert not {"lower", "upper", "origin", "weeks_ahead", "bucket"} & blended.keys()
    np.testing.assert_allclose(blend_rows(overlapping_rows, 1)["prediction"], blended["prediction"])
    np.testing.assert_allclose(blend_rows(overlapping_rows, 3)["prediction"], blended["prediction"])


def test_explicit_weights_normalize_over_present_leads(overlapping_rows):
    blended = blend_rows(overlapping_rows, 2, {1: 1, 2: 3, 3: 2})
    np.testing.assert_allclose(blended["prediction"], [130 / 6, 48, 90])
    np.testing.assert_allclose(blended["naive_prediction"], [118 / 6, 46, 88])


def test_explicit_zero_weights_do_not_include_nonfinite_predictions(overlapping_rows):
    overlapping_rows["prediction"][0] = np.nan
    blended = blend_rows(overlapping_rows, 2, {2: 1, 3: 1})
    np.testing.assert_allclose(blended["prediction"], [25, 50, 90])


def test_large_finite_weights_do_not_overflow(overlapping_rows):
    blended = blend_rows(overlapping_rows, 2, {1: 1e308, 2: 1e308, 3: 1e308})
    np.testing.assert_allclose(blended["prediction"], [20, 50, 90])


def test_series_are_never_blended_together(overlapping_rows):
    overlapping_rows["series"][0] = "b"
    blended = blend_rows(overlapping_rows, 2)
    np.testing.assert_array_equal(blended["series"], ["b", "a", "a", "a"])
    np.testing.assert_allclose(blended["prediction"], [10, 25, 50, 90])


def test_single_row_keeps_target_values(overlapping_rows):
    single = {name: values[-1:] for name, values in overlapping_rows.items()}
    blended = blend_rows(single, 2)
    for name in blended:
        np.testing.assert_array_equal(blended[name], single[name])


def test_empty_table_stays_empty(overlapping_rows):
    empty = {name: values[:0] for name, values in overlapping_rows.items()}
    assert all(len(values) == 0 for values in blend_rows(empty, 2).values())


def test_legacy_tables_pass_through(overlapping_rows):
    legacy = {name: values for name, values in overlapping_rows.items() if name != "weeks_ahead"}
    assert lead_rows(legacy, 2) is legacy
    assert blend_rows(legacy, 2) is legacy


@pytest.mark.parametrize("n", [0, -1, 1.5, "2", True])
def test_invalid_lead_is_rejected(overlapping_rows, n):
    with pytest.raises(ValueError, match="lead must be a positive integer"):
        lead_rows(overlapping_rows, n)
    with pytest.raises(ValueError, match="lead must be a positive integer"):
        blend_rows(overlapping_rows, n)


@pytest.mark.parametrize(
    "weights",
    [{}, {1: 0, 2: 0}, {1: -1}, {1: np.nan}, {1: np.inf}, {1: "1"}, {1: True}],
)
def test_invalid_weight_values_are_rejected(overlapping_rows, weights):
    with pytest.raises(ValueError, match="weights"):
        blend_rows(overlapping_rows, 2, weights)


def test_weight_integer_beyond_float_range_is_rejected(overlapping_rows):
    with pytest.raises(ValueError, match="weights must be finite and nonnegative"):
        blend_rows(overlapping_rows, 2, {1: 10**400, 2: 1, 3: 1})


@pytest.mark.parametrize("weights", [{0: 1}, {-1: 1}, {1.5: 1}, {"1": 1}, {True: 1}])
def test_invalid_weight_leads_are_rejected(overlapping_rows, weights):
    with pytest.raises(ValueError, match="lead"):
        blend_rows(overlapping_rows, 2, weights)


def test_target_without_a_positive_present_weight_is_rejected(overlapping_rows):
    with pytest.raises(ValueError, match="positive value for every target week"):
        blend_rows(overlapping_rows, 1, {1: 1})


def test_helpers_do_not_modify_input(overlapping_rows):
    original = {name: values.copy() for name, values in overlapping_rows.items()}
    lead_rows(overlapping_rows, 2)
    blend_rows(overlapping_rows, 2, {1: 1, 2: 2, 3: 3})
    for name in original:
        np.testing.assert_array_equal(overlapping_rows[name], original[name])
