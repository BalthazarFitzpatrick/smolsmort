import numpy as np

from smolsmort.forecast import evaluate as ev


def test_series_bands_use_twenty_residuals_and_fall_back_otherwise():
    prediction = np.full(35, 100.0)
    actual = prediction + np.r_[np.linspace(-10, 10, 25), np.full(10, 80)]
    leads = np.ones(35, dtype=int)
    series = np.array(["large"] * 25 + ["small"] * 10)
    bands = ev.fit_lead_bands(actual, prediction, leads, series)
    assert set(bands["series"]) == {"large"}
    lower, upper = ev.apply_lead_bands(bands, prediction, leads, series)
    own_lower, own_upper = ev.apply_band(
        ev.fit_band(actual[:25], prediction[:25]), prediction[:25], True
    )
    pooled_lower, pooled_upper = ev.apply_band(bands[1], prediction[25:], True)
    np.testing.assert_allclose(lower[:25], own_lower)
    np.testing.assert_allclose(upper[:25], own_upper)
    np.testing.assert_allclose(lower[25:], pooled_lower)
    np.testing.assert_allclose(upper[25:], pooled_upper)


def test_old_json_pooled_bands_apply_without_series_overrides():
    band = ev.fit_band(np.arange(20.0), np.full(20, 10.0))
    lower, upper = ev.apply_lead_bands({"1": band.__dict__}, [10], [1], ["a"])
    expected = ev.apply_band(band, [10], True)
    np.testing.assert_allclose(lower, expected[0])
    np.testing.assert_allclose(upper, expected[1])


def test_inner_band_is_narrower_and_verdict_reports_both_coverages():
    actual = np.arange(25.0) + 1
    prediction = np.full(25, 12.0)
    leads = np.ones(25, dtype=int)
    outer = ev.fit_lead_bands(actual, prediction, leads)
    inner = ev.fit_lead_bands(actual, prediction, leads, level=0.5)
    lo, hi = ev.apply_lead_bands(outer, prediction, leads)
    lo_50, hi_50 = ev.apply_lead_bands(inner, prediction, leads)
    assert np.all(lo_50 >= lo) and np.all(hi_50 <= hi)
    verdict = ev.judge(
        model_error=0.1,
        baseline_error=0.2,
        test_rows=25,
        band_coverage=ev.coverage(actual, lo, hi),
        band_coverage_50=ev.coverage(actual, lo_50, hi_50),
    )
    details = [check["detail"] for check in verdict["checks"]]
    assert "80% band covered 76% of test rows" in details
    assert "50% band covered 52% of test rows" in details
