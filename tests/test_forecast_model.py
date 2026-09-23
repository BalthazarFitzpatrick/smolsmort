"""fits and verdicts. which objective wins is data-dependent (measured 2026-09-23: in a volume
surge the naive model runs early, aft overshoots) - so the search picks it on validation, and these
tests pin only what must always hold"""

from __future__ import annotations

import numpy as np
import pytest

from smolsmort.forecast.evaluate import (
    apply_band,
    brier,
    coverage,
    fit_band,
    group_median,
    judge,
    log_loss,
    mae,
    open_consistency,
    wape,
)
from smolsmort.forecast.model import ModelError, fit_model, predict

pytest.importorskip("xgboost")

BASE = np.array([4.0, 7.0, 10.0, 5.5, 14.0])
FTYPES = ["c", "q"]


def _draw(rng, created):
    """rows with a branch effect, a size effect and a right-skewed lead in weeks"""
    branch = rng.integers(0, len(BASE), len(created))
    size = rng.lognormal(2.0, 0.8, len(created))
    lead = np.maximum(0.2, BASE[branch] + 0.08 * size + rng.gamma(2.0, 3.0, len(created)) - 3.0)
    return np.column_stack([branch, size]).astype(np.float32), lead


def _export(rng, created, cut=105.0):
    """what an export taken at `cut` says: a known lead, or open with the weeks waited so far"""
    x, lead = _draw(rng, np.sort(created))
    created = np.sort(created)
    known = created + lead <= cut
    lower = np.where(known, lead, np.maximum(cut - created, 0.1))
    upper = np.where(known, lead, np.inf)
    return x, lead, known, lower, upper


@pytest.fixture(scope="module")
def surge():
    """a thin early history, then most rows in the last 20 weeks - a surge that leaves many open"""
    rng = np.random.default_rng(0)
    created = np.concatenate([rng.uniform(0, 85, 800), rng.uniform(85, 105, 3200)])
    return _export(rng, created)


@pytest.fixture(scope="module")
def steady():
    rng = np.random.default_rng(1)
    x, lead, known, lower, upper = _export(rng, rng.uniform(60, 105, 4000))
    xv, yv = _draw(rng, np.zeros(800))
    xt, yt = _draw(rng, np.zeros(800))
    return x, lead, known, lower, upper, xv, yv, xt, yt


def test_aft_respects_what_open_rows_have_already_waited(surge):
    x, lead, known, lower, upper = surge
    assert 0.3 < 1 - known.mean() < 0.6, "the surge should leave a large share open"
    naive = fit_model(x[known], objective="absolute", y=lead[known], feature_types=FTYPES)
    aft = fit_model(x, objective="aft", lower=lower, upper=upper, feature_types=FTYPES)
    waited = lower[~known]
    naive_share = open_consistency(predict(naive, x[~known]), waited)
    aft_share = open_consistency(predict(aft, x[~known]), waited)
    assert aft_share > naive_share + 0.1, f"aft {aft_share:.2f} vs naive {naive_share:.2f}"


@pytest.mark.parametrize("objective", ["absolute", "aft"])
def test_the_band_lands_near_80_percent_on_unseen_rows(steady, objective):
    x, lead, known, lower, upper, xv, yv, xt, yt = steady
    if objective == "aft":
        fitted = fit_model(x, objective="aft", lower=lower, upper=upper, feature_types=FTYPES)
    else:
        fitted = fit_model(x[known], objective=objective, y=lead[known], feature_types=FTYPES)
    band = fit_band(yv, predict(fitted, xv))
    lo, hi = apply_band(band, predict(fitted, xt), nonnegative=True)
    assert 0.65 <= coverage(yt, lo, hi) <= 0.95
    assert mae(yt, predict(fitted, xt)) < 6


def test_rounds_come_from_the_end_of_train_not_the_cap(steady):
    x, lead, known, *_ = steady
    fitted = fit_model(x[known], objective="squared", y=lead[known], feature_types=FTYPES)
    assert 1 <= fitted.rounds < 2000


def test_a_negative_target_is_refused_for_a_count_objective():
    x = np.zeros((40, 1), dtype=np.float32)
    with pytest.raises(ModelError, match="non-negative"):
        fit_model(x, objective="poisson", y=np.full(40, -1.0))


def test_too_few_rows_are_refused_by_name():
    with pytest.raises(ModelError, match="too few"):
        fit_model(np.zeros((5, 1)), objective="squared", y=np.ones(5))


def test_classification_returns_probabilities_per_class():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 2)).astype(np.float32)
    labels = np.where(x[:, 0] > 0.5, "late", np.where(x[:, 0] < -0.5, "early", "on time"))
    fitted = fit_model(x, objective="softprob", labels=labels)
    probs = predict(fitted, x)
    assert fitted.classes == ["early", "late", "on time"]
    assert probs.shape == (300, 3) and np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    codes = np.array([fitted.classes.index(v) for v in labels])
    assert log_loss(codes, probs) < 0.5 and brier(codes, probs) < 0.3
    binary = fit_model(x, objective="logistic", labels=np.where(x[:, 1] > 0, "yes", "no"))
    assert predict(binary, x).shape == (300, 2)


def test_band_shapes_follow_the_number_of_residuals():
    assert fit_band([1, 2, 3], [1, 2, 3]) is None
    few = fit_band(np.arange(1, 11, dtype=float), np.arange(1, 11) + 0.5)
    assert few.lo == -few.hi, "under 20 residuals the band is symmetric"
    many = fit_band(np.arange(1, 41, dtype=float), np.arange(1, 41) * 0.9)
    assert many.n == 40 and many.lo != -many.hi
    lo, _ = apply_band(many, np.array([0.0, 1.0]), nonnegative=True)
    assert (lo >= 0).all()
    assert np.isnan(apply_band(None, np.array([1.0]))[0]).all()


def test_metrics_and_the_group_baseline():
    assert wape([10, 0, 10], [8, 1, 12]) == pytest.approx(5 / 20)
    assert mae([1, np.nan, 3], [2, 5, 3]) == pytest.approx(0.5)
    baseline = group_median(["a", "a", "b"], [1.0, 3.0, 10.0], ["a", "b", "c"])
    assert baseline.tolist() == [2.0, 10.0, 3.0]
    assert open_consistency([5, 1], [3, 3]) == 0.5


def test_the_verdict_names_each_failed_check_with_its_numbers():
    good = judge(model_error=7.0, baseline_error=10.0, band_coverage=0.8, test_rows=200)
    assert good["trusted"] and all(c["passed"] for c in good["checks"])
    bad = judge(
        model_error=10.0, baseline_error=10.5, band_coverage=0.5, test_rows=12, test_steps=8
    )
    assert not bad["trusted"]
    assert bad["summary"].startswith("do not act on this forecast")
    failed = {c["check"]: c["detail"] for c in bad["checks"] if not c["passed"]}
    assert set(failed) == {
        "beats the naive baseline",
        "band is calibrated",
        "enough history to judge",
    }
    assert "ratio 0.95" in failed["beats the naive baseline"]
    assert "50%" in failed["band is calibrated"]
    no_band = judge(model_error=1, baseline_error=2, band_coverage=None, test_rows=100)
    assert not no_band["trusted"]


def test_a_band_on_a_target_centred_at_zero_is_absolute_not_relative():
    y = np.array([0.0, 0.0, -1.0, 2.0, 0.0, 5.0, 0.0, 1.0] * 5)
    band = fit_band(y, np.zeros_like(y))
    assert band is not None and not band.relative
    lo, hi = apply_band(band, np.array([0.0]))
    assert hi[0] - lo[0] < 10, "an absolute band stays in the target's own unit"
    positive = fit_band(np.arange(1, 41, dtype=float), np.arange(1, 41) * 0.9)
    assert positive.relative


def test_a_band_as_wide_as_the_data_itself_is_not_trusted():
    common = {"model_error": 1.0, "baseline_error": 2.0, "band_coverage": 0.8, "test_rows": 500}
    sharp = judge(**common, band_width=3.0, spread=10.0)
    vague = judge(**common, band_width=40.0, spread=12.0)
    assert sharp["trusted"]
    assert not vague["trusted"]
    failed = [c for c in vague["checks"] if not c["passed"]]
    assert [c["check"] for c in failed] == ["band is sharper than the data's own spread"]
    assert "40" in failed[0]["detail"] and "12" in failed[0]["detail"]
