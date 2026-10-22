"""hand-checked survival scores, train-only baselines, and censored entity verdicts"""

from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import evaluate as ev
from smolsmort.forecast import hazard, pipeline
from smolsmort.forecast.hazard import HazardFitted, expand_periods, predict_hazard
from smolsmort.forecast.model import fit_model
from smolsmort.forecast.spec import ANCHOR, ROW


def fit_censor_km(lower, upper):
    knots = np.unique(lower)
    return knots, ev.kaplan_meier(lower, upper, knots, censoring=True)


def test_six_entity_ibs_and_concordance_match_hand_computation():
    lower = np.arange(1, 7, dtype=float)
    upper = np.array([1, np.inf, 3, 4, np.inf, 6])
    weeks = np.arange(1, 5)
    curves = np.tile([0.8, 0.6, 0.4, 0.2], (6, 1))
    censor_km = fit_censor_km(lower, upper)
    # g drops at 2 by 1/5, then at 5 by 1/2: g = [1, 4/5, 4/5, 4/5, 2/5, 2/5]
    # event weights at 1, 3, 4 are 1, 5/4, 5/4; all six rows remain in each denominator
    np.testing.assert_allclose(censor_km[1], [1, 0.8, 0.8, 0.8, 0.4, 0.4], atol=1e-9, rtol=0)
    # b1 = (.8^2 + 5*.2^2)/6 = .14; b2 = (.6^2 + 4*.4^2*5/4)/6 = 29/150
    # b3 = (.4^2*9/4 + 3*.6^2*5/4)/6 = .285; b4 = (.2^2*7/2 + 2*.8^2*5/4)/6 = .29
    # trapezoids over 1..4: (.14 + 2*29/150 + 2*.285 + .29)/6 = 52/225
    assert ev.integrated_brier(lower, upper, curves, weeks, censor_km) == pytest.approx(
        52 / 225, abs=1e-9, rel=0
    )
    # event 1: five concordant pairs; event 3: three; event 4: one wrong and one tied
    # censored 2 and 5 cannot be the earlier event: (5 + 3 + .5)/10 = .85
    risk = np.array([1, 4, 3, 6, 5, 6])
    assert ev.concordance(lower, upper, risk) == pytest.approx(0.85, abs=1e-9, rel=0)
    assert ev.concordance(lower, upper, -risk) == pytest.approx(0.15, abs=1e-9, rel=0)


def test_kaplan_meier_matches_tied_events_and_censoring_by_hand():
    lower = [1, 2, 2, 2, 3, 4]
    upper = [1, 2, 2, np.inf, np.inf, 4]
    weeks = np.arange(6)
    # event km: (1-1/6) at 1, then *(1-2/5) at 2 = 1/2; final event at 4 exhausts it
    np.testing.assert_allclose(
        ev.kaplan_meier(lower, upper, weeks), [1, 5 / 6, 1 / 2, 1 / 2, 0, 0], atol=1e-9, rtol=0
    )
    # reverse km: at 2 remove two events first, then 1-1/(5-2) = 2/3; at 3 halve it
    np.testing.assert_allclose(
        ev.kaplan_meier(lower, upper, weeks, censoring=True),
        [1, 1, 2 / 3, 1 / 3, 1 / 3, 1 / 3],
        atol=1e-9,
        rtol=0,
    )


def test_concordance_ignores_event_time_ties_but_keeps_censor_ties():
    # each event at 1 ties its prediction with the censor at 1 and correctly precedes event 2
    assert ev.concordance([1, 1, 1, 2], [1, 1, np.inf, 2], [2, 2, 2, 3]) == 0.75
    assert np.isnan(ev.concordance([1, 2], [np.inf, np.inf], [2, 1]))


def test_brier_uses_exact_train_knots_and_event_left_limits():
    censor_km = fit_censor_km([1.25, 1.5, 3], [1.25, np.inf, 3])
    # at week 2, event 1.5 uses g(1.5-) = 1; survivor 3 uses g(2) = 1/2
    score = ev.integrated_brier([1.5, 3], [1.5, 3], [[0.5], [0.5]], [2], censor_km)
    assert score == pytest.approx((0.25 + 0.25 * 2) / 2, abs=1e-9, rel=0)


def test_brier_counts_open_rows_and_omits_their_unobserved_future():
    censor_km = fit_censor_km([4, 5], [4, 5])
    curves = np.array([[0.5, 0.25, 0.125], [0.75, 0.5, 0.25]])
    # week scores are (.25 + .0625)/2, .0625/2, .015625/2; censor at 2 adds no later term
    expected = (0.15625 + 2 * 0.03125 + 0.0078125) / 4
    score = ev.integrated_brier([1, 2], [1, np.inf], curves, [1, 2, 3], censor_km)
    assert score == pytest.approx(expected, abs=1e-9, rel=0)
    closed_only = ev.integrated_brier([1], [1], curves[:1], [1, 2, 3], censor_km)
    assert score != pytest.approx(closed_only)


def test_brier_refuses_zero_censoring_support_instead_of_clipping_weights():
    censor_km = fit_censor_km([1, 2], [1, np.inf])
    with pytest.raises(ValueError, match="positive train censoring survival"):
        ev.integrated_brier([3], [3], [[0.5, 0.5]], [1, 2], censor_km)


def test_group_km_conditions_on_wait_and_pools_unseen_groups():
    curves = ev.group_kaplan_meier(
        ["a", "a", "b", "b"],
        [1, 3, 2, 4],
        [1, 3, 2, 4],
        ["a", "unseen", "a", "a"],
        [1, 2, 3, 4],
        waited=[0, 0, 2.5, 5],
    )
    np.testing.assert_allclose(
        curves,
        [
            [0.5, 0.5, 0, 0],
            [0.75, 0.5, 0.25, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ],
        atol=1e-9,
        rtol=0,
    )


def test_normal_aft_uses_log_prediction_scale_and_conditional_survival():
    weeks = 2 * np.exp([-1, 0, 1])
    expected = np.array([0.8413447460685429, 0.5, 0.15865525393145707])
    np.testing.assert_allclose(ev.aft_survival([2], weeks), expected[None, :], atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        ev.aft_survival([2], 2 * np.exp([-2, 0, 2]), scale=2),
        expected[None, :],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        ev.aft_survival([2], weeks, waited=[2]),
        [[1, 1, expected[2] / 0.5]],
        atol=1e-12,
        rtol=0,
    )


def test_extreme_aft_and_missing_scale_step_curve():
    np.testing.assert_allclose(
        ev.aft_survival([2], [1, 2, 3], distribution="extreme"),
        np.exp([[-0.5, -1, -1.5]]),
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_array_equal(ev.aft_survival([2], [1, 2, 3], scale=None), [[1, 0, 0]])


def make_workspace(unit="week"):
    units_per_week = 7 if unit == "day" else 1
    lower = np.array([1, 2, 4, 5, 2, 8, 1, 3], dtype=float) * units_per_week
    upper = np.where([True, True, True, False, True, False, True, False], lower, np.inf)
    train_lower, train_upper = lower.copy(), upper.copy()
    train_lower[4:], train_upper[4:] = np.nan, np.nan
    test_lower, test_upper = lower.copy(), upper.copy()
    test_lower[5] = units_per_week
    test_lower[6:], test_upper[6:] = np.nan, np.nan
    x = np.arange(8, dtype=float).reshape(-1, 1)
    return SimpleNamespace(
        mode="entity",
        unit=unit,
        spec=SimpleNamespace(
            task="regression",
            named=lambda _role: ["duration"],
            inputs=lambda: [],
            censor=SimpleNamespace(unit=unit),
        ),
        table={
            ROW: np.arange(8),
            ANCHOR: np.datetime64("2024-01-01") + np.arange(8) * 7,
            "duration": np.where(np.isfinite(upper), lower, np.nan),
        },
        features=SimpleNamespace(pick=lambda _families: (x, ["entity"], ["q"])),
        masks={
            "train": np.arange(8) < 4,
            "val": np.isin(np.arange(8), [4, 5]),
            "test": np.arange(8) > 5,
        },
        bounds={
            "val": (train_lower, train_upper),
            "test": (test_lower, test_upper),
            "final": (lower, upper),
        },
        predict_rows=np.isinf(upper),
    )


@pytest.mark.parametrize("unit", ["week", "day"])
@pytest.mark.parametrize(
    "objective,params",
    [
        ("hazard", {}),
        ("aft", {}),
        ("aft", {"aft_loss_distribution": "extreme", "aft_loss_distribution_scale": 0.7}),
    ],
)
def test_finish_includes_open_test_bounds_and_keeps_train_censoring_weights(
    monkeypatch, unit, objective, params
):
    ws = make_workspace(unit)
    fit_calls = []
    fitted = SimpleNamespace(eval_history=None)

    def capture_fit(_ws, _genome, rows, phase, _nthread):
        fit_calls.append((phase, rows.copy()))
        result = HazardFitted(fitted, cap=4) if objective == "hazard" else fitted
        return result, ws.features.pick(set())[0]

    monkeypatch.setattr(pipeline, "_entity_fit", capture_fit)
    monkeypatch.setattr(hazard, "predict", lambda _fitted, x: np.full(len(x), 0.2))
    units_per_week = 7 if unit == "day" else 1
    monkeypatch.setattr(pipeline, "predict", lambda _fitted, x: np.full(len(x), 2 * units_per_week))
    monkeypatch.setattr(pipeline, "measure_importance", lambda *_args: [])
    monkeypatch.setattr(pipeline, "combine_importance", lambda *_args: [])
    genome = pipeline.make_genome(["entity:measure"], objective, params)
    result = pipeline.finish(ws, genome)
    assert result["test"]["rows"] == 2
    assert len(result["test"]["prediction"]) == 2
    assert result["verdict"]["accuracy"]["metric"] == "ibs"
    # the curve's 80% band is reported as a fact next to the ibs verdict
    assert [c["check"] for c in result["verdict"]["checks"] if c["group"] == "band"] == [
        "band coverage"
    ]
    facts = {check["check"]: check["detail"] for check in result["verdict"]["checks"]}
    assert "c-index" in facts["concordance"]
    assert "1 open test entities included" in facts["survival scoring"]
    assert [phase for phase, _rows in fit_calls] == ["val", "test", "final"]
    np.testing.assert_array_equal(fit_calls[1][1], ws.masks["train"] | ws.masks["val"])
    assert np.isnan(result["validation"]["truth"][1])
    weeks = np.arange(1, 5)
    if objective == "hazard":
        curve = 0.8 ** (weeks + 1)
    else:
        curve = ev.aft_survival(
            [2],
            weeks,
            scale=params.get("aft_loss_distribution_scale", 1),
            distribution=params.get("aft_loss_distribution", "normal"),
        )[0]
    # open test row waited 3: zero errors before 3, then omitted, but denominator stays two
    # g is one on 1..4 from train, even though train+val contains a censor at 1
    expected = (curve[0] ** 2 + 2 * curve[1] ** 2 + 2 * curve[2] ** 2 + curve[3] ** 2) / 12
    assert result["model_error"] == pytest.approx(expected, abs=1e-9, rel=0)
    dropped = ev.integrated_brier([1], [1], curve[None, :], weeks, fit_censor_km([4], [4]))
    assert dropped == pytest.approx(2 * result["model_error"], abs=1e-9, rel=0)


def test_finish_selects_km_dimension_on_validation_and_pools_unseen_test_groups(monkeypatch):
    ws = make_workspace()
    duration = np.array([1, 1, 4, 4, 1, 4, 1, 4], dtype=float)
    train, test = duration.copy(), duration.copy()
    train[4:], test[6:] = np.nan, np.nan
    ws.bounds = {"val": (train, train), "test": (test, test), "final": (duration, duration)}
    ws.table["duration"] = duration
    ws.table["noise"] = np.array(["x", "y", "x", "y", "x", "y", "x", "x"], dtype=object)
    ws.table["segment"] = np.array(["a", "a", "b", "b", "a", "b", "a", "unseen"], dtype=object)
    ws.spec.inputs = lambda: ["noise", "segment"]
    monkeypatch.setattr(
        pipeline,
        "_entity_fit",
        lambda *_args: (
            HazardFitted(SimpleNamespace(eval_history=None), cap=4),
            ws.features.pick(set())[0],
        ),
    )
    monkeypatch.setattr(hazard, "predict", lambda _fitted, x: np.full(len(x), 0.2))
    genome = pipeline.make_genome(["entity:measure"], "hazard", {})
    _val, _test, verdict = pipeline._entity_survival_verdict(ws, genome, 1)
    facts = {check["check"]: check["detail"] for check in verdict["checks"]}
    assert facts["kaplan-meier baseline"] == "grouped by segment"
    # a is exact; unseen gets pooled s = [.5, .5, .5, 0], hence b = [.125, .125, .125, 0]
    assert verdict["accuracy"]["baseline_error"] == pytest.approx(5 / 48, abs=1e-9, rel=0)


def test_geometric_hazard_beats_pooled_km_and_matches_per_dimension_km():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(217)

    def draw_entities():
        dimension = np.tile([0, 1], 6000)
        events = rng.geometric(np.where(dimension == 0, 0.2, 0.05)) - 1
        ages = rng.integers(8, 81, len(dimension))
        closed = events <= ages
        lower = np.where(closed, events, ages)
        upper = np.where(closed, events, np.inf)
        anchors = np.datetime64("2026-01-05") - ages * np.timedelta64(7, "D")
        return dimension, lower, upper, anchors

    dim, lower, upper, anchors = draw_entities()
    periods = expand_periods(dim[:, None], lower, upper, anchors, cap=80)
    order = np.argsort(periods.date, kind="stable")
    fitted = fit_model(
        periods.x[order],
        objective="hazard",
        y=periods.y[order],
        feature_types=["q", "q", "q"],
        params={"max_depth": 1, "eta": 0.15, "min_child_weight": 100, "tree_method": "hist"},
        max_rounds=80,
        patience=10,
        nthread=1,
    )
    test_dim, test_lower, test_upper, test_anchors = draw_entities()
    assert np.isinf(test_upper).any()
    weeks = np.arange(1, 21)
    curves = predict_hazard(HazardFitted(fitted, periods.cap), test_dim[:, None], test_anchors)
    censor_km = fit_censor_km(lower, upper)
    model_ibs = ev.integrated_brier(
        test_lower, test_upper, curves["survival"][:, weeks], weeks, censor_km
    )
    pooled = ev.kaplan_meier(lower, upper, weeks)
    pooled_ibs = ev.integrated_brier(
        test_lower,
        test_upper,
        np.tile(pooled, (len(test_dim), 1)),
        weeks,
        censor_km,
    )
    per_dim = ev.group_kaplan_meier(dim, lower, upper, test_dim, weeks)
    per_dim_ibs = ev.integrated_brier(test_lower, test_upper, per_dim, weeks, censor_km)
    truth = (1 - np.where(test_dim == 0, 0.2, 0.05))[:, None] ** (weeks + 1)
    truth_ibs = ev.integrated_brier(test_lower, test_upper, truth, weeks, censor_km)
    assert model_ibs / pooled_ibs < 0.9
    assert model_ibs / per_dim_ibs == pytest.approx(1.0, abs=0.1)
    assert per_dim_ibs / truth_ibs == pytest.approx(1.0, abs=0.1)


def test_curve_band_coverage_counts_closed_events_inside_the_central_band():
    from smolsmort.forecast.evaluate import curve_band_coverage

    weeks = np.arange(1, 6, dtype=float)
    # one shared curve: survival 0.95, 0.85, 0.5, 0.2, 0.05 -> band from week 2 to week 5
    survival = np.tile([0.95, 0.85, 0.5, 0.2, 0.05], (4, 1))
    lower = np.array([3.0, 0.0, 9.0, 2.0])
    upper = np.array([3.0, 0.0, 9.0, np.inf])
    # closed events 3 (inside), 0 (before), 9 (after); the open entity is ignored
    assert curve_band_coverage(lower, upper, survival, weeks) == pytest.approx(1 / 3)
    assert curve_band_coverage(lower[3:], upper[3:], survival[3:], weeks) is None
