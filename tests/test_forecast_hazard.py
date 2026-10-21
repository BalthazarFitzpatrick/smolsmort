"""weekly risk sets, censoring, calibration, and entity forecasts from the hazard objective"""

from __future__ import annotations

import datetime as dt
import importlib
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from forecast_fixtures import AS_OF, write_rows

from smolsmort.forecast import hazard, model, pipeline
from smolsmort.forecast.hazard import HazardFitted, expand_periods, predict_hazard
from smolsmort.forecast.model import fit_model
from smolsmort.forecast.spec import ANCHOR, LOWER, ROW, UPPER, Censor, Column, PrepSpec
from smolsmort.forecast.splits import labels_as_of


def test_expansion_keeps_the_event_week_and_only_completed_open_weeks():
    anchors = np.array(["2024-01-01", "2024-02-05"], dtype="datetime64[D]")
    periods = expand_periods(
        np.array([[10.0], [20.0]]), [3, 4], [3, np.inf], anchors, cap=6, entity=[17, 29]
    )
    np.testing.assert_array_equal(periods.entity, [17] * 4 + [29] * 4)
    np.testing.assert_array_equal(periods.week, [0, 1, 2, 3] * 2)
    np.testing.assert_array_equal(periods.y, [0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(periods.x[:, 0], [10] * 4 + [20] * 4)
    np.testing.assert_array_equal(periods.x[:, -2], periods.week)
    np.testing.assert_array_equal(periods.x[:, -1], [1, 2, 3, 4, 6, 7, 8, 9])
    np.testing.assert_array_equal(
        periods.date, np.repeat(anchors, 4) + periods.week * np.timedelta64(7, "D")
    )


def test_expansion_pools_the_tail_once_and_never_exceeds_the_cap():
    periods = expand_periods(
        np.arange(3).reshape(-1, 1),
        [8.5, 9.5, 0],
        [8.5, np.inf, 0],
        np.full(3, np.datetime64("2024-01-01")),
        cap=3,
    )
    np.testing.assert_array_equal(periods.entity, [0] * 4 + [1] * 4 + [2])
    np.testing.assert_array_equal(periods.week, [0, 1, 2, 3] * 2 + [0])
    np.testing.assert_array_equal(periods.y, [0, 0, 0, 1, 0, 0, 0, 0, 1])
    assert periods.cap == 3
    assert periods.week.max() == periods.cap


def test_expansion_uses_train_p95_and_drops_unobserved_entities():
    lower = np.array([0, 1, 2, 3, 4, 30, np.nan, -1])
    upper = np.array([0, 1, 2, np.inf, 4, 30, np.nan, np.inf])
    periods = expand_periods(
        np.arange(len(lower)).reshape(-1, 1),
        lower,
        upper,
        np.full(len(lower), np.datetime64("2024-01-01")),
    )
    assert periods.cap == int(np.floor(np.percentile(lower[:6], 95)))
    assert set(periods.entity) == set(range(6))
    assert np.all(periods.week <= periods.cap)
    assert periods.y[periods.entity == 5][-1] == 1


def test_open_entities_with_no_completed_week_add_no_training_rows():
    periods = expand_periods(
        np.zeros((2, 1)),
        [0, 0.9],
        [np.inf, np.inf],
        np.full(2, np.datetime64("2024-01-01")),
        cap=0,
    )
    assert periods.x.shape == (0, 3)
    assert periods.y.size == periods.entity.size == periods.week.size == 0


def test_prediction_expansion_has_the_same_grid_for_every_entity():
    periods = expand_periods(
        np.array([[2.0], [7.0]]),
        None,
        None,
        np.array(["2020-12-28", "2021-01-04"], dtype="datetime64[D]"),
        cap=2,
        prediction=True,
    )
    assert periods.y is None
    np.testing.assert_array_equal(periods.entity, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(periods.week, [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(periods.x[:, -1], [53, 1, 2, 1, 2, 3])


def test_conditional_survival_starts_at_one_after_six_weeks(monkeypatch):
    monkeypatch.setattr(hazard, "predict", lambda _fitted, x: np.full(len(x), 0.2))
    fitted = HazardFitted(SimpleNamespace(eval_history=None), cap=12)
    curves = predict_hazard(
        fitted,
        np.zeros((2, 1)),
        np.full(2, np.datetime64("2024-01-01")),
        waited=np.array([0, 6]),
    )
    np.testing.assert_array_equal(curves["week"], np.arange(13))
    np.testing.assert_allclose(curves["hazard"], 0.2)
    np.testing.assert_allclose(curves["survival"][0], 0.8 ** np.arange(1, 14))
    np.testing.assert_array_equal(curves["survival"][1, :6], np.ones(6))
    np.testing.assert_allclose(curves["survival"][1, 6:], 0.8 ** np.arange(1, 8))
    assert curves["survival_start"][1, 6] == pytest.approx(1.0)
    np.testing.assert_allclose(curves["survival_start"][1, 6:], 0.8 ** np.arange(7))
    np.testing.assert_array_equal(curves["prediction"], [3, 9])
    np.testing.assert_array_equal(curves["lower"], [0, 6])
    # past the cap 12 the 0.2 tail repeats: 0.8^7 at week 12, four more weeks reach 0.8^11 <= 0.1
    np.testing.assert_array_equal(curves["upper"], [10, 16])


def test_quantiles_stop_at_the_tail_limit_when_survival_never_crosses(monkeypatch):
    monkeypatch.setattr(hazard, "predict", lambda _fitted, x: np.zeros(len(x)))
    curves = predict_hazard(
        HazardFitted(SimpleNamespace(eval_history=None), cap=4),
        np.zeros((1, 1)),
        np.array(["2024-01-01"], dtype="datetime64[D]"),
    )
    np.testing.assert_array_equal(curves["survival"], np.ones((1, 5)))
    # a zero tail never crosses: quantiles stop at the tail limit past the cap and are flagged
    for name in ("prediction", "lower", "upper"):
        np.testing.assert_array_equal(curves[name], [4 + hazard.TAIL_LIMIT_WEEKS])
    assert curves["beyond_limit"].tolist() == [True]


def test_events_after_a_phase_cut_cannot_change_that_phases_risk_set():
    anchors = np.array(["2024-01-01"] * 3 + ["2024-02-12"], dtype="datetime64[D]")
    events = np.array([1.0, 8.0, 20.0, 0.0])
    changed = np.array([1.0, 18.0, 90.0, 4.0])
    cut = dt.date(2024, 1, 29)
    first_bounds = labels_as_of(anchors, events, events, cut, 7)
    changed_bounds = labels_as_of(anchors, changed, changed, cut, 7)
    x = np.arange(4).reshape(-1, 1)
    before = expand_periods(x, *first_bounds, anchors, cap=5)
    after = expand_periods(x, *changed_bounds, anchors, cap=5)
    for name in ("x", "y", "entity", "week", "date"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    assert 3 not in before.entity
    assert np.all(before.date <= np.datetime64(cut))
    np.testing.assert_array_equal(before.y[before.entity == 1], np.zeros(4))


def make_workspace(unit="week"):
    days = 7 if unit == "week" else 1
    x = np.arange(4, dtype=float).reshape(-1, 1)
    train_lower = np.array([3.0, 1.0, 80.0, 90.0]) * 7 / days
    final_lower = np.array([30.0, 10.0, 80.0, 90.0]) * 7 / days
    anchors = np.array(["2024-01-01", "2024-01-08", "2024-02-05", "2024-02-12"])
    return SimpleNamespace(
        mode="entity",
        unit=unit,
        spec=SimpleNamespace(
            task="regression", named=lambda _role: ["lead"], censor=SimpleNamespace(unit=unit)
        ),
        table={
            ROW: np.arange(4),
            ANCHOR: anchors.astype("datetime64[D]"),
            LOWER: final_lower,
            UPPER: final_lower,
            "lead": final_lower,
        },
        features=SimpleNamespace(pick=lambda _families: (x, ["entity_id"], ["q"])),
        masks={
            "train": np.array([True, True, False, False]),
            "val": np.array([False, False, True, False]),
            "test": np.array([False, False, False, True]),
        },
        bounds={
            "val": (train_lower, train_lower),
            "test": (final_lower, final_lower),
            "final": (final_lower, final_lower),
        },
    )


@pytest.mark.parametrize("unit", ["week", "day"])
def test_pipeline_freezes_the_train_cap_and_orders_early_stopping_by_calendar(monkeypatch, unit):
    ws = make_workspace(unit)
    calls = []

    def capture_fit(x, **kwargs):
        calls.append((x, kwargs))
        return SimpleNamespace(eval_history={"metric": "logloss"})

    monkeypatch.setattr(pipeline, "fit_model", capture_fit)
    genome = pipeline.make_genome(["entity_id:measure"], "hazard", {"max_depth": 1})
    first, _ = pipeline._entity_fit(ws, genome, ws.masks["train"], "val", 1)
    final, _ = pipeline._entity_fit(ws, genome, np.ones(4, dtype=bool), "final", 1)
    assert first.cap == final.cap == 2
    assert first.eval_history == {"metric": "logloss"}
    for x, kwargs in calls:
        assert kwargs["objective"] == "hazard"
        assert kwargs["feature_types"] == ["q", "q", "q"]
        assert kwargs["nthread"] == 1
        assert "labels" not in kwargs
        assert "scale_pos_weight" not in kwargs["params"]
        assert x[:, -2].max() == 2
        dates = ws.table[ANCHOR][x[:, 0].astype(int)] + x[:, -2].astype(int) * np.timedelta64(
            7, "D"
        )
        assert np.all(dates[:-1] <= dates[1:])
    np.testing.assert_array_equal(calls[0][0][:, 0], [0, 0, 1, 0, 1])


def test_search_scores_log_loss_at_the_validation_cut_and_bins_by_entity_anchor(monkeypatch):
    ws = make_workspace()
    ws.masks["val"] = np.array([False, False, True, True])
    ws.masks["test"] = np.zeros(4, dtype=bool)
    ws.bounds["test"] = (np.array([3, 1, 3, 2]), np.array([3, 1, 3, np.inf]))
    x = ws.features.pick(set())[0]
    fitted = HazardFitted(SimpleNamespace(eval_history={"metric": "logloss"}), cap=4)
    fit_calls = []

    def capture_fit(_ws, _genome, rows, phase, nthread):
        fit_calls.append((rows.copy(), phase, nthread))
        return fitted, x

    monkeypatch.setattr(pipeline, "_entity_fit", capture_fit)
    monkeypatch.setattr(pipeline, "predict", lambda _fitted, features: np.full(len(features), 0.25))
    monkeypatch.setattr(hazard, "predict", lambda _fitted, features: np.full(len(features), 0.25))
    genome = pipeline.make_genome(["entity_id:measure"], "hazard", {})
    sample = np.array([True, False, False, False])
    result = pipeline.score(ws, genome, nthread=1, sample=sample)
    negative, positive = -np.log(0.75), -np.log(0.25)
    expected = (5 * negative + positive) / 6
    assert result.fitness == pytest.approx(expected)
    assert result.metrics["log_loss"] == pytest.approx(expected)
    np.testing.assert_allclose(result.contrib, [(3 * negative + positive) / 6, 2 * negative / 6])
    assert result.fitness == pytest.approx(result.contrib.sum())
    np.testing.assert_array_equal(result.prediction, [2, 2])
    np.testing.assert_array_equal(fit_calls[0][0], sample)
    assert fit_calls[0][1:] == ("val", 1)
    ws.bounds["final"] = (np.full(4, 900.0), np.full(4, 900.0))
    changed = pipeline.score(ws, genome, nthread=1, sample=sample)
    assert changed.fitness == result.fitness
    np.testing.assert_array_equal(changed.contrib, result.contrib)


def test_hazard_model_keeps_binary_logistic_unweighted_even_with_conflicting_params(monkeypatch):
    configs = []

    def capture_train(config, matrix, rounds, **kwargs):
        configs.append(config)
        assert rounds == 2
        return SimpleNamespace(predict=lambda values: np.full(len(values), 0.25))

    monkeypatch.setattr(model, "_xgb", lambda: SimpleNamespace(train=capture_train))
    monkeypatch.setattr(model, "_labelled", lambda x, *_args: x)
    monkeypatch.setattr(model, "_matrix", lambda x, _types: x)
    fitted = model.fit_model(
        np.zeros((40, 1)),
        objective="hazard",
        y=np.arange(40) % 2,
        params={"objective": "reg:squarederror", "eval_metric": "rmse", "scale_pos_weight": 99},
        max_rounds=2,
        patience=0,
        nthread=1,
    )
    assert len(configs) == 1
    assert configs[0]["objective"] == "binary:logistic"
    assert configs[0]["eval_metric"] == "logloss"
    assert configs[0]["scale_pos_weight"] == 1
    assert model.predict(fitted, np.zeros((3, 1))).shape == (3,)


def test_hazard_and_regression_contributions_align_when_observed_weeks_differ():
    ws = make_workspace()
    ws.masks["val"] = np.array([False, False, True, True])
    periods = pipeline._weeks(ws.table[ANCHOR][ws.masks["val"]])
    hazard_score = pipeline._period_fitness(
        ws, np.array([[0.75, 0.25]]), np.array([0]), periods[:1], classification=True
    )
    regression_score = pipeline._period_fitness(ws, np.array([3.0]), np.array([4.0]), periods[1:])
    assert hazard_score.contrib.shape == regression_score.contrib.shape == (2,)
    np.testing.assert_allclose(hazard_score.contrib, [-np.log(0.75), 0])
    np.testing.assert_allclose(regression_score.contrib, [0, 0.25])
    for result in (hazard_score, regression_score):
        assert result.fitness == pytest.approx(result.contrib.sum())


@pytest.mark.parametrize("invalid", ["timeseries", "classification", "uncensored"])
def test_hazard_rejects_workspaces_without_censored_entity_regression(invalid):
    ws = make_workspace()
    if invalid == "timeseries":
        ws.mode = "timeseries"
    elif invalid == "classification":
        ws.spec.task = "classification"
    else:
        ws.spec.censor = None
    genome = pipeline.make_genome(["entity_id:measure"], "hazard", {})
    with pytest.raises(pipeline.PipelineError, match="hazard requires censored entity regression"):
        pipeline.score(ws, genome)


def test_search_sampling_retains_every_period_of_each_selected_entity(monkeypatch):
    search_module = importlib.import_module("smolsmort.forecast.search")
    train = np.arange(60) < 50
    ws = SimpleNamespace(mode="entity", masks={"train": train})
    monkeypatch.setattr(search_module, "HALVING_ROWS", 10)
    selected = search_module._sample(ws, np.random.default_rng(7))
    assert selected.shape == train.shape and selected.dtype == bool
    assert 0 < selected.sum() < train.sum()
    assert not np.any(selected & ~train)
    x = np.arange(60, dtype=float).reshape(-1, 1)
    lower = 1 + np.arange(60) % 7
    upper = np.where(np.arange(60) % 2, lower, np.inf)
    anchors = np.full(60, np.datetime64("2024-01-01"))
    full = expand_periods(x, lower, upper, anchors, cap=6)
    sampled = expand_periods(
        x[selected],
        lower[selected],
        upper[selected],
        anchors[selected],
        cap=full.cap,
        entity=np.flatnonzero(selected),
    )
    np.testing.assert_array_equal(np.unique(sampled.entity), np.flatnonzero(selected))
    for entity in np.flatnonzero(selected):
        np.testing.assert_array_equal(
            sampled.week[sampled.entity == entity], full.week[full.entity == entity]
        )
        np.testing.assert_array_equal(
            sampled.y[sampled.entity == entity], full.y[full.entity == entity]
        )


def test_fitted_hazards_recover_two_censored_geometric_processes():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(17)
    dimension = np.tile([0, 1], 2000)
    probability = np.where(dimension == 0, 0.2, 0.05)
    events = rng.geometric(probability) - 0.5
    waited = rng.integers(1, 53, len(dimension))
    closed = events <= waited
    lower = np.where(closed, events, waited)
    upper = np.where(closed, events, np.inf)
    anchors = np.datetime64("2026-01-05") - waited * np.timedelta64(7, "D")
    assert (~closed).any()
    periods = expand_periods(dimension.reshape(-1, 1), lower, upper, anchors)
    order = np.argsort(periods.date, kind="stable")
    fitted = fit_model(
        periods.x[order],
        objective="hazard",
        y=periods.y[order],
        feature_types=["q", "q", "q"],
        params={"max_depth": 1, "eta": 0.2, "min_child_weight": 20, "tree_method": "hist"},
        max_rounds=50,
        patience=10,
        nthread=1,
    )
    query_dimension = np.repeat([0, 1], 52)
    query_anchors = np.tile(np.datetime64("2024-01-01") + np.arange(52) * np.timedelta64(7, "D"), 2)
    curves = predict_hazard(
        HazardFitted(fitted, periods.cap), query_dimension.reshape(-1, 1), query_anchors
    )
    for value, expected in ((0, 0.2), (1, 0.05)):
        # the pooled final bucket models a tail probability, not a single week's risk
        mean_hazard = curves["hazard"][query_dimension == value, : min(10, periods.cap)].mean()
        assert mean_hazard == pytest.approx(expected, abs=0.03)


@pytest.mark.parametrize("objective", ["hazard", "aft"])
def test_one_generation_search_finishes_with_one_forecast_per_open_entity(
    tmp_path, monkeypatch, objective
):
    pytest.importorskip("duckdb")
    pytest.importorskip("xgboost")
    from smolsmort.forecast.prep import prepare
    from smolsmort.forecast.search import Budget, search

    truth = write_rows(tmp_path, n=360, seed=4)
    spec = PrepSpec(
        str(truth.path),
        "entity",
        "regression",
        (
            Column("created", "anchor"),
            Column("branch", "dimension"),
            Column("size", "measure"),
            Column("planned_offset", "measure"),
            Column("leak_plus1", "ignore"),
            Column("line", "ignore"),
            Column("lead_weeks", "target"),
        ),
        censor=Censor("created", as_of=AS_OF.isoformat()),
    )
    prepared = prepare(spec, tmp_path / "cache")
    ws = pipeline.load_workspace(spec, prepared.folder, prepared.summary)
    assert {"hazard", "aft"} <= set(ws.profile["genes"]["objectives"])
    ws.profile["genes"]["objectives"] = [objective]
    monkeypatch.setattr(
        pipeline, "fit_model", partial(pipeline.fit_model, max_rounds=8, patience=2)
    )
    search_result = search(ws, Budget(population=4, max_generations=1, time_cap=60, nthread=1))
    best = search_result["leaderboard"][0]
    assert best["fitness"] is not None and np.isfinite(best["fitness"])
    genome = pipeline.genome_from_dict(best["genome"])
    assert genome.objective == objective
    result = pipeline.finish(ws, genome, nthread=1)
    forecast = result["forecast"]
    assert len(forecast["prediction"]) == prepared.summary["open_rows"] == truth.open_rows > 0
    assert np.isfinite(forecast["prediction"]).all()
    np.testing.assert_array_equal(forecast["row"], ws.table[ROW][ws.predict_rows])
    np.testing.assert_array_equal(
        forecast["predicted_date"],
        forecast["anchor"].astype("datetime64[D]")
        + (forecast["prediction"] * 7).round().astype("timedelta64[D]"),
    )
    lower, upper = ws.bounds["final"]
    test_rows = ws.masks["test"] & np.isfinite(lower) & ((lower == upper) | np.isposinf(upper))
    assert result["test"]["rows"] == int(test_rows.sum())
    test_prediction = np.asarray(result["test"]["prediction"])
    assert test_prediction.shape == lower[test_rows].shape
    assert result["verdict"]["accuracy"]["metric"] == "ibs"
    assert np.isfinite(result["model_error"])
    assert result["verdict"]["accuracy"]["model_error"] == result["model_error"]
    assert not any(check["group"] == "band" for check in result["verdict"]["checks"])
    if objective == "hazard":
        assert best["metrics"]["log_loss"] == pytest.approx(best["fitness"])
        curves = result["curves"]
        np.testing.assert_array_equal(curves["row"], forecast["row"])
        np.testing.assert_array_equal(curves["anchor"], forecast["anchor"])
        assert curves["survival"].shape == (truth.open_rows, curves["cap"] + 1)
        assert np.all(np.diff(curves["survival"], axis=1) <= 1e-12)
        crossed = curves["survival"] <= 0.5
        median = np.where(crossed.any(axis=1), crossed.argmax(axis=1), curves["cap"])
        np.testing.assert_array_equal(forecast["prediction"], median)
        assert np.all(forecast["lower"] <= forecast["prediction"])
        assert np.all(forecast["prediction"] <= forecast["upper"])


def test_quantiles_never_precede_a_wait_beyond_the_cap(monkeypatch):
    from smolsmort.forecast import hazard as hazard_module

    # a constant weekly hazard of 0.2 over a cap of 4 weeks [5 buckets]
    monkeypatch.setattr(hazard_module, "predict", lambda fitted, x: np.full(len(x), 0.2))
    fitted = hazard_module.HazardFitted(fitted=None, cap=4)
    anchors = np.array(["2026-01-05"] * 3, dtype="datetime64[D]")
    x = np.zeros((3, 1), dtype=np.float32)
    curves = hazard_module.predict_hazard(fitted, x, anchors, waited=[0, 3, 10])
    # geometric with h=0.2 from the wait w: first week k with 0.8^(k-w+1) <= 0.5, so k = w + 3
    assert curves["prediction"].tolist() == [3.0, 6.0, 13.0]
    assert (curves["upper"] >= np.array([0, 3, 10])).all()
    assert (curves["lower"] >= np.array([0, 3, 10])).all()
