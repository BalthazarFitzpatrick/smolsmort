"""target transforms preserve origins and keep evaluation on raw values"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_forecast_runs_v2 import make_request

from smolsmort.forecast import evaluate as ev
from smolsmort.forecast.families import get_family
from smolsmort.forecast.features import buckets_for, series_features, step_index
from smolsmort.forecast.pipeline import (
    PipelineError,
    Workspace,
    _series_baseline,
    _series_fit,
    _series_predict,
    finish,
    genome_from_dict,
    load_workspace,
    make_genome,
    score,
)
from smolsmort.forecast.runs import start_run, wait_run
from smolsmort.forecast.search import Budget, _FamilyVariation, _Variation, search
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec
from smolsmort.forecast.tables import read_table
from smolsmort.forecast.transform import TransformError, get_transform


@pytest.fixture
def workspace(tmp_path):
    count, horizon = 140, 4
    days = np.datetime64("2024-01-01") + np.arange(count).astype("timedelta64[D]")
    future = days[-1] + np.arange(1, horizon + 1).astype("timedelta64[D]")
    values = 50 + np.arange(count) / 10 + 8 * np.sin(2 * np.pi * np.arange(count) / 7)
    panel = {
        SERIES: np.repeat(["a", "b"], count),
        STEP: np.tile(days, 2),
        "y": np.concatenate([values, values * 2 + 3]),
    }
    scaffold = {SERIES: np.repeat(["a", "b"], horizon), STEP: np.tile(future, 2)}
    spec = PrepSpec(
        source="synthetic.csv",
        mode="series",
        task="regression",
        step="day",
        horizon=horizon,
        columns=(Column("day", "time"), Column("y", "target", "sum")),
    )
    ws = Workspace(
        spec, tmp_path, {}, {"genes": {"period": 7, "objectives": ["squared"]}}, [], "day"
    )
    ws.panel, ws.frames, ws.series_masks = panel, {}, {}
    ws.test_steps = 13
    for bucket in buckets_for(horizon):
        frame = series_features(
            spec, panel, scaffold, bucket, "day", panel[STEP] < days[100], period=7
        )
        ws.frames[bucket] = frame
        ws.series_masks[bucket] = {
            "train": ~frame.future & (frame.step < days[100]),
            "val": ~frame.future & (frame.step >= days[100]) & (frame.step < days[127]),
            "test": ~frame.future & (frame.step >= days[127]),
            "future": frame.future,
        }
    ws.families = sorted(set().union(*(frame.features.families for frame in ws.frames.values())))
    return ws


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
def test_random_nonnegative_roundtrip_including_zeros(name):
    rng = np.random.default_rng(0)
    values, base = rng.uniform(0, 100_000, size=(2, 500))
    values[::7] = 0
    base[::11] = 0
    transform = get_transform(name)

    restored = transform.inv(transform.fwd(values, base), base)

    np.testing.assert_allclose(restored, values, rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
def test_negative_targets_name_the_transform(name):
    with pytest.raises(TransformError, match=rf"{name}.*negative|negative.*{name}"):
        get_transform(name).fwd(np.array([0.0, -0.01, 10.0]), np.ones(3))


def test_none_preserves_values_and_missing_targets():
    values = np.array([-1.0, 0.0, 1.0, np.nan])
    transform = get_transform("none")

    np.testing.assert_array_equal(transform.fwd(values), values)
    np.testing.assert_array_equal(transform.inv(values), values)


def test_logdiff_uses_each_rows_origin_value():
    values = np.array([0.0, 4.0, 99.0])
    base = np.array([7.0, 0.0, 20.0])
    transform = get_transform("logdiff")

    np.testing.assert_array_equal(transform.fwd(values, base), np.log1p(values) - np.log1p(base))


def test_series_base_matches_every_origin_and_ignores_later_values(workspace):
    ws = workspace
    scaffold = {
        SERIES: np.repeat(["a", "b"], ws.spec.horizon),
        STEP: np.tile(
            ws.panel[STEP].max() + np.arange(1, ws.spec.horizon + 1).astype("timedelta64[D]"),
            2,
        ),
    }
    changed_panel = copy.deepcopy(ws.panel)
    pivot = np.datetime64("2024-04-20")
    changed_panel["y"][changed_panel[STEP] == pivot] += 1000
    lookup = {
        (key, index): value
        for key, index, value in zip(
            ws.panel[SERIES], step_index(ws.panel[STEP], ws.unit), ws.panel["y"], strict=True
        )
    }
    for bucket, frame in ws.frames.items():
        origin_index = step_index(frame.step, ws.unit) - bucket[1]
        expected = [
            lookup[(key, index)] for key, index in zip(frame.series, origin_index, strict=True)
        ]
        np.testing.assert_array_equal(frame.base, expected)
        changed = series_features(
            ws.spec,
            changed_panel,
            scaffold,
            bucket,
            ws.unit,
            ws.panel[STEP] < np.datetime64("2024-04-10"),
            period=7,
        )
        before = frame.origin < pivot
        np.testing.assert_array_equal(changed.base[before], frame.base[before])
        np.testing.assert_array_equal(
            changed.base[frame.origin == pivot], frame.base[frame.origin == pivot] + 1000
        )


def test_none_keeps_the_saved_genome_and_golden_key():
    golden_path = Path(__file__).parent / "golden" / "forecast_leaderboard.json"
    golden = json.loads(golden_path.read_text())
    for task in ("regression", "classification"):
        for entry in golden[task]:
            old_data = entry["genome"]
            old_key = hashlib.sha1(json.dumps(old_data, sort_keys=True).encode()).hexdigest()[:12]
            default = genome_from_dict(old_data)
            explicit = genome_from_dict({**old_data, "transform": "none"})

            assert default.transform == explicit.transform == "none"
            assert default.to_dict() == explicit.to_dict() == old_data
            assert default.key() == explicit.key() == old_key
            if "key" in entry:
                assert default.key() == entry["key"]
    assert genome_from_dict(golden["regression"][0]["genome"]).key() == "a0bafa1a7c3e"


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
def test_nondefault_transform_survives_recipe_and_changes_key(name):
    raw = make_genome(["lag+0"], "squared", {})
    transformed = make_genome(["lag+0"], "squared", {}, transform=name)

    assert transformed.to_dict()["transform"] == name
    assert transformed.key() != raw.key()
    assert genome_from_dict(transformed.to_dict()) == transformed


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
def test_lag_family_receives_transformed_targets(workspace, monkeypatch, name):
    ws = workspace
    bucket = (2, 4)
    frame, rows = ws.frames[bucket], ws.series_masks[bucket]["train"]
    family = get_family("ridge")
    original_fit = family.fit
    original_values = frame.y.copy()
    captured = {}

    def fit(x, **options):
        captured.update(options)
        return original_fit(x, **options)

    monkeypatch.setattr(family, "fit", fit)
    genome = make_genome(["lag+0"], "squared", {"alpha": 1.0}, family="ridge", transform=name)
    _series_fit(ws, genome, frame, rows, 1)
    order = np.flatnonzero(rows)[np.argsort(frame.step[rows], kind="stable")]

    expected = get_transform(name).fwd(frame.y[order], frame.base[order])
    np.testing.assert_array_equal(captured["y"], expected)
    np.testing.assert_array_equal(frame.y, original_values)


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
def test_stats_fit_sees_log_history_and_returns_raw_cutoff_predictions(
    workspace, monkeypatch, name
):
    pytest.importorskip("statsforecast")
    ws = workspace
    bucket = (2, 4)
    frame = ws.frames[bucket]
    fit_rows, eval_rows = ws.series_masks[bucket]["train"], ws.series_masks[bucket]["val"]
    family = get_family("ets")
    original_fit = family.fit
    captured = {}
    original_values = ws.panel["y"].copy()

    def fit(task, **options):
        captured["history"] = task.panel["y"].copy()
        fitted = original_fit(task, **options)
        captured["prediction"] = fitted.predict(task)
        return fitted

    monkeypatch.setattr(family, "fit", fit)
    genome = make_genome([], "squared", {}, family="ets", transform=name)
    predicted = _series_predict(ws, genome, frame, fit_rows, eval_rows, bucket, 1)

    visible = ws.panel[STEP] <= frame.step[eval_rows].max()
    np.testing.assert_array_equal(captured["history"], np.log1p(original_values[visible]))
    np.testing.assert_array_equal(ws.panel["y"], original_values)
    expected = np.expm1(captured["prediction"])
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-12)
    if name == "logdiff":
        base = frame.base[eval_rows]
        difference = captured["prediction"] - np.log1p(base)
        np.testing.assert_allclose(get_transform(name).inv(difference, base), expected, rtol=1e-12)


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
@pytest.mark.parametrize("family_name", ["xgboost", "ridge", "ets", "snaive"])
def test_real_families_score_and_finish_on_raw_scale(workspace, name, family_name):
    if family_name == "xgboost":
        pytest.importorskip("xgboost")
    if family_name == "ets":
        pytest.importorskip("statsforecast")
    ws = workspace
    ws.transform = name
    family = get_family(family_name)
    families = ["lag+0", "lag+1", "cal:weekday"] if family.needs == "lag_features" else []
    params = {"max_depth": 3} if family_name == "xgboost" else family.defaults()
    genome = make_genome(families, "squared", params, family=family_name, transform=name)

    scored, final = score(ws, genome), finish(ws, genome)

    validation = final["validation"]
    assert np.isfinite(scored.fitness)
    assert scored.metrics["wape"] == pytest.approx(
        ev.wape(validation["actual"], validation["prediction"])
    )
    assert final["model_error"] == pytest.approx(
        ev.wape(final["test"]["actual"], final["test"]["prediction"])
    )
    for phase in ("validation", "test", "forecast"):
        predictions = final[phase]["prediction"]
        assert np.isfinite(predictions).all()
        ratio = np.median(np.abs(predictions)) / np.median(ws.panel["y"])
        assert 0.2 < ratio < 5
    for label, band in final["bands"].items():
        rows = validation["bucket"] == label
        raw_band = ev.fit_band(validation["actual"][rows], validation["prediction"][rows])
        assert band == raw_band.__dict__
    baseline, actual = [], []
    for bucket, frame in ws.frames.items():
        rows = ws.series_masks[bucket]["test"]
        baseline.append(_series_baseline(frame, rows, bucket, 7))
        actual.append(frame.y[rows])
    assert final["baseline_error"] == ev.wape(np.concatenate(actual), np.concatenate(baseline))


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
def test_v2_panel_run_records_transform_and_raw_bands(tmp_path, name):
    pytest.importorskip("xgboost")
    pytest.importorskip("statsforecast")
    selections = [
        {"name": family, "method": "grid"} for family in ("xgboost", "ridge", "ets", "snaive")
    ]
    # three validation fits must finish before this test can inspect every family's transform
    request, spec, prepared = make_request(tmp_path, selections, duration=90)
    run_id = start_run(
        tmp_path / "runs",
        spec,
        prepared,
        version=2,
        families=request["families"],
        time_budget_s=request["time_budget_s"],
        nthread=4,
        transform=name,
    )
    state = wait_run(tmp_path / "runs", run_id, timeout=180)
    run = tmp_path / "runs" / run_id

    assert state["state"] == "done", state
    for selection in selections:
        family = selection["name"]
        status = json.loads((run / "families" / family / "status.json").read_text())
        assert status["state"] == "done", status
        winner = json.loads((run / "families" / family / "winner.json").read_text())
        assert winner["transform"] == name
    recipe = json.loads((run / "recipe.json").read_text())
    result = json.loads((run / "result.json").read_text())
    assert recipe["transform"] == result["transform"] == name
    assert recipe["genome"]["transform"] == name
    validation = read_table(run / "validation.parquet")
    ratio = np.median(np.abs(validation["prediction"])) / np.median(validation["actual"])
    assert 0.2 < ratio < 5
    for label, band in result["bands"].items():
        rows = validation["bucket"] == label
        raw_band = ev.fit_band(validation["actual"][rows], validation["prediction"][rows])
        assert band == raw_band.__dict__


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
@pytest.mark.parametrize("family_name", ["xgboost", "ridge"])
def test_search_variation_preserves_per_run_transform(workspace, name, family_name):
    ws = workspace
    ws.transform = name
    rng = np.random.default_rng(0)
    variation = (
        _Variation(ws, rng)
        if family_name == "xgboost"
        else _FamilyVariation(ws, rng, get_family(family_name), {})
    )
    first = variation.random()
    mutated = variation.mutate(first)
    crossed = variation.crossover(first, mutated)

    assert all(genome.transform == name for genome in (first, mutated, crossed))


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
def test_search_only_reuses_warm_genomes_with_the_same_transform(workspace, name):
    ws = workspace
    ws.transform, ws.families = name, ["lag+0"]
    matching = make_genome(["lag+0"], "squared", {"alpha": 19.0}, family="ridge", transform=name)
    other = make_genome(["lag+0"], "squared", {"alpha": 31.0}, family="ridge")

    result = search(
        ws,
        Budget(population=2, max_generations=1),
        family=get_family("ridge"),
        warm=[other.to_dict(), matching.to_dict()],
    )

    keys = {entry["key"] for entry in result["leaderboard"]}
    assert matching.key() in keys
    assert other.key() not in keys
    assert all(data["transform"] == name for data in result["population"])
    assert all(entry["genome"]["transform"] == name for entry in result["leaderboard"])


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
def test_resume_preserves_transform_and_reuses_cached_scores(workspace, monkeypatch, name):
    ws = workspace
    ws.transform, ws.families = name, ["lag+0"]
    checkpoints, calls = [], []
    module = importlib.import_module("smolsmort.forecast.search")
    original_score = module.score

    def score_candidate(*args, **kwargs):
        calls.append(args[1])
        return original_score(*args, **kwargs)

    monkeypatch.setattr(module, "score", score_candidate)
    budget = Budget(population=1, max_generations=1)
    options = {"family": get_family("ridge"), "method": "grid", "space": {"alpha": [1.0]}}
    first = search(
        ws, budget, on_checkpoint=lambda state: checkpoints.append(copy.deepcopy(state)), **options
    )
    fitted = len(calls)
    resumed = search(ws, budget, resume=checkpoints[-1], **options)

    assert resumed["best"] == first["best"]
    assert resumed["leaderboard"] == first["leaderboard"]
    assert resumed["best"]["transform"] == name
    assert len(calls) == fitted == 1
    mismatched = copy.deepcopy(checkpoints[-1])
    mismatched["population"][0].pop("transform")
    with pytest.raises(PipelineError, match="checkpoint transform differs"):
        search(ws, budget, resume=mismatched, **options)


@pytest.mark.parametrize("objective", ["poisson", "tweedie", "aft"])
def test_logdiff_refuses_positive_only_fit_objectives(workspace, objective):
    pytest.importorskip("xgboost")
    ws = workspace
    bucket = (1, 1)
    genome = make_genome(["lag+0"], objective, {}, transform="logdiff")

    with pytest.raises(PipelineError, match=rf"logdiff.*signed-target objective.*{objective}"):
        _series_fit(ws, genome, ws.frames[bucket], ws.series_masks[bucket]["train"], 1)


def test_logdiff_search_filters_positive_only_objectives_without_changing_none(workspace):
    ws = workspace
    ws.profile["genes"]["objectives"] = ["squared", "absolute", "poisson", "tweedie"]
    raw = _Variation(ws, np.random.default_rng(0))
    ws.transform = "logdiff"
    transformed = _Variation(ws, np.random.default_rng(0))

    assert raw.objectives == ["squared", "absolute", "poisson", "tweedie"]
    assert transformed.objectives == ["squared", "absolute"]


def test_v1_worker_ignores_an_unversioned_request_transform(tmp_path):
    pytest.importorskip("xgboost")
    request, spec, prepared = make_request(tmp_path, [{"name": "xgboost"}])
    genome = make_genome(["lag+0"], "squared", {"max_depth": 3})
    request.pop("version")
    request.pop("families")
    request.pop("time_budget_s")
    request.update(
        transform="logdiff", recipe=genome.to_dict(), budget={"population": 1}, nthread=1
    )
    run = tmp_path / "v1"
    run.mkdir()
    (run / "request.json").write_text(json.dumps(request))

    process = subprocess.run(
        [sys.executable, "-m", "smolsmort.forecast.worker", str(run)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    status = json.loads((run / "status.json").read_text())
    assert process.returncode == 0 and status["state"] == "done", (
        status,
        process.stdout,
        process.stderr,
    )
    assert json.loads((run / "recipe.json").read_text())["transform"] == "none"
    assert json.loads((run / "result.json").read_text())["transform"] == "none"
    ws = load_workspace(spec, prepared.folder, prepared.summary)
    assert ws.transform == "none"
    expected = finish(ws, genome)
    validation = read_table(run / "validation.parquet")
    np.testing.assert_array_equal(validation["prediction"], expected["validation"]["prediction"])


@pytest.mark.parametrize("unit", ["month", "quarter", "year"])
def test_calendar_base_uses_exact_origin_position(unit):
    count, horizon = 48, 4
    stride = {"month": 1, "quarter": 3, "year": 12}[unit]
    start = np.datetime64("2020-01", "M")
    steps = (start + np.arange(count) * stride).astype("datetime64[D]")
    future = (start + (count + np.arange(horizon)) * stride).astype("datetime64[D]")
    panel = {
        SERIES: np.repeat(["a", "b"], count),
        STEP: np.tile(steps, 2),
        "y": np.concatenate([np.arange(count) + 2.0, np.arange(count) * 3.0 + 5.0]),
    }
    scaffold = {SERIES: np.repeat(["a", "b"], horizon), STEP: np.tile(future, 2)}
    spec = PrepSpec(
        "calendar.csv",
        "series",
        "regression",
        (Column("day", "time"), Column("y", "target", "sum")),
        step=unit,
        horizon=horizon,
    )
    lookup = {
        (key, position): value
        for key, position, value in zip(
            panel[SERIES], step_index(panel[STEP], unit), panel["y"], strict=True
        )
    }
    for bucket in buckets_for(horizon):
        frame = series_features(spec, panel, scaffold, bucket, unit, panel[STEP] < steps[40])
        origins = step_index(frame.step, unit) - bucket[1]
        expected = [
            lookup[(key, position)] for key, position in zip(frame.series, origins, strict=True)
        ]

        np.testing.assert_array_equal(frame.base, expected)
        assert frame.future.sum() == (bucket[1] - bucket[0] + 1) * 2


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
@pytest.mark.parametrize("version", [None, 2])
def test_refit_refuses_recipe_transform_mismatch(tmp_path, name, version):
    pytest.importorskip("xgboost")
    _, spec, prepared = make_request(tmp_path, [{"name": "xgboost"}])
    genome = make_genome(["lag+0"], "squared", {"max_depth": 3}, transform=name)
    request_options = (
        {"version": 2, "families": [{"name": "xgboost"}], "time_budget_s": 2}
        if version == 2
        else {}
    )
    run_id = start_run(
        tmp_path / "runs", spec, prepared, recipe=genome.to_dict(), nthread=1, **request_options
    )

    state = wait_run(tmp_path / "runs", run_id, timeout=30)

    assert state["state"] == "failed", state
    assert "recipe transform differs from the requested transform" in state["reason"]
    assert "use a v2 request with the saved transform" in state["reason"]


@pytest.mark.parametrize("name", ["log1p", "logdiff"])
def test_v2_refit_preserves_matching_saved_transform(tmp_path, name):
    pytest.importorskip("xgboost")
    _, spec, prepared = make_request(tmp_path, [{"name": "xgboost"}])
    genome = make_genome(["lag+0"], "squared", {"max_depth": 3}, transform=name)
    run_id = start_run(
        tmp_path / "runs",
        spec,
        prepared,
        recipe=genome.to_dict(),
        version=2,
        families=[{"name": "xgboost"}],
        time_budget_s=2,
        transform=name,
        nthread=1,
    )
    state = wait_run(tmp_path / "runs", run_id, timeout=60)
    run = tmp_path / "runs" / run_id

    assert state["state"] == "done" and state["stopped"] == "refit", state
    recipe = json.loads((run / "recipe.json").read_text())
    result = json.loads((run / "result.json").read_text())
    assert recipe["genome"] == genome.to_dict()
    assert recipe["transform"] == result["transform"] == name
    validation = read_table(run / "validation.parquet")
    ratio = np.median(np.abs(validation["prediction"])) / np.median(validation["actual"])
    assert 0.2 < ratio < 5
    for label, band in result["bands"].items():
        rows = validation["bucket"] == label
        raw_band = ev.fit_band(validation["actual"][rows], validation["prediction"][rows])
        assert band == raw_band.__dict__
