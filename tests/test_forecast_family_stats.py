"""raw-series families preserve rolling origins and production baseline semantics"""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from smolsmort.forecast.families import get_family
from smolsmort.forecast.families.stats import (
    RawSeriesTask,
    build_model,
    join_predictions,
    season_length,
)
from smolsmort.forecast.features import buckets_for, series_features
from smolsmort.forecast.pipeline import (
    PipelineError,
    Workspace,
    _series_baseline,
    _series_raw_fit,
    finish,
    make_genome,
    score,
)
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec


@pytest.fixture
def workspace(tmp_path):
    count, horizon = 140, 13
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
    ws = Workspace(spec, tmp_path, {}, {"genes": {"period": 7}}, [], "day")
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
    return ws


def make_task(ws, bucket=(2, 4), phase="val"):
    return RawSeriesTask(
        ws.panel,
        "y",
        ws.unit,
        ws.frames[bucket],
        ws.series_masks[bucket][phase],
        bucket,
        season_length(ws.profile),
    )


def test_optional_imports_and_install_hint_are_lazy():
    script = (
        "import sys; sys.modules['statsforecast'] = None; "
        "from smolsmort.forecast.families import get_family; "
        "assert get_family('snaive').available() == (True, ''); "
        "assert not get_family('ets').available()[0]; "
        "assert 'uv sync --extra stats' in get_family('ets').available()[1]"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("period,expected", [(None, 1), (0, 1), (-3, 1), (6, 6)])
def test_season_uses_positive_train_profile(period, expected):
    assert season_length({"genes": {"period": period}}) == expected
    assert season_length({"period": period}) == expected


@pytest.mark.parametrize("name", ["ets", "theta", "arima", "snaive"])
def test_raw_family_contract(name):
    family = get_family(name)
    assert family.needs == "raw_series"
    assert family.space() == [] and family.defaults() == {}


def test_arima_search_is_capped():
    pytest.importorskip("statsforecast")
    model = build_model("arima", 6)
    for name, value in {
        "season_length": 6,
        "max_p": 2,
        "max_q": 2,
        "max_P": 1,
        "max_Q": 1,
        "max_order": 4,
        "nmodels": 20,
        "approximation": True,
    }.items():
        assert getattr(model, name) == value


@pytest.mark.parametrize("name", ["ets", "theta", "arima"])
def test_validation_maps_every_bucket_once_in_frame_order(workspace, name):
    pytest.importorskip("statsforecast")
    pd = pytest.importorskip("pandas")
    from statsforecast import StatsForecast

    ws = workspace
    genome = make_genome([], "squared", {}, family=name)
    for bucket, frame in ws.frames.items():
        task = make_task(ws, bucket)
        fitted, actual_task = _series_raw_fit(ws, genome, frame, task.rows, bucket, 1)
        model = build_model(name, 7)
        raw = pd.DataFrame(
            {
                "unique_id": ws.panel[SERIES],
                "ds": ws.panel[STEP].astype("datetime64[ns]"),
                "y": ws.panel["y"],
            }
        )
        raw = raw[raw.ds <= pd.Timestamp(frame.step[task.rows].max())]
        output = StatsForecast(models=[model], freq="D", n_jobs=1).cross_validation(
            df=raw, h=bucket[1], step_size=1, n_windows=27, refit=False
        )
        lookup = {
            (r.unique_id, np.datetime64(r.ds, "D"), np.datetime64(r.cutoff, "D")): getattr(
                r, str(model)
            )
            for r in output.itertuples(index=False)
        }
        expected = [
            lookup[(key, step, origin)]
            for key, step, origin in zip(
                frame.series[task.rows], frame.step[task.rows], frame.origin[task.rows], strict=True
            )
        ]
        assert len(expected) == int(task.rows.sum()) == 54
        np.testing.assert_array_equal(fitted.predict(actual_task), expected)
    result = score(ws, genome)
    assert np.isfinite(result.fitness) and result.metrics["season_length"] == 7


@pytest.mark.parametrize("name", ["ets", "theta", "arima"])
def test_first_last_origins_use_fixed_parameters_and_updated_history(workspace, name):
    pytest.importorskip("statsforecast")
    ws = workspace
    task = make_task(ws)
    fitted = get_family(name).fit(task, objective="squared")
    frame, rows = task.frame, task.rows
    selected = np.flatnonzero(rows & (frame.series == "a"))
    panel_rows = ws.panel[SERIES] == "a"
    days, values = ws.panel[STEP][panel_rows], ws.panel["y"][panel_rows]
    initial = values[days <= frame.origin[selected[0]]]
    model = build_model(name, 7).fit(initial)
    first = model.predict(h=4)["mean"][-1]
    last = model.forward(y=values[days <= frame.origin[selected[-1]]], h=4)["mean"][-1]
    prediction = fitted.predict(task)
    np.testing.assert_allclose(prediction[[0, len(selected) - 1]], [first, last], rtol=1e-10)
    changed = copy.deepcopy(task)
    pivot = frame.origin[selected[len(selected) // 2]]
    changed.panel["y"][changed.panel[STEP] == pivot] += 100
    updated = get_family(name).fit(changed, objective="squared").predict(changed)
    earlier = frame.origin[rows] < pivot
    np.testing.assert_array_equal(prediction[earlier], updated[earlier])
    assert np.max(np.abs(prediction[~earlier] - updated[~earlier])) > 0


def test_join_refuses_missing_duplicate_and_nonfinite_rows():
    pytest.importorskip("statsforecast")
    pd = pytest.importorskip("pandas")
    expected = pd.DataFrame(
        {
            "unique_id": ["b", "a"],
            "ds": pd.to_datetime(["2024-01-03", "2024-01-02"]),
            "cutoff": pd.to_datetime(["2024-01-02", "2024-01-01"]),
        }
    )
    output = expected.assign(AutoETS=[20.0, 10.0]).iloc[::-1]
    np.testing.assert_array_equal(join_predictions(output, expected, "AutoETS"), [20, 10])
    with pytest.raises(ValueError, match="missing"):
        join_predictions(output.iloc[:1], expected, "AutoETS")
    with pytest.raises(ValueError, match="duplicate"):
        join_predictions(pd.concat([output, output]), expected, "AutoETS")
    with pytest.raises(ValueError, match="duplicate"):
        join_predictions(output, pd.concat([expected, expected]), "AutoETS")
    with pytest.raises(ValueError, match="nonfinite"):
        join_predictions(output.assign(AutoETS=np.inf), expected, "AutoETS")


@pytest.mark.parametrize("name", ["ets", "theta", "arima"])
def test_insufficient_history_names_candidate(workspace, name):
    pytest.importorskip("statsforecast")
    task = make_task(workspace)
    cutoff = task.frame.origin[task.rows].min()
    keep = task.panel[STEP] >= cutoff
    task.panel = {name: values[keep] for name, values in task.panel.items()}
    with pytest.raises(
        PipelineError, match=rf"family '{name}'.*insufficient history.*1 observations"
    ):
        get_family(name).fit(task, objective="squared")


@pytest.mark.parametrize(
    "exception",
    [IndexError("too many indices"), KeyError("fitted"), NotImplementedError("tiny datasets")],
)
def test_library_history_errors_are_candidate_failures(workspace, monkeypatch, exception):
    pytest.importorskip("statsforecast")
    from smolsmort.forecast.families import stats

    def fail(*args):
        raise exception

    monkeypatch.setattr(stats, "forecast_rows", fail)
    with pytest.raises(PipelineError, match="family 'ets'.*check history at the earliest origin"):
        get_family("ets").fit(make_task(workspace), objective="squared")


def test_snaive_matches_production_baseline_in_every_phase(workspace):
    ws = workspace
    for bucket, frame in ws.frames.items():
        for phase in ("val", "test", "future"):
            task = make_task(ws, bucket, phase)
            predicted = get_family("snaive").fit(task, objective="squared").predict(task)
            np.testing.assert_array_equal(predicted, _series_baseline(frame, task.rows, bucket, 7))
    result = finish(ws, make_genome([], "squared", {}, family="snaive"))
    assert result["season_length"] == 7
    assert len(result["forecast"]["prediction"]) == 26
    assert result["model_error"] == result["baseline_error"]


def test_flat_snaive_stays_flat(workspace):
    ws = workspace
    for bucket, frame in ws.frames.items():
        for index, name in enumerate(frame.features.names):
            if name.startswith("lag_"):
                frame.features.x[:, index] = 42
        frame.y[~frame.future] = 42
        task = make_task(ws, bucket)
        np.testing.assert_array_equal(
            get_family("snaive").fit(task, objective="squared").predict(task), 42
        )
    assert score(ws, make_genome([], "squared", {}, family="snaive")).fitness == 0


def test_search_records_history_failure_and_continues(workspace):
    pytest.importorskip("statsforecast")
    from smolsmort.forecast.search import Budget, search

    ws = workspace
    ws.families = ["lag+0"]
    ws.profile["genes"]["objectives"] = ["squared"]
    cutoff = ws.frames[(1, 1)].origin[ws.series_masks[(1, 1)]["val"]].min()
    keep = ws.panel[STEP] >= cutoff
    ws.panel = {name: values[keep] for name, values in ws.panel.items()}
    genome = make_genome(["lag+0"], "squared", {}, family="ets")
    baseline = make_genome(["lag+0"], "squared", {}, family="snaive")
    result = search(
        ws, Budget(population=2, max_generations=1), warm=[genome.to_dict(), baseline.to_dict()]
    )
    assert result["best"] == baseline.to_dict()
    entry = next(row for row in result["leaderboard"] if row["genome"]["family"] == "ets")
    assert entry["fitness"] is None
    assert "failed: family 'ets'" in entry["note"]
    assert "insufficient history" in entry["note"]


@pytest.mark.parametrize("name", ["ets", "theta", "arima"])
def test_finish_runs_test_future_without_reading_later_observations(workspace, name):
    pytest.importorskip("statsforecast")
    ws = workspace
    genome = make_genome([], "squared", {}, family=name)
    result = finish(ws, genome)
    assert result["season_length"] == 7
    assert len(result["forecast"]["prediction"]) == 26
    assert np.isfinite(result["forecast"]["prediction"]).all()
    changed = copy.deepcopy(ws)
    changed.panel["y"][changed.panel[STEP] >= np.datetime64("2024-05-07")] += 500
    before, after = score(ws, genome), score(changed, genome)
    assert before.fitness == after.fitness


def test_result_writer_persists_effective_season(workspace, tmp_path):
    pytest.importorskip("duckdb")
    from smolsmort.forecast.worker import _write_result

    genome = make_genome([], "squared", {}, family="snaive")
    final = finish(workspace, genome)
    _write_result(tmp_path, workspace.spec, {"best": genome.to_dict(), "stopped": "refit"}, final)
    for name in ("recipe.json", "result.json"):
        assert json.loads((tmp_path / name).read_text())["season_length"] == 7
