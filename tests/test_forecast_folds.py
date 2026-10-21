"""walk-forward validation stays paired, equally weighted, and outside sealed test targets"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import evaluate, pipeline
from smolsmort.forecast.families import get_family
from smolsmort.forecast.families.stats import StatsFitted
from smolsmort.forecast.pipeline import finish, load_workspace, make_genome, score
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.search import Budget, Entry, _significant, search
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec
from smolsmort.forecast.splits import FOLDS, series_cuts


@pytest.fixture
def workspace(monkeypatch):
    weeks, horizon = 156, 4
    positions = np.arange(weeks)
    steps = np.datetime64("2023-01-02") + positions * np.timedelta64(7, "D")
    values = 20 + positions / 5 + 3 * np.sin(2 * np.pi * positions / 13)
    panel = {
        SERIES: np.repeat(["a", "b"], weeks),
        STEP: np.tile(steps, 2),
        "y": np.concatenate([values, 2 * values + 5]),
    }
    future = steps[-1] + np.arange(1, horizon + 1) * np.timedelta64(7, "D")
    scaffold = {SERIES: np.repeat(["a", "b"], horizon), STEP: np.tile(future, 2)}
    tables = {"panel.parquet": panel, "scaffold.parquet": scaffold}
    monkeypatch.setattr(pipeline, "read_table", lambda path, **kwargs: tables[path.name])
    spec = PrepSpec(
        "synthetic.csv",
        "series",
        "regression",
        (Column("day", "time"), Column("y", "target", "sum")),
        step="week",
        horizon=horizon,
    )
    ws = load_workspace(spec, Path("synthetic"), {"step": "week"})
    ws.families = ["y@day:lag+0", "y@day:lag+1"]
    return ws


def make_candidate(ws, family="ridge", transform="none"):
    provider = get_family(family)
    families = ws.families if provider.needs == "lag_features" else []
    return make_genome(families, "squared", provider.defaults(), family, transform)


def test_weekly_folds_are_disjoint_expand_history_and_preserve_test(workspace):
    ws = workspace
    cuts = series_cuts(156, ws.spec.horizon)
    assert len(cuts.folds) == len(ws.series_folds) == FOLDS == 3
    assert cuts.test_start == 143
    assert [end - start for start, end in cuts.folds] == [8, 9, 9]
    assert cuts.folds[0][0] == cuts.val_start
    assert cuts.folds[-1][1] == cuts.test_start
    for bucket, frame in ws.frames.items():
        masks = ws.series_masks[bucket]
        previous_train = np.zeros(len(frame.step), dtype=bool)
        previous_val = np.zeros(len(frame.step), dtype=bool)
        all_val = np.zeros(len(frame.step), dtype=bool)
        for (start, end), (train, val) in zip(
            ws.series_folds, pipeline._validation_masks(ws, bucket), strict=True
        ):
            assert train.any() and val.any()
            assert np.all(frame.step[train] < start)
            assert np.all((frame.step[val] >= start) & (frame.step[val] < end))
            assert np.all(frame.step[val] < frame.step[masks["test"]].min())
            assert not np.any((train | val) & masks["test"])
            assert not np.any(all_val & val)
            assert np.all(train[previous_train | previous_val])
            previous_train, previous_val = train, val
            all_val |= val
        np.testing.assert_array_equal(all_val, masks["val"])
        assert len(np.unique(frame.step[masks["test"]])) == 13


@pytest.mark.parametrize(("weeks", "expected_count"), [(34, 1), (50, 2), (156, 3)])
def test_thin_history_keeps_valid_fewer_folds(weeks, expected_count):
    cuts = series_cuts(weeks, 4)
    assert len(cuts.folds) == expected_count
    assert cuts.val_start >= 26
    assert cuts.steps - cuts.test_start >= 4
    assert cuts.folds[0][0] == cuts.val_start
    assert cuts.folds[-1][1] == cuts.test_start
    assert all(start < end <= cuts.test_start for start, end in cuts.folds)
    assert all(left[1] == right[0] for left, right in zip(cuts.folds, cuts.folds[1:], strict=False))


def test_fold_wapes_have_equal_weights_and_stable_paired_contributions():
    days = np.datetime64("2024-01-01") + np.arange(7) * np.timedelta64(1, "D")
    ws = SimpleNamespace(
        spec=SimpleNamespace(task="regression"),
        series_folds=tuple((days[index], days[index + 2]) for index in (0, 2, 4)),
    )
    periods = np.tile(days[:6].astype(int), 2)
    truth = np.tile([1, 1, 10, 10, 100, 100], 2).astype(float)
    errors = np.tile([1, 1, 1, 3, 10, 50], 2).astype(float)
    scored = pipeline._fitness(ws, truth + errors, truth, periods)
    expected_contrib = np.array([0.5, 0.5, 0.05, 0.15, 0.05, 0.25]) / 3
    np.testing.assert_allclose(scored.metrics["fold_scores"], [1.0, 0.2, 0.3])
    np.testing.assert_allclose(scored.contrib, expected_contrib)
    assert scored.fitness == pytest.approx(0.5)
    assert scored.metrics["wape"] == scored.fitness
    assert scored.metrics["fold_count"] == 3
    assert scored.fitness == pytest.approx(scored.contrib.sum())
    assert scored.fitness != pytest.approx(evaluate.wape(truth, truth + errors))
    order = np.arange(len(truth))[::-1]
    reversed_score = pipeline._fitness(ws, (truth + errors)[order], truth[order], periods[order])
    np.testing.assert_array_equal(reversed_score.contrib, scored.contrib)
    better = pipeline._fitness(ws, truth + errors / 2, truth, periods)
    genome = make_genome(["y@day:lag+0"], "squared", {}, "ridge")
    assert _significant(
        Entry(genome, better.fitness, better.contrib, better.metrics, 0),
        Entry(genome, scored.fitness, scored.contrib, scored.metrics, 0),
    )


def test_candidate_fitness_matches_independent_fold_errors(workspace):
    ws = workspace
    scored = score(ws, make_candidate(ws))
    truth = np.concatenate(
        [frame.y[ws.series_masks[bucket]["val"]] for bucket, frame in ws.frames.items()]
    )
    steps = np.concatenate(
        [frame.step[ws.series_masks[bucket]["val"]] for bucket, frame in ws.frames.items()]
    )
    expected = []
    for start, end in ws.series_folds:
        rows = (steps >= start) & (steps < end)
        expected.append(evaluate.wape(truth[rows], scored.prediction[rows]))
    np.testing.assert_allclose(scored.metrics["fold_scores"], expected)
    assert scored.metrics["fold_count"] == 3
    assert scored.fitness == pytest.approx(np.mean(expected))
    assert scored.contrib.sum() == pytest.approx(scored.fitness)


def test_ensemble_average_uses_the_same_equal_fold_ranking():
    days = np.datetime64("2024-01-01") + np.arange(7) * np.timedelta64(1, "D")
    ws = SimpleNamespace(
        spec=SimpleNamespace(task="regression"),
        series_folds=tuple((days[index], days[index + 2]) for index in (0, 2, 4)),
    )
    truth = np.array([1, 1, 10, 10, 100, 100], dtype=float)
    winner = truth * np.array([1.1, 1.1, 1.1, 1.1, 1.5, 1.5])
    runner_up = truth * np.array([1.6, 1.6, 1.1, 1.1, 1.1, 1.1])
    averaged = np.mean([winner, runner_up], axis=0)
    winner_score = pipeline._fitness(ws, winner, truth, days[:6].astype(int))
    ensemble_score = pipeline._fitness(ws, averaged, truth, days[:6].astype(int))
    assert winner_score.fitness < ensemble_score.fitness
    assert ensemble_score.fitness == pytest.approx(0.25)
    assert evaluate.wape(truth, winner) > evaluate.wape(truth, averaged)


@pytest.mark.parametrize("name", ["snaive", "ets", "theta", "arima"])
def test_raw_series_families_fit_every_fold(workspace, monkeypatch, name):
    ws = workspace
    calls = []

    def fit(task, **kwargs):
        steps = task.frame.step[task.rows]
        matching = [
            index
            for index, (start, end) in enumerate(ws.series_folds)
            if np.all((steps >= start) & (steps < end))
        ]
        assert len(matching) == 1
        assert task.panel[STEP].max() <= steps.max()
        calls.append((task.bucket, matching[0]))
        return StatsFitted(
            pipeline._series_baseline(task.frame, task.rows, task.bucket, task.season_length),
            task.season_length,
        )

    monkeypatch.setattr(get_family(name), "fit", fit)
    scored = score(ws, make_candidate(ws, name))
    assert calls == [(bucket, fold) for bucket in ws.frames for fold in range(3)]
    assert scored.metrics["fold_count"] == 3
    assert scored.fitness == pytest.approx(np.mean(scored.metrics["fold_scores"]))


class SealedTargets:
    def __init__(self, values, sealed):
        self.values = values
        self.sealed = sealed

    def __len__(self):
        return len(self.values)

    def __getitem__(self, rows):
        assert not np.any(self.sealed[rows]), "test-mask targets were read during search"
        return self.values[rows]

    def __array__(self, dtype=None, copy=None):
        assert not np.any(self.sealed), "test-mask targets were read during search"
        return np.asarray(self.values, dtype=dtype)


@pytest.mark.parametrize("family", ["ridge", "snaive", "ets"])
@pytest.mark.parametrize("transform", ["none", "log1p", "logdiff"])
def test_search_never_reads_sealed_test_targets(workspace, family, transform):
    if family == "ets":
        pytest.importorskip("statsforecast")
    ws = workspace
    ws.transform = transform
    test_start = ws.series_folds[-1][1]
    ws.panel["y"] = SealedTargets(ws.panel["y"], ws.panel[STEP] >= test_start)
    for bucket, frame in ws.frames.items():
        frame.y = SealedTargets(frame.y, ws.series_masks[bucket]["test"])
        with pytest.raises(AssertionError, match="test-mask targets"):
            frame.y[ws.series_masks[bucket]["test"]]
    with pytest.raises(AssertionError, match="test-mask targets"):
        ws.panel["y"][ws.panel[STEP] >= test_start]
    result = search(
        ws,
        Budget(population=1, max_generations=1, time_cap=60),
        family=family,
        defaults_only=True,
    )
    assert result["best"] is not None
    assert result["leaderboard"]
    for entry in result["leaderboard"]:
        assert entry["fitness"] is not None, entry["note"]
        assert entry["metrics"]["fold_count"] == 3
        assert entry["fitness"] == pytest.approx(np.mean(entry["metrics"]["fold_scores"]))


def test_search_asserts_if_a_validation_mask_includes_test_targets(workspace):
    ws = workspace
    bucket = next(iter(ws.frames))
    masks = ws.series_masks[bucket]
    index = np.flatnonzero(masks["test"])[0]
    masks["val"][index] = True
    start, _ = ws.series_folds[-1]
    ws.series_folds = (
        *ws.series_folds[:-1],
        (start, ws.frames[bucket].step[index] + np.timedelta64(1, "D")),
    )
    with pytest.raises(AssertionError, match="test targets are sealed during search"):
        score(ws, make_candidate(ws))


@pytest.mark.parametrize("family", ["ridge", "snaive"])
def test_cached_and_uncached_finish_share_fold_predictions_and_gap(workspace, family):
    ws = workspace
    genome = make_candidate(ws, family)
    scored = score(ws, genome)
    cached = finish(ws, genome, validation=scored.prediction)
    uncached = finish(ws, genome)
    for phase in ("validation", "test", "forecast"):
        for key in cached[phase]:
            np.testing.assert_array_equal(cached[phase][key], uncached[phase][key])
    assert cached["bands"] == uncached["bands"]
    assert cached["fold_count"] == uncached["fold_count"] == 3
    np.testing.assert_allclose(cached["fold_scores"], scored.metrics["fold_scores"])
    np.testing.assert_allclose(uncached["fold_scores"], cached["fold_scores"])
    assert cached["val_error"] == pytest.approx(scored.fitness)
    assert cached["val_test_gap"] == pytest.approx(cached["model_error"] - scored.fitness)
    assert cached["verdict"]["accuracy"]["val_error"] == pytest.approx(scored.fitness)
    assert cached["verdict"]["accuracy"]["val_test_gap"] == pytest.approx(cached["val_test_gap"])


@pytest.fixture
def checkpoint(workspace):
    saved = []
    search(
        workspace,
        Budget(population=1, max_generations=1, time_cap=60),
        family="ridge",
        on_checkpoint=saved.append,
    )
    assert saved[-1]["generation"] == 1
    assert saved[-1]["validation_folds"] == [
        [str(start), str(end)] for start, end in workspace.series_folds
    ]
    return json.loads(json.dumps(saved[-1], default=lambda value: value.tolist()))


def test_search_rejects_legacy_pooled_checkpoints(workspace, checkpoint):
    checkpoint.pop("validation_folds")
    for entry in checkpoint["entries"]:
        entry["metrics"].pop("fold_scores")
        entry["metrics"].pop("fold_count")
    with pytest.raises(
        pipeline.PipelineError, match="checkpoint validation folds differ; start a new search"
    ):
        search(workspace, Budget(max_generations=1), family="ridge", resume=checkpoint)


def test_search_rejects_changed_fold_checkpoints(workspace, checkpoint):
    (first_start, first_end), (_, second_end), third = workspace.series_folds
    moved_boundary = first_end + np.timedelta64(7, "D")
    workspace.series_folds = (
        (first_start, moved_boundary),
        (moved_boundary, second_end),
        third,
    )
    with pytest.raises(
        pipeline.PipelineError, match="checkpoint validation folds differ; start a new search"
    ):
        search(workspace, Budget(max_generations=1), family="ridge", resume=checkpoint)


def test_compatible_checkpoint_keeps_fold_scores_predictions_and_contributions(
    workspace, checkpoint, monkeypatch
):
    search_module = importlib.import_module("smolsmort.forecast.search")
    expected = {entry["key"]: entry for entry in checkpoint["entries"]}
    restored = []
    serialize_entry = Entry.to_dict

    def inspect_entry(entry):
        saved = expected[entry.genome.key()]
        assert entry.metrics == saved["metrics"]
        assert entry.fitness == saved["fitness"]
        assert entry.fitness == pytest.approx(np.mean(entry.metrics["fold_scores"]))
        np.testing.assert_array_equal(entry.prediction, saved["prediction"])
        np.testing.assert_array_equal(entry.contrib, saved["contrib"])
        restored.append(entry.genome.key())
        return serialize_entry(entry)

    def reject_rescore(*args, **kwargs):
        raise AssertionError("compatible exhausted checkpoint must not rescore")

    monkeypatch.setattr(search_module, "score", reject_rescore)
    monkeypatch.setattr(Entry, "to_dict", inspect_entry)
    result = search(
        workspace,
        Budget(population=1, max_generations=checkpoint["generation"]),
        family="ridge",
        resume=checkpoint,
    )
    assert set(restored) == set(expected)
    assert result["best"] == expected[checkpoint["best_key"]]["genome"]
    assert result["leaderboard"][0]["metrics"]["fold_count"] == 3


@pytest.fixture
def worker_run(workspace, checkpoint, tmp_path, monkeypatch):
    from smolsmort.forecast import worker

    request = parse_request(
        {
            "version": 2,
            "spec": asdict(workspace.spec),
            "prepared": str(tmp_path / "prepared"),
            "summary": {"step": "week"},
            "families": [{"name": "ridge", "method": "grid"}],
            "time_budget_s": 60,
            "nthread": 1,
            "ensemble": {"enabled": False, "top": 1},
        }
    )
    (tmp_path / "request.json").write_text(json.dumps(request))
    family_folder = tmp_path / "families" / "ridge"
    family_folder.mkdir(parents=True)
    (family_folder / "checkpoint.json").write_text(json.dumps(checkpoint))
    (family_folder / "status.json").write_text(
        json.dumps({"state": "done", "reason": "grid complete"})
    )
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker, "load_workspace", lambda *args, **kwargs: workspace)
    return tmp_path


@pytest.mark.parametrize(("family_state", "deadline"), [("done", 200), ("paused", 50)])
def test_v2_worker_rejects_legacy_completed_or_expired_checkpoints(
    worker_run, family_state, deadline, monkeypatch
):
    from smolsmort.forecast import launcher, worker

    checkpoint_path = worker_run / "families" / "ridge" / "checkpoint.json"
    saved = json.loads(checkpoint_path.read_text())
    saved.pop("validation_folds")
    checkpoint_path.write_text(json.dumps(saved))
    (worker_run / "coordinator.json").write_text(
        json.dumps(
            {
                "started_at": 1,
                "deadline": deadline,
                "phase": "deep",
                "families": {"ridge": {"state": family_state}},
            }
        )
    )
    launched = []

    def reject_launch(*args):
        launched.append(args)
        raise AssertionError("legacy checkpoints must fail before coordinator recovery")

    monkeypatch.setattr(worker.time, "time", lambda: 100)
    monkeypatch.setattr(launcher, "launch_families", reject_launch)
    assert worker.run(worker_run) == 1
    assert launched == []
    status = json.loads((worker_run / "status.json").read_text())
    assert status["state"] == "failed"
    assert "checkpoint validation folds differ; start a new search" in status["reason"]


def test_v2_worker_launches_compatible_completed_checkpoint(worker_run, monkeypatch):
    from smolsmort.forecast import launcher, worker

    (worker_run / "coordinator.json").write_text(
        json.dumps(
            {
                "started_at": 1,
                "deadline": 50,
                "phase": "deep",
                "families": {"ridge": {"state": "done"}},
            }
        )
    )
    launched = []

    def launch(folder, request, job):
        assert folder == worker_run
        assert request["families"] == [{"name": "ridge", "method": "grid"}]
        launched.append(folder)
        return {"stopped": "cancelled", "elapsed": 0}

    monkeypatch.setattr(worker.time, "time", lambda: 100)
    monkeypatch.setattr(launcher, "launch_families", launch)
    assert worker.run(worker_run) == 0
    assert launched == [worker_run]
    status = json.loads((worker_run / "status.json").read_text())
    assert status["state"] == "cancelled"
    assert status["stopped"] == "cancelled"
