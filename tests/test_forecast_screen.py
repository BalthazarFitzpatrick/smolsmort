"""feature screening finds a leading measure without reading held-out targets"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from smolsmort.forecast import pipeline, screen, worker
from smolsmort.forecast.features import step_index
from smolsmort.forecast.pipeline import load_workspace
from smolsmort.forecast.screen import screen_families
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec


def make_panel():
    """twelve seasonal series; orders reveal the next season's increment six weeks ahead"""
    rng = np.random.default_rng(42)
    weeks, series, horizon = 156, 12, 8
    positions = np.arange(weeks)
    steps = np.datetime64("2023-01-02") + positions * np.timedelta64(7, "D")
    keys = [f"series_{index}" for index in range(series)]
    targets, orders = [], []
    for index in range(series):
        increments = rng.normal(0, 4, weeks + 6)
        target = 100 + index * 5 + 30 * np.sin(2 * np.pi * positions / 52)
        for step in range(weeks):
            if step >= 52:
                target[step] = target[step - 52]
            target[step] += increments[step] + rng.normal(0, 0.1)
        targets.append(target)
        orders.append(increments[6:])
    panel = {
        SERIES: np.repeat(keys, weeks),
        STEP: np.tile(steps, series),
        "target": np.concatenate(targets),
        "orders": np.concatenate(orders),
        **{f"noise_{index:02}": rng.normal(size=weeks * series) for index in range(20)},
    }
    future = steps[-1] + np.arange(1, horizon + 1) * np.timedelta64(7, "D")
    scaffold = {SERIES: np.repeat(keys, horizon), STEP: np.tile(future, series)}
    spec = PrepSpec(
        "synthetic.csv",
        "series",
        "regression",
        (
            Column("day", "time"),
            Column("target", "target", "sum"),
            Column("orders", "measure", "sum"),
            *(Column(f"noise_{index:02}", "measure", "sum") for index in range(20)),
        ),
        step="week",
        horizon=horizon,
    )
    return spec, {"panel.parquet": panel, "scaffold.parquet": scaffold}


@pytest.fixture
def workspace(monkeypatch):
    spec, tables = make_panel()
    monkeypatch.setattr(pipeline, "read_table", lambda path, **kwargs: tables[path.name])
    return load_workspace(spec, Path("synthetic"), {"step": "week"})


def noise_median(ranking):
    return float(np.median([row["p_on"] for row in ranking if ":noise_" in row["family"]]))


def test_planted_measure_ranks_first(workspace):
    ranking = screen_families(workspace)
    assert ranking[0]["family"] in {"lead:orders", "measure:orders"}
    assert noise_median(ranking) <= 0.35
    assert [row["score"] for row in ranking] == sorted(
        (row["score"] for row in ranking), reverse=True
    )
    assert {row["family"] for row in ranking} == set(workspace.families)


def test_missing_lightgbm_uses_only_rank_correlation(workspace, monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    ranking = screen_families(workspace)
    assert ranking[0]["family"] in {"lead:orders", "measure:orders"}
    assert noise_median(ranking) <= 0.35
    assert all(row["gain"] == 0 and row["score"] == row["rank_corr"] for row in ranking)


def test_lightgbm_uses_one_small_fit(workspace, monkeypatch):
    pytest.importorskip("lightgbm")
    calls = []
    fit = screen.LightgbmFamily.fit

    def record_fit(self, x, **kwargs):
        calls.append(kwargs)
        return fit(self, x, **kwargs)

    monkeypatch.setattr(screen.LightgbmFamily, "fit", record_fit)
    ranking = screen_families(workspace)
    assert len(calls) == 1
    assert calls[0]["patience"] == 0
    assert calls[0]["max_rounds"] == 40
    assert sum(row["gain"] for row in ranking) == pytest.approx(1)
    for row in ranking:
        assert row["score"] == pytest.approx((row["rank_corr"] + row["gain"]) / 2)


def test_tampered_held_out_targets_do_not_change_ranking(workspace):
    expected = screen_families(workspace)
    held_out = workspace.panel[STEP] >= workspace.series_folds[0][0]
    workspace.panel["target"][held_out] = np.resize([np.nan, 1e12, -1e12], held_out.sum())
    rebuilt = load_workspace(workspace.spec, workspace.folder, workspace.summary)
    assert screen_families(rebuilt) == expected


class SealedTargets:
    def __init__(self, values, sealed):
        self.values, self.sealed = values, sealed

    def __getitem__(self, rows):
        assert not np.any(self.sealed[rows]), "sealed targets were read"
        return self.values[rows]

    def __array__(self, dtype=None, copy=None):
        raise AssertionError("sealed target array was converted")


def test_screen_never_reads_sealed_targets(workspace):
    expected = screen_families(workspace)
    cutoff = workspace.series_folds[0][0]
    workspace.panel["target"] = SealedTargets(
        workspace.panel["target"], workspace.panel[STEP] >= cutoff
    )
    for frame in workspace.frames.values():
        frame.y = SealedTargets(frame.y, frame.step >= cutoff)
    assert screen_families(workspace) == expected


@pytest.mark.parametrize("split", ["val", "test"])
def test_overlapping_train_masks_are_rejected(workspace, split):
    bucket = next(iter(workspace.frames))
    masks = workspace.series_masks[bucket]
    row = np.flatnonzero(masks[split])[0]
    masks["train"][row] = True
    workspace.series_folds = ()
    with pytest.raises(AssertionError, match="validation and test targets are sealed"):
        screen_families(workspace)


def test_priors_and_wildcards_are_bounded_and_seeded(workspace):
    ranking = screen_families(workspace, seed=12)
    assert screen_families(workspace, seed=12) == ranking
    assert all(0.1 <= row["p_on"] <= 0.9 for row in ranking)
    wildcards = [index for index, row in enumerate(ranking) if row["wildcard"]]
    assert len(wildcards) == pytest.approx(len(ranking) * 0.1, abs=0.5)
    assert min(wildcards) >= len(ranking) // 2
    assert all(ranking[index]["p_on"] == 0.5 for index in wildcards)
    other = screen_families(workspace, seed=13)
    assert {row["family"] for row in other if row["wildcard"]} != {
        ranking[index]["family"] for index in wildcards
    }


def test_invalid_fold_boundary_asserts_before_reading_targets(workspace):
    cutoff, end = workspace.series_folds[0]
    workspace.panel["target"] = SealedTargets(
        workspace.panel["target"], workspace.panel[STEP] >= cutoff
    )
    workspace.series_folds = ((end, workspace.series_folds[1][1]), *workspace.series_folds[2:])
    with pytest.raises(AssertionError, match="validation and test targets are sealed"):
        screen_families(workspace)


def test_missing_seasonal_history_returns_finite_zero_scores(workspace):
    workspace.profile["genes"]["period"] = 200
    ranking = screen_families(workspace)
    assert all(row["score"] == row["rank_corr"] == row["gain"] == 0 for row in ranking)
    json.dumps(ranking, allow_nan=False)


def test_constant_and_missing_features_share_finite_priors(workspace):
    for frame in workspace.frames.values():
        frame.features.x[:] = 1
        frame.features.x[:, ::2] = np.nan
    ranking = screen_families(workspace)
    assert all(row["score"] == row["rank_corr"] == row["gain"] == 0 for row in ranking)
    assert all(row["p_on"] == (0.5 if row["wildcard"] else 0.1) for row in ranking)
    json.dumps(ranking, allow_nan=False)


def test_spearman_uses_average_ranks_and_pairwise_finite_values():
    assert screen._rank_correlation(np.array([1, 1, 2, np.nan]), np.array([3, 3, 1, 0])) == 1
    assert screen._rank_correlation(np.ones(4), np.arange(4)) == 0
    assert screen._rank_correlation(np.array([1, np.nan, 2]), np.arange(3)) == 0


def test_seasonal_residual_and_row_cap(workspace):
    x, residual, _ = screen._training_data(workspace, 7, np.random.default_rng(0))
    assert len(x) == len(residual) <= 7
    # a shorter supplied season proves buckets use a multiple that reaches their origin
    workspace.profile["genes"]["period"] = 3
    _, residual, _ = screen._training_data(workspace, 20_000, np.random.default_rng(0))
    history = dict(
        zip(
            zip(workspace.panel[SERIES], step_index(workspace.panel[STEP], "week"), strict=True),
            workspace.panel["target"],
            strict=True,
        )
    )
    expected = []
    for bucket, frame in sorted(workspace.frames.items()):
        rows = np.flatnonzero(workspace.series_masks[bucket]["train"])
        lag = 3 * int(np.ceil(bucket[1] / 3))
        for row in rows:
            previous = history.get(
                (frame.series[row], step_index(frame.step[[row]], "week")[0] - lag)
            )
            if previous is not None:
                expected.append(frame.y[row] - previous)
    np.testing.assert_allclose(residual, expected)


@pytest.mark.parametrize("version", [1, 2])
def test_series_worker_writes_screen_before_search(workspace, tmp_path, monkeypatch, version):
    request = {
        "spec": asdict(workspace.spec),
        "prepared": "synthetic",
        "summary": {"step": "week"},
        "nthread": 1,
    }
    if version == 2:
        request.update(version=2, families=[{"name": "ridge", "method": "grid"}], time_budget_s=60)
    (tmp_path / "request.json").write_text(json.dumps(request))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(worker, "load_workspace", lambda *args, **kwargs: workspace)
    called = []

    def search(*args, **kwargs):
        saved = json.loads((tmp_path / "screen.json").read_text())
        assert saved == screen_families(workspace)
        called.append(True)
        return {"stopped": "cancelled", "leaderboard": [], "population": []}

    monkeypatch.setattr(worker, "search", search)
    from smolsmort.forecast import launcher

    monkeypatch.setattr(launcher, "launch_families", search)
    assert worker.run(tmp_path) == 0
    assert called == [True]


def measure_screen():
    spec, tables = make_panel()
    with patch.object(pipeline, "read_table", lambda path, **kwargs: tables[path.name]):
        ws = load_workspace(spec, Path("synthetic"), {"step": "week"})
    started = time.perf_counter()
    ranking = screen_families(ws)
    elapsed = time.perf_counter() - started
    return {
        "top_five": ranking[:5],
        "median_noise_p_on": noise_median(ranking),
        "seconds": elapsed,
        "families": len(ranking),
    }


if __name__ == "__main__":
    print(json.dumps(measure_screen(), indent=2))
