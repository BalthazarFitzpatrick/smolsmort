"""ended series keep their scored history and lose only their future rows"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from test_forecast_runs_v2 import make_request

from smolsmort.forecast.features import series_features
from smolsmort.forecast.pipeline import finish, load_workspace, make_genome
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.runs import start_run, wait_run
from smolsmort.forecast.spec import SERIES, STEP
from smolsmort.forecast.splits import series_cuts
from smolsmort.forecast.tables import read_table, write_table

pytest.importorskip("duckdb")


def test_workspace_removed_only_future_rows_of_ended_series(tmp_path):
    _, spec, prepared = make_request(tmp_path, [{"name": "snaive"}])
    ws = load_workspace(spec, prepared.folder, prepared.summary)
    scaffold = read_table(prepared.folder / "scaffold.parquet")
    steps = np.unique(ws.panel[STEP])
    cuts = series_cuts(len(steps), spec.horizon)
    train = ws.panel[STEP] < steps[cuts.val_start]
    ended = {f"C|p{product}" for product in range(1, 5)}
    assert [entry["series"] for entry in ws.excluded_series] == sorted(ended)
    for bucket, frame in ws.frames.items():
        before = series_features(
            spec,
            ws.panel,
            scaffold,
            bucket,
            ws.unit,
            train,
            period=ws.profile["genes"].get("period"),
            measure_lags=ws.profile["genes"].get("measure_lags"),
        )
        keep = ~(before.future & np.isin(before.series, list(ended)))
        for name in ("y", "step", "origin", "future"):
            np.testing.assert_array_equal(getattr(frame, name), getattr(before, name)[keep])
        assert np.array_equal(frame.series, before.series[keep])
        np.testing.assert_array_equal(frame.features.x, before.features.x[keep])
        assert frame.features.names == before.features.names
        known = ~before.future & np.isfinite(before.y)
        boundaries = {
            "train": before.step < steps[cuts.val_start],
            "val": (before.step >= steps[cuts.val_start]) & (before.step < steps[cuts.test_start]),
            "test": before.step >= steps[cuts.test_start],
        }
        for phase, boundary in boundaries.items():
            expected = known & boundary
            mask = ws.series_masks[bucket][phase]
            assert np.array_equal(frame.series[mask], before.series[expected])
            np.testing.assert_array_equal(frame.y[mask], before.y[expected])


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_workspace_used_last_finite_target_for_exclusions(tmp_path, missing):
    _, spec, prepared = make_request(tmp_path, [{"name": "snaive"}])
    panel = read_table(prepared.folder / "panel.parquet")
    origin = panel[STEP].max()
    row = (panel[SERIES] == "B|p1") & (panel[STEP] == origin)
    panel["units"][row] = missing
    zero = (panel[SERIES] == "A|p1") & (panel[STEP] == origin)
    panel["units"][zero] = 0
    write_table(prepared.folder / "panel.parquet", panel)
    ws = load_workspace(spec, prepared.folder, prepared.summary)
    assert ws.excluded_series[0] == {
        "series": "B|p1",
        "dims": {"project": "B", "product": "p1"},
        "last_seen": "2026-06-15",
    }
    assert len(ws.excluded_series) == 5
    for bucket, frame in ws.frames.items():
        future = ws.series_masks[bucket]["future"]
        assert "B|p1" not in frame.series[future]
        assert "A|p1" in frame.series[future]


def test_panel_without_ended_series_kept_identical_forecasts(tmp_path):
    _, spec, _ = make_request(tmp_path, [{"name": "snaive"}])
    spec = replace(spec, where="project <> 'C'")
    prepared = prepare(spec, tmp_path / "cache")
    ws = load_workspace(spec, prepared.folder, prepared.summary)
    assert ws.excluded_series == []
    scaffold = read_table(prepared.folder / "scaffold.parquet")
    steps = np.unique(ws.panel[STEP])
    cuts = series_cuts(len(steps), spec.horizon)
    train = ws.panel[STEP] < steps[cuts.val_start]
    original_frames = {
        bucket: series_features(
            spec,
            ws.panel,
            scaffold,
            bucket,
            ws.unit,
            train,
            period=ws.profile["genes"].get("period"),
            measure_lags=ws.profile["genes"].get("measure_lags"),
        )
        for bucket in ws.frames
    }
    original = replace(ws, frames=original_frames)
    genome = make_genome(ws.families, "squared", {}, family="snaive")
    expected = finish(original, genome)
    actual = finish(ws, genome)
    assert actual["excluded_series"] == []
    for name in expected["forecast"]:
        np.testing.assert_array_equal(actual["forecast"][name], expected["forecast"][name])
    run_id = start_run(tmp_path / "runs", spec, prepared, recipe=genome.to_dict(), nthread=1)
    state = wait_run(tmp_path / "runs", run_id, timeout=30)
    assert state["state"] == "done", state
    assert state["excluded_series"] == []
    run = tmp_path / "runs" / run_id
    assert json.loads((run / "result.json").read_text())["excluded_series"] == []
    forecast = read_table(run / "forecast.parquet")
    for name in expected["forecast"]:
        if name == "bucket":
            assert np.array_equal(forecast[name], expected["forecast"][name])
        else:
            np.testing.assert_array_equal(forecast[name], expected["forecast"][name])
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert not any(event["event"] == "excluded_series" for event in events)
