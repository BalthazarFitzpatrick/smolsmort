"""synthetic stacked forecasts use fresh origins and keep every lead"""

from pathlib import Path

import numpy as np
import pytest

from smolsmort.forecast import pipeline
from smolsmort.forecast.families import get_family
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec


def build_workspace(
    monkeypatch, *, weeks=165, series=2, holdout="seasonal", mode="single", values=None
):
    positions = np.arange(weeks)
    steps = np.datetime64("2020-01-06") + positions * np.timedelta64(7, "D")
    if values is None:
        values = np.concatenate(
            [
                40 + index * 5 + positions / 10 + 8 * np.sin(positions * 2 * np.pi / 52)
                for index in range(series)
            ]
        )
    panel = {
        SERIES: np.repeat([f"s{index}" for index in range(series)], weeks),
        STEP: np.tile(steps, series),
        "y": np.asarray(values),
        "market": np.tile(100 + positions, series).astype(float),
    }
    future = steps[-1] + np.arange(1, 9) * np.timedelta64(7, "D")
    scaffold = {SERIES: np.repeat(np.unique(panel[SERIES]), 8), STEP: np.tile(future, series)}
    tables = {"panel.parquet": panel, "scaffold.parquet": scaffold}
    monkeypatch.setattr(pipeline, "read_table", lambda path, **kwargs: tables[path.name])
    spec = PrepSpec(
        "synthetic.csv",
        "timeseries",
        "regression",
        (
            Column("day", "time"),
            Column("y", "target", "sum"),
            Column("market", "measure", "mean"),
        ),
        step="week",
        horizon=8,
    )
    workspace = pipeline.load_workspace(
        spec, Path("synthetic"), {"step": "week"}, holdout=holdout, horizon_mode=mode
    )
    return workspace


def candidate(workspace):
    return pipeline.make_genome(
        ["y@day:lag+0", "y@day:lag+1", "y@day:season"],
        "squared",
        get_family("ridge").defaults(),
        "ridge",
    )


@pytest.mark.parametrize("holdout", ["seasonal", "block"])
def test_stacked_slice_fits_scores_and_forecasts(monkeypatch, holdout):
    ws = build_workspace(monkeypatch, holdout=holdout)
    assert list(ws.frames) == [(1, 8)]
    genome = candidate(ws)
    scored = pipeline.score(ws, genome)
    final = pipeline.finish(ws, genome, validation=scored.prediction)
    assert np.isfinite(scored.fitness) and np.isfinite(final["model_error"])
    assert scored.metrics["fold_count"] == (4 if holdout == "seasonal" else 3)
    forecast = final["forecast"]
    for key in np.unique(forecast["series"]):
        rows = forecast["series"] == key
        assert forecast["weeks_ahead"][rows].tolist() == list(range(1, 9))
        assert len(np.unique(forecast["origin"][rows])) == 1
        assert np.all(
            forecast["step"][rows]
            == forecast["origin"][rows] + forecast["weeks_ahead"][rows] * np.timedelta64(7, "D")
        )
    frame = ws.frames[(1, 8)]
    for phase, windows, blocks in (
        ("val", ws.series_folds, pipeline._validation_masks(ws, (1, 8))),
        ("test", ws.test_folds, pipeline._test_windows(ws, (1, 8))),
    ):
        for (start, end), (train, evaluate) in zip(windows, blocks, strict=True):
            assert np.all(frame.step[train] < start)
            assert np.all((frame.origin[evaluate] >= start) & (frame.origin[evaluate] < end))
        rows = final["validation" if phase == "val" else "test"]
        assert len(rows["actual"]) == int(ws.series_masks[(1, 8)][phase].sum())
    assert final["horizon_mode"] == "single"
    assert final["origin_blocks"]["validation"][0]["origin_count"] > 1


def test_every_origin_has_every_known_lead(monkeypatch):
    ws = build_workspace(monkeypatch)
    frame = ws.frames[(1, 8)]
    last = ws.panel[STEP].max()
    for origin in np.unique(frame.origin[~frame.future]):
        rows = (frame.origin == origin) & (frame.series == "s0") & ~frame.future
        remaining = int((last - origin) / np.timedelta64(7, "D"))
        assert sorted(frame.weeks_ahead[rows]) == list(range(1, min(8, remaining) + 1))


@pytest.mark.parametrize("replacement", [np.nan, 1e12])
def test_future_data_cannot_change_origin_features_or_predictions(monkeypatch, replacement):
    ws = build_workspace(monkeypatch)
    frame = ws.frames[(1, 8)]
    start = ws.series_folds[0][0]
    train, _ = next(pipeline._validation_masks(ws, (1, 8)))
    rows = (frame.origin == start) & ~frame.future
    genome = candidate(ws)
    expected = pipeline._timeseries_predict(ws, genome, frame, train, rows, (1, 8), 1)
    original_features = frame.features.x[rows].copy()
    for name in ("y", "market"):
        ws.panel[name][ws.panel[STEP] > start] = replacement
    changed = pipeline.load_workspace(
        ws.spec, ws.folder, ws.summary, holdout="seasonal", horizon_mode="single"
    )
    changed_frame = changed.frames[(1, 8)]
    changed_rows = (changed_frame.origin == start) & ~changed_frame.future
    changed_train = (changed_frame.step < start) & np.isfinite(changed_frame.y)
    np.testing.assert_allclose(
        changed_frame.features.x[changed_rows], original_features, equal_nan=True
    )
    actual = pipeline._timeseries_predict(
        changed, genome, changed_frame, changed_train, changed_rows, (1, 8), 1
    )
    np.testing.assert_allclose(actual, expected)


def test_single_design_always_has_lead_and_ridge_one_hot(monkeypatch):
    ws = build_workspace(monkeypatch)
    frame = ws.frames[(1, 8)]
    _, names, _ = pipeline._timeseries_design(candidate(ws), frame)
    assert "weeks_ahead" in names
    assert all(f"weeks_ahead_{lead}" in names for lead in range(1, 9))
    rows = (frame.weeks_ahead == 1) & ~frame.future
    index = frame.features.names.index("lag_0")
    np.testing.assert_allclose(frame.features.x[rows, index], frame.base[rows], rtol=1e-6)
