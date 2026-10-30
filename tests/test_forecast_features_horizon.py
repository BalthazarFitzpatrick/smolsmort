"""one horizon frame drives fitting, saved forecasts and bucket-aware views"""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from test_forecast_seasonal_holdout import build_workspace, candidate

from smolsmort.forecast import pipeline, views
from smolsmort.forecast.features import buckets_for
from smolsmort.forecast.pipeline import finish, fit_importance, genome_from_dict, make_genome, score
from smolsmort.forecast.screen import screen_families
from smolsmort.forecast.search import Budget, search
from smolsmort.forecast.spec import Column
from smolsmort.forecast.tables import read_table, write_table
from smolsmort.forecast.worker import _write_result


@pytest.mark.parametrize(
    "horizon,expected",
    [
        (1, [(1, 1)]),
        (8, [(1, 1), (2, 4), (5, 8)]),
        (13, [(1, 1), (2, 4), (5, 8), (9, 13)]),
        (60, [(1, 1), (2, 4), (5, 8), (9, 13), (14, 26), (27, 52), (53, 60)]),
    ],
)
def test_bucket_modes_keep_grouped_boundaries_and_use_one_single_bucket(horizon, expected):
    assert buckets_for(horizon) == buckets_for(horizon, "grouped") == expected
    assert buckets_for(horizon, "single") == [(1, horizon)]


def test_entity_workspace_rejects_single_horizon_mode(monkeypatch):
    ws = build_workspace(monkeypatch, "block")
    spec = replace(ws.spec, mode="entity", columns=(Column("day", "anchor"), Column("y", "target")))
    with pytest.raises(pipeline.PipelineError, match="single horizon mode requires time series"):
        pipeline.load_workspace(spec, ws.folder, ws.summary, horizon_mode="single")


@pytest.mark.parametrize("holdout", ["block", "seasonal"])
def test_explicit_grouped_matches_default_frames_masks_and_forecasts(monkeypatch, holdout):
    default = build_workspace(monkeypatch, holdout, horizon=8)
    grouped = build_workspace(monkeypatch, holdout, horizon=8, horizon_mode="grouped")
    assert default.spec.digest() == grouped.spec.digest()
    assert default.frames.keys() == grouped.frames.keys()
    for bucket, frame in default.frames.items():
        other = grouped.frames[bucket]
        assert frame.features.names == other.features.names
        np.testing.assert_array_equal(frame.features.x, other.features.x)
        for name, mask in default.series_masks[bucket].items():
            np.testing.assert_array_equal(mask, grouped.series_masks[bucket][name])
    expected, actual = finish(default, candidate(default)), finish(grouped, candidate(grouped))
    for name in ("validation", "test", "forecast"):
        for column, values in expected[name].items():
            np.testing.assert_array_equal(values, actual[name][column])


@pytest.mark.parametrize("holdout", ["block", "seasonal"])
@pytest.mark.parametrize("family", ["ridge", "snaive"])
def test_single_horizon_finishes_with_one_future_fit_and_one_bucket(monkeypatch, holdout, family):
    ws = build_workspace(monkeypatch, holdout, horizon=8, horizon_mode="single")
    frame = ws.frames[(1, 8)]
    np.testing.assert_array_equal(frame.step - frame.origin, np.full(len(frame.step), 56))
    genome = candidate(ws) if family == "ridge" else make_genome([], "squared", {}, "snaive")
    scored = score(ws, genome)
    future_buckets = []
    original = pipeline._timeseries_predict

    def predict(ws_, genome_, frame_, fit_rows, eval_rows, bucket, nthread, **options):
        if np.array_equal(eval_rows, frame_.future):
            future_buckets.append(bucket)
        return original(ws_, genome_, frame_, fit_rows, eval_rows, bucket, nthread, **options)

    monkeypatch.setattr(pipeline, "_timeseries_predict", predict)
    final = finish(ws, genome, validation=scored.prediction, importance_cache={})
    assert future_buckets == [(1, 8)]
    assert final["horizon_mode"] == "single"
    assert set(final["bands"]) == {"1-8"}
    for name in ("validation", "test", "forecast"):
        table = final[name]
        pairs = list(zip(table["series"], table["step"], strict=True))
        assert len(pairs) == len(set(pairs))
        assert set(table["bucket"]) == {"1-8"}
    assert len(final["forecast"]["prediction"]) == 16
    assert final["band_summary"]["test"]["all"]["rows"] == len(final["test"]["actual"])
    if family == "ridge":
        assert final["importance"] and len(fit_importance(ws, genome)) == 1


def test_single_screen_search_ensemble_and_saved_views(monkeypatch, tmp_path):
    pytest.importorskip("duckdb")
    ws = build_workspace(monkeypatch, "seasonal", horizon=8, horizon_mode="single")
    priors = {entry["family"]: entry["p_on"] for entry in screen_families(ws)}
    result = search(
        ws, Budget(population=2, max_generations=1, seed=7), family="ridge", priors=priors
    )
    genome = genome_from_dict(result["best"])
    assert np.isfinite(result["leaderboard"][0]["fitness"])
    members = [genome, make_genome([], "squared", {}, "snaive")]
    final = finish(ws, genome, members=members)
    expected = np.mean([finish(ws, member)["forecast"]["prediction"] for member in members], axis=0)
    np.testing.assert_allclose(final["forecast"]["prediction"], expected)
    _write_result(tmp_path, ws.spec, result, final)
    saved = read_table(tmp_path / "forecast.parquet")
    assert set(saved["bucket"]) == {"1-8"}
    assert len(saved["prediction"]) == 16
    assert json.loads((tmp_path / "result.json").read_text())["horizon_mode"] == "single"
    write_table(tmp_path / "panel.parquet", ws.panel)
    request = {"spec": asdict(ws.spec), "prepared": str(tmp_path), "horizon_mode": "single"}
    view = views.timeseries_view(tmp_path, request, {})
    future = saved["step"] > ws.panel["__step"].max()
    line = next(series for series in view["series"] if series["id"] == "forecast")
    for step in np.unique(saved["step"][future]):
        index = view["x"].index(str(step.astype("datetime64[D]")))
        assert line["values"][index] == pytest.approx(
            saved["prediction"][saved["step"] == step].sum()
        )
        assert view["bands"][0]["lo"][index] is not None
    chart = views._read_tile_chart(tmp_path / "test.parquet")
    assert len(chart) == 8 * len(ws.test_folds)
    for row in chart:
        selected = final["test"]["step"] == np.datetime64(row["step"])
        assert row["prediction"] == pytest.approx(final["test"]["prediction"][selected].sum())
