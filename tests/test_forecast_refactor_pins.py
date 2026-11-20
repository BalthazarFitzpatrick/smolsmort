"""pin wape, ensemble blend and band outputs so shared-code refactors stay exact"""

import hashlib
import json
import math

import numpy as np
import pytest

from smolsmort.forecast import evaluate, export_wide, metrics, plan_sets, tiles, views
from smolsmort.forecast.tables import read_table, write_table

NAN = float("nan")


def digest(value) -> str:
    """hash every number to the last digit, nan and column order included"""
    parts = []

    def walk(item):
        if isinstance(item, dict):
            for key, inner in item.items():
                parts.append(f"k:{key}")
                walk(inner)
        elif isinstance(item, (list, tuple)):
            parts.append(f"n:{len(item)}")
            for inner in item:
                walk(inner)
        elif isinstance(item, np.ndarray):
            parts.append(f"a:{item.dtype}:{item.shape}")
            for inner in item.tolist():
                walk(inner)
        elif isinstance(item, float):
            parts.append("nan" if math.isnan(item) else item.hex())
        else:
            parts.append(repr(item))

    walk(value)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


WAPE_CASES = {
    "plain": ([10.0, 20.0], [12.0, 18.0]),
    "nan_pair": ([10.0, 100.0, 20.0], [12.0, NAN, 18.0]),
    "zero_actual_error": ([0.0, 0.0], [1.0, 2.0]),
    "zero_actual_exact": ([0.0, 0.0], [0.0, 0.0]),
    "all_nan": ([5.0, 6.0], [NAN, NAN]),
    "empty": ([], []),
    "mixed_sign": ([-4.0, 6.0], [-1.0, 9.0]),
}
# evaluate.wape nan, prediction_metrics wape none, plan_sets._wape inf, per zero-volume case
EXPECTED_WAPE = {
    "plain": (4 / 30, 4 / 30, 4 / 30),
    "nan_pair": (4 / 30, 4 / 30, 4 / 30),
    "zero_actual_error": (NAN, None, math.inf),
    "zero_actual_exact": (NAN, None, 0.0),
    "all_nan": (NAN, None, math.inf),
    "empty": (NAN, None, math.inf),
    "mixed_sign": (6 / 10, 6 / 10, 6 / 10),
}


@pytest.mark.parametrize("name", WAPE_CASES)
def test_wape_adapters_keep_their_missing_value_contracts(name):
    actual, prediction = (np.asarray(v, dtype=float) for v in WAPE_CASES[name])
    expected = EXPECTED_WAPE[name]
    values = (
        evaluate.wape(actual, prediction),
        metrics.prediction_metrics(actual, prediction)["wape"],
        plan_sets._wape(prediction, actual),
    )
    for got, want in zip(values, expected, strict=True):
        if want is None:
            assert got is None
        elif isinstance(want, float) and math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(want, rel=1e-15, abs=0)


def make_table(split, family, rows_per_cell, offset, *, with_nan=False):
    # members share actuals per split and differ only in predictions
    rng = np.random.default_rng(hash_seed(split))
    noise = np.random.default_rng(hash_seed(split, family))
    series, ahead = np.meshgrid(np.array(["a", "b"]), np.array([1, 2, 3]), indexing="ij")
    series = np.repeat(series.ravel(), rows_per_cell)
    ahead = np.repeat(ahead.ravel(), rows_per_cell)
    step = np.tile(np.arange(rows_per_cell), 6)
    actual = 50 + 10 * rng.standard_normal(len(series))
    prediction = actual + offset + 4 * noise.standard_normal(len(series))
    if with_nan:
        prediction[3] = np.nan
    return {
        "series": series,
        "step": step,
        "weeks_ahead": ahead,
        "actual": actual,
        "prediction": prediction,
    }


def fixture_tables():
    """validation 120 rows so per-series bands qualify, test and forecast small"""
    members = {}
    for family, offset in (("ridge", 2.0), ("snaive", -3.0), ("xgb", 0.5)):
        members[family] = {}
        for split, rows in (("validation", 20), ("test", 4), ("forecast", 2)):
            members[family][split] = make_table(
                split, family, rows, offset, with_nan=split == "test"
            )
    return members


def hash_seed(*names) -> int:
    return int(hashlib.sha256("/".join(names).encode()).hexdigest()[:8], 16)


def save_run(folder, members):
    for family, splits in members.items():
        genome = {"family": family, "families": ["y:lag"], "objective": "squared", "params": {}}
        target = folder / "families" / family
        target.mkdir(parents=True)
        (target / "winner.json").write_text(json.dumps(genome))
        (target / "status.json").write_text(json.dumps({"state": "done", "generation": 2}))
        (target / "leaderboard.json").write_text(
            json.dumps([{"genome": genome, "fitness": 0.2, "generation": 1}])
        )
        for split, table in splits.items():
            write_table(target / f"{split}.parquet", table)
    (folder / "request.json").write_text(
        json.dumps(
            {
                "version": 2,
                "spec": {"task": "regression", "mode": "timeseries", "horizon": 3},
                "families": [{"name": name} for name in members],
            }
        )
    )
    (folder / "status.json").write_text(json.dumps({"state": "done"}))
    (folder / "result.json").write_text(json.dumps({"winner": {"family": "ridge"}}))


def test_ensemble_views_pin(tmp_path):
    pytest.importorskip("duckdb")
    save_run(tmp_path, fixture_tables())
    rows = tiles.ensemble_rows(tmp_path)
    tiles.write_ensemble_views(tmp_path, rows)
    seen = {}
    for row in sorted(rows, key=tiles.rank_tile)[:5]:
        for split in ("validation", "test", "forecast"):
            path = tmp_path / "tile_views" / row["recipe_key"] / f"{split}.parquet"
            seen[f"{row['tile_id']}/{split}"] = read_table(path)
    assert len(seen) == 12
    assert digest(seen) == PIN_VIEWS


def test_tile_prediction_path_pin(tmp_path):
    pytest.importorskip("duckdb")
    save_run(tmp_path, fixture_tables())
    rows = tiles.ensemble_rows(tmp_path)
    seen = {}
    for row in sorted(rows, key=lambda r: r["tile_id"]):
        for kind in ("validation", "test"):
            path = views._tile_prediction_path(tmp_path, row["tile_id"], kind)
            seen[f"{row['tile_id']}/{kind}"] = read_table(path)
    assert digest(seen) == PIN_TILE_PATH


def test_pick_ensemble_pin(tmp_path):
    pytest.importorskip("duckdb")
    members = fixture_tables()
    for family, splits in members.items():
        (tmp_path / family).mkdir()
        for split, table in splits.items():
            write_table(tmp_path / family / f"winner_{split}.parquet", table)
    best, parts = plan_sets.pick_ensemble(tmp_path, {family: family for family in members})
    assert digest([best, parts]) == PIN_PICK


def test_calibrated_totals_pin():
    members = fixture_tables()["ridge"]
    tables = {}
    for phase, table in members.items():
        tables[phase] = {
            "step": np.asarray(table["step"]).astype("datetime64[D]") + np.timedelta64(100, "D"),
            "origin": np.asarray(table["step"]).astype("datetime64[D]"),
            "weeks_ahead": table["weeks_ahead"],
            "group": table["series"],
            "actual": table["actual"],
            "prediction": table["prediction"],
        }
    assert digest(export_wide.calibrated_totals(tables)) == PIN_TOTALS


def test_lead_band_helpers_pin():
    table = fixture_tables()["ridge"]["validation"]
    bands = evaluate.fit_lead_bands(
        table["actual"], table["prediction"], table["weeks_ahead"], table["series"]
    )
    lower, upper = evaluate.apply_lead_bands(
        bands, table["prediction"], table["weeks_ahead"], table["series"]
    )
    assert digest([lower, upper]) == PIN_BANDS


PIN_VIEWS = "f57cc7f3cc96d70b"
PIN_TILE_PATH = "10ab9f6fd83e9042"
PIN_PICK = "8a17ee353d6622dd"
PIN_TOTALS = "5f17d0f35014c825"
PIN_BANDS = "06584ce583939ccf"
