"""single-mode views preserve lead identity in saved synthetic artifacts"""

import json

import numpy as np
import pytest

from smolsmort.forecast import views
from smolsmort.forecast.tables import read_table, write_table

# saved synthetic artifacts need duckdb, which ci does not install
pytest.importorskip("duckdb")


def write_predictions(path, day, prediction):
    leads = np.array([1, 8, 1, 8])
    steps = np.full(4, day, dtype="datetime64[D]")
    write_table(
        path,
        {
            "step": steps,
            "origin": steps - leads * np.timedelta64(7, "D"),
            "weeks_ahead": leads,
            "series": np.array(["a", "a", "b", "b"]),
            "bucket": np.array(["1-8"] * 4),
            "actual": np.array([10.0, 10.0, 20.0, 20.0]),
            "prediction": np.array(prediction, dtype=float),
            "lower": np.array(prediction, dtype=float) - 1,
            "upper": np.array(prediction, dtype=float) + 1,
        },
    )


@pytest.fixture
def single_run(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    request = {
        "horizon_mode": "single",
        "prepared": str(prepared),
        "spec": {
            "mode": "timeseries",
            "task": "regression",
            "horizon": 8,
            "columns": [
                {"name": "date", "role": "time"},
                {"name": "unit", "role": "target"},
            ],
        },
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    (tmp_path / "result.json").write_text(json.dumps({"verdict": {}, "family": "ridge"}))
    write_table(
        prepared / "panel.parquet",
        {
            "__step": np.array(["2025-01-06", "2025-01-06"], dtype="datetime64[D]"),
            "__series": np.array(["a", "b"]),
            "unit": np.array([10.0, 20.0]),
        },
    )
    write_predictions(tmp_path / "validation.parquet", "2025-01-06", [9, 5, 19, 15])
    write_predictions(tmp_path / "test.parquet", "2025-01-13", [7, 6, 17, 16])
    leads = np.array([1, 8, 1, 8])
    origins = np.full(4, "2025-02-03", dtype="datetime64[D]")
    write_table(
        tmp_path / "forecast.parquet",
        {
            "step": origins + leads * np.timedelta64(7, "D"),
            "origin": origins,
            "weeks_ahead": leads,
            "series": np.array(["a", "a", "b", "b"]),
            "bucket": np.array(["1-8"] * 4),
            "prediction": np.array([10.0, 11.0, 20.0, 21.0]),
            "lower": np.array([9.0, 10.0, 19.0, 20.0]),
            "upper": np.array([11.0, 12.0, 21.0, 22.0]),
        },
    )
    return tmp_path, request


def test_tile_chart_selects_largest_lead_without_averaging(single_run):
    folder, request = single_run
    chart = views._read_tile_chart(folder / "test.parquet")
    assert chart == [
        {"step": "2025-01-13", "actual": 30.0, "prediction": 22.0, "lower": 20.0, "upper": 24.0}
    ]
    tile = next(iter(views.family_tiles(folder, request, {"state": "done"}).values()))
    assert tile["weeks_ahead_shown"] == 8
    assert tile["chart"] == chart


def test_chart_keeps_largest_historical_lead_and_all_future_weeks(single_run):
    folder, request = single_run
    payload = views.timeseries_view(folder, request, {})
    forecast = next(line for line in payload["series"] if line["id"] == "forecast")
    values = dict(zip(payload["x"], forecast["values"], strict=True))
    assert payload["weeks_ahead_shown"] == 8
    assert values["2025-01-06"] == 20
    assert values["2025-01-13"] == 22
    assert values["2025-02-10"] == 30
    assert values["2025-03-31"] == 32


def test_chart_bands_use_each_future_leads_own_validation_residuals(single_run):
    folder, request = single_run
    path = folder / "validation.parquet"
    source = read_table(path)
    columns = {name: np.tile(values, 4) for name, values in source.items()}
    offsets = np.repeat(np.arange(4), 4) * np.timedelta64(7, "D")
    columns["step"] -= offsets
    columns["origin"] -= offsets
    write_table(path, columns)
    payload = views.timeseries_view(folder, request, {})
    band = payload["bands"][0]
    lower = dict(zip(payload["x"], band["lo"], strict=True))
    upper = dict(zip(payload["x"], band["hi"], strict=True))
    assert lower["2025-02-10"] == pytest.approx(30 * (1 - 2 / 28))
    assert upper["2025-02-10"] == pytest.approx(30 * (1 + 2 / 28))
    assert lower["2025-03-31"] == 16
    assert upper["2025-03-31"] == 48


def test_table_preserves_selected_origins_and_all_future_leads(single_run):
    folder, request = single_run
    payload = views.timeseries_table(folder, request, {})
    assert payload["columns"][-2:] == ["origin", "weeks_ahead"]
    assert payload["total"] == 8
    assert sum(row[-1] == 1 for row in payload["rows"]) == 2
    assert sum(row[-1] == 8 for row in payload["rows"]) == 6


@pytest.mark.parametrize(
    ("mode", "label"),
    [
        ("single", "single model: all weeks ahead, one fit"),
        ("mixed", "mixed: one model per horizon group"),
        (None, "mixed: one model per horizon group"),
    ],
)
def test_run_payload_labels_saved_horizon_mode(single_run, mode, label):
    folder, request = single_run
    request = {**request, "horizon_mode": mode}
    assert views.run_metric_payload(folder, request)["horizon_mode_label"] == label
