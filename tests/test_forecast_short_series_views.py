"""saved n/a predictions stay missing through charts, tables, exports and view scores"""

import csv
import io
import json

import numpy as np
import pytest
from test_forecast_single_views import single_run as single_run

from smolsmort.forecast import views
from smolsmort.forecast.tables import connect, read_table, write_table


def skip_second_series(folder):
    for split in ("validation", "test", "forecast"):
        path = folder / f"{split}.parquet"
        table = read_table(path)
        skipped = table["series"] == "b"
        for name in ("prediction", "lower", "upper"):
            table[name][skipped] = np.nan
        table["naive_prediction"] = np.full(len(skipped), 5.0)
        write_table(path, table)


@pytest.mark.parametrize("horizon_mode", ["single", "mixed"])
def test_chart_table_export_preserve_skipped_series(single_run, horizon_mode):
    folder, request = single_run
    request["horizon_mode"] = horizon_mode
    skip_second_series(folder)
    result_path = folder / "result.json"
    result = json.loads(result_path.read_text())
    result["verdict"] = {"accuracy": {"ratio": 0.5}}
    result_path.write_text(json.dumps(result))
    payload = views.timeseries_view(folder, request, {})
    forecast = next(line for line in payload["series"] if line["id"] == "forecast")
    values = dict(zip(payload["x"], forecast["values"], strict=True))
    assert values["2025-02-10"] == 10
    table = views.timeseries_table(folder, request, {"size": 100})
    index = table["columns"].index("prediction")
    assert any(row[index] is None for row in table["rows"])
    exported = views.export_csv(folder, request).decode()
    rows = list(csv.reader(io.StringIO(exported.split("\n", 1)[1])))
    assert any(row[index] == "" for row in rows[1:])
    assert all(row[-1] == "" for row in rows[1:] if row[index] == "")
    assert all(row[-1] == "0.5" for row in rows[1:] if row[index] != "")


def test_entirely_skipped_chart_has_null_future_values(single_run):
    folder, request = single_run
    for split in ("validation", "test", "forecast"):
        path = folder / f"{split}.parquet"
        table = read_table(path)
        for name in ("prediction", "lower", "upper"):
            table[name][:] = np.nan
        write_table(path, table)
    payload = views.timeseries_view(folder, request, {})
    forecast = next(line for line in payload["series"] if line["id"] == "forecast")
    assert all(value is None for value in forecast["values"])


def test_horizon_metrics_use_covered_rows_and_matching_baseline(single_run):
    folder, request = single_run
    skip_second_series(folder)
    with connect() as con:
        metrics = views._horizon_metrics(
            con, folder, request, {}, folder / "test.parquet", {"mode": "lead", "n": 1}
        )
    assert metrics["wape"]["value"] == pytest.approx(3 / 10)
    assert metrics["mae"]["value"] == 3
    assert metrics["test_ratio"]["value"] == pytest.approx(3 / 5)
