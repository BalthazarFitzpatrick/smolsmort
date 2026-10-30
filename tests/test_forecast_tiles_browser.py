"""flip tiles use saved synthetic artifacts and preserve controls while polling"""

from __future__ import annotations

import copy
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("playwright")
pytest.importorskip("xgboost")

from test_forecast_progress_browser import write_multifamily_run
from test_forecast_ui_browser import session, show_tab, switch_to_regression, with_error_metrics

from smolsmort.forecast.tables import write_table

pytest.importorskip("duckdb")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_saved_tiles(root: Path) -> str:
    """save completed artifacts directly; no worker or training run is started"""
    run_id = "20261030-120000-tiles"
    run = root / ".forecast-runs" / run_id
    families = {
        "xgboost": {"state": "done", "best_error": 0.3},
        "ridge": {"state": "done", "best_error": 0.2, "elapsed": 780, "generation": 19},
        "snaive": {"state": "pruned", "pruned_reason": "paired validation error behind ridge"},
    }
    genome = {"family": "ridge", "families": ["lag"], "params": {"alpha": 1.0}}
    write_json(
        run / "request.json",
        {
            "version": 2,
            "families": [{"name": name} for name in families],
            "time_budget_s": 1200,
            "spec": {
                "source": "panel.csv",
                "task": "regression",
                "mode": "timeseries",
                "step": "week",
                "horizon": 4,
                "columns": [
                    {"name": "day", "role": "time"},
                    {"name": "project", "role": "dimension"},
                    {"name": "units", "role": "target", "aggregation": "sum"},
                ],
            },
        },
    )
    write_json(run / "status.json", {"state": "done", "families": families})
    write_json(
        run / "result.json",
        {
            "winner": {
                "family": "ridge",
                "fitness": 0.2,
                "model_error": 0.182,
                "verdict": {"accuracy": {"ratio": 0.71, "baseline_error": 0.256}},
            },
            "model_error": 0.182,
            "verdict": {"accuracy": {"ratio": 0.71}},
            "families": {
                name: {**status, "fitness": status.get("best_error")}
                for name, status in families.items()
            },
        },
    )
    write_json(run / "recipe.json", {"family": "ridge", "genome": genome})
    write_json(
        run / "families/ridge/leaderboard.json",
        [
            {
                "fitness": 0.2,
                "genome": genome,
                "metrics": {"fold_scores": [0.19, 0.2, 0.21], "fold_count": 3, "wape": 0.2},
                "generation": 19,
            }
        ],
    )
    write_json(
        run / "families/ridge/importance.json",
        {
            "method": "absolute standardised coefficient",
            "features": [
                {"feature": "lag_1", "importance": 0.7},
                {"feature": "lag_2", "importance": 0.3},
            ],
        },
    )
    write_table(
        run / "winner_test.parquet",
        {
            "step": np.array(
                ["2025-01-06", "2025-01-06", "2025-01-13", "2025-01-13"],
                dtype="datetime64[D]",
            ),
            "series": np.array(["a", "b", "a", "b"]),
            "actual": np.array([6.0, 4.0, 8.0, 7.0]),
            "prediction": np.array([5.0, 3.0, 7.0, 6.0]),
            "lower": np.array([4.0, 2.0, 6.0, 5.0]),
            "upper": np.array([6.0, 4.0, 8.0, 7.0]),
        },
    )
    return run_id


def open_tiles(page) -> None:
    switch_to_regression(page)
    page.select_option("#topic-flavour", "timeseries")
    show_tab(page, "regression-search")
    page.locator("#regression-tiles .forecast-tile").first.wait_for()


def make_search_curve(values: list[float]) -> list[dict]:
    return [
        {
            "generation": index + 1,
            "validation_error": value,
            "wape": value / 2,
            "mae": value * (index + 2) * 10,
            "rmse": value * (index + 3) * 10,
        }
        for index, value in enumerate(values)
    ]


def make_live_tiles() -> dict:
    ridge = {
        "state": "done",
        "best_error": 0.2,
        "tile": {
            "validation_error": 0.2,
            "test_ratio": 0.71,
            "test_error": 0.182,
            "test_error_metric": "wape",
            "test_naive_wape": 0.256,
            "search_curve": make_search_curve([0.5, 0.3, 0.2]),
            "folds": [
                {"label": "fold 1", "error": 0.19, "ratio": 0.66},
                {"label": "fold 2", "error": 0.2, "ratio": 0.83},
                {"label": "2025-01-20", "error": 0.21, "ratio": 0.74},
            ],
            "importance": {
                "method": "absolute standardised coefficient",
                "features": [
                    {"feature": "lag_1", "importance": 0.7},
                    {"feature": "lag_2", "importance": 0.3},
                ],
            },
            "genome": {"family": "ridge", "params": {"alpha": 1.0}},
            "elapsed": 780,
            "generation": 3,
        },
    }
    worse = copy.deepcopy(ridge)
    worse["best_error"] = 0.35
    worse["tile"].update(
        {
            "test_ratio": 1.12,
            "validation_error": 0.35,
            "genome": {"params": {"season": 52}},
            "search_curve": make_search_curve([0.35]),
            "search_single": True,
            "generation": None,
        }
    )
    return {
        "id": "20261030-120000-live",
        "kind": "multi",
        "version": 2,
        "state": "running",
        "task": "regression",
        "time_budget_s": 1200,
        "families": {
            "xgboost": {
                "state": "running",
                "tile": {
                    "search_curve": make_search_curve([0.8, 0.7, 0.6, 0.5, 0.4]),
                    "folds": [],
                    "elapsed": 420,
                    "generation": 5,
                },
            },
            "snaive": worse,
            "lightgbm": {
                "state": "pruned",
                "pruned_reason": "paired validation error behind ridge",
                "tile": {
                    "search_curve": make_search_curve([0.7, 0.55]),
                    "folds": [],
                    "elapsed": 120,
                    "generation": 2,
                },
            },
            "ridge": ridge,
        },
        "generations": [],
        "generation_curves": [],
        "leaderboard": [],
        "leaderboard_generation": 0,
    }


def mock_live_tiles(page, snapshot: dict) -> list:
    calls = []
    page.route(
        "**/api/forecast-runs",
        lambda route: route.fulfill(
            json={"runs": [{key: snapshot[key] for key in ("id", "kind", "state")}]}
        ),
    )

    def read_run(route):
        calls.append(route.request.url)
        response = with_error_metrics(snapshot)
        if snapshot.get("metric_registry"):
            response["metric_registry"] = snapshot["metric_registry"]
        route.fulfill(json=response)

    page.route("**/api/forecast-run?**", read_run)
    return calls


def find_tile(page, family: str):
    return page.locator(f'#regression-tiles .forecast-tile[data-family="{family}"]')


def save_tiles(page, name: str) -> None:
    folder = Path(__file__).resolve().parents[1] / "outputs" / "forecast-tile-generations"
    folder.mkdir(parents=True, exist_ok=True)
    viewport = page.viewport_size
    page.set_viewport_size({**viewport, "height": 1800})
    page.locator("#regression-tiles").screenshot(path=folder / f"{name}.png", animations="disabled")
    page.set_viewport_size(viewport)


def select_chart_metrics(page, selected: list[str]) -> None:
    page.locator("#regression-error-metrics").click()
    menu = page.locator("#regression-error-metrics-menu")
    for metric in selected:
        menu.locator(f'input[data-metric-id="{metric}"]').check()
    for checkbox in menu.locator("input[data-metric-id]").all():
        if checkbox.get_attribute("data-metric-id") not in selected:
            checkbox.uncheck()
    page.keyboard.press("Escape")


def test_chart_faces_compare_running_done_pruned_and_one_shot_generations(tmp_path):
    snapshot = make_live_tiles()
    with session(tmp_path, port=8095) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        charts = page.locator("#regression-tiles .forecast-tile-front .forecast-tile-chart")
        scales = charts.evaluate_all(
            "nodes => nodes.map(node => [node.dataset.scaleMin, node.dataset.scaleMax, "
            "node.dataset.generationMax])"
        )
        assert len(scales) == 4 and all(scale == scales[0] for scale in scales)
        assert float(scales[0][0]) <= 0.2 and float(scales[0][1]) >= 0.8
        assert scales[0][2] == "5"
        for name, generation, value in (
            ("xgboost", 5, 0.4),
            ("ridge", 3, 0.2),
            ("lightgbm", 2, 0.55),
            ("snaive", 1, 0.35),
        ):
            chart = find_tile(page, name).locator(".forecast-tile-front .forecast-tile-chart")
            assert chart.get_attribute("data-chart-metric") == "validation_error"
            marker = chart.locator(".forecast-tile-current")
            assert marker.count() == 1
            assert marker.get_attribute("data-generation") == str(generation)
            assert float(marker.get_attribute("data-value")) == value
            assert "validation error" in chart.locator(".forecast-tile-chart-label").inner_text()
            assert f"{value * 100:.1f}%" in chart.inner_text()
        assert find_tile(page, "lightgbm").locator(".forecast-tile-pruned").count() == 1
        assert find_tile(page, "xgboost").locator(".forecast-tile-search-line").count() == 1
        assert find_tile(page, "ridge").locator(".forecast-tile-search-line").count() == 1
        assert (
            "test forecast not saved for this family"
            not in page.locator("#regression-tiles").inner_text()
        )
        save_tiles(page, "four-family-generations")


def test_chart_metric_switch_first_choice_and_untracked_fallback(tmp_path):
    snapshot = make_live_tiles()
    registry = with_error_metrics(snapshot)["metric_registry"]
    snapshot["metric_registry"] = [
        *registry,
        {
            "id": "rmse",
            "label": "rmse",
            "unit": "number",
            "scope": "validation",
            "direction": "lower",
            "explanation": "saved validation loss",
        },
    ]
    with session(tmp_path, port=8095) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        chart = find_tile(page, "ridge").locator(".forecast-tile-front .forecast-tile-chart")
        validation_line = chart.locator(".forecast-tile-search-line").get_attribute("points")
        select_chart_metrics(page, ["mae", "wape"])
        assert chart.get_attribute("data-chart-metric") == "mae"
        assert "mae" in chart.locator(".forecast-tile-chart-label").inner_text()
        assert float(chart.locator(".forecast-tile-current").get_attribute("data-value")) == 8
        assert (
            chart.locator(".forecast-tile-search-line").get_attribute("points") != validation_line
        )
        select_chart_metrics(page, ["wape"])
        assert chart.get_attribute("data-chart-metric") == "wape"
        assert chart.locator(".forecast-tile-current").get_attribute("data-value") == "0.1"
        for metric, label in (("mape", "mape"), ("test_ratio", "test ratio")):
            select_chart_metrics(page, [metric])
            assert chart.get_attribute("data-chart-metric") == "validation_error"
            assert chart.locator(".forecast-tile-chart-note").inner_text() == (
                f"{label}: not tracked per generation"
            )
            assert "validation error" in chart.locator(".forecast-tile-chart-label").inner_text()
            assert (
                chart.locator(".forecast-tile-search-line").get_attribute("points")
                == validation_line
            )
        select_chart_metrics(page, ["rmse"])
        assert chart.get_attribute("data-chart-metric") == "rmse"
        assert float(chart.locator(".forecast-tile-current").get_attribute("data-value")) == 10
        assert "rmse" in chart.locator(".forecast-tile-chart-label").inner_text()
        scales = page.locator("#regression-tiles .forecast-tile-chart").evaluate_all(
            "nodes => nodes.map(node => [node.dataset.scaleMin, node.dataset.scaleMax])"
        )
        assert all(scale == scales[0] for scale in scales)
        save_tiles(page, "selected-loss-generations")


def test_chart_broad_phase_single_value_and_missing_curve_message(tmp_path):
    snapshot = make_live_tiles()
    snapshot["families"]["xgboost"]["tile"].update(
        search_curve=make_search_curve([0.45]), search_single=True, generation=None
    )
    snapshot["families"]["ridge"]["tile"]["search_curve"] = []
    with session(tmp_path, port=8095) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        running = find_tile(page, "xgboost").locator(".forecast-tile-chart")
        assert running.locator(".forecast-tile-current").count() == 1
        assert "45.0%" in running.inner_text()
        missing = find_tile(page, "ridge").locator(".forecast-tile-chart")
        assert "no search curve saved for this family" in missing.inner_text()
        assert missing.locator("svg").count() == 0


def make_live_legacy_tiles() -> dict:
    return {
        "id": "20261030-130000-legacy-live",
        "version": 1,
        "kind": "search",
        "state": "running",
        "family": "xgboost",
        "families": {},
        "budget": {"time_cap": 1200, "max_generations": 10, "plateau": 5},
        "progress": {"generation": 1, "elapsed": 42},
        "generations": [
            {"generation": 0, "best": 0.5, "metrics": {"wape": 0.5, "mae": 12}},
            {"generation": 1, "best": 0.25, "metrics": {"wape": 0.25, "mae": 8}},
        ],
        "generation_curves": [
            {"generation": 0, "key": "legacy-0", "metric": "rmse", "validation": [20, 15]},
            {"generation": 1, "key": "legacy-1", "metric": "rmse", "validation": [15, 10]},
        ],
        "leaderboard": [
            {
                "key": "legacy-1",
                "fitness": 0.25,
                "generation": 1,
                "genome": {"params": {"max_depth": 2}},
                "metrics": {"wape": 0.25, "mae": 8},
            }
        ],
        "leaderboard_generation": 1,
    }


def test_live_legacy_chart_polled_generations_and_kept_menu_focus(tmp_path):
    snapshot = make_live_legacy_tiles()
    with session(tmp_path, port=8095) as (page, _base):
        calls = mock_live_tiles(page, snapshot)
        open_tiles(page)
        tile = find_tile(page, "xgboost")
        chart = tile.locator(".forecast-tile-chart")
        marker = chart.locator(".forecast-tile-current")
        assert marker.get_attribute("data-generation") == "2"
        assert marker.get_attribute("data-value") == "0.25"
        assert chart.locator(".forecast-tile-search-line").count() == 1
        select_chart_metrics(page, ["mae"])
        assert chart.get_attribute("data-chart-metric") == "mae"
        assert marker.get_attribute("data-value") == "8"
        menu = tile.locator('.forecast-tile-front details[data-menu="param"]')
        summary = menu.locator("summary")
        summary.focus()
        page.keyboard.press("Enter")
        assert menu.get_attribute("open") is not None
        before = len(calls)
        snapshot["progress"]["generation"] = 2
        snapshot["generations"].append(
            {"generation": 2, "best": 0.2, "metrics": {"wape": 0.2, "mae": 6}}
        )
        snapshot["generation_curves"].append(
            {"generation": 2, "key": "legacy-2", "metric": "rmse", "validation": [10, 7]}
        )
        page.wait_for_function(
            "() => document.querySelector('#regression-tiles .forecast-tile-current')"
            "?.dataset.generation === '3'"
        )
        assert len(calls) > before
        assert marker.get_attribute("data-value") == "6"
        assert menu.get_attribute("open") is not None
        assert summary.evaluate("node => node === document.activeElement")


def test_live_legacy_chart_without_generations_showed_single_saved_value(tmp_path):
    snapshot = make_live_legacy_tiles()
    snapshot.update(generations=[], generation_curves=[], progress={"elapsed": 10})
    with session(tmp_path, port=8095) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        chart = find_tile(page, "xgboost").locator(".forecast-tile-chart")
        assert chart.locator(".forecast-tile-current").get_attribute("data-value") == "0.25"
        assert chart.locator(".forecast-tile-current-value").text_content() == "single value: 25.0%"
        assert chart.locator(".forecast-tile-search-line").count() == 0
        select_chart_metrics(page, ["mae"])
        assert chart.locator(".forecast-tile-current").get_attribute("data-value") == "8"


def test_chart_missing_latest_metric_did_not_highlight_older_generation(tmp_path):
    snapshot = make_live_tiles()
    for name in ("ridge", "lightgbm"):
        snapshot["families"][name]["tile"]["search_curve"][-1]["mae"] = None
    with session(tmp_path, port=8095) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        select_chart_metrics(page, ["mae"])
        for name, label in (
            ("ridge", "gen 3: not saved"),
            ("lightgbm", "pruned · gen 2: not saved"),
        ):
            chart = find_tile(page, name).locator(".forecast-tile-chart")
            assert chart.locator(".forecast-tile-current").count() == 0
            assert chart.locator(".forecast-tile-pruned").count() == 0
            assert chart.locator(".forecast-tile-current-value").text_content() == label
            assert chart.locator(".forecast-tile-search-point").count() > 0


def test_tiles_read_saved_winner_artifacts_and_kept_tables(tmp_path):
    with session(tmp_path, port=8095) as (page, _base):
        make_saved_tiles(tmp_path / "forecast_root")
        with page.expect_response("**/api/forecast-run?**") as response:
            open_tiles(page)
        data = response.value.json()
        tile_data = data["families"]["ridge"]["tile"]
        assert tile_data["search_curve"][-1]["validation_error"] == 0.2
        assert "chart" not in tile_data
        assert page.locator("#regression-run-view-tiles").get_attribute("aria-pressed") == "true"
        assert page.locator("#regression-tiles .forecast-tile").count() == 3
        assert not page.locator("#regression-family-progress").is_visible()
        save_tiles(page, "saved-charts")
        ridge = find_tile(page, "ridge")
        front = ridge.locator(".forecast-tile-front")
        assert "test ratio: 0.71" in front.inner_text()
        assert front.locator(".forecast-tile-current").get_attribute("data-value") == "0.2"
        assert front.locator(".forecast-tile-split").count() == 0
        assert "no search curve saved for this family" in find_tile(page, "snaive").inner_text()
        assert (
            "test forecast not saved for this family"
            not in page.locator("#regression-tiles").inner_text()
        )
        assert find_tile(page, "xgboost").get_attribute("data-state") == "unknown"
        feature_menu = front.locator('details[data-menu="feat"]')
        feature_menu.locator("summary").click()
        assert "lag_1" in feature_menu.inner_text() and "70%" in feature_menu.inner_text()
        params = front.locator('details[data-menu="param"]')
        params.locator("summary").click()
        assert "alpha" in params.inner_text()
        page.locator("#regression-tiles-flip").click()
        back = ridge.locator(".forecast-tile-back")
        assert "fold 1" in back.inner_text() and "fold 3" in back.inner_text()
        assert "test" in back.inner_text()
        assert "wape per validation window; mark = naive on test" in back.inner_text()
        fold = back.locator('.forecast-tile-bar[data-label="fold 1"]')
        assert fold.get_attribute("aria-label") == "fold 1: wape 19.0%; lower is better"
        assert float(
            fold.locator(".forecast-tile-fill").evaluate("node => parseFloat(node.style.width)")
        ) == pytest.approx(0.19 / 0.26 * 100)
        save_tiles(page, "saved-folds")
        page.locator("#regression-run-view-tables").click()
        assert page.locator("#regression-family-progress").is_visible()
        assert page.locator("#regression-family-progress tbody tr").count() == 3
        assert not page.locator("#regression-tiles").is_visible()
        page.locator("#regression-run-view-tiles").click()
        assert page.locator("#regression-tiles").is_visible()


def test_tiles_polled_states_without_losing_flip_menu_or_focus(tmp_path):
    snapshot = make_live_tiles()
    with session(tmp_path, port=8095) as (page, _base):
        calls = mock_live_tiles(page, snapshot)
        open_tiles(page)
        families = page.locator("#regression-tiles-grid .forecast-tile").evaluate_all(
            "tiles => tiles.map(tile => tile.dataset.family)"
        )
        assert families[:2] == ["ridge", "snaive"]
        save_tiles(page, "live-charts")
        for name, score, color in (
            ("ridge", "test ratio: 0.71", "rgb(199, 237, 95)"),
            ("snaive", "test ratio: 1.12", "rgb(173, 121, 111)"),
            ("xgboost", "test ratio: -", "rgb(119, 183, 199)"),
            ("lightgbm", "test ratio: -", "rgb(173, 121, 111)"),
        ):
            front = find_tile(page, name).locator(".forecast-tile-front")
            assert score in front.inner_text()
            assert front.evaluate("node => getComputedStyle(node).borderTopColor") == color
        running = find_tile(page, "xgboost").locator(".forecast-tile-front")
        assert running.locator(".forecast-tile-search-line").count() == 1
        assert "running 7 min" in running.inner_text() and "gen 5" in running.inner_text()
        flip = page.locator("#regression-tiles-flip")
        flip.focus()
        page.keyboard.press("Enter")
        assert flip.get_attribute("aria-pressed") == "true"
        assert "show charts" in flip.inner_text()
        ridge = find_tile(page, "ridge")
        assert ridge.locator(".forecast-tile-front").evaluate("node => node.inert")
        assert not ridge.locator(".forecast-tile-back").evaluate("node => node.inert")
        back = ridge.locator(".forecast-tile-back")
        assert "2025-01-20" in back.inner_text()
        fold = back.locator('.forecast-tile-bar[data-label="fold 1"]')
        assert fold.get_attribute("aria-label") == "fold 1: wape 19.0%; lower is better"
        assert float(
            fold.locator(".forecast-tile-fill").evaluate("node => parseFloat(node.style.width)")
        ) == pytest.approx(0.19 / 0.26 * 100)
        origin = back.locator(
            '.forecast-tile-bar[data-label="2025-01-20"] .forecast-tile-bar-label'
        )
        assert origin.evaluate("node => node.scrollWidth <= node.clientWidth")
        assert "wape per validation window; mark = naive on test" in back.inner_text()
        save_tiles(page, "live-folds")
        menu = back.locator('details[data-menu="feat"]')
        summary = menu.locator("summary")
        summary.focus()
        page.keyboard.press("Enter")
        assert menu.get_attribute("open") is not None
        before = len(calls)
        snapshot["families"]["xgboost"]["tile"]["generation"] = 6
        snapshot["families"]["xgboost"]["tile"]["search_curve"].append(
            {"generation": 6, "validation_error": 0.3, "wape": 0.15, "mae": 21}
        )
        snapshot["families"]["xgboost"]["tile"]["folds"] = [{"label": "fold 1", "error": 0.633}]
        snapshot["families"]["ridge"]["tile"]["test_ratio"] = 0.69
        page.wait_for_function(
            "() => document.querySelector('[data-family=\"xgboost\"] .forecast-tile-front')"
            ".textContent.includes('gen 6')"
        )
        assert len(calls) > before
        assert running.locator(".forecast-tile-current").get_attribute("data-generation") == "6"
        assert running.locator(".forecast-tile-current").get_attribute("data-value") == "0.3"
        assert (
            ridge.locator('.forecast-tile-front [data-metric-id="test_ratio"]').inner_text()
            == "test ratio: 0.69"
        )
        assert flip.get_attribute("aria-pressed") == "true"
        assert menu.get_attribute("open") is not None
        assert summary.evaluate("node => node === document.activeElement")
        assert back.locator(".forecast-tile-bars").get_attribute("data-scale-max") == "0.64"
        assert float(
            back.locator('.forecast-tile-bar[data-label="fold 1"] .forecast-tile-fill').evaluate(
                "node => parseFloat(node.style.width)"
            )
        ) == pytest.approx(0.19 / 0.64 * 100)
        page.keyboard.press("Enter")
        assert menu.get_attribute("open") is None
        flip.click()
        assert flip.get_attribute("aria-pressed") == "false"
        assert ridge.locator(".forecast-tile-back").evaluate("node => node.inert")


@pytest.mark.parametrize("seasonal", [False, True])
def test_back_folds_fill_each_family_on_one_wape_scale(tmp_path, seasonal):
    with session(tmp_path, port=8095) as (page, _base):
        root = tmp_path / "forecast_root"
        run_id = make_saved_tiles(root)
        run = root / ".forecast-runs" / run_id
        scores = {
            "ridge": [0.193, 0.21, 0.24, *([0.266] if seasonal else [])],
            "xgboost": [0.35, 0.29],
            "snaive": [0.42],
        }
        origins = ["2023-12-04", "2024-01-01", "2024-02-05", "2024-03-04"]
        result = json.loads((run / "result.json").read_text())
        if seasonal:
            result["holdout"] = {"validation_origins": origins}
        result["winner"]["verdict"]["accuracy"]["baseline_error"] = 0.88
        result["families"]["xgboost"]["state"] = "running"
        write_json(run / "result.json", result)
        status = json.loads((run / "status.json").read_text())
        status["state"] = status["families"]["xgboost"]["state"] = "running"
        write_json(run / "status.json", status)
        for name, values in scores.items():
            write_json(
                run / "families" / name / "leaderboard.json",
                [
                    {"fitness": 0.9, "metrics": {"fold_scores": [0.9] * 4}},
                    {
                        "fitness": sum(values) / len(values),
                        "genome": {"family": name, "params": {}},
                        "metrics": {"fold_scores": values, "fold_count": len(values)},
                    },
                ],
            )
        open_tiles(page)
        page.locator("#regression-error-metrics").click()
        page.locator('#regression-error-metrics-menu input[data-metric-id="test_ratio"]').uncheck()
        page.keyboard.press("Escape")
        page.locator("#regression-tiles-flip").click()
        for name, values in scores.items():
            back = find_tile(page, name).locator(".forecast-tile-back")
            bars = back.locator(".forecast-tile-bars")
            assert bars.get_attribute("data-scale-max") == "0.88"
            assert bars.locator(".forecast-tile-bar").count() == len(values) + 1
            assert bars.evaluate(
                "node => [...node.children].every(row => "
                "row.getBoundingClientRect().bottom <= node.getBoundingClientRect().bottom + 1)"
            )
            for index, value in enumerate(values):
                label = origins[index] if seasonal else f"fold {index + 1}"
                bar = bars.locator(f'.forecast-tile-bar[data-label="{label}"]')
                assert bar.locator(".forecast-tile-bar-value").inner_text() == f"{value * 100:.1f}%"
                assert float(
                    bar.locator(".forecast-tile-fill").evaluate(
                        "node => parseFloat(node.style.width)"
                    )
                ) == pytest.approx(value / 0.88 * 100, abs=0.001)
                assert bar.locator(".forecast-tile-mark").count() == 0
            assert "wape per validation window; mark = naive on test" in back.inner_text()
        ridge = find_tile(page, "ridge")
        test = ridge.locator('.forecast-tile-back .forecast-tile-bar[data-label="test"]')
        assert test.locator(".forecast-tile-bar-value").inner_text() == "18.2%"
        assert test.locator(".forecast-tile-mark").evaluate("node => node.style.left") == "100%"
        assert test.locator(".forecast-tile-mark").get_attribute("title") == "naive test wape 88.0%"
        assert page.locator("#regression-tiles .forecast-tile-mark").count() == 1
        assert ridge.locator(".forecast-tile-back .forecast-tile-score").inner_text() == "+29%"
        save_tiles(page, "seasonal-wape-folds" if seasonal else "block-wape-folds")


def test_tiles_mobile_layout_motion_and_palette_tokens(tmp_path):
    with session(tmp_path, width=390, port=8095) as (page, _base):
        mock_live_tiles(page, make_live_tiles())
        open_tiles(page)
        tiles = page.locator("#regression-tiles-grid .forecast-tile")
        boxes = [tile.bounding_box() for tile in tiles.all()]
        assert all(box and abs(box["width"] - 232) < 1 for box in boxes)
        assert all(abs(box["height"] - 306) < 1 for box in boxes)
        assert all(box["x"] >= 0 and box["x"] + box["width"] <= 390 for box in boxes)
        assert all(a["y"] + a["height"] <= b["y"] for a, b in pairwise(boxes))
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        save_tiles(page, "mobile-charts")
        inner = tiles.first.locator(".forecast-tile-inner")
        assert inner.evaluate("node => getComputedStyle(node).transitionDuration") == "0.42s"
        page.emulate_media(reduced_motion="reduce")
        assert inner.evaluate("node => getComputedStyle(node).transitionDuration") == "0s"
        front = tiles.first.locator(".forecast-tile-front")
        dark = front.evaluate("node => getComputedStyle(node).backgroundColor")
        page.evaluate(
            """() => {
                const root = document.documentElement;
                root.style.setProperty('--panel-row', '#f5f5f5');
                root.style.setProperty('--text', '#222222');
                root.style.setProperty('--cream', '#222222');
                root.style.setProperty('--text-dim', '#666666');
                root.style.setProperty('--bg', '#ffffff');
                root.style.setProperty('--grey-border', '#aaaaaa');
            }"""
        )
        assert front.evaluate("node => getComputedStyle(node).backgroundColor") != dark
        assert front.evaluate("node => getComputedStyle(node).color") == "rgb(34, 34, 34)"
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        save_tiles(page, "mobile-light-tokens")
        page.locator("#regression-tiles-flip").click()
        assert tiles.first.locator(".forecast-tile-front").evaluate("node => node.inert")


def test_legacy_tiles_loaded_saved_search_curves_and_kept_live_metadata(tmp_path):
    with session(tmp_path, port=8095) as (page, _base):
        write_multifamily_run(tmp_path / "forecast_root", version=1)
        with page.expect_response("**/api/forecast-view?**") as response:
            open_tiles(page)
        assert list(response.value.json()["tiles"]) == ["xgboost"]
        tile = find_tile(page, "xgboost")
        tile.locator(".forecast-tile-front svg").wait_for()
        assert (
            tile.locator('.forecast-tile-front [data-metric-id="test_ratio"]').inner_text()
            == "test ratio: 0.95"
        )
        assert tile.locator(".forecast-tile-front .forecast-tile-current").count() == 1
        assert tile.locator(".forecast-tile-search-line").count() == 1
        assert [
            point["generation"]
            for point in response.value.json()["tiles"]["xgboost"]["search_curve"]
        ] == [1, 2]
        value = page.evaluate(
            """() => forecastTileEntries({version: 1, state: 'running',
              families: {xgboost: {state: 'running'}}, progress: {elapsed: 42, generation: 5},
              leaderboard: [{genome: {params: {max_depth: 2}}, fitness: 0.3}]})[0]"""
        )
        assert value["elapsed"] == 42 and value["generation"] == 5
        assert value["genome"]["params"] == {"max_depth": 2}
        assert page.evaluate("forecastTileState({state: 'paused'})") == "running"
