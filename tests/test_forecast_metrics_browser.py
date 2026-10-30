"""shared error metric choices over saved artifacts, with no forecast workers"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from forecast_ui_audit import settle_menu
from test_forecast_progress_browser import write_multifamily_run
from test_forecast_ui_browser import session, show_tab, switch_to_regression

from smolsmort.forecast.tables import write_table

pytest.importorskip("duckdb")

METRICS = ["validation_error", "wape", "mae", "mape", "test_ratio"]
DEFAULTS = ["validation_error", "test_ratio"]
PICKER = "#regression-error-metrics"
MENU = "#regression-error-metrics-menu"
STORAGE_KEY = "smolsmort:forecast:regression:error-metrics"


def write_metric_run(root: Path) -> str:
    run_id = write_multifamily_run(root)
    folder = root / ".forecast-runs" / run_id
    for index, name in enumerate(("xgboost", "ridge", "snaive")):
        leaderboard = folder / "families" / name / "leaderboard.json"
        rows = json.loads(leaderboard.read_text(encoding="utf-8"))
        rows[0]["metrics"] = {"wape": 0.12 + index * 0.06, "mae": 1.5 + index}
        leaderboard.write_text(json.dumps(rows), encoding="utf-8")
    predictions = {
        "series": np.array(["all"] * 4),
        "step": np.array(
            ["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26"],
            dtype="datetime64[D]",
        ),
        "bucket": np.array(["1-2"] * 4),
        "actual": np.array([0.0, 4.0, 8.0, -2.0]),
        "prediction": np.array([1.0, 3.0, 6.0, -1.0]),
    }
    write_table(folder / "winner_test.parquet", predictions)
    write_table(folder / "families/xgboost/test.parquet", predictions)
    write_table(
        folder / "families/ridge/test.parquet",
        {**predictions, "prediction": np.array([0.0, 4.0, 7.0, -1.0])},
    )
    return run_id


def open_metric_run(page, root: Path) -> str:
    run_id = write_metric_run(root)
    switch_to_regression(page)
    page.select_option("#topic-flavour", "timeseries")
    show_tab(page, "regression-search")
    page.locator(f'#regression-runs-list [data-run-id="{run_id}"]').click()
    page.locator("#regression-tiles .forecast-tile").first.wait_for()
    return run_id


def select_metrics(page, selected: list[str]) -> None:
    page.locator(PICKER).click()
    settle_menu(page)
    assert page.locator(f'{MENU} input[type="checkbox"]').count() == 0
    for metric in METRICS:
        item = page.locator(f'{MENU} .menu-item[data-id="{metric}"]')
        checked = "on" in item.get_attribute("class").split()
        if checked != (metric in selected):
            item.click()
        assert page.locator(MENU).is_visible()
    page.keyboard.press("Escape")


def assert_metric_headers(page, selector: str, selected: list[str]) -> None:
    headers = page.locator(f"{selector} th[data-metric-id]")
    assert headers.evaluate_all("nodes => nodes.map(node => node.dataset.metricId)") == selected
    assert all(headers.nth(index).get_attribute("title") for index in range(headers.count()))


def assert_metric_lines(locator, selected: list[str]) -> None:
    metrics = locator.locator(".forecast-metric[data-metric-id]")
    assert metrics.evaluate_all("nodes => nodes.map(node => node.dataset.metricId)") == selected
    assert all(metrics.nth(index).get_attribute("title") for index in range(metrics.count()))


def save_metrics(page, name: str) -> None:
    folder = Path(__file__).resolve().parents[1] / "outputs" / "error-metric-picker"
    folder.mkdir(parents=True, exist_ok=True)
    viewport = page.viewport_size
    if not name.startswith("phone-"):
        page.set_viewport_size({**viewport, "height": 1800})
    page.locator(PICKER).scroll_into_view_if_needed()
    page.screenshot(path=folder / f"{name}.png", full_page=True, animations="disabled")
    selectors = (
        ("#regression-tiles",)
        if "tiles" in name
        else ("#regression-family-progress", "#regression-leaderboard-table")
        if "tables" in name
        else ("#regression-family-comparison",)
    )
    for selector in selectors:
        locator = page.locator(selector)
        if locator.is_visible():
            locator.screenshot(
                path=folder / f"{name}-{selector.removeprefix('#regression-')}.png",
                animations="disabled",
            )
    page.set_viewport_size(viewport)


def test_error_metric_selection_updates_every_display_from_saved_predictions(tmp_path):
    with session(tmp_path, width=1440) as (page, _base):
        run_id = open_metric_run(page, tmp_path / "forecast_root")
        picker = page.locator(PICKER).element_handle()
        assert page.locator(f".forecast-run-controls {PICKER}").count() == 1
        assert page.locator(PICKER).count() == 1
        assert page.locator('.forecast-metric-controls input[type="checkbox"]').count() == 0
        run_row = page.locator(f'#regression-runs-list [data-run-id="{run_id}"]')
        tile = page.locator('#regression-tiles .forecast-tile[data-family="xgboost"]')
        assert_metric_lines(tile.locator(".forecast-tile-front"), DEFAULTS)
        assert_metric_lines(run_row, DEFAULTS)
        page.locator("#regression-run-view-tables").click()
        for selector in ("#regression-family-progress", "#regression-leaderboard-table"):
            assert_metric_headers(page, selector, DEFAULTS)

        select_metrics(page, METRICS)
        for selector in ("#regression-family-progress", "#regression-leaderboard-table"):
            assert_metric_headers(page, selector, METRICS)
        progress = page.locator("#regression-family-progress tbody tr").first
        values = {
            metric: progress.locator(f'.forecast-metric[data-metric-id="{metric}"]').inner_text()
            for metric in METRICS
        }
        assert values == {
            "validation_error": "12.0%",
            "wape": "35.7%",
            "mae": "1.25",
            "mape": "25.0%",
            "test_ratio": "0.95",
        }
        candidate_mape = page.locator(
            '#regression-leaderboard-table tbody tr:first-child [data-metric-id="mape"]'
        )
        assert candidate_mape.inner_text() == "-"
        assert candidate_mape.get_attribute("title")
        mape = progress.locator('[data-metric-id="mape"]')
        assert "2" in mape.get_attribute("title")
        objective = page.locator('#regression-leaderboard-table th:text-is("loss (objective)")')
        assert "l1" in objective.get_attribute("title")
        assert "squared" in objective.get_attribute("title")
        assert page.locator(".forecast-metric-help").count() == 0
        page.locator(PICKER).click()
        settle_menu(page)
        validation = page.locator(f'{MENU} .menu-item[data-id="validation_error"]')
        assert "held-out validation windows" in validation.get_attribute("title")
        mape_help = page.locator(f'{MENU} .menu-item[data-id="mape"]').get_attribute("title")
        assert "actual > 0" in mape_help and "zero" in mape_help
        for metric in METRICS:
            assert page.locator(f'{MENU} .menu-item[data-id="{metric}"]').get_attribute("title")
        page.keyboard.press("Escape")
        save_metrics(page, "tables")

        page.locator("#regression-run-view-tiles").click()
        for face in (".forecast-tile-front", ".forecast-tile-back"):
            assert_metric_lines(tile.locator(face), METRICS)
            for metric, value in values.items():
                assert (
                    tile.locator(f'{face} [data-metric-id="{metric}"]').inner_text().endswith(value)
                )
        assert_metric_lines(run_row, METRICS)
        save_metrics(page, "tiles-front")
        page.locator("#regression-tiles-flip").click()
        save_metrics(page, "tiles-back")

        page.locator("#regression-run-open").click()
        page.locator("#regression-family-comparison tbody tr").first.wait_for()
        assert page.locator(f".forecast-results-header {PICKER}").count() == 1
        assert page.locator(PICKER).evaluate("(node, original) => node === original", picker)
        assert page.locator(PICKER).count() == 1
        assert_metric_headers(page, "#regression-family-comparison", METRICS)
        comparison = page.locator("#regression-family-comparison tbody tr").first
        for metric, value in values.items():
            assert comparison.locator(f'[data-metric-id="{metric}"]').inner_text() == value
        save_metrics(page, "comparison")

        selected = ["wape", "mae", "mape"]
        select_metrics(page, selected)
        assert_metric_headers(page, "#regression-family-comparison", selected)
        show_tab(page, "regression-search")
        assert page.locator(f".forecast-run-controls {PICKER}").count() == 1
        assert page.locator(PICKER).evaluate("(node, original) => node === original", picker)
        assert_metric_lines(tile.locator(".forecast-tile-front"), selected)
        assert_metric_lines(tile.locator(".forecast-tile-back"), selected)
        assert_metric_lines(run_row, selected)
        page.locator("#regression-run-view-tables").click()
        for selector in ("#regression-family-progress", "#regression-leaderboard-table"):
            assert_metric_headers(page, selector, selected)


def test_error_metric_picker_keyboard_close_and_reload_persistence(tmp_path):
    with session(tmp_path) as (page, _base):
        open_metric_run(page, tmp_path / "forecast_root")
        button = page.locator(PICKER)
        button.focus()
        page.keyboard.press("Enter")
        settle_menu(page)
        assert page.locator(MENU).is_visible()
        assert button.get_attribute("aria-expanded") == "true"
        page.keyboard.press("ArrowDown")
        assert page.locator(f'{MENU} .menu-item[data-id="validation_error"]').evaluate(
            "node => node === document.activeElement"
        )
        for _ in range(3):
            page.keyboard.press("ArrowDown")
        mape = page.locator(f'{MENU} .menu-item[data-id="mape"]')
        assert mape.evaluate("node => node === document.activeElement")
        page.keyboard.press("Space")
        assert "on" in mape.get_attribute("class").split()
        assert page.locator(MENU).is_visible()
        assert mape.evaluate("node => node === document.activeElement")
        page.keyboard.press("Escape")
        assert not page.locator(MENU).is_visible()
        assert button.evaluate("node => node === document.activeElement")
        button.click()
        settle_menu(page)
        page.locator("#regression-run-title").click()
        assert not page.locator(MENU).is_visible()
        assert button.get_attribute("aria-expanded") == "false"
        assert set(page.evaluate("key => JSON.parse(localStorage.getItem(key))", STORAGE_KEY)) == {
            "validation_error",
            "mape",
            "test_ratio",
        }
        page.reload()
        show_tab(page, "regression-search")
        page.locator("#regression-tiles .forecast-tile").first.wait_for()
        assert_metric_lines(
            page.locator("#regression-tiles .forecast-tile-front").first,
            ["validation_error", "mape", "test_ratio"],
        )
        page.locator(PICKER).click()
        settle_menu(page)
        assert page.locator(f'{MENU} .menu-item[data-id="mape"].on').count() == 1


@pytest.mark.parametrize("storage", ["malformed", "blocked"])
def test_error_metric_picker_tolerates_unavailable_storage(tmp_path, storage):
    with session(tmp_path) as (page, _base):
        run_id = open_metric_run(page, tmp_path / "forecast_root")
        if storage == "malformed":
            page.evaluate("key => localStorage.setItem(key, '{broken')", STORAGE_KEY)
        else:
            page.add_init_script(
                """(() => {
                    for (const method of ['getItem', 'setItem']) {
                        const original = Storage.prototype[method];
                        Storage.prototype[method] = function (key, ...args) {
                            if (key.endsWith(':error-metrics')) {
                                throw new DOMException('storage disabled', 'SecurityError');
                            }
                            return original.call(this, key, ...args);
                        };
                    }
                })();"""
            )
        page.reload()
        show_tab(page, "regression-search")
        page.locator(f'#regression-runs-list [data-run-id="{run_id}"]').click()
        front = page.locator("#regression-tiles .forecast-tile-front").first
        assert_metric_lines(front, DEFAULTS)
        select_metrics(page, ["mape"])
        assert_metric_lines(front, ["mape"])


def test_error_metric_picker_keeps_phone_document_within_viewport(tmp_path):
    with session(tmp_path, width=390) as (page, _base):
        open_metric_run(page, tmp_path / "forecast_root")
        select_metrics(page, METRICS)

        def assert_width():
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

        assert_width()
        save_metrics(page, "phone-tiles-front")
        page.locator("#regression-tiles-flip").click()
        assert_width()
        save_metrics(page, "phone-tiles-back")
        page.locator("#regression-run-view-tables").click()
        assert_width()
        save_metrics(page, "phone-tables")
        page.locator("#regression-run-open").click()
        page.locator("#regression-family-comparison tbody tr").first.wait_for()
        assert_width()
        page.locator(PICKER).click()
        settle_menu(page)
        assert_width()
        menu = page.locator(MENU).bounding_box()
        assert menu and menu["x"] >= 0 and menu["x"] + menu["width"] <= 390
        save_metrics(page, "phone-picker")
