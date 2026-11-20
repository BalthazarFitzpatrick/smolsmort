"""saved model columns use shared menu selection and dismissal"""

from pathlib import Path

import pytest
from forecast_ui_audit import settle_menu
from test_forecast_horizon_browser import horizon_session, open_horizon_run


def test_results_exports_use_dialog_only(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        assert page.locator("#regression-export-dialog").is_visible()
        assert page.locator("#regression-export").count() == 0
        assert page.locator("#regression-export-wide").count() == 0
        assert page.get_by_role("link", name="csv", exact=True).count() == 0
        assert page.get_by_role("link", name="csv wide", exact=True).count() == 0


@pytest.mark.parametrize("selection", [None, ["test_ratio", "mae"], ["test_mae"], []])
def test_results_model_resolves_tile_metric(tmp_path, selection):
    with horizon_session(tmp_path) as (page, run_id):
        page.evaluate("localStorage.clear()")
        page.reload()
        open_horizon_run(page, run_id)
        assert page.evaluate("forecastTopics.get('regression').selectedMetricIds") == [
            "validation_error",
            "test_ratio",
        ]
        page.evaluate(
            """selection => {
              const topic = forecastTopics.get('regression');
              if (selection !== null) topic.selectedMetricIds = selection;
              topic.metricRegistry.push({id: 'test_mae', name: 'mae', scope: 'test'});
              topic.resultsView.ensemble = null;
              topic.resultsView.run_tiles = [
                {tile_id: 'ridge', family: 'ridge', kind: 'model', test_ratio: 0.9,
                  selection_metrics: {wape: 0.18, mae: 100},
                  metrics: {ape: {}, mape: {}, ae: {}, wape: {validation: 0.3, test: 0.18}}},
                {tile_id: 'snaive', family: 'snaive', kind: 'model',
                  metrics: {ape: {}, mape: {}, ae: {}, wape: {validation: 0.2, test: 0.2}}},
              ];
              if (selection?.length) {
                topic.resultsView.run_tiles[0].metrics = {mae: {validation: 150, test: 100}};
                topic.resultsView.run_tiles[1].metrics = {mae: {validation: 200, test: 200}};
              } else if (selection !== null) {
                topic.resultsView.run_tiles.forEach(tile => {
                  tile.metrics = {unknown: {validation: 3, test: 2}};
                  delete tile.test_ratio;
                });
              }
            }""",
            selection,
        )
        page.click("#regression-results-model")
        settle_menu(page)
        ridge = page.locator('.forecast-results-menu .menu-item[data-id="ridge"]')
        values = [
            ridge.locator(f".forecast-results-{column}").inner_text()
            for column in ("val", "test", "ratio")
        ]
        if selection is None:
            assert values == ["0.3000", "0.1800", "0.90"]
            assert ridge.get_attribute("title") == "wape 0.1800 at lead 3"
            screenshot = (
                Path(__file__).resolve().parents[1] / "artifacts/results-model-default-metric.png"
            )
            screenshot.parent.mkdir(exist_ok=True)
            page.screenshot(path=str(screenshot))
        elif selection:
            assert values == ["150", "100", "0.50"]
            assert ridge.get_attribute("title") == "mae 100.0000 at lead 3"
        else:
            assert values == ["–", "–", "–"]


def test_results_model_columns(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        page.evaluate(
            """() => {
              const topic = forecastTopics.get('regression');
              topic.selectedMetricIds = ['wape'];
              topic.resultsModel = 'ridge';
              topic.resultsView.ensemble = null;
              topic.resultsView.view = null;
              topic.resultsView.run_tiles = [
                {tile_id: 'ridge', family: 'ridge', kind: 'model',
                  selection_metrics: {wape: 0.1847}, test_ratio: 0.9,
                  metrics: {wape: {validation: 0.3, test: 0.18}, mae: {validation: 1234.56, test: 100}}},
                {tile_id: 'snaive', family: 'snaive', kind: 'model',
                  metrics: {wape: {validation: 0.2, test: 0.2}, mae: {validation: 200, test: 200}}},
                {tile_id: 'ensemble', family: 'ensemble', kind: 'ensemble', members: ['ridge', 'snaive'],
                  metrics: {wape: {validation: 0.1, test: 0.19}, mae: {validation: 150, test: 50}}},
                {tile_id: 'worse', family: 'ets', kind: 'model', metrics: {wape: {test: 0.3}}},
                {tile_id: 'missing', family: 'theta', kind: 'model', test_ratio: 0.4,
                  metrics: {wape: {validation: 0.05}}},
              ];
              topic.paintModelResults({...topic.resultsView, model: 'ridge'});
            }"""
        )
        page.click("#regression-results-model")
        settle_menu(page)
        menu = page.locator(".forecast-results-menu")
        rows = menu.locator(".menu-item")

        def order():
            return rows.evaluate_all("rows => rows.map(row => row.dataset.id)")

        assert order() == ["ridge", "ensemble", "snaive", "worse", "missing"]
        ridge = menu.locator('.menu-item[data-id="ridge"]')
        assert ridge.locator(".forecast-results-val").inner_text() == "0.3000"
        assert ridge.locator(".forecast-results-test").inner_text() == "0.1800"
        assert ridge.locator(".forecast-results-ratio").inner_text() == "0.90"
        assert ridge.get_attribute("title") == "wape 0.1847 at saved test"
        ensemble = menu.locator('.menu-item[data-id="ensemble"]')
        assert ensemble.locator(".name").inner_text() == "ensemble of ridge, snaive"
        assert ensemble.locator(".forecast-results-val").inner_text() == "0.1000"
        assert ensemble.locator(".forecast-results-test").inner_text() == "0.1900"
        assert ensemble.locator(".forecast-results-ratio").inner_text() == "0.95"
        for model, tone in [
            ("ridge", "better"),
            ("snaive", "identical"),
            ("worse", "worse"),
            ("missing", "none"),
        ]:
            assert (
                menu.locator(f'[data-id="{model}"] .forecast-results-ratio.tone-{tone}').count()
                == 1
            )
        assert menu.locator('[data-id="missing"] .forecast-results-test').inner_text() == "–"
        assert ridge.evaluate("row => row.getBoundingClientRect().height") == 31
        assert ridge.evaluate("row => getComputedStyle(row).color") == ridge.locator(
            ".forecast-results-ratio"
        ).evaluate("cell => getComputedStyle(cell).color")
        assert ridge.evaluate("""row => {
          const probe = document.createElement('span');
          probe.style.backgroundColor = 'var(--accent-on)';
          row.append(probe);
          const matches = getComputedStyle(row).backgroundColor === getComputedStyle(probe).backgroundColor;
          probe.remove();
          return matches;
        }""")
        screenshot = Path(__file__).resolve().parents[1] / "artifacts/results-model-columns.png"
        screenshot.parent.mkdir(exist_ok=True)
        page.screenshot(path=str(screenshot))
        for column, expected in [
            ("val", ["missing", "ensemble", "snaive", "ridge", "worse"]),
            ("test", ["ridge", "ensemble", "snaive", "worse", "missing"]),
        ]:
            for _ in range(2):
                menu.locator(f'[data-sort="{column}"]').click()
                assert menu.is_visible()
                assert order() == expected
        page.keyboard.press("Escape")
        page.click("#regression-results-model")
        settle_menu(page)
        assert menu.locator('[data-sort="test"]').get_attribute("aria-pressed") == "true"
        page.keyboard.press("Escape")
        page.click("#regression-error-metrics")
        settle_menu(page)
        picker = page.locator("#regression-error-metrics-menu")
        picker.locator('.menu-item[data-id="wape"]').click()
        picker.locator('.menu-item[data-id="mae"]').click()
        page.keyboard.press("Escape")
        page.click("#regression-results-model")
        settle_menu(page)
        assert order() == ["ensemble", "ridge", "snaive", "worse", "missing"]
        assert ridge.locator(".forecast-results-val").inner_text() == "1235"
        assert ridge.locator(".forecast-results-test").inner_text() == "100"
        assert ridge.locator(".forecast-results-ratio").inner_text() == "0.50"
        assert order()[-2:] == ["worse", "missing"]
        page.keyboard.press("Escape")
        page.evaluate(
            "forecastTopics.get('regression').resultsView.run_tiles = forecastTopics.get('regression').resultsView.run_tiles.filter(tile => tile.family !== 'snaive')"
        )
        page.click("#regression-results-model")
        settle_menu(page)
        assert ridge.locator(".forecast-results-ratio").inner_text() == "–"
        page.keyboard.press("Escape")
        page.evaluate("forecastTopics.get('regression').selectedMetricIds = ['wape']")
        page.click("#regression-results-model")
        settle_menu(page)
        assert ridge.locator(".forecast-results-ratio").inner_text() == "0.90"
        assert ensemble.locator(".forecast-results-ratio").inner_text() == "–"
        page.set_viewport_size({"width": 375, "height": 900})
        page.keyboard.press("Escape")
        page.click("#regression-results-model")
        settle_menu(page)
        bounds = menu.bounding_box()
        assert bounds["x"] >= 0
        assert bounds["x"] + bounds["width"] <= 375
        page.keyboard.press("Escape")
        assert menu.count() == 0
