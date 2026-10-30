"""stacked results and dimension menus over synthetic saved runs"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from test_forecast_horizon_browser import horizon_session
from test_forecast_ui_browser import session, show_tab, switch_to_regression, with_error_metrics

pytest.importorskip("duckdb")


@pytest.mark.parametrize("width", [1280, 390])
def test_results_stacked_report_and_shared_metric_picker(tmp_path, width):
    with horizon_session(tmp_path, width=width) as (page, run_id):
        switch_to_regression(page)
        page.select_option("#topic-flavour", "timeseries")
        show_tab(page, "regression-search")
        rail = page.locator("#regression-new-search").locator("..")
        assert rail.locator("#regression-new-search").is_visible()
        assert rail.locator("#regression-run-sort .dropdown-head").inner_text() == "sort"
        assert rail.locator("#regression-error-metrics").is_visible()
        page.evaluate(
            "window.sharedMetricHead = document.getElementById('regression-error-metrics')"
        )
        page.locator(f'.forecast-run-rail [data-run-id="{run_id}"]').click()
        page.locator("#regression-run-open").click()
        page.locator("#regression-lead-errors tbody tr").first.wait_for()

        results = page.locator('.tab-panel[data-panel="regression-results"]')
        header = results.locator(".forecast-results-header")
        for control in (
            "results-run",
            "results-model",
            "error-metrics",
            "lead",
            "blend",
            "breakdown-head",
            "info",
            "export",
        ):
            assert header.locator(f"#regression-{control}").is_visible()
        assert page.evaluate(
            "window.sharedMetricHead === document.getElementById('regression-error-metrics')"
        )
        assert page.locator("#regression-error-metrics").count() == 1
        assert header.locator('input[type="checkbox"]').count() == 0
        assert results.locator("#regression-verdict, #regression-excluded-series").count() == 0
        assert page.locator(".forecast-metric-help").count() == 0

        chart = results.locator(".forecast-chart-panel")
        assert chart.locator("#regression-chart").is_visible()
        assert chart.locator("#regression-view-metrics").is_visible()
        filter_head = chart.locator("#regression-filter")
        title = filter_head.locator("..").locator(".field-label").first
        title_box = title.bounding_box()
        filter_box = filter_head.bounding_box()
        chart_box = chart.bounding_box()
        title_middle = title_box["y"] + title_box["height"] / 2
        filter_middle = filter_box["y"] + filter_box["height"] / 2
        assert abs(title_middle - filter_middle) < 1
        assert title_box["x"] + title_box["width"] <= filter_box["x"]
        assert filter_box["x"] + filter_box["width"] >= chart_box["x"] + chart_box["width"] - 16

        order = page.evaluate(
            """() => {
                const root = document.querySelector('.tab-panel[data-panel="regression-results"]');
                const nodes = [
                    '.forecast-results-header', '.forecast-chart-panel',
                    '#regression-family-comparison-wrap', '.forecast-results-models',
                    '#regression-results-table', '#regression-page-info'
                ].map(selector => root.querySelector(selector));
                return nodes.slice(1).every((node, index) =>
                    nodes[index].compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING);
            }"""
        )
        assert order
        family_label = page.locator("#regression-family-comparison-wrap").inner_text()
        assert "family test results" in family_label
        row_height = page.locator("#regression-family-comparison th").first.bounding_box()["height"]
        cells = page.locator("#regression-lead-errors th, #regression-lead-errors td")
        assert cells.evaluate_all(
            "(cells, height) => cells.every(cell => "
            "Math.abs(cell.getBoundingClientRect().height - height) < 1 "
            "&& getComputedStyle(cell).boxSizing === 'border-box')",
            row_height,
        )
        panels = results.locator(".forecast-results-models")
        importance = panels.locator("#regression-importance-wrap").bounding_box()
        errors = panels.locator("#regression-lead-errors-wrap").bounding_box()
        if width == 1280:
            assert abs(importance["y"] - errors["y"]) < 1
            assert importance["x"] + importance["width"] <= errors["x"]
        else:
            assert importance["y"] + importance["height"] <= errors["y"]
            assert abs(importance["x"] - errors["x"]) < 1
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        artifacts = Path(__file__).resolve().parents[1] / "outputs" / "results-layout"
        artifacts.mkdir(parents=True, exist_ok=True)
        page.screenshot(
            path=artifacts / f"results-{width}.png", full_page=True, animations="disabled"
        )
        panels.screenshot(path=artifacts / f"panels-{width}.png", animations="disabled")

        show_tab(page, "regression-search")
        assert rail.locator("#regression-error-metrics").is_visible()
        assert page.evaluate(
            "window.sharedMetricHead === document.getElementById('regression-error-metrics')"
        )


@pytest.mark.parametrize("dimensions", [("project",), ("project", "product")])
@pytest.mark.parametrize("width", [1280, 390])
def test_filter_menu_kept_dimension_selections_and_refreshed_chart_and_table(
    tmp_path, dimensions, width
):
    records = [
        {"project": "north", "product": "a", "prediction": 10},
        {"project": "north", "product": "b", "prediction": 20},
        {"project": "south", "product": "a", "prediction": 30},
        {"project": "south", "product": "b", "prediction": 40},
    ]
    spec = {
        "mode": "timeseries",
        "task": "regression",
        "columns": [
            {"name": "day", "role": "time"},
            *({"name": name, "role": "dimension"} for name in dimensions),
            {"name": "prediction", "role": "target"},
        ],
    }
    run_id = "20261030-120000-filter"

    def selected_records(route):
        params = parse_qs(urlparse(route.request.url).query)
        filters = json.loads(params.get("filters", ["{}"])[0])
        selected = [
            row for row in records if all(row[name] in values for name, values in filters.items())
        ]
        return params, selected

    def read_table(route):
        params, selected = selected_records(route)
        columns = [*dimensions, "prediction"]
        route.fulfill(
            json={
                "columns": columns,
                "rows": [[row[name] for name in columns] for row in selected],
                "page": int(params.get("page", ["0"])[0]),
                "total": len(selected),
            }
        )

    def read_view(route):
        _, selected = selected_records(route)
        total = sum(row["prediction"] for row in selected)
        route.fulfill(
            json=with_error_metrics(
                {
                    "mode": "timeseries",
                    "spec": spec,
                    "x": ["2026-01-05", "2026-01-12"],
                    "series": [{"id": "forecast", "label": "forecast", "values": [0, total]}],
                    "bands": [],
                    "markers": [],
                    "verdict": {"accuracy": {"ratio": 0.9}},
                }
            )
        )

    with session(tmp_path, width=width) as (page, _base):
        page.route(
            "**/api/forecast-runs",
            lambda route: route.fulfill(json={"runs": [{"id": run_id, "state": "done"}]}),
        )
        page.route(
            "**/api/forecast-run?**",
            lambda route: route.fulfill(json=with_error_metrics({"id": run_id, "spec": spec})),
        )
        page.route("**/api/forecast-table?**", read_table)
        page.route("**/api/forecast-view?**", read_view)
        switch_to_regression(page)
        page.select_option("#topic-flavour", "timeseries")
        show_tab(page, "regression-results")
        page.locator("#regression-results-run").click()
        page.locator(f'.menu-panel .menu-item[data-id="{run_id}"]').click()
        page.locator("#regression-results-table tr").nth(4).wait_for()
        page.locator("#regression-filter").click()
        menu = page.locator("#regression-filter-menu")
        assert menu.locator(".menu-list").count() == len(dimensions)
        assert menu.locator('input[type="checkbox"]').count() == 0
        box = menu.bounding_box()
        assert box["x"] >= 0 and box["x"] + box["width"] <= width
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        artifacts = Path(__file__).resolve().parents[1] / "outputs" / "results-layout"
        artifacts.mkdir(parents=True, exist_ok=True)
        page.screenshot(
            path=artifacts / f"filter-{len(dimensions)}-{width}.png",
            full_page=True,
            animations="disabled",
        )
        expected = {}

        def pick(dimension, value):
            section = menu.locator(".menu-list").nth(dimensions.index(dimension))
            item = section.locator(f'.menu-item[data-id="{value}"]')
            values = expected.setdefault(dimension, [])
            if value in values:
                values.remove(value)
            else:
                values.append(value)
            if not values:
                del expected[dimension]
            with (
                page.expect_response("**/api/forecast-view?**") as view_response,
                page.expect_response("**/api/forecast-table?**") as table_response,
            ):
                item.click()
            for response in (view_response.value, table_response.value):
                params = parse_qs(urlparse(response.url).query)
                assert params["id"] == [run_id]
                filters = json.loads(params.get("filters", ["{}"])[0])
                assert filters == expected
                if "/api/forecast-table?" in response.url:
                    assert params["page"] == ["0"]
            selected = [
                row
                for row in records
                if all(row[name] in values for name, values in expected.items())
            ]
            total = sum(row["prediction"] for row in selected)
            page.wait_for_function(
                "([rows, total]) => document.querySelectorAll('#regression-results-table tr').length === rows + 1"
                " && forecastTopics.get('regression').resultsView.series[0].values[1] === total",
                arg=[len(selected), total],
            )
            assert menu.is_visible()
            assert ("on" in item.get_attribute("class").split()) == (
                value in expected.get(dimension, [])
            )

        pick("project", "north")
        pick("project", "south")
        if "product" in dimensions:
            pick("product", "a")
        pick("project", "north")
        pick("project", "south")
        if "product" in dimensions:
            pick("product", "a")
        page.keyboard.press("Escape")
        assert not menu.is_visible()
