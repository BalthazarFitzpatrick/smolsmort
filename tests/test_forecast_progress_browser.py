"""candidate progress and saved loss history in headless chromium"""

from __future__ import annotations

import time
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("playwright")
pytest.importorskip("xgboost")
pytest.importorskip("duckdb")

from test_forecast_ui_browser import session, show_tab, switch_to_regression


def make_run(*, state="running", progress=None, curves=None):
    return {
        "id": "20261009-120000-progress",
        "kind": "search",
        "state": state,
        "budget": {"time_cap": 1200, "max_generations": 10, "plateau": 5},
        "progress": progress
        or {
            "generation": 0,
            "candidate": 1,
            "population": 10,
            "elapsed": 10,
            "flat_generations": 0,
        },
        "generations": [],
        "generation_curves": curves or [],
        "leaderboard": [],
    }


def mock_saved_run(page, run, polls=None):
    def read_run(route):
        if polls is not None:
            polls.append(time.monotonic())
        cursor = int(parse_qs(urlparse(route.request.url).query).get("since_generation", ["-1"])[0])
        payload = {
            **run,
            "generation_curves": [
                curve for curve in run["generation_curves"] if curve["generation"] > cursor
            ],
        }
        route.fulfill(json=payload)

    page.route("**/api/forecast-runs", lambda route: route.fulfill(json={"runs": [run]}))
    page.route("**/api/forecast-run?**", read_run)
    switch_to_regression(page)
    show_tab(page, "regression-search")
    page.wait_for_function(
        "document.getElementById('regression-run-title').innerText.includes('progress')"
    )


@pytest.mark.parametrize(
    ("progress", "budget", "expected"),
    [
        ({"elapsed": 600}, {}, 0.5),
        ({"generation": 2, "candidate": 5}, {}, 0.25),
        ({"flat_generations": 3}, {}, 0.6),
        ({"elapsed": 600, "generation": 8, "candidate": 5, "flat_generations": 3}, {}, 0.85),
        ({"elapsed": 2000}, {}, 1),
        ({"elapsed": -10, "candidate": -1}, {}, 0),
        ({"elapsed": 30}, {"time_cap": 60}, 0.5),
    ],
)
def test_search_progress_used_nearest_stop_condition(tmp_path, progress, budget, expected):
    with session(tmp_path) as (page, _):
        run = make_run()
        run["progress"].update(elapsed=0, candidate=0)
        run["progress"].update(progress)
        run["budget"].update(budget)
        value = page.evaluate("run => forecastSearchProgress(run)", run)
        assert value["fill"] == pytest.approx(expected)
        assert value["done"] is False


@pytest.mark.parametrize(
    ("state", "stopped", "fill", "done", "suffix"),
    [
        ("running", None, 0.4, False, ""),
        ("done", "plateau", 1, True, "done"),
        ("done", "cancelled", 0.4, False, "cancelled"),
        ("cancelled", "cancelled", 0.4, False, "cancelled"),
        ("failed", None, 0.4, False, "failed"),
    ],
)
def test_search_progress_label_and_terminal_state(tmp_path, state, stopped, fill, done, suffix):
    with session(tmp_path) as (page, _):
        run = make_run(
            state=state,
            progress={
                "generation": 3,
                "candidate": 7,
                "population": 10,
                "elapsed": 54,
                "flat_generations": 2,
            },
        )
        if stopped:
            run["stopped"] = stopped
        value = page.evaluate("run => forecastSearchProgress(run)", run)
        assert value["fill"] == pytest.approx(fill)
        assert value["done"] is done
        label = "generation 3 - candidate 7 of 10 - 54s of 20 min - 2 of 5 flat generations"
        assert value["label"] == label + (f" - {suffix}" if suffix else "")


def test_failed_search_kept_progress_after_an_older_generation_summary(tmp_path):
    with session(tmp_path) as (page, _):
        run = make_run(state="failed")
        run["elapsed"] = 10
        run["budget"]["time_cap"] = 120
        run["progress"]["elapsed"] = 90
        value = page.evaluate("run => forecastSearchProgress(run)", run)
        assert value["fill"] == pytest.approx(0.75)
        assert "90s of 2 min" in value["label"]
        assert value["label"].endswith(" - failed")
        assert value["done"] is False


def test_running_search_bar_grew_between_one_second_polls(tmp_path):
    with session(tmp_path) as (page, _):
        run = make_run()
        polls = []
        mock_saved_run(page, run, polls)
        fill = page.locator("#regression-run-bar-fill")
        rail_fill = page.locator(".forecast-run-row .bar-fill")
        page.wait_for_function(
            "document.getElementById('regression-run-bar-fill').getBoundingClientRect().width > 0"
        )
        before = fill.bounding_box()["width"]
        rail_before = rail_fill.bounding_box()["width"]
        run["progress"].update(candidate=8, elapsed=54)
        page.wait_for_function(
            "width => document.getElementById('regression-run-bar-fill').getBoundingClientRect().width > width",
            arg=before,
            timeout=1800,
        )
        assert rail_fill.bounding_box()["width"] > rail_before
        assert "candidate 8 of 10" in page.locator("#regression-run-progress-label").inner_text()
        assert len(polls) >= 2
        assert 0.75 <= polls[-1] - polls[-2] <= 1.6
        run.update(state="done", stopped="max_generations")
        page.wait_for_function(
            "document.getElementById('regression-run-title').innerText.endsWith('· done')"
        )
        assert fill.bounding_box()["width"] == pytest.approx(
            fill.locator("..").bounding_box()["width"], abs=1
        )
        assert "good" in fill.get_attribute("class").split()


def test_saved_loss_chart_filtered_metrics_and_collapsed_repeated_keys(tmp_path):
    with session(tmp_path) as (page, _):
        curves = [
            {"generation": 0, "key": "a", "metric": "mae", "validation": [4, 3, 2]},
            {"generation": 1, "key": "b", "metric": "rmse", "validation": [7, 6, 5]},
            {"generation": 2, "key": "c", "metric": "mae", "validation": [3, 2, 1]},
            {"generation": 3, "key": "d", "metric": "rmse", "validation": [6, 5, 4]},
            {"generation": 4, "key": "e", "metric": "mae", "validation": [2, 1, 0.5]},
            {"generation": 5, "key": "e", "metric": "mae", "validation": [2, 1, 0.5]},
        ]
        run = make_run(state="done", curves=curves)
        mock_saved_run(page, run)
        loss = page.locator("#regression-search-loss")
        chart = page.locator("#regression-loss-chart")
        page.wait_for_function(
            "document.querySelectorAll('#regression-loss-chart .chart-line').length === 3"
        )
        assert page.locator("#regression-loss-metric").inner_text() == "mae (lower is better)"
        assert loss.locator(".chart-legend-item:visible").all_text_contents() == [
            "current generation",
            "previous generations",
        ]
        assert page.locator("#regression-loss-hidden").inner_text() == (
            "2 earlier generations used a different metric and are hidden"
        )
        colors = page.evaluate(
            "() => { const css = getComputedStyle(document.documentElement);"
            " return ['--kingfisher', '--cream'].map(token => css.getPropertyValue(token).trim()); }"
        )
        assert loss.locator(".chart-legend-swatch:visible").evaluate_all(
            "nodes => nodes.map(node => getComputedStyle(node).borderTopColor)"
        ) == page.evaluate(
            "colors => colors.map(color => { const node = document.createElement('span');"
            " node.style.color = color; document.body.append(node);"
            " const resolved = getComputedStyle(node).color; node.remove(); return resolved; })",
            colors,
        )
        assert chart.locator(".chart-line").evaluate_all(
            "(lines, cream) => lines.every(line => line.getAttribute('stroke') === cream)",
            colors[1],
        )
        show_tab(page, "regression-data")
        show_tab(page, "regression-search")
        assert chart.locator(".chart-line").count() == 3
        assert loss.locator(".chart-legend-item:visible").count() == 2
