"""headless chromium against the REAL review server, over the synthetic row fixture: the
regression topic end to end (data -> search -> results). run with:
uv run --with playwright pytest -q tests/test_forecast_ui_browser.py"""

from __future__ import annotations

import csv
import json
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("playwright")
pytest.importorskip("xgboost")
pytest.importorskip("duckdb")

import review_world
from forecast_fixtures import write_panel, write_rows
from forecast_ui_audit import CONTROL_SIZES, visit_control_states
from playwright.sync_api import Page, sync_playwright

from smolsmort.forecast.tab import forecast_tab
from smolsmort.review.server import build_app, serve
from smolsmort.review_ui.server import STATIC

TOPIC = "regression"


def show_tab(page: Page, name: str) -> None:
    page.click(f'.nav-tab[data-tab="{name}"]')
    page.wait_for_timeout(250)


@contextmanager
def session(tmp_path: Path, *, width: int = 1280):
    """the real review server, its forecast root holding the synthetic rows fixture, and a
    headless page on it"""
    errors: list[str] = []
    review_world.register_world_backend()
    root = tmp_path / "forecast_root"
    root.mkdir()
    write_rows(root, n=600)
    panel = write_panel(root, weeks=60)
    with panel.path.open(newline="") as source:
        rows = list(csv.reader(source))
    with panel.path.open("w", newline="") as source:
        writer = csv.writer(source)
        writer.writerow([*rows[0], "other_day"])
        writer.writerows([*row, row[0]] for row in rows[1:])
    with pytest.MonkeyPatch.context() as patch:
        world = review_world.make_world(tmp_path / "review", patch)
        app = build_app(
            backend="world",
            ui_dir=STATIC,
            tabs=[forecast_tab()],
            pool=world.tiles,
            bases_file=world.root / "bases.json",
            crop_file=world.root / "crop.json",
        )
        app.state.set_bases({"forecast": str(root)})
        server = serve(app)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            time.sleep(0.2)
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={"width": width, "height": 900})
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                page.goto(base + "/")
                page.wait_for_timeout(400)
                yield page, base
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            review_world.unregister_world_backend()
    assert not errors, errors


def switch_to_regression(page: Page) -> None:
    page.click('#topic-switch .topic-tab:has-text("regression")')
    page.wait_for_timeout(150)
    page.select_option("#topic-flavour", "row")
    page.wait_for_timeout(150)


def set_role(page: Page, column: str, role: str) -> None:
    row = page.locator("tr", has=page.locator(f"td:text-is('{column}')"))
    row.locator(".dropdown-head").click()
    page.locator(f'.menu-panel .menu-item[data-id="{role}"]').click()


def uncheck_known(page: Page, column: str) -> None:
    row = page.locator("tr", has=page.locator(f"td:text-is('{column}')"))
    row.locator('[role="checkbox"][aria-checked="true"]').click()


def run_search(page: Page) -> None:
    show_tab(page, "regression-search")
    page.click(f"#{TOPIC}-new-search")
    for key, value in (("population", 6), ("plateau", 1), ("generations", 2)):
        head = page.locator(f'#{TOPIC}-budget [data-stepper="{key}"]')
        current = int(head.locator(".field-value").inner_text())
        delta = int(head.locator("[data-step]").first.get_attribute("data-step"))
        clicks = round((value - current) / delta)
        button = head.locator(f'[data-step="{delta if clicks >= 0 else -delta}"]')
        for _ in range(abs(clicks)):
            button.click()
    page.click(f"#{TOPIC}-search-start")
    page.wait_for_function(
        f'/· (done|failed)$/.test(document.getElementById("{TOPIC}-run-title").innerText)',
        timeout=180000,
    )
    status = page.inner_text(f"#{TOPIC}-run-title")
    assert status.endswith("· done"), status


def test_forecast_regression_row_flow(tmp_path):
    with session(tmp_path) as (page, base):
        switch_to_regression(page)
        show_tab(page, "regression-data")

        page.click(f"#{TOPIC}-data-source")
        page.wait_for_selector(".menu-panel .menu-item")
        page.click('.menu-panel .menu-item:has-text("rows.csv")')
        page.wait_for_function(
            f'document.getElementById("{TOPIC}-data-columns").children.length > 0'
        )

        set_role(page, "created", "anchor")
        set_role(page, "branch", "dimension")
        set_role(page, "size", "measure")
        set_role(page, "planned_offset", "measure")
        set_role(page, "leak_plus1", "measure")
        uncheck_known(page, "leak_plus1")
        set_role(page, "lead_weeks", "target")
        set_role(page, "line", "ignore")

        page.select_option("#topic-flavour", "series")
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{TOPIC}-data-status").innerText)'
        )
        page.select_option("#topic-flavour", "row")
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{TOPIC}-data-status").innerText)'
        )
        target = page.locator("tr", has=page.locator('td:text-is("lead_weeks")'))
        assert target.locator(".dropdown-head").inner_text() == "target"
        leak = page.locator("tr", has=page.locator('td:text-is("leak_plus1")'))
        assert leak.locator('[role="checkbox"]').get_attribute("aria-checked") == "false"

        page.click(f"#{TOPIC}-censor-toggle")
        page.click(f"#{TOPIC}-censor-unit .dropdown-head")
        page.locator('.menu-panel .menu-item[data-id="week"]').click()
        page.wait_for_function(
            f'document.getElementById("{TOPIC}-censor-anchor").innerText == "created"'
        )

        page.click(f"#{TOPIC}-prepare")
        page.wait_for_function(
            f'document.getElementById("{TOPIC}-data-status").innerText.startsWith("prepared")',
            timeout=30000,
        )

        page.reload()
        page.wait_for_selector("#forecast-dataset:not(.hidden)")
        assert page.locator("#forecast-dataset .field-value").last.inner_text() == "prepared"
        with page.expect_response(
            lambda response: (
                response.request.method == "POST" and response.url.endswith("/api/forecast-runs")
            )
        ) as started:
            run_search(page)
        assert started.value.json()["prepared"]["reused"] is True

        page.click(f"#{TOPIC}-run-open")
        page.wait_for_timeout(300)

        page.wait_for_selector(f"#{TOPIC}-verdict:not(.hidden)")
        verdict_text = page.inner_text(f"#{TOPIC}-verdict")
        assert verdict_text.strip()
        # a failing check names itself, whatever the fixture's actual quality turns out to be
        if "warn" in (page.get_attribute(f"#{TOPIC}-verdict", "class") or ""):
            assert ":" in verdict_text

        page.wait_for_function(
            f'document.querySelectorAll("#{TOPIC}-chart svg path.chart-band").length > 0',
            timeout=15000,
        )
        assert page.locator(f"#{TOPIC}-results-table tr").count() > 1

        export_href = page.get_attribute(f"#{TOPIC}-export", "href")
        assert export_href
        with urllib.request.urlopen(base + export_href, timeout=20) as response:
            csv_bytes = response.read()
        first_line = csv_bytes.decode().splitlines()[0]
        assert first_line.startswith("# verdict:"), first_line


def test_stored_filters_with_quotes_fill_their_fields_intact(tmp_path):
    """sql filters quote column names; stored text must come back verbatim, never break markup"""
    where = '"branch" <> \'x"y\''
    with session(tmp_path) as (page, base):
        page.evaluate(
            "([key, where]) => localStorage.setItem(key, JSON.stringify("
            "{state: {where: where, predictWhere: where}}))",
            [f"smolsmort:forecast:{TOPIC}", where],
        )
        page.reload()
        page.wait_for_timeout(400)
        switch_to_regression(page)
        show_tab(page, "regression-data")
        assert page.locator(f"#{TOPIC}-where").input_value() == where
        assert page.locator(f"#{TOPIC}-predict-where").input_value() == where


@pytest.mark.parametrize(
    ("topic", "flavour"),
    [("regression", "row"), ("regression", "series"), ("classification", "row")],
)
def test_data_controls_use_smortui_styles(tmp_path, topic, flavour):
    with session(tmp_path) as (page, base):
        page.click(f'#topic-switch .topic-tab:has-text("{topic}")')
        page.select_option("#topic-flavour", flavour)
        show_tab(page, f"{topic}-data")
        page.click(f"#{topic}-data-source")
        page.locator('.menu-panel .menu-item[data-id="rows.csv"]').click()
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{topic}-data-status").innerText)'
        )
        panel = page.locator(f'.tab-panel[data-panel="{topic}-data"]')
        assert panel.locator("select").count() == 0
        for field in panel.locator("input").all():
            assert "text-field" in field.get_attribute("class").split()
            assert field.evaluate(
                "el => getComputedStyle(el).backgroundColor === getComputedStyle(document.body).backgroundColor"
            )
            assert field.evaluate("el => getComputedStyle(el).borderWidth") == "2px"
            assert field.evaluate("el => getComputedStyle(el).borderStyle") == "solid"
            assert field.evaluate(
                "el => getComputedStyle(el).fontFamily === getComputedStyle(document.body).fontFamily"
            )
        for check in panel.locator('[role="checkbox"]').all():
            assert "toggle" in check.get_attribute("class").split()


@pytest.mark.parametrize("stored_step", [None, "auto"])
def test_unsaved_time_setup_shows_week_and_saved_plural(tmp_path, stored_step):
    with session(tmp_path) as (page, base):
        if stored_step is not None:
            page.evaluate(
                "step => localStorage.setItem('smolsmort:forecast:regression', JSON.stringify({state: {step}}))",
                stored_step,
            )
            page.reload()
        switch_to_regression(page)
        page.select_option("#topic-flavour", "series")
        show_tab(page, "regression-data")
        head = page.locator(f"#{TOPIC}-setup-time")
        readout = page.locator(f"#{TOPIC}-setup-readout")
        assert head.inner_text() == "choose date · week · +8"
        assert "8 weeks ahead" in readout.inner_text()
        assert "auto" not in readout.inner_text()
        head.click()
        menu = page.locator(".forecast-setup-menu")
        assert menu.locator(".dropdown-head").inner_text() == "week"
        menu.locator(".dropdown-head").click()
        menu.locator('.forecast-inline-list .menu-item:has-text("month")').click()
        menu.locator('input[type="number"]').fill("1")
        menu.locator('.menu-buttons [data-id="save"]').click()
        assert head.inner_text() == "choose date · month · +1"
        assert "1 month ahead" in readout.inner_text()


def test_scaffold_defaults_empty_and_preserves_saved_picks(tmp_path):
    with session(tmp_path) as (page, base):
        switch_to_regression(page)
        page.select_option("#topic-flavour", "series")
        show_tab(page, "regression-data")

        def load_source(name):
            page.click(f"#{TOPIC}-data-source")
            page.locator(f'.menu-panel .menu-item[data-id="{name}"]').click()
            page.wait_for_function(
                f'/^\\d+ columns/.test(document.getElementById("{TOPIC}-data-status").innerText)'
            )

        def assert_empty_scaffold():
            assert (
                page.locator(f"#{TOPIC}-setup-scaffold").inner_text() == "none (one total series)"
            )
            assert "total = 1 series" in page.locator(f"#{TOPIC}-setup-readout").inner_text()
            page.click(f"#{TOPIC}-setup-scaffold")
            menu = page.locator(".forecast-setup-menu")
            assert menu.locator(".menu-item.on").count() == 0
            assert menu.locator(".forecast-series-product").inner_text() == "1 series (the total)"
            menu.locator('.menu-buttons [data-id="close"]').click()

        assert_empty_scaffold()
        load_source("panel.csv")
        assert_empty_scaffold()
        page.select_option("#topic-flavour", "row")
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{TOPIC}-data-status").innerText)'
        )
        page.select_option("#topic-flavour", "series")
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{TOPIC}-data-status").innerText)'
        )
        assert_empty_scaffold()
        load_source("rows.csv")
        assert_empty_scaffold()
        load_source("panel.csv")
        assert_empty_scaffold()

        page.click(f"#{TOPIC}-setup-scaffold")
        page.locator('.forecast-setup-menu .menu-item[data-id="project"]').click()
        page.locator('.forecast-setup-menu .menu-buttons [data-id="save"]').click()
        page.select_option("#topic-flavour", "row")
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{TOPIC}-data-status").innerText)'
        )
        page.select_option("#topic-flavour", "series")
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{TOPIC}-data-status").innerText)'
        )
        load_source("panel.csv")
        page.click(f"#{TOPIC}-setup-scaffold")
        menu = page.locator(".forecast-setup-menu")
        assert menu.locator(".menu-item.on").get_attribute("data-id") == "project"
        menu.locator('.menu-item[data-id="project"]').click()
        menu.locator('.menu-buttons [data-id="save"]').click()
        page.reload()
        switch_to_regression(page)
        page.select_option("#topic-flavour", "series")
        show_tab(page, "regression-data")
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{TOPIC}-data-status").innerText)'
        )
        assert_empty_scaffold()


@pytest.mark.parametrize(
    ("topic", "metric"), [("regression", "wape"), ("classification", "log_loss")]
)
def test_search_charts_accumulate_validation_generations_and_reset(tmp_path, topic, metric):
    with session(tmp_path) as (page, base):
        snapshot = {"count": 0, "state": "running", "invalid": False}
        first_fit = {
            "key": "first",
            "metric": "rmse" if topic == "regression" else "logloss",
            "validation": [4.0, 3.0, 2.0, 1.0],
        }
        second_fit = {
            "key": "second",
            "metric": "mae" if topic == "regression" else "mlogloss",
            "validation": [0.9, 0.6],
            "bucket": [1, 4],
        }
        run_number = 0
        best_values = [0.8, 0.6, 0.6, 0.4]
        cursors = []

        def start_run(route):
            nonlocal run_number
            if route.request.method == "POST":
                run_number += 1
                route.fulfill(json={"run_id": f"progress-{run_number}"})
            else:
                route.fulfill(json={"runs": []})

        def poll_run(route):
            query = parse_qs(urlparse(route.request.url).query)
            cursor = int(query.get("since_generation", ["-1"])[0])
            cursors.append(cursor)
            generations = [
                {"generation": index, "best": best, "evaluated": index + 1, "elapsed": index}
                for index, best in enumerate(best_values[: snapshot["count"]])
            ]
            curves = [
                {"generation": index, **(first_fit if index == 0 else second_fit)}
                for index in range(snapshot["count"])
                if index > cursor
            ]
            if snapshot["invalid"]:
                curves.extend(
                    [
                        {"generation": "bad", "validation": [1, 2]},
                        {"generation": 500, "validation": "bad"},
                        {"generation": 501, "validation": [None, "bad"]},
                    ]
                )
            route.fulfill(
                json={
                    "state": snapshot["state"],
                    "generations": generations,
                    "verdict": {"trusted": True},
                    "leaderboard": [],
                    "generation_curves": curves,
                }
            )

        page.route("**/api/forecast-runs", start_run)
        page.route("**/api/forecast-run?**", poll_run)
        page.route(
            "**/api/forecast-prep",
            lambda route: route.fulfill(json={"summary": {}, "sql": "", "reused": False}),
        )
        page.click(f'#topic-switch .topic-tab:has-text("{topic}")')
        show_tab(page, f"{topic}-data")
        page.click(f"#{topic}-data-source")
        page.locator('.menu-panel .menu-item[data-id="rows.csv"]').click()
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{topic}-data-status").innerText)'
        )
        set_role(page, "lead_weeks", "target")
        page.click(f"#{topic}-prepare")
        page.wait_for_function(
            f'document.getElementById("{topic}-data-status").innerText.startsWith("prepared")'
        )
        show_tab(page, f"{topic}-search")
        progress = page.locator(f"#{topic}-search-progress")
        chart = page.locator(f"#{topic}-search-chart")
        loss = page.locator(f"#{topic}-search-loss")
        loss_chart = page.locator(f"#{topic}-loss-chart")
        if topic == "classification":
            page.add_style_tag(
                content=":root { --kingfisher: rgb(11, 122, 133); --cream: rgb(231, 211, 181); }"
            )
        colors = page.evaluate(
            "() => { const css = getComputedStyle(document.documentElement);"
            " return ['--cream', '--kingfisher'].map(token => css.getPropertyValue(token).trim()); }"
        )
        assert not progress.is_visible()
        assert not loss.is_visible()
        page.click(f"#{topic}-search-start")
        page.wait_for_function(
            f'document.getElementById("{topic}-search-status").innerText === "running"'
        )
        assert not progress.is_visible()
        assert not loss.is_visible()

        def wait_points(count):
            page.wait_for_function(
                "([id, count]) => {"
                " const chart = document.getElementById(id);"
                " const points = chart.querySelectorAll('.chart-point').length;"
                " const lines = [...chart.querySelectorAll('.chart-line')];"
                " return points + lines.reduce((n, line) => n + line.getAttribute('d').split(' L ').length, 0) === count;"
                "}",
                arg=[f"{topic}-search-chart", count],
            )

        for count in (1, 2):
            snapshot["count"] = count
            wait_points(count)
            assert progress.is_visible()
            assert (
                page.locator(f"#{topic}-search-metric").inner_text()
                == f"{metric} (lower is better)"
            )
            history = first_fit if count == 1 else second_fit
            page.wait_for_function(
                "([id, count]) => document.getElementById(id).querySelectorAll('.chart-line').length === count",
                arg=[f"{topic}-loss-chart", count],
            )
            assert loss.is_visible()
            label = f"{history['metric']} (lower is better)"
            if history.get("bucket"):
                label += " - horizon bucket 1-4"
            assert page.locator(f"#{topic}-loss-metric").inner_text() == label
            assert loss.locator(".field-label").last.inner_text() == "boosting round"
            assert "training loss" not in loss_chart.inner_text()
            assert loss_chart.locator(".chart-legend-item").all_text_contents() == [
                f"generation {index}" for index in range(count)
            ]
            assert loss_chart.locator(".chart-x-label").all_text_contents() == ["1", "2", "3", "4"]
            lines = loss_chart.locator(".chart-line")
            assert lines.last.get_attribute("stroke") == colors[1]
            assert lines.last.get_attribute("opacity") == "1"
            assert lines.first.get_attribute("d").count(" L ") == 3
            if count == 2:
                assert lines.first.get_attribute("stroke") == colors[0]
                assert float(lines.first.get_attribute("opacity")) == pytest.approx(1 - 1 / 60)
                assert lines.last.get_attribute("d").count(" L ") == 1
            assert loss_chart.locator(".chart-legend-swatch").evaluate_all(
                "swatches => swatches.every(swatch => getComputedStyle(swatch).opacity === '1')"
            )
            overlay = loss_chart.locator(".chart-overlay").bounding_box()
            page.mouse.move(overlay["x"] + 1, overlay["y"] + overlay["height"] / 2)
            assert loss_chart.locator(".chart-tooltip-value").all_text_contents() == [
                *(["4"] if count == 2 else []),
                str(history["validation"][0]).removesuffix(".0"),
            ]
            assert chart.locator(".chart-x-label").all_text_contents() == [
                str(i) for i in range(count)
            ]
            assert (
                f"generation {count - 1} - best"
                in page.locator(f"#{topic}-search-status").inner_text()
            )

        with page.expect_response("**/api/forecast-run?**"):
            page.wait_for_timeout(2100)
        wait_points(2)
        assert cursors[-1] == 1
        snapshot.update(count=3, invalid=True)
        wait_points(3)
        assert loss.is_visible()
        assert loss_chart.locator(".chart-line").count() == 3
        snapshot.update(count=65, state="done", invalid=False)
        wait_points(4)
        page.wait_for_function(
            f'document.getElementById("{topic}-run-title").innerText.endsWith("· done")'
        )
        assert chart.locator(".chart-x-label").all_text_contents() == ["0", "1", "2", "3"]
        progress.screenshot(path=tmp_path / f"{topic}-search-progress.png")
        assert loss.is_visible()
        assert loss_chart.locator(".chart-line").count() == 60
        assert loss_chart.locator(".chart-legend-item").first.inner_text() == "generation 5"
        assert loss_chart.locator(".chart-legend-item").last.inner_text() == "generation 64"
        assert loss_chart.locator(".chart-line").first.get_attribute("opacity") == "0.2"
        assert loss_chart.locator(".chart-line").last.get_attribute("stroke") == colors[1]
        assert loss_chart.locator(".chart-line").evaluate_all(
            "(lines, cream) => lines.slice(0, -1).every(line => line.getAttribute('stroke') === cream)",
            colors[0],
        )
        loss.screenshot(path=tmp_path / f"{topic}-generation-loss.png")
        assert page.evaluate(
            "topic => { const ids = ['search-progress', 'search-loss', 'leaderboard'];"
            " const nodes = ids.map(id => document.getElementById(`${topic}-${id}`));"
            " return nodes[0].nextElementSibling === nodes[1] && nodes[1].parentElement.nextElementSibling === nodes[2]; }",
            topic,
        )
        overlay = chart.locator(".chart-overlay").bounding_box()
        page.mouse.move(overlay["x"] + 1, overlay["y"] + overlay["height"] / 2)
        assert chart.locator(".chart-tooltip-value").inner_text() == "0.8"
        page.mouse.move(overlay["x"] + overlay["width"] - 1, overlay["y"] + overlay["height"] / 2)
        assert chart.locator(".chart-tooltip-value").inner_text() == "0.4"
        show_tab(page, f"{topic}-data")
        show_tab(page, f"{topic}-search")
        wait_points(4)
        assert progress.is_visible()
        assert loss.is_visible()
        snapshot.update(count=0, state="running")
        page.click(f"#{topic}-new-search")
        page.click(f"#{topic}-search-start")
        page.wait_for_function(
            f'document.getElementById("{topic}-search-status").innerText === "running"'
        )
        assert not progress.is_visible()
        assert chart.locator("svg").count() == 0
        assert not loss.is_visible()
        assert loss_chart.locator("svg").count() == 0
        snapshot.update(count=1, state="done")
        wait_points(1)
        assert chart.locator(".chart-x-label").all_text_contents() == ["0"]
        assert loss.is_visible()
        assert loss_chart.locator(".chart-line").count() == 1
        assert loss_chart.locator(".chart-line").first.get_attribute("stroke") == colors[1]
        assert cursors[-1] == -1
        assert run_number == 2


def test_generation_loss_ignores_response_from_previous_run(tmp_path):
    with session(tmp_path) as (page, base):
        run_number = 0
        delayed = []

        def start_run(route):
            nonlocal run_number
            if route.request.method == "POST":
                run_number += 1
                route.fulfill(json={"run_id": f"delayed-{run_number}"})
            else:
                route.fulfill(json={"runs": []})

        def poll_run(route):
            query = parse_qs(urlparse(route.request.url).query)
            if query["id"] == ["delayed-1"]:
                delayed.append(route)
            else:
                route.fulfill(json={"state": "running", "generations": [], "generation_curves": []})

        page.route("**/api/forecast-runs", start_run)
        page.route("**/api/forecast-run?**", poll_run)
        page.route(
            "**/api/forecast-prep",
            lambda route: route.fulfill(json={"summary": {}, "sql": "", "reused": False}),
        )
        switch_to_regression(page)
        show_tab(page, "regression-data")
        page.click(f"#{TOPIC}-data-source")
        page.locator('.menu-panel .menu-item[data-id="rows.csv"]').click()
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{TOPIC}-data-status").innerText)'
        )
        set_role(page, "lead_weeks", "target")
        page.click(f"#{TOPIC}-prepare")
        page.wait_for_function(
            f'document.getElementById("{TOPIC}-data-status").innerText.startsWith("prepared")'
        )
        show_tab(page, "regression-search")
        with page.expect_request("**/api/forecast-run?**"):
            page.click(f"#{TOPIC}-search-start")
        page.wait_for_timeout(100)
        assert len(delayed) == 1
        page.click(f"#{TOPIC}-new-search")
        page.click(f"#{TOPIC}-search-start")
        page.wait_for_function(
            f'document.getElementById("{TOPIC}-search-status").innerText === "running"'
        )
        delayed[0].fulfill(
            json={
                "state": "done",
                "generations": [{"generation": 99, "best": 0.1, "evaluated": 1, "elapsed": 1}],
                "generation_curves": [
                    {"generation": 99, "key": "stale", "metric": "rmse", "validation": [2, 1]}
                ],
            }
        )
        page.wait_for_timeout(100)
        assert page.locator(f"#{TOPIC}-search-status").inner_text() == "running"
        assert not page.locator(f"#{TOPIC}-search-loss").is_visible()
        assert page.locator(f"#{TOPIC}-loss-chart svg").count() == 0
        assert not page.locator(f"#{TOPIC}-search-progress").is_visible()


@pytest.mark.parametrize(
    ("topic", "metric"), [("regression", "wape"), ("classification", "log_loss")]
)
def test_search_leaderboard_explains_recorded_recipes(tmp_path, topic, metric):
    recorded = json.loads((Path(__file__).parent / "golden/forecast_leaderboard.json").read_text())[
        topic
    ]
    snapshot = {"rows": recorded}
    with session(tmp_path) as (page, base):
        page.route(
            "**/api/forecast-runs",
            lambda route: route.fulfill(
                json={"run_id": "leaderboard"} if route.request.method == "POST" else {"runs": []}
            ),
        )
        page.route(
            "**/api/forecast-run?**",
            lambda route: route.fulfill(
                json={
                    "state": "done",
                    "generations": [],
                    "generation_curves": [],
                    "verdict": {"trusted": True},
                    "leaderboard": snapshot["rows"],
                }
            ),
        )
        page.route(
            "**/api/forecast-prep",
            lambda route: route.fulfill(json={"summary": {}, "sql": "", "reused": False}),
        )
        page.click(f'#topic-switch .topic-tab:has-text("{topic}")')
        show_tab(page, f"{topic}-data")
        page.click(f"#{topic}-data-source")
        page.locator('.menu-panel .menu-item[data-id="rows.csv"]').click()
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{topic}-data-status").innerText)'
        )
        set_role(page, "lead_weeks", "target")
        page.click(f"#{topic}-prepare")
        page.wait_for_function(
            f'document.getElementById("{topic}-data-status").innerText.startsWith("prepared")'
        )
        show_tab(page, f"{topic}-search")
        page.click(f"#{topic}-search-start")
        table = page.locator(f"#{topic}-leaderboard-table")
        table.locator("tbody tr").first.wait_for()
        headers = [
            "rank",
            "generation found",
            f"validation error ({metric})",
            "gap to best",
            "mae",
            "objective",
            "tree depth",
            "learning rate (eta)",
            "features",
        ]
        assert table.locator("th").all_text_contents() == headers
        caption = table.locator("caption")
        assert caption.inner_text() == (
            f"best ten recipes the search tried, ranked by validation error ({metric}); lower is better; the winner is the first row"
        )
        assert caption.evaluate("node => getComputedStyle(node).whiteSpace") == "nowrap"
        rows = table.locator("tbody tr")
        assert rows.count() == len(recorded)
        assert table.locator("tbody tr.on").count() == 1
        assert "on" in rows.first.get_attribute("class").split()
        selected = rows.first.locator("td").first.evaluate(
            "node => getComputedStyle(node).backgroundColor"
        )
        regular = (
            rows.nth(1)
            .locator("td")
            .first.evaluate("node => getComputedStyle(node).backgroundColor")
        )
        assert selected != regular
        winner = recorded[0]
        cells = rows.first.locator("td").all_text_contents()
        assert cells[:5] == [
            "1",
            str(winner["generation"]),
            f"{winner['fitness']:.4g}",
            "+0.00%",
            f"{winner['metrics']['mae']:.4g}" if topic == "regression" else "-",
        ]
        assert cells[6:9] == [
            str(winner["genome"]["params"]["max_depth"]),
            "0.125",
            f"{winner['families']} features",
        ]
        rows.first.locator(".dropdown-head").click()
        details = page.locator(".menu-panel")
        assert details.locator(".popup-title").inner_text() == "recipe 1"
        if topic == "regression":
            assert cells[5] == "absolute error"
            assert details.locator(".forecast-recipe-field").all_text_contents()[:5] == [
                "lags0,1,3",
                "rolling8,13,52",
                "ewm0.1,0.3,0.5",
                "calendarweek,quarter",
                "dimsproject",
            ]
        else:
            assert cells[5] in {"multiclass log loss", "log loss"}
            assert details.locator(".forecast-recipe-field").all_text_contents()[:3] == [
                "calendarweek,quarter",
                "measuresplanned_offset,size",
                "log measuressize",
            ]
        assert "tree params" in details.inner_text()
        assert "metrics" in details.inner_text()
        labels = details.locator(".forecast-recipe-field .field-label").all_text_contents()
        assert all(
            label in labels
            for label in ["colsample", "eta", "lambda", "depth", "min child weight", "subsample"]
        )
        page.keyboard.press("Escape")
        gap = (recorded[1]["fitness"] - winner["fitness"]) / winner["fitness"] * 100
        assert rows.nth(1).locator("td").nth(3).inner_text() == f"+{gap:.2f}%"
        assert "[object Object]" not in table.inner_text()
        assert not any(entry["key"] in table.inner_text() for entry in recorded)
        page.locator(f"#{topic}-leaderboard").screenshot(path=tmp_path / f"{topic}-leaderboard.png")

        # missing fields stay readable; notes belong to the recipe menu
        snapshot["rows"] = (
            recorded + [{"note": "failed: too few usable rows"}] + [recorded[-1]] * 12
        )
        page.click(f"#{topic}-new-search")
        page.click(f"#{topic}-search-start")
        table.locator("tbody tr").nth(9).wait_for()
        assert table.locator("th").all_text_contents() == headers
        assert table.locator("tbody tr").count() == 10
        missing = table.locator("tbody tr").nth(len(recorded)).locator("td").all_text_contents()
        assert missing == [str(len(recorded) + 1)] + ["-"] * 7 + ["- features"]
        table.locator("tbody tr").nth(len(recorded)).locator(".dropdown-head").click()
        assert "failed: too few usable rows" in page.locator(".menu-panel").inner_text()
        page.keyboard.press("Escape")
        assert "[object Object]" not in table.inner_text()

        snapshot["rows"] = [{"fitness": 0, "metrics": {metric: 0}}, {"fitness": 1}, {}]
        page.click(f"#{topic}-new-search")
        page.click(f"#{topic}-search-start")
        page.wait_for_function(
            "id => document.getElementById(id).querySelectorAll('tbody tr').length === 3",
            arg=f"{topic}-leaderboard-table",
        )
        assert rows.first.locator("td").nth(3).inner_text() == "+0.00%"
        assert rows.nth(1).locator("td").nth(3).inner_text() == "-"
        assert table.locator("th").all_text_contents() == headers
        snapshot["rows"] = [{"fitness": 1}, {"fitness": 1.0002}]
        page.click(f"#{topic}-new-search")
        page.click(f"#{topic}-search-start")
        table.locator('td:text-is("+0.02%")').wait_for()
        snapshot["rows"] = []
        page.click(f"#{topic}-new-search")
        page.click(f"#{topic}-search-start")
        page.wait_for_function(
            "id => document.getElementById(id).classList.contains('hidden')",
            arg=f"{topic}-leaderboard",
        )


def test_search_runs_reattach_sort_and_keep_live_detail(tmp_path):
    recipes = json.loads((Path(__file__).parent / "golden/forecast_leaderboard.json").read_text())[
        "regression"
    ]
    best = "20261001-120000-best01"
    recent = "20261003-120000-later1"
    missing = "20261004-120000-noerr1"
    live = "20261005-120000-live01"
    snapshot = {"count": 1, "state": "running"}
    board_cursors = []
    with session(tmp_path) as (page, base):
        summaries = [
            {
                "id": best,
                "kind": "search",
                "state": "done",
                "stopped": "plateau",
                "test_error": 0.1,
            },
            {
                "id": recent,
                "kind": "warm start",
                "state": "done",
                "stopped": "time cap",
                "test_error": 0.3,
            },
            {
                "id": missing,
                "kind": "refit",
                "state": "failed",
                "reason": "too few usable rows",
                "test_error": None,
            },
        ]

        def list_runs(route):
            route.fulfill(
                json={
                    "runs": [
                        *summaries,
                        {
                            "id": live,
                            "kind": "search",
                            "state": snapshot["state"],
                            "generation": snapshot["count"] - 1,
                            "best_error": 0.2 / snapshot["count"],
                            "best_wape": 0.2 / snapshot["count"],
                            "stopped": "plateau" if snapshot["state"] == "done" else None,
                            "test_error": 0.05 if snapshot["state"] == "done" else None,
                        },
                    ]
                }
            )

        def read_run(route):
            query = parse_qs(urlparse(route.request.url).query)
            run_id = query["id"][0]
            count = snapshot["count"] if run_id == live else 2
            cursor = int(query.get("since_generation", ["-1"])[0])
            revision = count - 1
            board_cursor = int(query.get("since_leaderboard", ["-1"])[0])
            if run_id == live:
                board_cursors.append(board_cursor)
            run = next((row for row in summaries if row["id"] == run_id), {})
            state = snapshot["state"] if run_id == live else run["state"]
            board = [dict(row) for row in recipes]
            if run_id == live and count > 1:
                board.reverse()
                board[0]["fitness"] = 0.09
            board[0]["note"] = "recorded best recipe"
            board[0]["genome"] = {
                **board[0]["genome"],
                "params": {
                    "colsample_bytree": 0.75,
                    "eta": 0.125,
                    "lambda": 2,
                    "max_depth": 4,
                    "min_child_weight": 5,
                    "subsample": 0.8,
                },
            }
            response = {
                **run,
                "state": state,
                "kind": run.get("kind", "search"),
                "spec": {"task": "regression"},
                "budget": {"plateau": 2},
                "stopped": "plateau" if state == "done" and run_id == live else run.get("stopped"),
                "verdict": {"trusted": True} if state == "done" else None,
                "generations": [
                    {
                        "generation": index,
                        "best": 0.2 / (index + 1),
                        "evaluated": 4 + index,
                        "elapsed": index + 1,
                    }
                    for index in range(count)
                ],
                "generation_curves": [
                    {
                        "generation": index,
                        "metric": "rmse",
                        "bucket": [1, 4],
                        "validation": [2 / (index + 1), 1 / (index + 1)],
                    }
                    for index in range(count)
                    if index > cursor
                ],
                "leaderboard_generation": revision,
            }
            if board_cursor < revision:
                response["leaderboard"] = board
            route.fulfill(json=response)

        page.route("**/api/forecast-runs", list_runs)
        page.route("**/api/forecast-run?**", read_run)
        switch_to_regression(page)
        show_tab(page, "regression-search")
        rail = page.locator(f"#{TOPIC}-runs-list")
        running = page.locator(f"#{TOPIC}-running-list")
        table = page.locator(f"#{TOPIC}-leaderboard-table")
        table.locator("tbody tr").first.wait_for()
        assert running.locator(".forecast-run-row").count() == 1
        assert rail.locator(".forecast-run-row").evaluate_all(
            "rows => rows.map(row => row.dataset.runId)"
        ) == [missing, recent, best]
        assert running.locator(".forecast-run-row > span").all_text_contents() == [
            "live01",
            "running",
            "0.2",
        ]
        for row in page.locator(".forecast-run-row").all():
            assert row.locator("span").count() == 3
            assert row.evaluate("node => getComputedStyle(node).flexDirection") == "row"
        assert running.locator(".forecast-run-row.on .name").evaluate(
            "node => getComputedStyle(node).color === getComputedStyle(node.parentElement).color"
        )
        assert page.locator(f"#{TOPIC}-run-title").inner_text().endswith("search · running")
        assert table.locator("tbody tr").count() == len(recipes)
        assert not page.locator(f"#{TOPIC}-run-verdict").is_visible()
        assert page.locator('.tab-panel[data-panel="regression-search"] select').count() == 0
        page.locator(f"#{TOPIC}-run-sort .dropdown-head").click()
        page.locator('.menu-panel .menu-item[data-id="best performance"]').click()
        assert rail.locator(".forecast-run-row").evaluate_all(
            "rows => rows.map(row => row.dataset.runId)"
        ) == [best, recent, missing]

        rail.locator(f'[data-run-id="{recent}"]').click()
        page.wait_for_function(
            "id => document.getElementById(id).innerText.includes('warm start')",
            arg=f"{TOPIC}-run-title",
        )
        assert page.locator(f"#{TOPIC}-run-stop").inner_text() == "time cap"
        page.wait_for_function(
            "id => document.getElementById(id).querySelectorAll('.chart-line').length === 2",
            arg=f"{TOPIC}-loss-chart",
        )
        assert page.locator(f"#{TOPIC}-loss-chart .chart-line").count() == 2
        snapshot["count"] = 2
        page.wait_for_function(
            "id => document.getElementById(id).querySelector('.forecast-run-wape').innerText === '0.1'",
            arg=f"{TOPIC}-running-list",
        )
        assert "warm start" in page.locator(f"#{TOPIC}-run-title").inner_text()
        running.locator(f'[data-run-id="{live}"]').click()
        table.locator('td:text-is("0.09")').wait_for()
        assert page.locator(f"#{TOPIC}-loss-chart .chart-line").count() == 2
        assert table.locator("tbody tr.on td").nth(1).inner_text() == str(recipes[-1]["generation"])
        assert 0 in board_cursors and 1 in board_cursors
        table.locator("tbody tr").first.locator(".dropdown-head").click()
        menu = page.locator(".menu-panel")
        fields = menu.locator(".forecast-recipe-field").all_text_contents()
        assert all(
            field in fields
            for field in [
                "colsample0.75",
                "eta0.125",
                "lambda2",
                "depth4",
                "min child weight5",
                "subsample0.8",
            ]
        )
        assert "recorded best recipe" in menu.inner_text()
        assert "loss function" in menu.inner_text() and "wape" in menu.inner_text()
        assert "[object Object]" not in menu.inner_text()
        page.keyboard.press("Escape")

        snapshot["state"] = "done"
        page.locator(f"#{TOPIC}-running-section").wait_for(state="hidden")
        assert running.locator(".forecast-run-row").count() == 0
        assert rail.locator(".forecast-run-row").evaluate_all(
            "rows => rows.map(row => row.dataset.runId)"
        ) == [live, best, recent, missing]
        assert (
            page.locator(f"#{TOPIC}-run-stop").inner_text()
            == "no gain larger than the noise for 2 generations in a row"
        )
        assert page.locator(f"#{TOPIC}-run-verdict").inner_text() == "trusted"
        assert not page.locator(f"#{TOPIC}-search-status").is_visible()
        assert (
            "done - trusted"
            not in page.locator('.tab-panel[data-panel="regression-search"]').inner_text()
        )
        assert page.locator(f"#{TOPIC}-search-progress").is_visible()
        assert page.locator(f"#{TOPIC}-search-loss").is_visible()
        page.locator(f"#{TOPIC}-run-sort .dropdown-head").click()
        page.locator('.menu-panel .menu-item[data-id="date, newest first"]').click()
        assert rail.locator(".forecast-run-row").evaluate_all(
            "rows => rows.map(row => row.dataset.runId)"
        ) == [live, missing, recent, best]
        rail.locator(f'[data-run-id="{missing}"]').click()
        page.wait_for_function(
            "id => document.getElementById(id).innerText === 'failed: too few usable rows'",
            arg=f"{TOPIC}-run-stop",
        )
        assert "refit" in page.locator(f"#{TOPIC}-run-title").inner_text()
        rail.locator(f'[data-run-id="{best}"]').click()
        page.wait_for_function(
            "id => document.getElementById(id).innerText.includes('best01')",
            arg=f"{TOPIC}-run-title",
        )
        page.wait_for_function(
            "id => document.getElementById(id).querySelectorAll('.chart-line').length === 2",
            arg=f"{TOPIC}-loss-chart",
        )
        assert page.locator(f"#{TOPIC}-loss-chart .chart-line").count() == 2
        page.locator(f"#{TOPIC}-leaderboard .forecast-leaderboard-wrap").evaluate(
            "node => { node.scrollLeft = 0; }"
        )
        page.locator('.tab-panel[data-panel="regression-search"]').screenshot(
            path=tmp_path / "runs-first.png"
        )
        page.click(f"#{TOPIC}-new-search")
        assert page.locator(f"#{TOPIC}-new-search-pane").is_visible()
        assert not page.locator(f"#{TOPIC}-search-loss").is_visible()


@pytest.mark.parametrize("width", [1440, 1100])
def test_forecast_controls_follow_the_shared_row_height(tmp_path, width):
    recorded = json.loads((Path(__file__).parent / "golden/forecast_leaderboard.json").read_text())[
        "regression"
    ]
    old = "20261001-120000-older1"
    cancelled = "20261002-120000-stop01"
    refit = "20261003-120000-refit1"
    spec = {
        "mode": "series",
        "task": "regression",
        "step": "week",
        "horizon": 3,
        "columns": [
            {"name": "day", "role": "time"},
            {"name": "project", "role": "dimension"},
            {"name": "units", "role": "target", "aggregation": "sum"},
        ],
    }
    observations = []
    states = []
    view_mode = ["series"]
    with session(tmp_path, width=width) as (page, base):
        for run_id in (old, cancelled, refit):
            folder = tmp_path / "forecast_root/.forecast-runs" / run_id
            folder.mkdir(parents=True)
            request = {
                "spec": spec,
                "prepared": str(tmp_path),
                "budget": {"plateau": 2},
                "recipe": recorded[0]["genome"] if run_id == refit else None,
            }
            (folder / "request.json").write_text(json.dumps(request))
            (folder / "status.json").write_text(
                json.dumps(
                    {
                        "state": "done",
                        "verdict": {"trusted": True},
                        "stopped": "cancelled"
                        if run_id == cancelled
                        else "refit"
                        if run_id == refit
                        else "plateau",
                    }
                )
            )
            events = [
                {"event": "generation", "generation": 0, "best": 0.3, "elapsed": 1},
                {"event": "generation", "generation": 1, "best": 0.2, "elapsed": 2},
            ]
            (folder / "events.jsonl").write_text(
                "" if run_id == refit else "".join(json.dumps(event) + "\n" for event in events)
            )
            (folder / "leaderboard.json").write_text(json.dumps(recorded))
            curves = [
                {
                    "generation": index,
                    "metric": "rmse",
                    "bucket": [1, 3],
                    "validation": [2 / (index + 1), 1 / (index + 1)],
                }
                for index in range(2)
            ]
            (folder / "generation_curves.jsonl").write_text(
                "".join(json.dumps(curve) + "\n" for curve in curves)
            )
        page.route(
            "**/api/forecast-view?**",
            lambda route: route.fulfill(
                json={
                    "mode": view_mode[0],
                    "x": ["2026-01-01", "2026-01-08"],
                    "series": [{"id": "actual", "label": "actual", "values": [1, 2]}],
                    "bands": [],
                    "markers": [],
                    "verdict": {"trusted": True},
                }
            ),
        )
        page.route(
            "**/api/forecast-table?**",
            lambda route: route.fulfill(
                json={
                    "columns": ["day", "project", "units"],
                    "rows": [["2026-01-01", "one", 2]],
                    "page": 0,
                    "total": 1,
                }
            ),
        )

        def capture(state):
            if state.endswith(("breakdown-menu", "group-menu")):
                assert page.locator(".forecast-field-menu .menu-item").count() > 0
            measured = page.evaluate(CONTROL_SIZES)
            controls = measured["controls"]
            assert controls, state
            for control in controls:
                assert control["height"] == pytest.approx(measured["target"], abs=1e-6), (
                    state,
                    control,
                )
                assert control["size"] == page.evaluate(
                    "getComputedStyle(document.body).fontSize"
                ), (state, control)
                if control["kind"] == "data-field":
                    assert control["width"] == pytest.approx(measured["fieldWidth"]), (
                        state,
                        control,
                    )
            for kind in {control["kind"] for control in controls} - {None}:
                widths = {
                    round(control["width"], 4) for control in controls if control["kind"] == kind
                }
                assert len(widths) == 1, (state, kind, widths)
            for text in measured["texts"]:
                assert text["lines"] <= 1, (state, text)
                assert text["textHeight"] <= text["lineHeight"] + 1, (state, text)
                if text["lineBoxHeight"] is not None:
                    assert text["lineBoxHeight"] <= text["lineHeight"] + 1, (state, text)
            observations.extend({"state": state, **control} for control in controls)
            states.append(state)
            page.screenshot(path=tmp_path / f"audit-{state}.png", full_page=True)

        visit_control_states(
            page,
            source="panel.csv",
            targets=["units", "orders"],
            run_id=old,
            capture=capture,
            set_results_mode=lambda mode: view_mode.__setitem__(0, mode),
        )
        assert len(observations) > 300
        assert {
            "setup-predict",
            "setup-aggregate",
            "setup-time",
            "setup-scaffold",
            "settings",
            "search-regression-new",
            "search-classification-finished",
            "recipe-regression",
            "results-classification",
            "data-regression-row",
            "data-regression-series",
        } <= set(states)
        (tmp_path / "control-sizes.json").write_text(json.dumps(observations, indent=2))
        show_tab(page, "regression-search")
        rows = page.locator(f"#{TOPIC}-runs-list .forecast-run-row")
        assert rows.count() == 3
        assert page.locator(
            f'#{TOPIC}-runs-list [data-run-id="{old}"] > span'
        ).all_text_contents() == ["older1", "done", "0.2"]
        assert page.locator(
            f'#{TOPIC}-runs-list [data-run-id="{cancelled}"] > span'
        ).all_text_contents() == ["stop01", "cancelled", "0.2"]
        assert page.locator(
            f'#{TOPIC}-runs-list [data-run-id="{refit}"] > span'
        ).all_text_contents() == ["refit1", "done", "-"]
        columns = []
        for row in rows.all():
            spans = row.locator("span")
            assert spans.count() == 3
            positions = [span.bounding_box() for span in spans.all()]
            assert len({position["y"] for position in positions}) == 1
            columns.append(positions[-1])
            assert spans.last.evaluate("node => getComputedStyle(node).textAlign") == "right"
        assert len({(column["x"], column["width"]) for column in columns}) == 1
        assert not page.locator(f"#{TOPIC}-search-status").is_visible()
        settings = page.locator("#open-settings").bounding_box()
        nav = page.locator(".nav-tab.active").bounding_box()
        assert settings["y"] + settings["height"] / 2 == nav["y"] + nav["height"] / 2


def test_series_setup_saves_discards_and_prepares(tmp_path):
    with session(tmp_path) as (page, base):
        switch_to_regression(page)
        page.select_option("#topic-flavour", "series")
        show_tab(page, "regression-data")
        page.click(f"#{TOPIC}-data-source")
        page.locator('.menu-panel .menu-item[data-id="panel.csv"]').click()
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{TOPIC}-data-status").innerText)'
        )
        assert page.locator('.tab-panel[data-panel="regression-data"] select').count() == 0
        heads = page.locator(f"#{TOPIC}-setup-heads .dropdown-head")
        assert heads.count() == 4
        for head in heads.all():
            assert head.bounding_box()["width"] == 260
        saved_readout = page.locator(f"#{TOPIC}-setup-readout").inner_text()

        def open_picker(key):
            page.click(f"#{TOPIC}-setup-{key}")
            menu = page.locator(".forecast-setup-menu")
            menu.screenshot(path=tmp_path / f"picker-{key}.png")
            for header in menu.locator(".field-label, .popup-title").all():
                assert header.evaluate("el => getComputedStyle(el).whiteSpace") == "nowrap"
            return menu

        def pick(name):
            page.locator(f'.forecast-setup-menu .menu-item[data-id="{name}"]').click()

        def finish(action):
            page.locator(f'.forecast-setup-menu .menu-buttons [data-id="{action}"]').click()

        menu = open_picker("predict")
        wide_width = menu.bounding_box()["width"]
        assert menu.locator(".menu-list").bounding_box()["width"] == 260
        for field in menu.locator(".menu-list .menu-item").all():
            assert field.bounding_box()["width"] == 260
        pick("units")
        finish("close")
        assert page.locator(f"#{TOPIC}-setup-readout").inner_text() == saved_readout
        open_picker("predict")
        assert (
            page.locator('.menu-item[data-id="units"]').get_attribute("class") == "toggle menu-item"
        )
        pick("units")
        pick("orders")
        finish("save")

        menu = open_picker("aggregate")
        assert menu.locator(".forecast-agg-row").count() == 2
        for field in menu.locator(".forecast-field").all():
            assert field.bounding_box()["width"] == 260
        row = menu.locator(".forecast-agg-row", has=page.locator('.name:text-is("units")'))
        row.locator(".dropdown-head").click()
        row.locator('.menu-item:has-text("median")').click()
        finish("close")
        assert "median" not in page.locator(f"#{TOPIC}-setup-aggregate").inner_text()
        menu = open_picker("aggregate")
        row = menu.locator(".forecast-agg-row", has=page.locator('.name:text-is("units")'))
        row.locator(".dropdown-head").click()
        row.locator('.menu-item:has-text("median")').click()
        finish("save")
        open_picker("predict")
        pick("units")
        pick("units")
        finish("save")
        assert "median(units)" in page.locator(f"#{TOPIC}-setup-aggregate").inner_text()

        menu = open_picker("time")
        assert menu.bounding_box()["width"] < wide_width
        assert menu.locator(".menu-list").bounding_box()["width"] == 260
        pick("other_day")
        assert menu.locator(".menu-list .menu-item.on").count() == 1
        menu.locator(".dropdown-head").click()
        menu.locator('.forecast-inline-list .menu-item:has-text("month")').click()
        menu.locator('input[type="number"]').fill("3")
        finish("close")
        assert "+8" in page.locator(f"#{TOPIC}-setup-time").inner_text()
        assert "other_day" not in page.locator(f"#{TOPIC}-setup-time").inner_text()
        menu = open_picker("time")
        pick("other_day")
        pick("other_day")
        assert menu.locator(".menu-list .menu-item.on").count() == 1
        menu.locator(".dropdown-head").click()
        menu.locator('.forecast-inline-list .menu-item:has-text("week")').click()
        menu.locator('input[type="number"]').fill("3")
        finish("save")

        menu = open_picker("scaffold")
        assert menu.locator('.menu-item[data-id="project"] .stats').inner_text() == "[3]"
        assert menu.locator('.menu-item[data-id="product"] .stats').inner_text() == "[4]"
        assert menu.locator(".menu-item.on").count() == 0
        assert menu.locator(".forecast-series-product").inner_text() == "1 series (the total)"
        pick("project")
        pick("product")
        assert menu.locator(".forecast-series-product").inner_text() == "3 x 4 = 12 series"
        pick("product")
        assert menu.locator(".forecast-series-product").inner_text() == "3 = 3 series"
        finish("close")
        menu = open_picker("scaffold")
        assert menu.locator(".forecast-series-product").inner_text() == "1 series (the total)"
        pick("project")
        finish("save")

        page.reload()
        switch_to_regression(page)
        page.select_option("#topic-flavour", "series")
        show_tab(page, "regression-data")
        page.wait_for_function(
            f'/^\\d+ columns/.test(document.getElementById("{TOPIC}-data-status").innerText)'
        )
        assert "median(units)" in page.locator(f"#{TOPIC}-setup-readout").inner_text()
        with page.expect_request("**/api/forecast-prep") as request:
            page.click(f"#{TOPIC}-prepare")
        spec = request.value.post_data_json["spec"]
        columns = {col["name"]: col for col in spec["columns"]}
        assert columns["other_day"]["role"] == "time"
        assert columns["day"]["role"] == "ignore"
        assert columns["units"]["role"] == columns["orders"]["role"] == "target"
        assert columns["units"]["aggregation"] == "median"
        assert columns["orders"]["aggregation"] == "sum"
        assert columns["project"]["role"] == "dimension"
        assert columns["product"]["role"] == "ignore"
        assert (spec["step"], spec["horizon"]) == ("week", 3)
        page.wait_for_function(
            f'document.getElementById("{TOPIC}-data-status").innerText.startsWith("prepared")',
            timeout=30000,
        )
        assert 'MEDIAN("units")' in page.locator(f"#{TOPIC}-sql-text").text_content()
        with page.expect_request(
            lambda request: request.method == "POST" and request.url.endswith("/api/forecast-runs")
        ) as request:
            run_search(page)
        assert request.value.post_data_json["spec"] == spec
        page.click(f"#{TOPIC}-run-open")
        page.wait_for_selector(f"#{TOPIC}-verdict:not(.hidden)")
        page.locator(f"#{TOPIC}-results-table tr").nth(1).wait_for()
        assert page.locator(f"#{TOPIC}-results-table tr").count() > 1


def mock_prepared_searches(page):
    requests = []
    preparations = []
    runs = {}
    prepared = {
        "summary": {"source_rows": 600, "source_columns": 7, "encoding": "utf-8"},
        "sql": "select * from source",
        "reused": True,
    }

    def prepare(route):
        preparations.append(route.request.post_data_json["spec"])
        route.fulfill(json=prepared)

    def list_or_start(route):
        if route.request.method == "GET":
            route.fulfill(json={"runs": list(runs.values())})
            return
        start(route)

    def start(route):
        payload = route.request.post_data_json
        requests.append((urlparse(route.request.url).path, payload))
        run_id = f"20261008-120000-test{len(requests):02d}"
        kind = (
            "refit" if "run_id" in payload else "warm start" if "warm_from" in payload else "search"
        )
        runs[run_id] = {
            "id": run_id,
            "state": "done",
            "kind": kind,
            "spec": payload["spec"],
            "stopped": "generation cap",
            "generations": [],
            "generation_curves": [],
            "leaderboard": [],
            "verdict": {"trusted": True},
        }
        route.fulfill(json={"run_id": run_id, "prepared": prepared})

    def read(route):
        run_id = parse_qs(urlparse(route.request.url).query)["id"][0]
        route.fulfill(json=runs[run_id])

    page.route("**/api/forecast-prep", prepare)
    page.route("**/api/forecast-runs", list_or_start)
    page.route("**/api/forecast-refit", start)
    page.route("**/api/forecast-run?**", read)
    return requests, preparations


def choose_forecast_source(page, source, topic="regression"):
    show_tab(page, f"{topic}-data")
    page.click(f"#{topic}-data-source")
    page.locator(f'.menu-panel .menu-item[data-id="{source}"]').click()
    page.wait_for_function(
        "id => /^\\d+ columns/.test(document.getElementById(id).innerText)",
        arg=f"{topic}-data-status",
    )


def assert_dataset_state(page, state, *, source="rows.csv", topic="regression"):
    readout = page.locator("#forecast-dataset")
    assert readout.is_visible()
    assert readout.locator(".field-value").last.inner_text() == state
    if source:
        assert readout.locator(".forecast-dataset-file").inner_text() == source
        assert readout.locator(".forecast-dataset-file").get_attribute("title") == source
    for tab in ("data", "search", "results"):
        show_tab(page, f"{topic}-{tab}")
        assert readout.is_visible()
        assert readout.locator(".field-value").last.inner_text() == state


def choose_series_target(page, name):
    show_tab(page, "regression-data")
    page.click("#regression-setup-predict")
    option = page.locator(f'.forecast-setup-menu .menu-item[data-id="{name}"]')
    if "on" not in option.get_attribute("class").split():
        option.click()
    page.locator('.forecast-setup-menu .menu-buttons [data-id="save"]').click()


def test_forecast_reloaded_setup_started_with_current_picks(tmp_path):
    with session(tmp_path) as (page, _):
        requests, preparations = mock_prepared_searches(page)
        switch_to_regression(page)
        assert_dataset_state(page, "no dataset selected - pick one on the data tab", source=None)
        choose_forecast_source(page, "rows.csv")
        set_role(page, "lead_weeks", "target")
        assert_dataset_state(page, "not prepared")
        show_tab(page, "regression-data")
        page.click("#regression-prepare")
        page.wait_for_function(
            "document.getElementById('forecast-dataset').innerText.includes('600 rows')"
        )
        assert_dataset_state(page, "prepared")
        assert "7 columns" in page.locator("#forecast-dataset").inner_text()
        assert "utf-8" in page.locator("#forecast-dataset").inner_text()
        initial_spec = preparations[0]

        page.reload()
        page.wait_for_selector("#forecast-dataset:not(.hidden)")
        assert_dataset_state(page, "prepared")
        show_tab(page, "regression-search")
        page.click("#regression-new-search")
        page.click("#regression-search-start")
        page.wait_for_function(
            "document.getElementById('regression-run-title').innerText.endsWith('· done')"
        )
        assert requests[0][1]["spec"] == initial_spec
        assert len(preparations) == 1

        for index, action in enumerate(("start", "warm", "refit")):
            show_tab(page, "regression-data")
            page.locator("#regression-where").fill(f"size > {index}")
            page.locator("#regression-where").blur()
            assert_dataset_state(page, "changed since prepare")
            show_tab(page, "regression-search")
            if action == "start":
                page.click("#regression-new-search")
                page.click("#regression-search-start")
            else:
                page.click(f"#regression-run-{action}")
            page.wait_for_function(
                "shortId => { const title = document.getElementById('regression-run-title').innerText; "
                "return title.includes(shortId) && title.endsWith('· done'); }",
                arg=f"test{index + 2:02d}",
            )
            assert requests[-1][1]["spec"]["where"] == f"size > {index}"
            assert_dataset_state(page, "prepared")
        assert requests[-2][1]["warm_from"]
        assert requests[-1][0] == "/api/forecast-refit"
        assert requests[-1][1]["run_id"]
        assert len(preparations) == 1
        choose_forecast_source(page, "panel.csv")
        assert_dataset_state(page, "changed since prepare", source="panel.csv")
        assert "600 rows" not in page.locator("#forecast-dataset").inner_text()


def test_forecast_prepared_setup_stayed_per_topic_and_flavour(tmp_path):
    with session(tmp_path) as (page, _):
        _, preparations = mock_prepared_searches(page)
        switch_to_regression(page)
        choose_forecast_source(page, "panel.csv")
        set_role(page, "units", "target")
        page.click("#regression-prepare")
        page.wait_for_function(
            "document.getElementById('forecast-dataset').innerText.includes('600 rows')"
        )
        page.select_option("#topic-flavour", "series")
        page.wait_for_function(
            "document.getElementById('regression-data-status').innerText.match(/^\\d+ columns/)"
        )
        assert_dataset_state(page, "not prepared", source="panel.csv")
        choose_series_target(page, "units")
        page.click("#regression-prepare")
        page.wait_for_function(
            "document.getElementById('forecast-dataset').innerText.includes('600 rows')"
        )
        assert preparations[-1]["mode"] == "series"
        page.select_option("#topic-flavour", "row")
        assert_dataset_state(page, "prepared", source="panel.csv")

        page.click('#topic-switch .topic-tab:has-text("classification")')
        assert_dataset_state(
            page,
            "no dataset selected - pick one on the data tab",
            source=None,
            topic="classification",
        )
        choose_forecast_source(page, "rows.csv", "classification")
        assert_dataset_state(page, "not prepared", topic="classification")
        page.click('#topic-switch .topic-tab:has-text("regression")')
        assert_dataset_state(page, "prepared", source="panel.csv")
        page.select_option("#topic-flavour", "series")
        assert_dataset_state(page, "prepared", source="panel.csv")
        page.reload()
        page.wait_for_selector("#forecast-dataset:not(.hidden)")
        assert page.locator("#topic-flavour").input_value() == "series"
        assert_dataset_state(page, "prepared", source="panel.csv")
        saved = page.evaluate(
            "JSON.parse(localStorage.getItem('smolsmort:forecast:regression')).state"
        )
        assert set(saved["preparedByMode"]) == {"row", "series"}


def test_forecast_search_named_missing_setup_pieces(tmp_path):
    with session(tmp_path) as (page, _):
        requests, preparations = mock_prepared_searches(page)
        switch_to_regression(page)

        def refused(message):
            show_tab(page, "regression-search")
            page.click("#regression-new-search")
            page.click("#regression-search-start")
            assert page.locator("#regression-search-status").inner_text() == message

        refused("no dataset selected")
        choose_forecast_source(page, "rows.csv")
        refused("pick a variable to predict")
        (tmp_path / "forecast_root/no_dates.csv").write_text("units,group\n1,a\n2,b\n3,a\n")
        choose_forecast_source(page, "no_dates.csv")
        set_role(page, "units", "target")
        page.select_option("#topic-flavour", "series")
        page.wait_for_function(
            "document.getElementById('regression-data-status').innerText.match(/^\\d+ columns/)"
        )
        choose_series_target(page, "units")
        refused("pick a date to extend in time")
        assert requests == preparations == []
        assert "prepare the data first" not in page.locator("body").inner_text()


@pytest.mark.parametrize("flavour", ["row", "series"])
def test_forecast_unprepared_saved_setup_started_after_reload(tmp_path, flavour):
    with session(tmp_path) as (page, _):
        requests, preparations = mock_prepared_searches(page)
        switch_to_regression(page)
        if flavour == "series":
            page.select_option("#topic-flavour", flavour)
        source = "panel.csv" if flavour == "series" else "rows.csv"
        choose_forecast_source(page, source)
        if flavour == "series":
            choose_series_target(page, "units")
        else:
            set_role(page, "lead_weeks", "target")
        page.reload()
        page.wait_for_selector("#forecast-dataset:not(.hidden)")
        assert_dataset_state(page, "not prepared", source=source)
        show_tab(page, "regression-search")
        page.click("#regression-new-search")
        page.click("#regression-search-start")
        page.wait_for_function(
            "document.getElementById('regression-run-title').innerText.endsWith('· done')"
        )
        assert requests[0][1]["spec"]["mode"] == flavour
        assert any(col["role"] == "target" for col in requests[0][1]["spec"]["columns"])
        assert preparations == []
        assert_dataset_state(page, "prepared", source=source)
