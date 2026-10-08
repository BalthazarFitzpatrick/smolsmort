"""headless chromium against the REAL review server, over the synthetic row fixture: the
regression topic end to end (data -> search -> results). run with:
uv run --with playwright pytest -q tests/test_forecast_ui_browser.py"""

from __future__ import annotations

import csv
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("playwright")
pytest.importorskip("xgboost")
pytest.importorskip("duckdb")

import review_world
from forecast_fixtures import write_panel, write_rows
from playwright.sync_api import Page, sync_playwright

from smolsmort.forecast.tab import forecast_tab
from smolsmort.review.server import build_app, serve
from smolsmort.review_ui.server import STATIC

TOPIC = "regression"


def show_tab(page: Page, name: str) -> None:
    page.click(f'.nav-tab[data-tab="{name}"]')
    page.wait_for_timeout(250)


@contextmanager
def session(tmp_path: Path):
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
                page = browser.new_page(viewport={"width": 1280, "height": 900})
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
        f'document.getElementById("{TOPIC}-search-status").innerText.startsWith("done")'
        f' || document.getElementById("{TOPIC}-search-status").innerText.startsWith("failed")',
        timeout=180000,
    )
    status = page.inner_text(f"#{TOPIC}-search-status")
    assert status.startswith("done"), status


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

        run_search(page)

        page.wait_for_selector(f'#{TOPIC}-runs-list .toggle:has-text("open results")')
        page.click(f'#{TOPIC}-runs-list .toggle:has-text("open results")')
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
        page.locator(f'#{TOPIC}-runs-list .toggle:has-text("open results")').first.click()
        page.wait_for_selector(f"#{TOPIC}-verdict:not(.hidden)")
        assert page.locator(f"#{TOPIC}-results-table tr").count() > 1
