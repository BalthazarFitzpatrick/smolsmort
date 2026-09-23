"""headless chromium against the REAL review server, over the synthetic row fixture: the
regression topic end to end (data -> search -> results). run with:
uv run --with playwright pytest -q tests/test_forecast_ui_browser.py"""

from __future__ import annotations

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
from forecast_fixtures import write_rows
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
    row.locator("select.field-select").first.select_option(role)


def uncheck_known(page: Page, column: str) -> None:
    row = page.locator("tr", has=page.locator(f"td:text-is('{column}')"))
    row.locator("input[type=checkbox]").uncheck()


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

        page.click(f"#{TOPIC}-censor-toggle")
        page.select_option(f"#{TOPIC}-censor-unit", "week")
        page.wait_for_function(
            f'document.getElementById("{TOPIC}-censor-anchor").innerText == "created"'
        )

        page.click(f"#{TOPIC}-prepare")
        page.wait_for_function(
            f'document.getElementById("{TOPIC}-data-status").innerText.startsWith("prepared")',
            timeout=30000,
        )

        show_tab(page, "regression-search")
        for key, value in (("population", 6), ("plateau", 1), ("generations", 2)):
            head = page.locator(f'#{TOPIC}-budget [data-stepper="{key}"]')
            current = int(head.locator(".field-value").inner_text())
            step = head.locator("[data-step]").first
            delta = int(step.get_attribute("data-step"))
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
