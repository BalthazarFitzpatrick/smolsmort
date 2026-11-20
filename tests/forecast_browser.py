"""the one headless chromium harness the forecast browser tests share"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
import review_world
from forecast_ui_audit import settle_menu
from playwright.sync_api import Page, sync_playwright

from smolsmort.forecast import runs
from smolsmort.forecast.tab import forecast_tab
from smolsmort.review.server import build_app, serve
from smolsmort.review_ui.server import STATIC

TOPIC = "regression"


@contextmanager
def forecast_page(
    tmp_path: Path, root: Path, *, width: int = 1280, start_run=None, alive=None, settle_s=0.0
):
    """the real review server over a forecast root and a headless page on it.
    start_run replaces the worker launch, alive replaces the pid check; both optional"""
    import time

    errors: list[str] = []
    review_world.register_world_backend()
    with pytest.MonkeyPatch.context() as patch:
        if start_run is not None:
            patch.setattr(runs, "start_run", start_run)
        if alive is not None:
            patch.setattr(runs, "_alive", alive)
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
        server = serve(app, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            time.sleep(settle_s)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": width, "height": 900})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base + "/")
                yield page, base
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            review_world.unregister_world_backend()
    assert not errors, errors


def show_tab(page: Page, name: str) -> None:
    page.click(f'.nav-tab[data-tab="{name}"]')
    page.locator(f'.tab-panel[data-panel="{name}"]:not(.hidden)').wait_for()


def open_info(page: Page, topic: str = TOPIC):
    """open the read-only results menu before checking its facts"""
    panel = page.locator(f"#{topic}-info-menu")
    if not panel.is_visible():
        page.locator(f"#{topic}-info").click()
    panel.wait_for()
    settle_menu(page)
    return panel


def switch_to_regression(page: Page) -> None:
    page.click('#topic-switch .topic-tab:has-text("regression")')
    page.locator('#topic-switch .topic-tab.on:has-text("regression")').wait_for()
    page.select_option("#topic-flavour", "entity")
    page.wait_for_function(
        "document.getElementById('topic-flavour').value === 'entity'"
        " && !!document.querySelector('#nav-bar .nav-tab.active')"
        " && typeof forecastTopics !== 'undefined' && forecastTopics.has('regression')"
    )
