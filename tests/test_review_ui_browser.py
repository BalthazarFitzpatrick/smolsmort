"""the find, select and train tabs in headless chromium against the real review server. skipped unless
playwright is importable: uv run --with playwright pytest tests/test_review_ui_browser.py

ONE SESSION, ONE TEST PER TAB, IN LOOP ORDER. the checks build on each other (select judges what
find drew, train learns from what select promoted), so they share a module-scoped session and run
in definition order; a failure names the tab it happened in, and the tabs after it fail too."""

from __future__ import annotations

import pytest

pytest.importorskip("playwright")

import browser_check  # noqa: E402 - must follow the importorskip above


@pytest.fixture(scope="module")
def live():
    with browser_check.session() as running:
        yield running


def test_tabs_and_a_host_extension_register(live):
    browser_check.check_tabs_and_extension(live.page)


def test_find_tab(live):
    browser_check.check_find(live.page, live.base, live.log)


def test_select_tab(live):
    browser_check.check_select(live.page, live.base, live.log)


def test_train_tab(live):
    browser_check.check_train(live.page, live.base, live.log, live.urls, live.app)
