"""the find, select and train tabs in headless chromium against the real review server. skipped unless
playwright is importable: uv run --with playwright pytest tests/test_review_ui_browser.py"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright")

import browser_check  # noqa: E402 - must follow the importorskip above


def test_the_review_tabs_behave_in_a_browser():
    browser_check.run()
