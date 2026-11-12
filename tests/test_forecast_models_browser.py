"""the training config box sends v2 requests from synthetic data in headless chromium"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright")
pytest.importorskip("xgboost")
pytest.importorskip("duckdb")

from test_forecast_ui_browser import (
    choose_forecast_source,
    choose_timeseries_target,
    mock_prepared_searches,
    session,
    set_config_number,
    show_tab,
    switch_to_regression,
)


def make_catalog():
    return [
        {
            "name": "xgboost",
            "label": "xgboost",
            "available": True,
            "suitable": True,
            "reason": None,
            "install_command": None,
        },
        {
            "name": "ridge",
            "label": "ridge regression",
            "available": True,
            "suitable": True,
            "reason": None,
            "install_command": None,
        },
        {
            "name": "snaive",
            "label": "seasonal naive",
            "available": True,
            "suitable": True,
            "reason": None,
            "install_command": None,
        },
        {
            "name": "lightgbm",
            "label": "lightgbm",
            "available": False,
            "suitable": False,
            "reason": "library unavailable; uv sync --extra lightgbm",
            "install_command": "uv sync --extra lightgbm",
        },
        {
            "name": "ets",
            "label": "exponential smoothing",
            "available": True,
            "suitable": False,
            "reason": "requires at least two seasons of training history",
            "install_command": None,
        },
    ]


def mock_catalog(page, catalog=None):
    catalog = catalog if catalog is not None else make_catalog()
    page.route(
        "**/api/forecast-families?**",
        lambda route: route.fulfill(json={"families": catalog}),
    )


def select_timeseries(page):
    switch_to_regression(page)
    page.select_option("#topic-flavour", "timeseries")
    choose_forecast_source(page, "panel.csv")
    choose_timeseries_target(page, "units")
    show_tab(page, "regression-search")


def open_families(page, topic="regression"):
    page.click(f"#{topic}-config-families")
    return page.locator(".menu-panel")


def test_families_menu_shows_short_series_note(tmp_path):
    with session(tmp_path, width=1600) as (page, _base):
        mock_prepared_searches(page)
        catalog = make_catalog()
        catalog[2].update(
            skipped_series=1,
            total_series=13,
            coverage_note="n/a for 1 of 13 series: needs two seasons",
        )
        mock_catalog(page, catalog)
        select_timeseries(page)
        item = open_families(page).locator('.menu-item[data-id="snaive"]')
        assert "n/a for 1 of 13 series: needs two seasons" in item.inner_text()
        assert "disabled" not in (item.get_attribute("class") or "").split()
        item.click()


def save_families(page, names):
    menu = open_families(page)
    for name in names:
        menu.locator(f'.menu-item[data-id="{name}"]').click()
    menu.locator('.menu-buttons [data-id="save"]').click()


def assert_config_rows(page):
    heads = page.locator("#regression-config .dropdown-head")
    assert heads.count() == 6
    positions = [head.bounding_box() for head in heads.all()]
    assert len({round(position["y"]) for position in positions}) == 2
    assert positions[0]["y"] == positions[1]["y"] == positions[2]["y"]
    assert positions[3]["y"] == positions[4]["y"] == positions[5]["y"]


def test_training_config_shared_new_and_warm_request_values(tmp_path):
    with session(tmp_path, width=1600) as (page, _base):
        requests, _preparations = mock_prepared_searches(page)
        mock_catalog(page)
        select_timeseries(page)
        assert_config_rows(page)
        page.locator("#regression-config").screenshot(path="/private/tmp/training-config-new.png")
        assert page.locator("#regression-config-mode-warm").is_disabled()
        for key, value in (
            ("minutes", 30),
            ("atOnce", 1),
            ("population", 24),
            ("patience", 8),
            ("generations", 100),
        ):
            set_config_number(page, key, value)
        page.click("#regression-config-population")
        assert (
            "on"
            in page.locator('.menu-panel .menu-item[data-id="24"]').get_attribute("class").split()
        )
        page.locator('.menu-panel .menu-item[data-id="24"]').click()
        page.click("#regression-config-start")
        page.wait_for_function(
            "document.getElementById('regression-run-title').innerText.includes('test01')"
        )
        payload = requests[0][1]
        assert {
            key: payload[key]
            for key in (
                "time_budget_s",
                "max_concurrent_families",
                "population",
                "plateau",
                "max_generations",
            )
        } == {
            "time_budget_s": 1800,
            "max_concurrent_families": 1,
            "population": 24,
            "plateau": 8,
            "max_generations": 100,
        }
        page.click("#regression-config-mode-warm")
        assert_config_rows(page)
        page.locator("#regression-config").screenshot(path="/private/tmp/training-config-warm.png")
        assert page.locator("#regression-config-families").is_disabled()
        assert "xgboost" in page.locator("#regression-config-families").inner_text()
        for key, value in (
            ("minutes", 30),
            ("population", 24),
            ("patience", 8),
            ("generations", 100),
        ):
            assert str(value) in page.locator(f"#regression-config-{key}").inner_text()
        for key, value in (
            ("minutes", 30),
            ("population", 24),
            ("patience", 8),
            ("generations", 100),
        ):
            set_config_number(page, key, value)
        page.click("#regression-run-warm")
        page.wait_for_function(
            "document.getElementById('regression-run-title').innerText.includes('test02')"
        )
        assert requests[-1][1]["warm_from"].endswith("test01")
        assert requests[-1][1]["budget"] == {
            "population": 24,
            "plateau": 8,
            "max_generations": 100,
            "time_cap": 1800,
        }
        saved = page.evaluate(
            "JSON.parse(localStorage.getItem('smolsmort:forecast:regression')).state.modelPlan"
        )
        assert saved["population"] == 24
        assert saved["minutes"] == 30
        page.set_viewport_size({"width": 800, "height": 1000})
        page.screenshot(path="/private/tmp/training-config-narrow.png", full_page=True)


def test_training_catalog_reloaded_when_data_changed_and_guarded_stale_spec(tmp_path):
    with session(tmp_path) as (page, _base):
        requests, _preparations = mock_prepared_searches(page)
        mock_catalog(page)
        select_timeseries(page)
        with page.expect_request("**/api/forecast-families?**"):
            page.evaluate("""() => {
                const ctl = forecastTopics.get('regression');
                ctl.legacyWhere = '1 = 1';
                ctl.saveStored();
            }""")
        page.wait_for_function(
            "document.getElementById('regression-config-status').innerText !== 'checking families...'"
        )
        page.evaluate("forecastTopics.get('regression').legacyWhere = '1 = 0'")
        page.click("#regression-config-start")
        assert "data changed" in page.locator("#regression-config-status").inner_text()
        assert requests == []


def test_training_picked_run_loaded_late_without_resetting_edits_on_poll(tmp_path):
    with session(tmp_path) as (page, _base):
        mock_prepared_searches(page)
        mock_catalog(page)
        select_timeseries(page)
        page.evaluate("""() => {
            const ctl = forecastTopics.get('regression');
            ctl.selectedRunId = 'late-run';
            ctl.runCache.set('late-run', {id: 'late-run', state: 'done'});
            ctl.models.paintControls();
        }""")
        page.click("#regression-config-mode-warm")
        page.evaluate("""() => {
            const ctl = forecastTopics.get('regression');
            ctl.runCache.set('late-run', {id: 'late-run', state: 'done',
                families: {ridge: {}, snaive: {}}, time_budget_s: 3600,
                population: 40, plateau: 8, max_generations: 100,
                max_concurrent_families: 2});
            ctl.models.paintControls();
        }""")
        assert "ridge + snaive" in page.locator("#regression-config-families").inner_text()
        assert "60 min" in page.locator("#regression-config-minutes").inner_text()
        assert page.locator("#regression-config-population").inner_text() == "40"
        set_config_number(page, "population", 24)
        page.evaluate("forecastTopics.get('regression').models.paintControls()")
        assert page.locator("#regression-config-population").inner_text() == "24"


def test_training_families_menu_saved_and_discarded_draft(tmp_path):
    with session(tmp_path) as (page, _base):
        mock_catalog(page)
        select_timeseries(page)
        menu = open_families(page)
        assert menu.locator('.menu-buttons [data-id="save"]').is_visible()
        assert menu.locator('.menu-buttons [data-id="close"]').is_visible()
        disabled = menu.locator('.menu-item[data-id="lightgbm"]')
        assert "disabled" in disabled.get_attribute("class").split()
        assert "uv sync --extra lightgbm" in menu.inner_text()
        menu.locator('.menu-item[data-id="ridge"]').click()
        menu.locator('.menu-buttons [data-id="close"]').click()
        assert "ridge" not in page.locator("#regression-config-families").inner_text()
        save_families(page, ["ridge"])
        assert "xgboost" in page.locator("#regression-config-families").inner_text()
        assert "ridge" in page.locator("#regression-config-families").inner_text()
        page.reload()
        page.wait_for_selector("#regression-config-families")
        assert "ridge" in page.locator("#regression-config-families").inner_text()


def test_training_families_all_suitable_and_empty_selection(tmp_path):
    with session(tmp_path) as (page, _base):
        mock_catalog(page)
        select_timeseries(page)
        save_families(page, ["xgboost"])
        assert page.locator("#regression-config-start").is_disabled()
        menu = open_families(page)
        menu.locator('.menu-item[data-id="all-suitable"]').click()
        menu.locator('.menu-buttons [data-id="save"]').click()
        assert page.locator("#regression-config-start").is_enabled()
        label = page.locator("#regression-config-families").inner_text()
        assert all(name in label for name in ("xgboost", "ridge", "snaive"))
        assert "lightgbm" not in label


@pytest.mark.parametrize("minutes", [1, 1440])
def test_training_time_cap_custom_bounds(tmp_path, minutes):
    with session(tmp_path) as (page, _base):
        requests, _preparations = mock_prepared_searches(page)
        mock_catalog(page)
        select_timeseries(page)
        set_config_number(page, "minutes", minutes)
        page.click("#regression-config-start")
        page.wait_for_function(
            "document.getElementById('regression-run-title').innerText.includes('test01')"
        )
        assert requests[-1][1]["time_budget_s"] == minutes * 60


@pytest.mark.parametrize("parallel,expected", [(False, 1), (True, 3)])
def test_training_config_migrated_old_saved_plan(tmp_path, parallel, expected):
    with session(tmp_path) as (page, _base):
        mock_catalog(page)
        page.evaluate(
            "state => localStorage.setItem('smolsmort:forecast:regression', JSON.stringify({state}))",
            {
                "modelPlan": {
                    "families": ["ridge"],
                    "minutes": 30,
                    "parallel": parallel,
                    "atOnce": 3,
                },
            },
        )
        page.reload()
        select_timeseries(page)
        assert "ridge" in page.locator("#regression-config-families").inner_text()
        assert "30" in page.locator("#regression-config-minutes").inner_text()
        saved = page.evaluate(
            "JSON.parse(localStorage.getItem('smolsmort:forecast:regression')).state.modelPlan"
        )
        assert saved["atOnce"] == expected
        assert "parallel" not in saved


@pytest.mark.parametrize("topic", ["regression", "classification"])
def test_training_entity_only_offered_xgboost_and_removed_tab(tmp_path, topic):
    with session(tmp_path) as (page, _base):
        mock_catalog(page)
        page.click(f'#topic-switch .topic-tab:has-text("{topic}")')
        page.select_option("#topic-flavour", "entity")
        show_tab(page, f"{topic}-search")
        menu = open_families(page, topic)
        assert menu.locator('.menu-item[data-id="xgboost"]').is_visible()
        assert menu.locator('.menu-item[data-id="ridge"]').count() == 0
        page.keyboard.press("Escape")
        assert page.locator("#nav-bar .nav-tab").all_text_contents() == [
            "data",
            "training",
            "results",
        ]
        page.click(f"#{topic}-new-search")
        assert page.locator(f"#{topic}-config-start").is_visible()
        assert page.locator(f'.nav-tab.active[data-tab="{topic}-search"]').count() == 1
