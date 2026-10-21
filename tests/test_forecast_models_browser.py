"""the models page sends v2 requests from synthetic data in headless chromium"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("playwright")
pytest.importorskip("xgboost")
pytest.importorskip("duckdb")

from forecast_fixtures import write_panel
from test_forecast_ui_browser import (
    choose_forecast_source,
    choose_timeseries_target,
    mock_prepared_searches,
    session,
    show_tab,
    switch_to_regression,
)

from smolsmort.forecast import runs


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


def family_toggle(page, name, topic="regression"):
    return page.locator(
        f'#{topic}-models-families [data-family="{name}"] button.toggle[role="checkbox"]'
    )


def selected_families(page, topic="regression"):
    return page.locator(f"#{topic}-models-families [data-family]").evaluate_all(
        "rows => rows.filter(row => row.querySelector('[aria-checked=\"true\"]'))"
        ".map(row => row.dataset.family)"
    )


def select_timeseries(page):
    switch_to_regression(page)
    page.select_option("#topic-flavour", "timeseries")
    choose_forecast_source(page, "panel.csv")
    choose_timeseries_target(page, "units")


def test_models_saved_v2_requests_and_restored_plan(tmp_path, monkeypatch):
    """only the worker launch is replaced; the route persists the actual browser request"""
    launched = []

    def start_worker(command, **_kwargs):
        folder = Path(command[-1])
        launched.append(folder)
        (folder / "status.json").write_text(json.dumps({"state": "done"}))
        return SimpleNamespace(pid=900000 + len(launched), poll=lambda: 0)

    monkeypatch.setattr(
        runs, "subprocess", SimpleNamespace(Popen=start_worker, STDOUT=runs.subprocess.STDOUT)
    )
    monkeypatch.setattr(runs, "_PROCESSES", {})
    with session(tmp_path) as (page, _base):
        write_panel(tmp_path / "forecast_root", weeks=156)
        select_timeseries(page)
        page.click("#regression-prepare")
        page.wait_for_function(
            "document.getElementById('regression-data-status').innerText.startsWith('prepared')",
            timeout=30000,
        )
        with page.expect_response("**/api/forecast-families?**") as catalog_response:
            show_tab(page, "regression-models")
        catalog = catalog_response.value.json()["families"]
        assert all(
            row["available"] and row["suitable"]
            for row in catalog
            if row["name"] in {"ridge", "snaive"}
        )
        query = parse_qs(urlparse(catalog_response.value.url).query)
        assert json.loads(query["spec"][0])["mode"] == "timeseries"
        family_toggle(page, "xgboost").click()
        family_toggle(page, "ridge").click()
        family_toggle(page, "snaive").click()
        assert selected_families(page) == ["ridge", "snaive"]
        minutes = page.locator("#regression-models-minutes")
        minutes.fill("1")
        minutes.blur()
        at_once = page.locator("#regression-models-at-once")
        assert at_once.is_disabled()
        assert page.locator("#regression-models-parallel").get_attribute("aria-checked") == "false"
        page.click("#regression-models-parallel")
        assert at_once.is_enabled()
        assert at_once.get_attribute("max") == "2"
        at_once.fill("2")
        at_once.blur()

        for concurrency in (2, 1):
            with page.expect_response(
                lambda response: (
                    response.request.method == "POST"
                    and response.url.endswith("/api/forecast-runs")
                )
            ) as started:
                page.click("#regression-models-start")
            assert started.value.status == 200, started.value.text()
            run_id = started.value.json()["run_id"]
            request_path = tmp_path / "forecast_root" / ".forecast-runs" / run_id / "request.json"
            stored = json.loads(request_path.read_text())
            expected = {
                "version": 2,
                "families": [
                    {"name": "ridge", "method": "genetic"},
                    {"name": "snaive", "method": "genetic"},
                ],
                "time_budget_s": 60,
                "max_concurrent_families": concurrency,
            }
            assert {key: stored[key] for key in expected} == expected
            assert "budget" not in stored
            assert stored["transform"] == "none"
            assert stored["spec"]["mode"] == "timeseries"
            page.wait_for_selector('.nav-tab.active[data-tab="regression-search"]')
            page.wait_for_function(
                "id => document.getElementById('regression-run-title').innerText.includes(id)",
                arg=run_id.rsplit("-", 1)[-1],
            )
            show_tab(page, "regression-models")
            if concurrency == 2:
                page.reload()
                page.wait_for_selector("#regression-models-families [data-family]")
                assert selected_families(page) == ["ridge", "snaive"]
                assert minutes.input_value() == "1"
                assert at_once.input_value() == "2"
                assert (
                    page.locator("#regression-models-parallel").get_attribute("aria-checked")
                    == "true"
                )
                page.click("#regression-models-parallel")
                assert at_once.is_disabled()
        saved = page.evaluate(
            "JSON.parse(localStorage.getItem('smolsmort:forecast:regression')).state.modelPlan"
        )
        assert saved == {
            "families": ["ridge", "snaive"],
            "minutes": 1,
            "parallel": False,
            "atOnce": 2,
        }
        assert len(launched) == 2


def test_models_catalog_explained_skips_and_selected_only_suitable_families(tmp_path):
    with session(tmp_path, width=800) as (page, _base):
        mock_catalog(page)
        select_timeseries(page)
        show_tab(page, "regression-models")
        family_toggle(page, "xgboost").wait_for()
        assert page.locator("#regression-models-families [data-family]").count() == 5
        assert selected_families(page) == ["xgboost"]
        for name, reason in (
            ("lightgbm", "uv sync --extra lightgbm"),
            ("ets", "requires at least two seasons of training history"),
        ):
            toggle = family_toggle(page, name)
            assert toggle.is_disabled()
            assert "disabled" in toggle.get_attribute("class").split()
            row = page.locator(f'#regression-models-families [data-family="{name}"]')
            assert reason in row.inner_text()
            toggle.evaluate("node => node.click()")
            assert toggle.get_attribute("aria-checked") == "false"
        page.click("#regression-models-all-suitable")
        assert selected_families(page) == ["xgboost", "ridge", "snaive"]
        at_once = page.locator("#regression-models-at-once")
        assert at_once.input_value() == "3"
        assert at_once.get_attribute("max") == "3"
        family_toggle(page, "snaive").click()
        assert at_once.input_value() == "2"
        assert at_once.get_attribute("max") == "2"
        panel = page.locator('.tab-panel[data-panel="regression-models"]')
        assert panel.locator("select").count() == 0
        assert panel.locator("[data-stepper]").count() == 0
        for field in panel.locator("input").all():
            assert "text-field" in field.get_attribute("class").split()
        page.screenshot(path=tmp_path / "models-page.png", full_page=True)


@pytest.mark.parametrize("topic", ["regression", "classification"])
def test_models_entity_mode_only_offered_xgboost_and_kept_tab_ids(tmp_path, topic):
    with session(tmp_path) as (page, _base):
        mock_catalog(page)
        page.click(f'#topic-switch .topic-tab:has-text("{topic}")')
        page.select_option("#topic-flavour", "entity")
        show_tab(page, f"{topic}-models")
        family_toggle(page, "xgboost", topic).wait_for()
        assert page.locator(f"#{topic}-models-families [data-family]").evaluate_all(
            "rows => rows.map(row => row.dataset.family)"
        ) == ["xgboost"]
        assert page.locator("#nav-bar .nav-tab").all_text_contents() == [
            "data",
            "models",
            "training",
            "results",
        ]
        assert page.locator("#nav-bar .nav-tab").evaluate_all(
            "tabs => tabs.map(tab => tab.dataset.tab)"
        ) == [f"{topic}-{name}" for name in ("data", "models", "search", "results")]
        page.click(f"#{topic}-models-all-suitable")
        assert selected_families(page, topic) == ["xgboost"]
        show_tab(page, f"{topic}-search")
        assert page.locator(f"#{topic}-search-start").count() == 0
        page.click(f"#{topic}-new-search")
        assert page.locator(f"#{topic}-models-start").is_visible()
        page.evaluate(
            "topic => { localStorage.setItem('smolsmort:tab:' + topic, topic + '-removed');"
            " localStorage.setItem('ui-base:tab', topic + '-removed'); }",
            topic,
        )
        page.reload()
        page.wait_for_selector(f'.nav-tab.active[data-tab="{topic}-data"]')


@pytest.mark.parametrize(("minutes", "seconds"), [("1.5", 90), ("1440", 86400)])
def test_models_duration_supported_fractional_minutes_and_one_day(tmp_path, minutes, seconds):
    with session(tmp_path) as (page, _base):
        requests, _preparations = mock_prepared_searches(page)
        mock_catalog(page)
        select_timeseries(page)
        show_tab(page, "regression-models")
        duration = page.locator("#regression-models-minutes")
        assert duration.get_attribute("min") == "1"
        assert duration.get_attribute("max") == "1440"
        assert duration.get_attribute("step") == "any"
        duration.fill(minutes)
        duration.blur()
        page.click("#regression-models-start")
        page.wait_for_selector('.nav-tab.active[data-tab="regression-search"]')
        assert len(requests) == 1
        assert requests[0][1]["time_budget_s"] == seconds
        assert requests[0][1]["families"] == [{"name": "xgboost", "method": "genetic"}]


@pytest.mark.parametrize("minutes", ["0", "1441"])
def test_models_refused_duration_outside_the_form_bounds(tmp_path, minutes):
    with session(tmp_path) as (page, _base):
        requests, _preparations = mock_prepared_searches(page)
        mock_catalog(page)
        select_timeseries(page)
        show_tab(page, "regression-models")
        duration = page.locator("#regression-models-minutes")
        duration.fill(minutes)
        duration.blur()
        assert page.locator("#regression-models-start").is_disabled()
        assert duration.evaluate("node => node.checkValidity()") is False
        assert requests == []
        assert page.locator('.nav-tab.active[data-tab="regression-models"]').count() == 1


@pytest.mark.parametrize("at_once", ["0", "1.5", "3"])
def test_models_refused_invalid_parallelism(tmp_path, at_once):
    with session(tmp_path) as (page, _base):
        requests, _preparations = mock_prepared_searches(page)
        mock_catalog(page)
        select_timeseries(page)
        show_tab(page, "regression-models")
        family_toggle(page, "ridge").click()
        field = page.locator("#regression-models-at-once")
        assert field.get_attribute("min") == "1"
        assert field.get_attribute("max") == "2"
        assert field.get_attribute("step") == "1"
        field.fill(at_once)
        assert field.evaluate("node => node.checkValidity()") is False
        assert page.locator("#regression-models-start").is_disabled()
        assert requests == []


def test_models_flavour_switch_restored_timeseries_selection(tmp_path):
    with session(tmp_path) as (page, _base):
        mock_catalog(page)
        select_timeseries(page)
        show_tab(page, "regression-models")
        page.click("#regression-models-all-suitable")
        assert selected_families(page) == ["xgboost", "ridge", "snaive"]
        page.select_option("#topic-flavour", "entity")
        page.wait_for_function(
            "document.querySelectorAll('#regression-models-families [data-family]').length === 1"
        )
        assert selected_families(page) == ["xgboost"]
        page.select_option("#topic-flavour", "timeseries")
        family_toggle(page, "ridge").wait_for()
        assert selected_families(page) == ["xgboost", "ridge", "snaive"]


def test_models_kept_new_run_selected_when_the_first_training_list_arrived_late(tmp_path):
    new_id = "20261008-120000-test01"
    saved_id = "20990101-120000-imported"
    held_lists = []
    with session(tmp_path) as (page, _base):
        requests, _preparations = mock_prepared_searches(page)
        mock_catalog(page)

        def list_or_start(route):
            if route.request.method == "GET":
                held_lists.append(route)
            else:
                route.fallback()

        def read_run(route):
            run_id = parse_qs(urlparse(route.request.url).query)["id"][0]
            if run_id == saved_id:
                route.fulfill(json={"id": saved_id, "kind": "search", "state": "done"})
            else:
                route.fallback()

        page.route("**/api/forecast-runs", list_or_start)
        page.route("**/api/forecast-run?**", read_run)
        select_timeseries(page)
        show_tab(page, "regression-models")
        with page.expect_request(
            lambda request: request.method == "GET" and request.url.endswith("/api/forecast-runs")
        ):
            page.click("#regression-models-start")
        page.wait_for_function(
            "document.getElementById('regression-run-title').innerText === 'test01 · search · done'"
        )
        assert len(requests) == 1
        assert len(held_lists) == 1
        with page.expect_request("**/api/forecast-run?**") as selected:
            held_lists.pop().fulfill(
                json={
                    "runs": [
                        {"id": saved_id, "kind": "search", "state": "done"},
                        {"id": new_id, "kind": "search", "state": "done"},
                    ]
                }
            )
        assert parse_qs(urlparse(selected.value.url).query)["id"] == [new_id]
        page.wait_for_function(
            "document.getElementById('regression-run-title').innerText === 'test01 · search · done'"
        )
