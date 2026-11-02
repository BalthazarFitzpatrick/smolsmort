"""search plan controls use saved artifacts without a forecast worker"""

import copy
import json

import pytest

pytest.importorskip("playwright")

from playwright.sync_api import expect
from test_forecast_ui_browser import session, show_tab, switch_to_regression


def test_plan_controls_saved_edits_and_polled_without_replacing_form(tmp_path):
    snapshot = {
        "plan": {
            "hours": 4,
            "families": [{"name": "ridge", "budget_s": 600, "patience": 6}],
            "sets": [{"label": "clients", "horizon": 8, "spec": "requests/clients.json"}],
            "warm": {"clients": {"ridge": "previous"}},
        },
        "status": {
            "state": "stopped",
            "jobs_done": 1,
            "jobs_total": 3,
            "seconds_left": 7380,
            "current": {"set": "clients", "family": "ridge", "pass": 1},
            "warnings": ["synthetic warning"],
        },
        "running": False,
    }
    calls = []
    saved = []
    with session(tmp_path, port=0) as (page, _base):

        def handle(route):
            request = route.request
            path = request.url.split("/api/")[-1]
            calls.append((path, request.method))
            if path == "forecast-plan" and request.method == "POST":
                snapshot["plan"] = request.post_data_json
                saved.append(copy.deepcopy(snapshot["plan"]))
                result = {"plan": snapshot["plan"]}
            elif path == "forecast-plan-start":
                snapshot["running"] = True
                snapshot["status"]["state"] = "running"
                result = {"pid": 12345}
            elif path == "forecast-plan-stop":
                snapshot["running"] = False
                snapshot["status"]["state"] = "stopped"
                result = {"ok": True}
            else:
                result = snapshot
            route.fulfill(content_type="application/json", body=json.dumps(result))

        page.route("**/api/forecast-plan*", handle)
        switch_to_regression(page)
        show_tab(page, "regression-search")
        section = page.locator(".forecast-plan:visible")
        expect(section).not_to_have_attribute("open", "")
        section.locator("summary").click()
        expect(section.locator("[data-plan-status]")).to_contain_text("1 of 3 jobs")
        expect(section.locator("[data-plan-status]")).to_contain_text("2 h 3 min left")
        expect(section.locator("[data-plan-status]")).to_contain_text("clients · ridge · pass 1")
        expect(section).to_contain_text("synthetic warning")
        start = section.locator('[data-plan-action="start"]')
        stop = section.locator('[data-plan-action="stop"]')
        save = section.locator('[data-plan-action="save"]')
        expect(start).to_have_text("resume")
        expect(start).to_be_enabled()
        expect(stop).to_be_disabled()
        budget = section.locator('[data-plan-field="budget (minutes)"]')
        budget.fill("12")
        budget.focus()
        snapshot["status"]["jobs_done"] = 2
        expect(section.locator("[data-plan-status]")).to_contain_text("2 of 3 jobs", timeout=7000)
        expect(budget).to_have_value("12")
        expect(budget).to_be_focused()
        expect(section).to_have_attribute("open", "")
        section.locator('[data-plan-field="horizon"]').fill("12")
        section.locator('[data-plan-field="second pass"]').check()
        save.click()
        expect(section).to_contain_text("saved")
        assert saved[-1]["families"][0]["budget_s"] == 720
        assert saved[-1]["sets"][0] == {
            "label": "clients",
            "horizon": 12,
            "spec": "requests/clients.json",
        }
        assert saved[-1]["warm"] == {"clients": {"ridge": "previous"}}
        assert saved[-1]["passes"]["second_pass"] is True
        start.click()
        expect(start).to_be_disabled()
        expect(save).to_be_disabled()
        expect(stop).to_be_enabled()
        section.locator("summary").click()
        before = len(calls)
        page.wait_for_timeout(5200)
        assert len(calls) > before
        section.locator("summary").click()
        stop.click()
        expect(stop).to_be_disabled()
        expect(start).to_be_enabled()
        expect(save).to_be_enabled()
        section.locator("summary").click()
        before = len(calls)
        page.wait_for_timeout(5200)
        assert len(calls) == before
