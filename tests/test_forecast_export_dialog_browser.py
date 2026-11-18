"""run export uses the picked model and keeps its written path visible"""

from pathlib import Path

from forecast_ui_audit import settle_menu
from test_forecast_horizon_browser import horizon_session, open_horizon_run


def test_export_dialog_writes_selected_model(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        page.locator("#regression-results-model").click()
        settle_menu(page)
        model = page.locator('.forecast-results-menu .menu-item[data-id="ridge"]')
        if not model.count():
            model = page.locator(".forecast-results-menu .menu-item[data-id]").last
        chosen = model.get_attribute("data-id")
        model.click()
        page.wait_for_function(
            "model => forecastTopics.get('regression').resultsModel === model", arg=chosen
        )
        page.locator("#regression-export-dialog").click()
        dialog = page.get_by_role("dialog", name="export run")
        dialog.wait_for()
        assert dialog.is_visible()
        assert page.locator("#regression-export-model").get_attribute("data-value") == chosen
        page.locator("#regression-export-close").click()
        assert not dialog.count()
        page.locator("#regression-export-dialog").click()
        folder = tmp_path / "exports"
        page.locator("#regression-export-folder").fill(str(folder))
        with page.expect_response("**/api/forecast-export-bundle"):
            page.locator("#regression-export-submit").click()
        status = page.locator("#regression-export-status")
        page.wait_for_function(
            "() => document.querySelector('#regression-export-status').textContent !== 'exporting'"
        )
        path = status.inner_text()
        assert path.startswith(str(folder)), path
        assert (Path(path) / "README.md").is_file()
        assert (Path(path) / "recipe.json").is_file()
        assert dialog.is_visible()
        page.locator("#regression-export-close").click()
        page.locator("#regression-export-dialog").click()
        assert page.locator("#regression-export-folder").input_value() == str(folder)


def test_export_dialog_shows_bad_folder_error(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        page.locator("#regression-export-dialog").click()
        page.locator("#regression-export-folder").fill("")
        with page.expect_response("**/api/forecast-export-bundle"):
            page.locator("#regression-export-submit").click()
        page.wait_for_function(
            "() => document.querySelector('#regression-export-status').textContent !== 'exporting'"
        )
        assert page.locator("#regression-export-status").inner_text() == (
            "folder must be a nonempty absolute path"
        )
        assert page.get_by_role("dialog", name="export run").is_visible()
