"""run export uses the picked model and keeps its written path visible"""

import json
from pathlib import Path

import pytest
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
        open_export(page)
        open_export_models(page).first.click()
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


def make_tiles_viewable(tmp_path, run_id):
    """give every family saved predictions so ensemble tiles open in the results picker"""
    from smolsmort.forecast.tables import read_table, write_table
    from smolsmort.forecast.tiles import ensemble_rows, write_ensemble_views

    folder = tmp_path / "forecast_root/.forecast-runs" / run_id
    future = read_table(folder / "winner_forecast.parquet")
    for family in ("xgboost", "ridge", "snaive"):
        for split in ("validation", "test"):
            path = folder / "families" / family / f"{split}.parquet"
            if not path.exists():
                write_table(path, read_table(folder / f"winner_{split}.parquet"))
        write_table(folder / "families" / family / "forecast.parquet", future)
    ensembles = ensemble_rows(folder)
    (folder / "ensembles.json").write_text(json.dumps(ensembles), encoding="utf-8")
    write_ensemble_views(folder, ensembles)


def pick_results_model(page, model_id):
    page.locator("#regression-results-model").click()
    settle_menu(page)
    row = page.locator(f'.forecast-results-menu .menu-item[data-id="{model_id}"]')
    label = row.locator(".name").inner_text()
    row.click()
    page.wait_for_function(
        "model => forecastTopics.get('regression').resultsModel === model", arg=model_id
    )
    return label


def results_row_ids(page):
    page.locator("#regression-results-model").click()
    settle_menu(page)
    rows = page.locator(".forecast-results-menu .menu-item[data-id]")
    ids = rows.evaluate_all("rows => rows.map(row => row.dataset.id)")
    lit = page.locator(".forecast-results-menu .menu-item.on").evaluate_all(
        "rows => rows.map(row => row.dataset.id)"
    )
    page.keyboard.press("Escape")
    return ids, lit


def open_export(page):
    page.locator("#regression-export-dialog").click()
    page.get_by_role("dialog", name="export run").wait_for()
    return page.locator("#regression-export-model")


def open_export_models(page):
    page.locator("#regression-export-model").click()
    settle_menu(page)
    return page.locator(".menu-panel .menu-item[data-id]")


def lit_ids(rows):
    return rows.evaluate_all(
        "rows => rows.filter(row => row.classList.contains('on')).map(row => row.dataset.id)"
    )


def export_posts(page, tmp_path):
    page.locator("#regression-export-folder").fill(str(tmp_path / "exports"))
    with page.expect_request("**/api/forecast-export-bundle") as request:
        page.locator("#regression-export-submit").click()
    page.wait_for_function(
        "() => document.querySelector('#regression-export-status').textContent !== 'exporting'"
    )
    return request.value.post_data_json


@pytest.mark.parametrize("kind", ["ensemble", "model"])
def test_export_dialog_opens_on_picked_tile(tmp_path, kind):
    with horizon_session(tmp_path) as (page, run_id):
        make_tiles_viewable(tmp_path, run_id)
        open_horizon_run(page, run_id)
        ids, _ = results_row_ids(page)
        prefix = "ensemble:" if kind == "ensemble" else "model:ridge:"
        chosen = next(item for item in ids if item.startswith(prefix))
        label = pick_results_model(page, chosen)
        head = open_export(page)
        assert head.get_attribute("data-value") == chosen
        assert head.inner_text() == label
        assert page.locator("#regression-export-submit").is_enabled()
        rows = open_export_models(page)
        assert lit_ids(rows) == [chosen]
        page.keyboard.press("Escape")
        payload = export_posts(page, tmp_path)
        assert payload["model"] == chosen
        status = page.locator("#regression-export-status").inner_text()
        recipe = json.loads((Path(status) / "recipe.json").read_text())
        assert recipe["model"] == chosen
        assert recipe["selected"]["tile_id"] == chosen


def test_export_dialog_resolves_ensemble_alias_to_lit_tile(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        make_tiles_viewable(tmp_path, run_id)
        open_horizon_run(page, run_id)
        page.evaluate("""async () => {
          const topic = forecastTopics.get('regression');
          topic.resultsModel = 'ensemble';
          await topic.refreshView();
        }""")
        ids, lit = results_row_ids(page)
        assert "ensemble" not in ids
        assert len(lit) == 1 and lit[0].startswith("ensemble:")
        head = open_export(page)
        assert head.get_attribute("data-value") == lit[0]
        rows = open_export_models(page)
        assert lit_ids(rows) == lit
        assert "ensemble" not in rows.evaluate_all("rows => rows.map(row => row.dataset.id)")


def test_export_dialog_without_pick_asks_for_model(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        _, lit = results_row_ids(page)
        assert lit == []
        head = open_export(page)
        assert head.inner_text() == "pick a model"
        assert head.get_attribute("data-value") == ""
        submit = page.locator("#regression-export-submit")
        assert submit.is_disabled()
        rows = open_export_models(page)
        assert lit_ids(rows) == []
        chosen = rows.first.get_attribute("data-id")
        rows.first.click()
        assert head.get_attribute("data-value") == chosen
        assert head.inner_text() != "pick a model"
        assert submit.is_enabled()
        assert export_posts(page, tmp_path)["model"] == chosen


def test_export_dialog_model_list_draws_over_dialog(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        open_export(page)
        rows = open_export_models(page)
        panel = page.locator(".menu-panel")
        box = panel.bounding_box()
        on_top = page.evaluate(
            """([x, y]) => !!document.elementFromPoint(x, y)?.closest('.menu-panel')""",
            [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2],
        )
        assert on_top
        chosen = rows.last.get_attribute("data-id")
        label = rows.last.inner_text()
        rows.last.click()
        head = page.locator("#regression-export-model")
        assert head.get_attribute("data-value") == chosen
        assert head.inner_text() == label.strip()
        assert not panel.count()
        page.keyboard.press("Escape")
        assert not page.get_by_role("dialog", name="export run").count()
        page.click("#open-settings")
        page.locator("#settings-popup").wait_for()
        page.locator("#settings-popup .dropdown-head").first.click()
        settle_menu(page)
        box = page.locator(".menu-panel").bounding_box()
        assert page.evaluate(
            """([x, y]) => !!document.elementFromPoint(x, y)?.closest('.menu-panel')""",
            [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2],
        )
