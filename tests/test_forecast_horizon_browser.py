"""lead and blend controls over saved synthetic predictions, with no workers"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from shutil import copytree
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import review_world
from forecast_fixtures import write_panel
from forecast_ui_audit import settle_menu
from test_forecast_progress_browser import write_multifamily_run
from test_forecast_ui_browser import open_info, show_tab, switch_to_regression

from smolsmort.forecast import runs
from smolsmort.forecast.tab import forecast_tab
from smolsmort.forecast.tables import write_table
from smolsmort.review.server import build_app, serve
from smolsmort.review_ui.server import STATIC

pytest.importorskip("duckdb")
sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

STORAGE_KEY = "smolsmort:forecast:regression:horizon-view"
LEAD = "#regression-lead"
BLEND = "#regression-blend"
BLEND_MENU = "#regression-blend-menu"
CONTROLS = "#regression-horizon-controls"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_horizon_run(
    root: Path, *, legacy: bool = False, running: bool = False, horizon: int = 3
) -> str:
    run_id = write_multifamily_run(root)
    if legacy:
        return run_id
    folder = root / ".forecast-runs" / run_id
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    request["spec"].update(horizon=horizon, horizon_mode="single")
    write_json(folder / "request.json", request)
    dates = np.arange(
        np.datetime64("2025-12-15"), np.datetime64("2026-02-09"), np.timedelta64(7, "D")
    )
    write_table(
        Path(request["prepared"]) / "panel.parquet",
        {"__step": dates, "__series": np.array(["all"] * 8), "units": np.arange(10.0, 81, 10)},
    )
    leads = np.array([1, 2, 3, 1, 2, 1])
    steps = np.array(
        ["2026-01-19"] * 3 + ["2026-01-26"] * 2 + ["2026-02-02"], dtype="datetime64[D]"
    )
    predictions = np.array([54.0, 48.0, 42.0, 63.0, 56.0, 72.0])
    history = {
        "series": np.array(["all"] * 6),
        "step": steps,
        "origin": steps - (leads * 7).astype("timedelta64[D]"),
        "weeks_ahead": leads,
        "bucket": np.array(["1-2"] * 6),
        "actual": np.array([60.0, 60.0, 60.0, 70.0, 70.0, 80.0]),
        "prediction": predictions,
        "lower": predictions - 2,
        "upper": predictions + 2,
    }
    future = {
        "series": np.array(["all"] * 3),
        "step": np.array(["2026-02-09", "2026-02-16", "2026-02-23"], dtype="datetime64[D]"),
        "origin": np.array(["2026-02-02"] * 3, dtype="datetime64[D]"),
        "weeks_ahead": np.array([1, 2, 3]),
        "bucket": np.array(["1-2"] * 3),
        "prediction": np.array([81.0, 90.0, 99.0]),
        "lower": np.array([79.0, 88.0, 97.0]),
        "upper": np.array([83.0, 92.0, 101.0]),
    }
    for prefix in ("", "winner_", "ensemble_"):
        for kind in ("test", "validation"):
            write_table(folder / f"{prefix}{kind}.parquet", history)
        write_table(folder / f"{prefix}forecast.parquet", future)
    for family in ("xgboost", "ridge", "snaive"):
        for kind in ("test", "validation"):
            write_table(folder / "families" / family / f"{kind}.parquet", history)
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    result["lead_errors"] = [
        {"weeks_ahead": 1, "model_wape": 0.1, "naive_wape": 1 / 7, "ratio": 0.7, "rows": 3},
        {"weeks_ahead": 2, "model_wape": 0.2, "naive_wape": 4 / 13, "ratio": 0.65, "rows": 2},
        {"weeks_ahead": 3, "model_wape": 0.3, "naive_wape": 0.5, "ratio": 0.6, "rows": 1},
    ]
    write_json(folder / "result.json", result)
    if running:
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        status["state"] = status["families"]["ridge"]["state"] = "running"
        write_json(folder / "status.json", status)
    return run_id


@contextmanager
def horizon_session(tmp_path: Path, *, width: int = 1280, legacy=False, running=False, horizon=3):
    errors = []
    review_world.register_world_backend()
    root = tmp_path / "forecast_root"
    root.mkdir()
    write_panel(root, weeks=60)
    run_id = write_horizon_run(root, legacy=legacy, running=running, horizon=horizon)

    def reject_worker(*_args, **_kwargs):
        pytest.fail("browser checks must not launch forecast workers")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runs, "start_run", reject_worker)
        patch.setattr(runs, "_alive", lambda folder: running and folder.name == run_id)
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
        server = serve(app, port=8111)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": width, "height": 900})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto("http://127.0.0.1:8111/")
                yield page, run_id
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            review_world.unregister_world_backend()
    assert not errors, errors


def open_horizon_run(page, run_id: str) -> None:
    switch_to_regression(page)
    page.select_option("#topic-flavour", "timeseries")
    show_tab(page, "regression-search")
    page.locator(f'.forecast-run-rail [data-run-id="{run_id}"]').click()
    page.locator("#regression-run-open").click()
    page.locator("#regression-family-comparison tbody tr").first.wait_for()


def choose_horizon(page, mode: str, value: int | str):
    page.locator(f"#regression-{mode}").click()
    settle_menu(page)
    assert page.locator(f'#regression-{mode}-menu input[type="checkbox"]').count() == 0
    if mode == "blend":
        page.locator(f'{BLEND_MENU} [data-column="0"] .menu-item[data-id="{value}"]').click()
        return save_blend(page)
    with page.expect_response("**/api/forecast-view?**") as response:
        page.locator(f'#regression-{mode}-menu .menu-item[data-id="{value}"]').click()
    assert not page.locator(f"#regression-{mode}-menu").is_visible()
    data = response.value.json()
    page.wait_for_function(
        "view => JSON.stringify(forecastTopics.get('regression').resultsView.view) === JSON.stringify(view)",
        arg=data["view"],
    )
    assert_horizon_titles(page)
    return data


def open_blend(page) -> None:
    page.locator(BLEND).click()
    settle_menu(page)
    assert page.locator(BLEND_MENU).is_visible()


def pick_blend_weight(page, week: int, weight: int) -> None:
    page.locator(f'{BLEND_MENU} [data-column="0"] .menu-item[data-id="{week}"]').click()
    page.locator(f'{BLEND_MENU} [data-column="1"] .menu-item[data-id="{weight}x"]').click()
    assert page.locator(BLEND_MENU).is_visible()
    row = page.locator(f'{BLEND_MENU} [data-column="0"] .menu-item[data-id="{week}"]')
    assert row.locator(".stats").inner_text() == f"{weight}x"


def save_blend(page) -> dict:
    with page.expect_response("**/api/forecast-view?**") as response:
        page.locator(f'{BLEND_MENU} .menu-buttons [data-id="save"]').click()
    data = response.value.json()
    page.wait_for_function(
        "view => JSON.stringify(forecastTopics.get('regression').resultsView.view) === JSON.stringify(view)",
        arg=data["view"],
    )
    assert not page.locator(BLEND_MENU).is_visible()
    assert_horizon_titles(page)
    return data


def choose_blend_weights(page, weights: dict[int, int]) -> dict:
    open_blend(page)
    for week, weight in weights.items():
        pick_blend_weight(page, week, weight)
    return save_blend(page)


def blend_view(weights: dict[int, int] | None = None) -> dict:
    return {
        "mode": "blend",
        "weights": {str(week): (weights or {}).get(week, 1) for week in range(1, 4)},
    }


def assert_horizon_titles(page) -> None:
    assert "show only the forecast made n weeks before each week" in page.locator(
        LEAD
    ).get_attribute("title")
    assert "nothing is averaged" in page.locator(LEAD).get_attribute("title")
    assert page.locator(BLEND).get_attribute("title") == (
        "average of all forecasts of a week; each time part counts with its weight. "
        "not a forecast the model made."
    )


def assert_blend_columns(page, *, week: int | str | None = None, weights=None) -> None:
    menu = page.locator(BLEND_MENU)
    columns = menu.locator(".col")
    assert columns.count() == 2
    assert columns.nth(0).locator(".menu-item .name").all_text_contents() == [
        "off",
        "1 week ahead",
        "2 weeks ahead",
        "3 weeks ahead",
    ]
    assert columns.nth(0).locator(".menu-item.on").count() == 1
    focused = columns.nth(0).locator(".menu-item.on").get_attribute("data-id")
    if week is not None:
        assert focused == str(week)
    expected = blend_view(weights)["weights"]
    assert columns.nth(0).locator(".menu-item .stats").all_text_contents() == [
        f"{weight}x" for weight in expected.values()
    ]
    if focused == "off":
        assert columns.nth(1).locator(".menu-item").count() == 0
    else:
        assert columns.nth(1).locator(".menu-item .name").all_text_contents() == [
            "1x",
            "2x",
            "4x",
            "8x",
        ]
        assert (
            columns.nth(1).locator(".menu-item.on").get_attribute("data-id")
            == f"{expected[focused]}x"
        )
    assert columns.evaluate_all("nodes => nodes.every(node => node.dataset.multi === 'false')")
    assert menu.locator('input[type="checkbox"]').count() == 0
    buttons = menu.locator(".menu-buttons [data-id]")
    assert buttons.all_text_contents() == ["save", "discard", "close"]
    assert menu.locator('.menu-buttons [data-id="save"]').get_attribute("data-primary") == "true"
    assert menu.locator(".menu-item .name").evaluate_all(
        """nodes => nodes.every(node => {
            const name = node.getBoundingClientRect();
            const row = node.closest('.menu-item').getBoundingClientRect();
            return node.scrollWidth <= node.clientWidth && name.right <= row.right - 2;
        })"""
    )
    body = columns.first.bounding_box()
    footer = menu.locator(".menu-section.menu-buttons").bounding_box()
    assert body and footer and body["y"] + body["height"] <= footer["y"]


def choose_wape(page) -> None:
    page.locator("#regression-error-metrics").click()
    settle_menu(page)
    page.locator('#regression-error-metrics-menu .menu-item[data-id="wape"]').click()
    assert page.locator("#regression-error-metrics-menu").is_visible()
    page.keyboard.press("Escape")


def read_wape(page) -> str:
    return page.locator(
        '#regression-family-comparison tbody tr:first-child [data-metric-id="wape"]'
    ).inner_text()


def read_forecasts(payload: dict) -> dict:
    series = next(series for series in payload["series"] if series["id"] == "forecast")
    return dict(zip(payload["x"], series["values"], strict=True))


def save_horizon(page, name: str) -> None:
    folder = Path(__file__).resolve().parents[1] / "outputs" / "lead-blend"
    folder.mkdir(parents=True, exist_ok=True)
    page.locator(CONTROLS).scroll_into_view_if_needed()
    page.screenshot(path=folder / f"{name}.png", full_page=True, animations="disabled")
    page.locator("#regression-chart").screenshot(
        path=folder / f"{name}-chart.png", animations="disabled"
    )
    page.locator("#regression-lead-errors").screenshot(
        path=folder / f"{name}-errors.png", animations="disabled"
    )


def save_blend_menu(page, name: str) -> None:
    folder = Path(__file__).resolve().parents[1] / "outputs"
    folder.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=folder / f"blend-weights-{name}.png", animations="disabled")


@pytest.mark.parametrize("mode", ["lead", "blend"])
def test_horizon_metrics_belong_only_to_selected_model(tmp_path, mode):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        page.locator("#regression-error-metrics").click()
        settle_menu(page)
        for metric in ("wape", "mae", "mape"):
            page.locator(f'#regression-error-metrics-menu .menu-item[data-id="{metric}"]').click()
        page.keyboard.press("Escape")
        choose_horizon(page, mode, 2)
        if mode == "blend":
            choose_blend_weights(page, {2: 4})
        rows = page.locator("#regression-family-comparison tbody tr")
        validation = {
            row.locator("td").first.inner_text(): row.locator(
                '[data-metric-id="validation_error"]'
            ).inner_text()
            for row in rows.all()
        }
        assert validation["xgboost"] == "12.0%"
        assert validation["ridge"] == "18.0%"
        for model in ("winner", "ensemble", "winner"):
            if page.evaluate("forecastTopics.get('regression').resultsView.model") != model:
                page.locator("#regression-results-model").click()
                settle_menu(page)
                with page.expect_response("**/api/forecast-view?**"):
                    page.locator(f'.menu-panel .menu-item[data-id="{model}"]').click()
                page.wait_for_function(
                    "model => forecastTopics.get('regression').resultsView.model === model",
                    arg=model,
                )
            selected = "xgboost" if model == "winner" else "ensemble of xgboost, ridge"
            assert rows.count() == 4
            for row in rows.all():
                family = row.locator("td").first.inner_text()
                assert (
                    row.locator('[data-metric-id="validation_error"]').inner_text()
                    == (validation[family])
                )
                for metric in ("wape", "mae", "mape", "test_ratio"):
                    cell = row.locator(f'[data-metric-id="{metric}"]')
                    if family == selected:
                        assert cell.inner_text() != "-"
                        assert "not computed for this view" not in cell.get_attribute("title")
                    else:
                        assert cell.inner_text() == "-", (model, family, metric)
                        assert "not computed for this view" in cell.get_attribute("title")


@pytest.mark.parametrize("mode", ["lead", "blend"])
def test_horizon_chart_keeps_one_caption_and_distinct_legend(tmp_path, mode):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        choose_wape(page)
        choose_horizon(page, mode, 2)
        if mode == "blend":
            choose_blend_weights(page, {2: 4})
        assert page.locator("#regression-chart-caption").is_visible()
        assert page.locator("#regression-horizon-help").count() == 0
        assert page.locator("#regression-chart .chart-legend-item").all_text_contents() == [
            "actual",
            "forecast",
        ]
        assert_horizon_titles(page)
        if mode == "blend":
            save_horizon(page, "desktop-reviewed-blend")
            page.locator(BLEND).click()
            settle_menu(page)
            save_horizon(page, "desktop-reviewed-blend-menu")


def test_horizon_controls_select_rows_metrics_and_bands(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        choose_wape(page)
        assert page.locator(CONTROLS).is_visible()
        assert "3 weeks ahead" in page.locator(LEAD).inner_text()
        assert "off" in page.locator(BLEND).inner_text()
        assert_horizon_titles(page)
        assert page.locator(f'{CONTROLS} input[type="checkbox"]').count() == 0
        assert page.locator(f".forecast-results-header {CONTROLS}").count() == 1
        for selector in (LEAD, BLEND):
            assert {"toggle", "dropdown-head"}.issubset(
                page.locator(selector).get_attribute("class").split()
            )
            assert page.locator(f"{CONTROLS} {selector}").count() == 1
        assert read_wape(page) == "30.0%"
        open_info(page)
        verdict = page.locator("#regression-info-verdict").inner_text()
        assert "pooled verdict" in verdict and "0.95" in verdict
        assert "pooled" in page.locator(".forecast-info-context").inner_text()
        page.keyboard.press("Escape")
        assert "lead 3" in page.locator(
            '#regression-family-comparison th[data-metric-id="wape"]'
        ).get_attribute("title")
        assert page.locator("#regression-lead-errors tbody tr").count() == 3
        assert page.locator("#regression-lead-errors tbody tr").first.locator(
            "td"
        ).all_text_contents() == ["1", "10.0%", "14.3%", "0.70", "3"]
        for label in ("model wape", "naive wape"):
            title = page.locator(f'#regression-lead-errors th:text-is("{label}")').get_attribute(
                "title"
            )
            assert "absolute error as a share of absolute actuals" in title
        ratio_title = page.locator('#regression-lead-errors th:text-is("ratio")').get_attribute(
            "title"
        )
        assert "model error / naive error" in ratio_title and "below 1 beats naive" in ratio_title
        line_before = page.locator("#regression-chart .chart-line").last.get_attribute("d")
        page.locator(LEAD).click()
        settle_menu(page)
        assert page.locator("#regression-lead-menu .menu-item .name").all_text_contents() == [
            "1 week ahead",
            "2 weeks ahead",
            "3 weeks ahead",
        ]
        page.keyboard.press("Escape")
        lead = choose_horizon(page, "lead", 1)
        assert lead["view"] == {"mode": "lead", "n": 1}
        predictions = read_forecasts(lead)
        assert [predictions[step] for step in ("2026-01-19", "2026-01-26", "2026-02-02")] == [
            54.0,
            63.0,
            72.0,
        ]
        assert read_wape(page) == "10.0%"
        assert line_before != page.locator("#regression-chart .chart-line").last.get_attribute("d")
        assert page.locator("#regression-chart .chart-band").count() > 0
        assert page.locator("#regression-chart-caption").inner_text() == (
            "each week shows the forecast made 1 weeks before that week."
        )
        save_horizon(page, "desktop-lead")
        page.locator(BLEND).click()
        settle_menu(page)
        assert_blend_columns(page, week="off")
        save_horizon(page, "desktop-blend-menu")
        page.keyboard.press("Escape")
        blend = choose_horizon(page, "blend", 2)
        assert blend["view"] == blend_view()
        assert blend["bands"] == []
        predictions = read_forecasts(blend)
        assert [predictions[step] for step in ("2026-01-19", "2026-01-26", "2026-02-02")] == [
            48.0,
            59.5,
            72.0,
        ]
        assert read_wape(page) == "14.5%"
        assert "off" in page.locator(LEAD).inner_text()
        assert "blend: equal weights" in page.locator(BLEND).inner_text()
        open_info(page)
        assert "no band for a blend" in page.locator("#regression-info-notes").inner_text()
        assert "no band for a blend" not in page.locator("#regression-chart-status").inner_text()
        assert page.locator("#regression-info").inner_text() == "info · 1 note"
        assert page.locator("#regression-chart .chart-band").count() == 0
        assert "every week counts 1x" in page.locator("#regression-chart-caption").inner_text()
        assert "blend weighted" in page.locator(
            '#regression-family-comparison th[data-metric-id="wape"]'
        ).get_attribute("title")
        assert page.locator("#regression-info-verdict").inner_text() == verdict
        save_horizon(page, "desktop-blend")
        page.keyboard.press("Escape")
        weighted = choose_blend_weights(page, {2: 4})
        assert weighted["view"] == blend_view({2: 4})
        assert weighted["bands"] == []
        predictions = read_forecasts(weighted)
        assert [predictions[step] for step in ("2026-01-19", "2026-01-26", "2026-02-02")] == (
            pytest.approx([48.0, 57.4, 72.0])
        )
        assert read_wape(page) == "15.5%"
        assert "blend: week 2 4x" in page.locator(BLEND).inner_text()
        assert "week 2" in page.locator("#regression-chart-caption").inner_text()
        assert "4" in page.locator("#regression-chart-caption").inner_text()
        assert "blend weighted" in page.locator(
            '#regression-family-comparison th[data-metric-id="wape"]'
        ).get_attribute("title")
        assert (
            "blend weighted"
            in page.locator('#regression-family-comparison th[data-metric-id="wape"]').inner_text()
        )
        page.locator(BLEND).click()
        settle_menu(page)
        assert_blend_columns(page, weights={2: 4})
        save_horizon(page, "desktop-weighted-blend-menu")
        page.keyboard.press("Escape")
        choose_horizon(page, "lead", 2)
        assert "blend: off" in page.locator(BLEND).inner_text()
        assert read_wape(page) == "20.0%"
        choose_horizon(page, "blend", 3)
        default = choose_horizon(page, "blend", "off")
        assert default["view"] == {"mode": "lead", "n": 3}
        assert "blend: off" in page.locator(BLEND).inner_text()
        open_blend(page)
        assert_blend_columns(page, week="off")
        page.keyboard.press("Escape")


def test_horizon_blend_drafts_save_discard_close_and_dismiss(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        choose_wape(page)
        requests = []
        page.on(
            "request",
            lambda request: (
                requests.append(request.url) if "/api/forecast-view?" in request.url else None
            ),
        )
        open_blend(page)
        assert_blend_columns(page, week="off")
        pick_blend_weight(page, 2, 4)
        pick_blend_weight(page, 3, 2)
        assert_blend_columns(page, week=3, weights={2: 4, 3: 2})
        assert not requests
        assert "blend: off" in page.locator(BLEND).inner_text()
        saved = save_blend(page)
        expected = blend_view({2: 4, 3: 2})
        assert saved["view"] == expected
        assert "blend: week 2 4x, week 3 2x" in page.locator(BLEND).inner_text()
        assert page.locator("#regression-chart-caption").inner_text() == (
            "each week shows the weighted average of all its forecasts: week 2 counts 4x, "
            "week 3 counts 2x, the others 1x. not a forecast the model made."
        )
        assert (
            "blend weighted"
            in page.locator('#regression-family-comparison th[data-metric-id="wape"]').inner_text()
        )
        open_blend(page)
        pick_blend_weight(page, 3, 2)
        assert_blend_columns(page, week=3, weights={2: 4, 3: 2})
        save_blend_menu(page, "desktop")
        pick_blend_weight(page, 1, 8)
        pick_blend_weight(page, 2, 1)
        page.locator(f'{BLEND_MENU} .menu-buttons [data-id="discard"]').click()
        assert page.locator(BLEND_MENU).is_visible()
        assert_blend_columns(page, weights={2: 4, 3: 2})
        assert page.evaluate("forecastTopics.get('regression').resultsView.view") == expected
        page.locator(f'{BLEND_MENU} .menu-buttons [data-id="menu-close"]').click()
        for dismissal in ("close", "escape", "outside"):
            open_blend(page)
            pick_blend_weight(page, 1, 8)
            pick_blend_weight(page, 2, 1)
            if dismissal == "close":
                page.locator(f'{BLEND_MENU} .menu-buttons [data-id="menu-close"]').click()
            elif dismissal == "escape":
                page.keyboard.press("Escape")
            else:
                page.locator("#regression-chart-caption").click()
            assert not page.locator(BLEND_MENU).is_visible()
            assert page.evaluate("forecastTopics.get('regression').resultsView.view") == expected
            open_blend(page)
            assert_blend_columns(page, weights={2: 4, 3: 2})
            page.keyboard.press("Escape")
        assert len(requests) == 1
        page.reload()
        open_horizon_run(page, run_id)
        assert page.evaluate("forecastTopics.get('regression').resultsView.view") == expected
        open_blend(page)
        assert_blend_columns(page, weights={2: 4, 3: 2})
        page.locator(f'{BLEND_MENU} [data-column="0"] [data-id="off"]').click()
        assert_blend_columns(page, week="off", weights={2: 4, 3: 2})
        page.locator(f'{BLEND_MENU} .menu-buttons [data-id="discard"]').click()
        assert_blend_columns(page, weights={2: 4, 3: 2})
        page.locator(f'{BLEND_MENU} [data-column="0"] [data-id="off"]').click()
        off = save_blend(page)
        assert off["view"] == {"mode": "lead", "n": 3}
        assert "blend: off" in page.locator(BLEND).inner_text()


def test_results_info_kept_open_and_focused_across_polling(tmp_path):
    with horizon_session(tmp_path, running=True) as (page, run_id):
        open_horizon_run(page, run_id)
        head = page.locator("#regression-info")
        head.focus()
        page.keyboard.press("Enter")
        settle_menu(page)
        panel = page.locator("#regression-info-menu")
        assert panel.is_visible()
        assert panel.evaluate("node => node === document.activeElement")
        page.evaluate("window.savedInfoPanel = document.getElementById('regression-info-menu')")
        result_file = tmp_path / "forecast_root" / ".forecast-runs" / run_id / "result.json"
        result = json.loads(result_file.read_text(encoding="utf-8"))
        result["winner"]["verdict"] = {
            "accuracy": {"ratio": 0.75},
            "checks": [{"check": "history", "detail": "updated saved history"}],
        }
        result["excluded_series"] = [
            {"series": "ended", "dims": {"series": "ended"}, "last_seen": "2026-01-05"}
        ]
        write_json(result_file, result)
        page.wait_for_function(
            "document.querySelector('#regression-info-verdict .forecast-verdict-summary')"
            "?.innerText.includes('0.75') && "
            "document.getElementById('regression-info').innerText === 'info · 1 note'",
            timeout=10000,
        )
        assert panel.is_visible()
        assert page.evaluate(
            "window.savedInfoPanel === document.getElementById('regression-info-menu')"
        )
        assert head.get_attribute("aria-expanded") == "true"
        assert panel.evaluate("node => node === document.activeElement")
        assert panel.locator(".forecast-verdict-check").inner_text() == (
            "history: updated saved history"
        )
        assert panel.locator("#regression-info-notes").inner_text() == (
            "excluded from the forecast, no data before the forecast start: "
            "[series=ended] (last seen 2026-01-05)"
        )
        page.keyboard.press("Escape")
        assert not panel.is_visible()
        assert head.evaluate("node => node === document.activeElement")


def test_horizon_controls_keyboard_reload_and_polling(tmp_path):
    with horizon_session(tmp_path, running=True) as (page, run_id):
        polls = []
        page.on(
            "response",
            lambda response: (
                polls.append(response.url) if "/api/forecast-run?" in response.url else None
            ),
        )
        open_horizon_run(page, run_id)
        page.locator(LEAD).focus()
        page.keyboard.press("Enter")
        settle_menu(page)
        page.keyboard.press("ArrowDown")
        assert page.locator('#regression-lead-menu .menu-item[data-id="1"]').evaluate(
            "node => node === document.activeElement"
        )
        page.keyboard.press("Escape")
        button = page.locator(BLEND)
        button.focus()
        page.keyboard.press("Enter")
        settle_menu(page)
        assert page.locator("#regression-blend-menu").is_visible()
        assert button.get_attribute("aria-expanded") == "true"
        for _ in range(3):
            page.keyboard.press("ArrowDown")
        option = page.locator('#regression-blend-menu .menu-item[data-id="2"]')
        assert option.evaluate("node => node === document.activeElement")
        page.keyboard.press("Space")
        assert_blend_columns(page, week=2)
        page.keyboard.press("ArrowRight")
        assert page.locator('#regression-blend-menu .menu-item[data-id="1x"]').evaluate(
            "node => node === document.activeElement"
        )
        page.keyboard.press("ArrowDown")
        page.keyboard.press("ArrowDown")
        strength = page.locator('#regression-blend-menu .menu-item[data-id="4x"]')
        assert strength.evaluate("node => node === document.activeElement")
        page.keyboard.press("Space")
        assert_blend_columns(page, week=2, weights={2: 4})
        assert "blend: off" in button.inner_text()
        with page.expect_response("**/api/forecast-view?**") as response:
            page.keyboard.press("Enter")
        assert response.value.json()["view"] == blend_view({2: 4})
        assert not page.locator("#regression-blend-menu").is_visible()
        assert button.evaluate("node => node === document.activeElement")
        open_blend(page)
        for _ in range(3):
            page.keyboard.press("ArrowDown")
        page.keyboard.press("ArrowRight")
        focused = page.locator('#regression-blend-menu .menu-item[data-id="4x"]')
        assert focused.evaluate("node => node === document.activeElement")
        page.keyboard.press("ArrowLeft")
        page.keyboard.press("ArrowDown")
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowDown")
        page.keyboard.press("Space")
        focused = page.locator('#regression-blend-menu .menu-item[data-id="2x"]')
        assert focused.evaluate("node => node === document.activeElement")
        before = len(polls)
        with page.expect_response(
            lambda response: (
                "/api/forecast-run?" in response.url
                and parse_qs(urlparse(response.url).query).get("id") == [run_id]
            ),
            timeout=10000,
        ):
            pass
        assert len(polls) > before
        assert page.locator("#regression-blend-menu").is_visible()
        assert focused.evaluate("node => node === document.activeElement")
        assert_blend_columns(page, week=3, weights={2: 4, 3: 2})
        page.keyboard.press("Escape")
        assert "blend: week 2 4x" in button.inner_text()
        assert "off" in page.locator(LEAD).inner_text()
        saved = page.evaluate("key => JSON.parse(localStorage.getItem(key))", STORAGE_KEY)
        assert saved[run_id] == blend_view({2: 4})
        show_tab(page, "regression-search")
        metrics = page.locator("#regression-error-metrics")
        metrics.focus()
        page.keyboard.press("Enter")
        settle_menu(page)
        page.keyboard.press("ArrowDown")
        page.keyboard.press("ArrowDown")
        wape = page.locator('#regression-error-metrics-menu .menu-item[data-id="wape"]')
        assert wape.evaluate("node => node === document.activeElement")
        page.keyboard.press("Space")
        page.evaluate("() => forecastTopics.get('regression').pollSearch()")
        assert page.locator("#regression-error-metrics-menu").is_visible()
        assert wape.evaluate("node => node === document.activeElement")
        assert "on" in wape.get_attribute("class").split()
        page.keyboard.press("Escape")
        page.reload()
        open_horizon_run(page, run_id)
        assert "blend: week 2 4x" in page.locator(BLEND).inner_text()
        assert "off" in page.locator(LEAD).inner_text()
        page.locator(BLEND).click()
        settle_menu(page)
        assert_blend_columns(page, weights={2: 4})


@pytest.mark.parametrize("strength", [None, 4])
def test_horizon_controls_migrate_older_blend_weights(tmp_path, strength):
    with horizon_session(tmp_path) as (page, run_id):
        selection = {"mode": "blend", "n": 2}
        if strength is not None:
            selection["strength"] = strength
        page.evaluate(
            "([key, run, selection]) => localStorage.setItem(key, JSON.stringify({[run]: selection}))",
            [STORAGE_KEY, run_id, selection],
        )
        page.reload()
        open_horizon_run(page, run_id)
        caption = "blend: week 2 4x" if strength else "blend: equal weights"
        assert caption in page.locator(BLEND).inner_text()
        assert "off" in page.locator(LEAD).inner_text()
        page.locator(BLEND).click()
        settle_menu(page)
        assert_blend_columns(page, weights={2: strength or 1})
        assert page.evaluate("forecastTopics.get('regression').resultsView.view") == blend_view(
            {2: strength or 1}
        )
        saved = page.evaluate("key => JSON.parse(localStorage.getItem(key))", STORAGE_KEY)
        assert saved[run_id] == blend_view({2: strength or 1})


def test_horizon_controls_keep_pending_weight_changes(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        page.evaluate(
            """() => {
                const topic = forecastTopics.get('regression');
                const refresh = topic.refreshView.bind(topic);
                window.horizonRefreshes = [];
                topic.refreshView = (...args) => {
                    const result = refresh(...args);
                    window.horizonRefreshes.push(result);
                    return result;
                };
            }"""
        )
        held = []

        def hold_first(route):
            if not held:
                held.append(route)
            else:
                route.continue_()

        page.route("**/api/forecast-view?**", hold_first)
        open_blend(page)
        pick_blend_weight(page, 2, 4)
        with page.expect_request("**/api/forecast-view?**"):
            page.locator(f'{BLEND_MENU} .menu-buttons [data-id="save"]').click()
        open_blend(page)
        assert len(held) == 1
        assert_blend_columns(page, weights={2: 4})
        pick_blend_weight(page, 3, 2)
        with page.expect_response("**/api/forecast-view?**") as response:
            page.locator(f'{BLEND_MENU} .menu-buttons [data-id="save"]').click()
        expected = blend_view({2: 4, 3: 2})
        assert response.value.json()["view"] == expected
        query = parse_qs(urlparse(response.value.url).query)
        assert query["blend_weights"] == ["1:1,2:4,3:2"]
        assert "blend" not in query and "strength" not in query
        held[0].continue_()
        page.evaluate("() => Promise.all(window.horizonRefreshes)")
        page.unroute("**/api/forecast-view?**", hold_first)
        assert page.evaluate("forecastTopics.get('regression').resultsView.view") == expected
        assert "blend: week 2 4x, week 3 2x" in page.locator(BLEND).inner_text()


def test_horizon_controls_wait_for_new_run_before_opening_blend(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        run_root = tmp_path / "forecast_root" / ".forecast-runs"
        next_run = "20261017-120000-synthetic"
        copytree(run_root / run_id, run_root / next_run)
        open_horizon_run(page, run_id)
        choose_horizon(page, "blend", 2)
        errors = []
        held = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def hold_next_run(route):
            if parse_qs(urlparse(route.request.url).query).get("id") == [next_run]:
                held.append(route)
            else:
                route.continue_()

        page.route("**/api/forecast-run?**", hold_next_run)
        page.locator("#regression-results-run").click()
        settle_menu(page)
        with page.expect_request("**/api/forecast-run?**"):
            page.locator(f'.menu-panel .menu-item[data-id="{next_run}"]').click()
        page.locator("#regression-results-model").click()
        assert not page.locator(".menu-panel").is_visible()
        page.locator(BLEND).click()
        assert len(held) == 1
        assert not errors, errors
        assert not page.locator("#regression-blend-menu").is_visible()
        assert page.locator(BLEND).get_attribute("aria-expanded") == "false"
        with page.expect_response("**/api/forecast-view?**") as response:
            held[0].continue_()
        assert response.value.json()["view"] == {"mode": "lead", "n": 3}
        page.unroute("**/api/forecast-run?**", hold_next_run)
        page.wait_for_function(
            """run => {
                const topic = forecastTopics.get('regression');
                return topic.resultsRunId === run && topic.resultsView.view.mode === 'lead';
            }""",
            arg=next_run,
        )
        chosen = choose_horizon(page, "blend", 1)
        assert chosen["view"] == blend_view()
        assert (
            page.evaluate(
                "run => forecastTopics.get('regression').horizonSelections[run]", next_run
            )
            == chosen["view"]
        )


def test_horizon_controls_wait_for_cached_run_metadata_before_opening_blend(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        run_root = tmp_path / "forecast_root" / ".forecast-runs"
        next_run = "20261017-120000-synthetic"
        copytree(run_root / run_id, run_root / next_run)
        request_path = run_root / next_run / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["spec"]["horizon"] = 8
        write_json(request_path, request)
        open_horizon_run(page, next_run)
        expected = {"mode": "blend", "weights": {str(week): 1 for week in range(1, 9)}}
        expected["weights"]["8"] = 2
        assert choose_blend_weights(page, {8: 2})["view"] == expected
        page.locator("#regression-results-run").click()
        settle_menu(page)
        with page.expect_response("**/api/forecast-view?**"):
            page.locator(f'.menu-panel .menu-item[data-id="{run_id}"]').click()
        page.wait_for_function("forecastTopics.get('regression').resultsView.view.n === 3")
        held = []

        def hold_next_run(route):
            if parse_qs(urlparse(route.request.url).query).get("id") == [next_run]:
                held.append(route)
            else:
                route.continue_()

        page.route("**/api/forecast-run?**", hold_next_run)
        page.locator("#regression-results-run").click()
        settle_menu(page)
        with page.expect_request("**/api/forecast-run?**"):
            page.locator(f'.menu-panel .menu-item[data-id="{next_run}"]').click()
        page.locator(BLEND).click()
        pending_open = page.locator(BLEND_MENU).is_visible()
        pending_rows = page.locator(f'{BLEND_MENU} [data-column="0"] .menu-item').evaluate_all(
            "rows => rows.map(row => row.dataset.id)"
        )
        assert len(held) == 1
        with page.expect_response("**/api/forecast-view?**") as response:
            held[0].continue_()
        page.unroute("**/api/forecast-run?**", hold_next_run)
        assert response.value.json()["view"] == expected
        page.wait_for_function(
            "view => JSON.stringify(forecastTopics.get('regression').resultsView.view) === JSON.stringify(view)",
            arg=expected,
        )
        assert not pending_open, {"pending_rows": pending_rows, "saved_view": expected}
        open_blend(page)
        assert page.locator(f'{BLEND_MENU} [data-column="0"] .menu-item').count() == 9
        assert (
            page.locator(
                f'{BLEND_MENU} [data-column="0"] .menu-item[data-id="8"] .stats'
            ).inner_text()
            == "2x"
        )
        assert save_blend(page)["view"] == expected
        saved = page.evaluate("key => JSON.parse(localStorage.getItem(key))", STORAGE_KEY)
        assert saved[next_run] == expected


def test_horizon_blend_long_phone_picker_keeps_footer_and_far_week_weights(tmp_path):
    with horizon_session(tmp_path, width=390, horizon=52) as (page, run_id):
        open_horizon_run(page, run_id)
        open_blend(page)
        menu = page.locator(BLEND_MENU)
        assert menu.locator('[data-column="0"] .menu-item').count() == 53

        def assert_viewport():
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            box = menu.bounding_box()
            footer = menu.locator(".menu-section.menu-buttons").bounding_box()
            assert box and box["x"] >= 0 and box["x"] + box["width"] <= 390
            assert footer and footer["y"] >= 0 and footer["y"] + footer["height"] <= 900
            for button in menu.locator(".menu-buttons [data-id]").all():
                assert button.is_visible()
                assert button.evaluate("""node => {
                    const box = node.getBoundingClientRect();
                    return document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2) === node;
                }""")

        def assert_selected_week_visible(week):
            selected = menu.locator(f'[data-column="0"] .menu-item[data-id="{week}"]')
            visible = selected.evaluate("""node => {
                const box = node.getBoundingClientRect();
                return node.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2));
            }""")
            truncated = menu.locator(".menu-item .name").evaluate_all("""nodes => nodes
                .filter(node => node.scrollWidth > node.clientWidth)
                .map(node => node.textContent)""")
            children_visible = menu.locator('[data-column="1"] .menu-item').evaluate_all(
                """nodes => nodes.every(node => {
                    const box = node.getBoundingClientRect();
                    return node.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2));
                })"""
            )
            assert visible and not truncated and children_visible, {
                "selected_visible": visible,
                "truncated_labels": truncated,
                "children_visible": children_visible,
            }

        assert_viewport()
        for week, weight in ((3, 4), (8, 2), (52, 8)):
            row = menu.locator(f'[data-column="0"] .menu-item[data-id="{week}"]')
            row.click()
            assert_viewport()
            menu.locator(f'[data-column="1"] .menu-item[data-id="{weight}x"]').click()
            assert row.locator(".stats").inner_text() == f"{weight}x"
            assert_viewport()
        assert_selected_week_visible(52)
        expected = {"mode": "blend", "weights": {str(week): 1 for week in range(1, 53)}}
        expected["weights"].update({"3": 4, "8": 2, "52": 8})
        assert save_blend(page)["view"] == expected
        assert "blend: week 3 4x, week 8 2x +1 more" in page.locator(BLEND).inner_text()
        assert "week 52 counts 8x" in page.locator("#regression-chart-caption").inner_text()
        open_blend(page)
        far_week = menu.locator('[data-column="0"] .menu-item[data-id="52"]')
        far_week.click()
        assert far_week.locator(".stats").inner_text() == "8x"
        assert_viewport()
        assert_selected_week_visible(52)
        save_blend_menu(page, "phone-long")
        page.keyboard.press("Escape")
        open_blend(page)
        for _ in range(53):
            page.keyboard.press("ArrowDown")
        assert far_week.evaluate("node => node === document.activeElement")
        assert_selected_week_visible(52)
        assert_viewport()
        page.keyboard.press("ArrowRight")
        weight = menu.locator('[data-column="1"] .menu-item[data-id="8x"]')
        assert weight.evaluate("node => node === document.activeElement")
        page.keyboard.press("Space")
        assert_selected_week_visible(52)
        with page.expect_response("**/api/forecast-view?**") as response:
            page.keyboard.press("Enter")
        assert response.value.json()["view"] == expected
        assert not menu.is_visible()


@pytest.mark.parametrize("storage", ["malformed", "blocked"])
def test_horizon_controls_tolerate_unavailable_storage(tmp_path, storage):
    with horizon_session(tmp_path) as (page, run_id):
        if storage == "malformed":
            page.evaluate("key => localStorage.setItem(key, '{broken')", STORAGE_KEY)
        else:
            page.add_init_script(
                """(() => {
                    for (const method of ['getItem', 'setItem']) {
                        const original = Storage.prototype[method];
                        Storage.prototype[method] = function (key, ...args) {
                            if (key.includes(':horizon-view')) {
                                throw new DOMException('storage disabled', 'SecurityError');
                            }
                            return original.call(this, key, ...args);
                        };
                    }
                })();"""
            )
        page.reload()
        open_horizon_run(page, run_id)
        assert "3 weeks ahead" in page.locator(LEAD).inner_text()
        choose_horizon(page, "blend", 1)
        assert page.locator("#regression-chart-caption").is_visible()
        assert page.locator("#regression-horizon-help").count() == 0
        assert "blend: equal weights" in page.locator(BLEND).inner_text()


def test_horizon_controls_phone_layout_and_palette_tokens(tmp_path):
    with horizon_session(tmp_path, width=390) as (page, run_id):
        open_horizon_run(page, run_id)
        choose_wape(page)
        for mode in ("lead", "blend"):
            page.locator(f"#regression-{mode}").click()
            settle_menu(page)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            menu = page.locator(f"#regression-{mode}-menu")
            box = menu.bounding_box()
            assert box and box["x"] >= 0 and box["x"] + box["width"] <= 390
            if mode == "blend":
                assert_blend_columns(page, week="off")
            save_horizon(page, f"phone-dark-{mode}")
            page.keyboard.press("Escape")
        choose_blend_weights(page, {2: 4, 3: 2})
        open_blend(page)
        pick_blend_weight(page, 3, 2)
        assert_blend_columns(page, week=3, weights={2: 4, 3: 2})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        box = page.locator(BLEND_MENU).bounding_box()
        assert box and box["x"] >= 0 and box["x"] + box["width"] <= 390
        save_blend_menu(page, "phone")
        page.keyboard.press("Escape")
        dark = page.locator(BLEND).evaluate("node => getComputedStyle(node).backgroundColor")
        page.evaluate(
            """() => {
                const root = document.documentElement;
                root.style.setProperty('--panel-row', '#f5f5f5');
                root.style.setProperty('--text', '#222222');
                root.style.setProperty('--cream', '#222222');
                root.style.setProperty('--text-dim', '#666666');
                root.style.setProperty('--bg', '#ffffff');
                root.style.setProperty('--grey-border', '#aaaaaa');
            }"""
        )
        assert (
            page.locator(BLEND).evaluate("node => getComputedStyle(node).backgroundColor") != dark
        )
        assert (
            page.locator(BLEND).evaluate("node => getComputedStyle(node).color")
            == "rgb(34, 34, 34)"
        )
        page.locator(BLEND).click()
        settle_menu(page)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert_blend_columns(page, weights={2: 4, 3: 2})
        save_horizon(page, "phone-light-blend")


def test_horizon_controls_hidden_for_legacy_runs(tmp_path):
    with horizon_session(tmp_path, legacy=True) as (page, run_id):
        open_horizon_run(page, run_id)
        assert not page.locator(CONTROLS).is_visible()
        assert not page.locator("#regression-lead-errors").is_visible()
        assert not page.locator("#regression-chart-caption").is_visible()
        open_info(page)
        assert "pooled verdict" not in page.locator("#regression-info-verdict").inner_text()


def test_results_info_recovered_initial_view_failure_on_poll(tmp_path):
    with horizon_session(tmp_path, running=True) as (page, run_id):
        unavailable = [True]

        def read_view(route):
            if unavailable[0]:
                route.fulfill(status=503, json={"error": "saved chart temporarily unavailable"})
            else:
                route.continue_()

        page.route("**/api/forecast-view?**", read_view)
        switch_to_regression(page)
        page.select_option("#topic-flavour", "timeseries")
        show_tab(page, "regression-search")
        page.locator(f'.forecast-run-rail [data-run-id="{run_id}"]').click()
        page.locator("#regression-run-open").click()
        page.wait_for_function(
            "document.getElementById('regression-info').innerText === 'info · 1 note'"
        )
        panel = open_info(page)
        assert panel.locator("#regression-info-verdict").inner_text() == "no verdict yet"
        assert panel.locator(".forecast-info-note").inner_text() == (
            "saved chart temporarily unavailable"
        )
        unavailable[0] = False
        page.wait_for_function(
            "document.querySelector('#regression-info-verdict .forecast-verdict-summary')"
            "?.innerText.includes('0.95') && "
            "document.getElementById('regression-info').innerText === 'info'",
            timeout=10000,
        )
        assert panel.is_visible()
        assert panel.evaluate("node => node === document.activeElement")
        assert panel.locator("#regression-info-notes").inner_text() == "no notes"
        assert page.locator("#regression-chart .chart-line").count() > 0
