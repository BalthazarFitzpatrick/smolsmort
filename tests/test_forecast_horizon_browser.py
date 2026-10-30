"""lead and blend controls over saved synthetic predictions, with no workers"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import review_world
from test_forecast_progress_browser import write_multifamily_run
from test_forecast_ui_browser import show_tab, switch_to_regression

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
CONTROLS = "#regression-horizon-controls"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_horizon_run(root: Path, *, legacy: bool = False, running: bool = False) -> str:
    run_id = write_multifamily_run(root)
    if legacy:
        return run_id
    folder = root / ".forecast-runs" / run_id
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    request["spec"].update(horizon=3, horizon_mode="single")
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
def horizon_session(tmp_path: Path, *, width: int = 1280, legacy=False, running=False):
    errors = []
    review_world.register_world_backend()
    root = tmp_path / "forecast_root"
    root.mkdir()
    run_id = write_horizon_run(root, legacy=legacy, running=running)

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
        server = serve(app, port=8094)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": width, "height": 900})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto("http://127.0.0.1:8094/")
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
    with page.expect_response("**/api/forecast-view?**") as response:
        page.locator(f'#regression-{mode}-menu input[data-{mode}="{value}"]').check()
    page.keyboard.press("Escape")
    data = response.value.json()
    page.wait_for_function(
        "view => JSON.stringify(forecastTopics.get('regression').resultsView.view) === JSON.stringify(view)",
        arg=data["view"],
    )
    return data


def choose_wape(page) -> None:
    page.locator("#regression-error-metrics").click()
    page.locator('#regression-error-metrics-menu input[data-metric-id="wape"]').check()
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


def test_horizon_controls_select_rows_metrics_and_bands(tmp_path):
    with horizon_session(tmp_path) as (page, run_id):
        open_horizon_run(page, run_id)
        choose_wape(page)
        assert page.locator(CONTROLS).is_visible()
        assert "3 weeks ahead" in page.locator(LEAD).inner_text()
        assert "off" in page.locator(BLEND).inner_text()
        assert page.locator(LEAD).get_attribute("title")
        assert page.locator(BLEND).get_attribute("title")
        for selector in (LEAD, BLEND):
            assert {"toggle", "dropdown-head"}.issubset(
                page.locator(selector).get_attribute("class").split()
            )
            assert (
                "forecast-metric-picker"
                in page.locator(selector).locator("..").get_attribute("class").split()
            )
        assert read_wape(page) == "30.0%"
        verdict = page.locator("#regression-verdict").inner_text()
        assert "pooled verdict" in verdict and "0.95" in verdict
        assert "lead 3" in page.locator(
            '#regression-family-comparison th[data-metric-id="wape"]'
        ).get_attribute("title")
        assert page.locator("#regression-lead-errors tbody tr").count() == 3
        assert page.locator("#regression-lead-errors tbody tr").first.locator(
            "td"
        ).all_text_contents() == ["1", "10.0%", "14.3%", "0.70", "3"]
        line_before = page.locator("#regression-chart .chart-line").last.get_attribute("d")
        page.locator(LEAD).click()
        assert page.locator("#regression-lead-menu label").all_text_contents() == [
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
        assert page.locator("#regression-blend-menu label").all_text_contents() == [
            "off",
            "around 1",
            "around 2",
            "around 3",
        ]
        page.keyboard.press("Escape")
        blend = choose_horizon(page, "blend", 2)
        assert blend["view"] == {"mode": "blend", "n": 2}
        assert blend["bands"] == []
        predictions = read_forecasts(blend)
        assert [predictions[step] for step in ("2026-01-19", "2026-01-26", "2026-02-02")] == [
            48.0,
            59.5,
            72.0,
        ]
        assert read_wape(page) == "14.5%"
        assert "off" in page.locator(LEAD).inner_text()
        assert "around 2" in page.locator(BLEND).inner_text()
        assert "no band for a blend" in page.locator("#regression-chart-status").inner_text()
        assert page.locator("#regression-chart .chart-band").count() == 0
        assert "equal weights for now" in page.locator("#regression-horizon-help").inner_text()
        assert (
            "centred on 2 for later weighting"
            in page.locator("#regression-chart-caption").inner_text()
        )
        assert "blend around 2" in page.locator(
            '#regression-family-comparison th[data-metric-id="wape"]'
        ).get_attribute("title")
        assert page.locator("#regression-verdict").inner_text() == verdict
        save_horizon(page, "desktop-blend")
        choose_horizon(page, "lead", 2)
        assert "off" in page.locator(BLEND).inner_text()
        assert read_wape(page) == "20.0%"
        choose_horizon(page, "blend", 3)
        default = choose_horizon(page, "blend", "off")
        assert default["view"] == {"mode": "lead", "n": 3}


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
        page.keyboard.press("ArrowDown")
        assert page.locator('#regression-lead-menu input[data-lead="1"]').evaluate(
            "node => node === document.activeElement"
        )
        page.keyboard.press("Escape")
        button = page.locator(BLEND)
        button.focus()
        page.keyboard.press("Enter")
        assert page.locator("#regression-blend-menu").is_visible()
        assert button.get_attribute("aria-expanded") == "true"
        option = page.locator('#regression-blend-menu input[data-blend="2"]')
        option.focus()
        with page.expect_response("**/api/forecast-view?**"):
            page.keyboard.press("Space")
        page.keyboard.press("Escape")
        assert not page.locator("#regression-blend-menu").is_visible()
        assert button.evaluate("node => node === document.activeElement")
        button.click()
        page.locator("#regression-chart-caption").click()
        assert not page.locator("#regression-blend-menu").is_visible()
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
        assert "around 2" in button.inner_text()
        assert "off" in page.locator(LEAD).inner_text()
        saved = page.evaluate("key => JSON.parse(localStorage.getItem(key))", STORAGE_KEY)
        assert saved[run_id] == {"mode": "blend", "n": 2}
        page.reload()
        open_horizon_run(page, run_id)
        assert "around 2" in page.locator(BLEND).inner_text()
        assert "off" in page.locator(LEAD).inner_text()
        page.locator(BLEND).click()
        assert page.locator('#regression-blend-menu input[data-blend="2"]').is_checked()


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
        for selector in ("#regression-horizon-help", "#regression-chart-caption"):
            assert page.locator(selector).is_visible()
        assert "around 1" in page.locator(BLEND).inner_text()


def test_horizon_controls_phone_layout_and_palette_tokens(tmp_path):
    with horizon_session(tmp_path, width=390) as (page, run_id):
        open_horizon_run(page, run_id)
        choose_wape(page)
        for mode in ("lead", "blend"):
            page.locator(f"#regression-{mode}").click()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            menu = page.locator(f"#regression-{mode}-menu")
            box = menu.bounding_box()
            assert box and box["x"] >= 0 and box["x"] + box["width"] <= 390
            save_horizon(page, f"phone-dark-{mode}")
            page.keyboard.press("Escape")
        choose_horizon(page, "blend", 1)
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
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        save_horizon(page, "phone-light-blend")


def test_horizon_controls_hidden_for_legacy_runs(tmp_path):
    with horizon_session(tmp_path, legacy=True) as (page, run_id):
        open_horizon_run(page, run_id)
        assert not page.locator(CONTROLS).is_visible()
        assert not page.locator("#regression-lead-errors").is_visible()
        assert not page.locator("#regression-chart-caption").is_visible()
        assert "pooled verdict" not in page.locator("#regression-verdict").inner_text()
