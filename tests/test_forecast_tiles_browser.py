"""flip tiles use saved synthetic artifacts and preserve controls while polling"""

from __future__ import annotations

import copy
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("playwright")
pytest.importorskip("xgboost")
pytest.importorskip("duckdb")

from test_forecast_progress_browser import write_multifamily_run
from test_forecast_ui_browser import session, show_tab, switch_to_regression

from smolsmort.forecast.tables import write_table


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_saved_tiles(root: Path) -> str:
    """save completed artifacts directly; no worker or training run is started"""
    run_id = "20261030-120000-tiles"
    run = root / ".forecast-runs" / run_id
    families = {
        "xgboost": {"state": "done", "best_error": 0.3},
        "ridge": {"state": "done", "best_error": 0.2, "elapsed": 780, "generation": 19},
        "snaive": {"state": "pruned", "pruned_reason": "paired validation error behind ridge"},
    }
    genome = {"family": "ridge", "families": ["lag"], "params": {"alpha": 1.0}}
    write_json(
        run / "request.json",
        {
            "version": 2,
            "families": [{"name": name} for name in families],
            "time_budget_s": 1200,
            "spec": {
                "source": "panel.csv",
                "task": "regression",
                "mode": "timeseries",
                "step": "week",
                "horizon": 4,
                "columns": [
                    {"name": "day", "role": "time"},
                    {"name": "project", "role": "dimension"},
                    {"name": "units", "role": "target", "aggregation": "sum"},
                ],
            },
        },
    )
    write_json(run / "status.json", {"state": "done", "families": families})
    write_json(
        run / "result.json",
        {
            "winner": {
                "family": "ridge",
                "fitness": 0.2,
                "model_error": 0.182,
                "verdict": {"accuracy": {"ratio": 0.71}},
            },
            "model_error": 0.182,
            "verdict": {"accuracy": {"ratio": 0.71}},
            "families": {
                name: {**status, "fitness": status.get("best_error")}
                for name, status in families.items()
            },
        },
    )
    write_json(run / "recipe.json", {"family": "ridge", "genome": genome})
    write_json(
        run / "families/ridge/leaderboard.json",
        [
            {
                "fitness": 0.2,
                "genome": genome,
                "metrics": {"fold_scores": [0.19, 0.2, 0.21], "fold_count": 3, "wape": 0.2},
                "generation": 19,
            }
        ],
    )
    write_json(
        run / "families/ridge/importance.json",
        {
            "method": "absolute standardised coefficient",
            "features": [
                {"feature": "lag_1", "importance": 0.7},
                {"feature": "lag_2", "importance": 0.3},
            ],
        },
    )
    write_table(
        run / "winner_test.parquet",
        {
            "step": np.array(
                ["2025-01-06", "2025-01-06", "2025-01-13", "2025-01-13"],
                dtype="datetime64[D]",
            ),
            "series": np.array(["a", "b", "a", "b"]),
            "actual": np.array([6.0, 4.0, 8.0, 7.0]),
            "prediction": np.array([5.0, 3.0, 7.0, 6.0]),
            "lower": np.array([4.0, 2.0, 6.0, 5.0]),
            "upper": np.array([6.0, 4.0, 8.0, 7.0]),
        },
    )
    return run_id


def open_tiles(page) -> None:
    switch_to_regression(page)
    page.select_option("#topic-flavour", "timeseries")
    show_tab(page, "regression-search")
    page.locator("#regression-tiles .forecast-tile").first.wait_for()


def make_live_tiles() -> dict:
    chart = [
        {"step": "2025-01-06", "actual": 10, "prediction": 8, "lower": 6, "upper": 10},
        {"step": "2025-01-13", "actual": 15, "prediction": 13, "lower": 11, "upper": 15},
        {"step": "2025-01-20", "actual": 12, "prediction": 11, "lower": 9, "upper": 13},
    ]
    ridge = {
        "state": "done",
        "best_error": 0.2,
        "tile": {
            "validation_error": 0.2,
            "test_ratio": 0.71,
            "test_error": 0.182,
            "test_error_metric": "wape",
            "chart": chart,
            "folds": [
                {"label": "fold 1", "error": 0.19, "ratio": 0.66},
                {"label": "fold 2", "error": 0.2, "ratio": 0.83},
                {"label": "2025-01-20", "error": 0.21, "ratio": 0.74},
            ],
            "importance": {
                "method": "absolute standardised coefficient",
                "features": [
                    {"feature": "lag_1", "importance": 0.7},
                    {"feature": "lag_2", "importance": 0.3},
                ],
            },
            "genome": {"family": "ridge", "params": {"alpha": 1.0}},
            "elapsed": 780,
            "generation": 19,
        },
    }
    worse = copy.deepcopy(ridge)
    worse["best_error"] = 0.35
    worse["tile"].update(
        {"test_ratio": 1.12, "validation_error": 0.35, "genome": {"params": {"season": 52}}}
    )
    return {
        "id": "20261030-120000-live",
        "kind": "multi",
        "version": 2,
        "state": "running",
        "task": "regression",
        "time_budget_s": 1200,
        "families": {
            "xgboost": {
                "state": "running",
                "tile": {"chart": [], "folds": [], "elapsed": 420, "generation": 5},
            },
            "snaive": worse,
            "ets": {
                "state": "pruned",
                "pruned_reason": "paired validation error behind ridge",
                "tile": {"chart": [], "folds": [], "elapsed": 120, "generation": 2},
            },
            "ridge": ridge,
        },
        "generations": [],
        "generation_curves": [],
        "leaderboard": [],
        "leaderboard_generation": 0,
    }


def mock_live_tiles(page, snapshot: dict) -> list:
    calls = []
    page.route(
        "**/api/forecast-runs",
        lambda route: route.fulfill(
            json={"runs": [{key: snapshot[key] for key in ("id", "kind", "state")}]}
        ),
    )

    def read_run(route):
        calls.append(route.request.url)
        route.fulfill(json=snapshot)

    page.route("**/api/forecast-run?**", read_run)
    return calls


def find_tile(page, family: str):
    return page.locator(f'#regression-tiles .forecast-tile[data-family="{family}"]')


def save_tiles(page, name: str) -> None:
    folder = Path(__file__).resolve().parents[1] / "outputs" / "forecast-tiles"
    folder.mkdir(parents=True, exist_ok=True)
    viewport = page.viewport_size
    page.set_viewport_size({**viewport, "height": 1800})
    page.locator("#regression-tiles").screenshot(path=folder / f"{name}.png", animations="disabled")
    page.set_viewport_size(viewport)


def test_tiles_read_saved_winner_artifacts_and_kept_tables(tmp_path):
    with session(tmp_path) as (page, _base):
        make_saved_tiles(tmp_path / "forecast_root")
        with page.expect_response("**/api/forecast-run?**") as response:
            open_tiles(page)
        data = response.value.json()
        assert data["families"]["ridge"]["tile"]["chart"] == [
            {"step": "2025-01-06", "actual": 10, "prediction": 8, "lower": 6, "upper": 10},
            {"step": "2025-01-13", "actual": 15, "prediction": 13, "lower": 11, "upper": 15},
        ]
        assert page.locator("#regression-run-view-tiles").get_attribute("aria-pressed") == "true"
        assert page.locator("#regression-tiles .forecast-tile").count() == 3
        assert not page.locator("#regression-family-progress").is_visible()
        save_tiles(page, "saved-charts")
        ridge = find_tile(page, "ridge")
        front = ridge.locator(".forecast-tile-front")
        assert "+29%" in front.inner_text()
        assert front.locator("svg polyline").count() == 2
        assert (
            front.locator(".forecast-tile-split").evaluate(
                "node => getComputedStyle(node).strokeDasharray"
            )
            == "3px, 3px"
        )
        assert find_tile(page, "xgboost").locator(".forecast-tile-front svg polyline").count() == 0
        assert find_tile(page, "xgboost").get_attribute("data-state") == "unknown"
        feature_menu = front.locator('details[data-menu="feat"]')
        feature_menu.locator("summary").click()
        assert "lag_1" in feature_menu.inner_text() and "70%" in feature_menu.inner_text()
        params = front.locator('details[data-menu="param"]')
        params.locator("summary").click()
        assert "alpha" in params.inner_text()
        page.locator("#regression-tiles-flip").click()
        back = ridge.locator(".forecast-tile-back")
        assert "fold 1" in back.inner_text() and "fold 3" in back.inner_text()
        assert "test" in back.inner_text()
        assert "fold ratios not saved" in back.inner_text()
        fold = back.locator('.forecast-tile-bar[data-label="fold 1"]')
        assert fold.get_attribute("aria-label") == (
            "fold 1: ratio unavailable; validation error 0.19"
        )
        assert fold.locator(".forecast-tile-fill").evaluate("node => node.style.width") == "0%"
        save_tiles(page, "saved-folds")
        page.locator("#regression-run-view-tables").click()
        assert page.locator("#regression-family-progress").is_visible()
        assert page.locator("#regression-family-progress tbody tr").count() == 3
        assert not page.locator("#regression-tiles").is_visible()
        page.locator("#regression-run-view-tiles").click()
        assert page.locator("#regression-tiles").is_visible()


def test_tiles_polled_states_without_losing_flip_menu_or_focus(tmp_path):
    snapshot = make_live_tiles()
    with session(tmp_path) as (page, _base):
        calls = mock_live_tiles(page, snapshot)
        open_tiles(page)
        families = page.locator("#regression-tiles-grid .forecast-tile").evaluate_all(
            "tiles => tiles.map(tile => tile.dataset.family)"
        )
        assert families[:2] == ["ridge", "snaive"]
        save_tiles(page, "live-charts")
        for name, score, color in (
            ("ridge", "+29%", "rgb(199, 237, 95)"),
            ("snaive", "−12%", "rgb(173, 121, 111)"),
            ("xgboost", "…", "rgb(119, 183, 199)"),
            ("ets", "—", "rgb(173, 121, 111)"),
        ):
            front = find_tile(page, name).locator(".forecast-tile-front")
            assert score in front.inner_text()
            assert front.evaluate("node => getComputedStyle(node).borderTopColor") == color
        running = find_tile(page, "xgboost").locator(".forecast-tile-front")
        assert running.locator("svg polyline").count() == 0
        assert "running 7 min" in running.inner_text() and "gen 5" in running.inner_text()
        flip = page.locator("#regression-tiles-flip")
        flip.focus()
        page.keyboard.press("Enter")
        assert flip.get_attribute("aria-pressed") == "true"
        assert "show charts" in flip.inner_text()
        ridge = find_tile(page, "ridge")
        assert ridge.locator(".forecast-tile-front").evaluate("node => node.inert")
        assert not ridge.locator(".forecast-tile-back").evaluate("node => node.inert")
        back = ridge.locator(".forecast-tile-back")
        assert "2025-01-20" in back.inner_text()
        fold = back.locator('.forecast-tile-bar[data-label="fold 1"]')
        assert fold.get_attribute("aria-label") == "fold 1: ratio 0.66"
        assert fold.locator(".forecast-tile-fill").evaluate("node => node.style.width") == "44%"
        origin = back.locator(
            '.forecast-tile-bar[data-label="2025-01-20"] .forecast-tile-bar-label'
        )
        assert origin.evaluate("node => node.scrollWidth <= node.clientWidth")
        assert "ratio vs naive · mark at 1.0" in back.inner_text()
        save_tiles(page, "live-folds")
        menu = back.locator('details[data-menu="feat"]')
        summary = menu.locator("summary")
        summary.focus()
        page.keyboard.press("Enter")
        assert menu.get_attribute("open") is not None
        before = len(calls)
        snapshot["families"]["xgboost"]["tile"]["generation"] = 6
        snapshot["families"]["ridge"]["tile"]["test_ratio"] = 0.69
        page.wait_for_function(
            "() => document.querySelector('[data-family=\"xgboost\"] .forecast-tile-front')"
            ".textContent.includes('gen 6')"
        )
        assert len(calls) > before
        assert ridge.locator(".forecast-tile-front .forecast-tile-score").inner_text() == "+31%"
        assert flip.get_attribute("aria-pressed") == "true"
        assert menu.get_attribute("open") is not None
        assert summary.evaluate("node => node === document.activeElement")
        page.keyboard.press("Enter")
        assert menu.get_attribute("open") is None
        flip.click()
        assert flip.get_attribute("aria-pressed") == "false"
        assert ridge.locator(".forecast-tile-back").evaluate("node => node.inert")


def test_tiles_mobile_layout_motion_and_palette_tokens(tmp_path):
    with session(tmp_path, width=390) as (page, _base):
        mock_live_tiles(page, make_live_tiles())
        open_tiles(page)
        tiles = page.locator("#regression-tiles-grid .forecast-tile")
        boxes = [tile.bounding_box() for tile in tiles.all()]
        assert all(box and abs(box["width"] - 232) < 1 for box in boxes)
        assert all(abs(box["height"] - 232) < 1 for box in boxes)
        assert all(box["x"] >= 0 and box["x"] + box["width"] <= 390 for box in boxes)
        assert all(a["y"] + a["height"] <= b["y"] for a, b in pairwise(boxes))
        save_tiles(page, "mobile-charts")
        inner = tiles.first.locator(".forecast-tile-inner")
        assert inner.evaluate("node => getComputedStyle(node).transitionDuration") == "0.42s"
        page.emulate_media(reduced_motion="reduce")
        assert inner.evaluate("node => getComputedStyle(node).transitionDuration") == "0s"
        front = tiles.first.locator(".forecast-tile-front")
        dark = front.evaluate("node => getComputedStyle(node).backgroundColor")
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
        assert front.evaluate("node => getComputedStyle(node).backgroundColor") != dark
        assert front.evaluate("node => getComputedStyle(node).color") == "rgb(34, 34, 34)"
        save_tiles(page, "mobile-light-tokens")
        page.locator("#regression-tiles-flip").click()
        assert tiles.first.locator(".forecast-tile-front").evaluate("node => node.inert")


def test_legacy_tiles_loaded_saved_charts_and_kept_live_metadata(tmp_path):
    with session(tmp_path) as (page, _base):
        write_multifamily_run(tmp_path / "forecast_root", version=1)
        with page.expect_response("**/api/forecast-view?**") as response:
            open_tiles(page)
        assert list(response.value.json()["tiles"]) == ["xgboost"]
        tile = find_tile(page, "xgboost")
        tile.locator(".forecast-tile-front svg").wait_for()
        assert tile.locator(".forecast-tile-score").first.inner_text() == "+5%"
        assert tile.locator(".forecast-tile-front svg circle").count() == 2
        assert response.value.json()["tiles"]["xgboost"]["chart"][0]["actual"] == 16
        value = page.evaluate(
            """() => forecastTileEntries({version: 1, state: 'running',
              families: {xgboost: {state: 'running'}}, progress: {elapsed: 42, generation: 5},
              leaderboard: [{genome: {params: {max_depth: 2}}, fitness: 0.3}]})[0]"""
        )
        assert value["elapsed"] == 42 and value["generation"] == 5
        assert value["genome"]["params"] == {"max_depth": 2}
        assert page.evaluate("forecastTileState({state: 'paused'})") == "running"
