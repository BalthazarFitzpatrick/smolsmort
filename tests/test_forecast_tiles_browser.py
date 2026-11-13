"""flip tiles use saved synthetic artifacts and preserve controls while polling"""

from __future__ import annotations

import copy
import json
import re
import shutil
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("playwright")
pytest.importorskip("xgboost")

from forecast_ui_audit import settle_menu
from test_forecast_progress_browser import write_multifamily_run
from test_forecast_ui_browser import session, show_tab, switch_to_regression, with_error_metrics

from smolsmort.forecast.tables import write_table

pytest.importorskip("duckdb")


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
                "verdict": {"accuracy": {"ratio": 0.71, "baseline_error": 0.256}},
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


def open_tiles(page, *, contract=False) -> None:
    if not contract:
        # legacy coverage exercises runs whose artifacts cannot serve the new contract
        page.route(
            "**/api/forecast-run-tiles?**",
            lambda route: route.fulfill(status=404, json={"error": "legacy artifacts"}),
        )
    switch_to_regression(page)
    page.select_option("#topic-flavour", "timeseries")
    show_tab(page, "regression-search")
    page.locator("#regression-tiles .forecast-tile").first.wait_for()


def make_contract_tiles(root):
    from smolsmort.forecast.tiles import recipe_key

    run_id = make_saved_tiles(root)
    run = root / ".forecast-runs" / run_id
    names = ["xgboost", "lightgbm", "ridge", "ets", "theta", "arima", "snaive"]
    request = json.loads((run / "request.json").read_text())
    request["families"] = [{"name": name} for name in names]
    write_json(run / "request.json", request)
    write_json(
        run / "status.json",
        {
            "state": "running",
            "families": {name: {"state": "running", "generation": 3} for name in names},
        },
    )
    for index, name in enumerate(names):
        genome = {"family": name, "families": ["lag"], "params": {}}
        key = recipe_key(genome)
        write_json(
            run / f"families/{name}/leaderboard.json",
            [{"genome": genome, "fitness": 0.2, "generation": 1}],
        )
        records = [
            {
                "generation": generation,
                "key": key,
                "train_loss": [9, 4, 2] if index < 2 else [index + generation],
                "curves": {
                    "wape": [3, 2, 1] if index < 2 else [index + generation],
                    "mae": [1, 2, 3],
                },
                "loss": {"name": "rmse"},
            }
            for generation in range(4 if index < 3 else 0)
        ]
        target = run / f"families/{name}/generation_curves.jsonl"
        target.write_text("\n".join(json.dumps(record) for record in records))
        for split in ("validation", "test"):
            write_table(
                run / f"families/{name}/{split}.parquet",
                {
                    "series": np.array(["a", "b"]),
                    "step": np.array([1, 1]),
                    "actual": np.array([100.0, 1.0]),
                    "prediction": np.array([100.0 + index, 8.0 - index]),
                },
            )
    return run_id


def test_contract_tiles_masonry_curves_filters_and_ensembles(tmp_path, monkeypatch):
    from smolsmort.forecast import runs

    monkeypatch.setattr(runs, "_alive", lambda _folder: True)
    with session(tmp_path, port=0) as (page, _base):
        run_id = make_contract_tiles(tmp_path / "forecast_root")
        open_tiles(page, contract=True)
        page.locator(".forecast-run-tile").first.wait_for()
        tiles = page.locator(".forecast-run-tile")
        assert tiles.count() == 12
        model = page.locator('.forecast-run-tile[data-family="xgboost"]')
        naive = page.locator('.forecast-run-tile[data-family="snaive"]')
        assert model.bounding_box()["height"] > naive.bounding_box()["height"] + 100
        assert naive.locator("svg, details").count() == 0
        assert "one candidate, no curve" in naive.inner_text()
        assert model.locator(".forecast-run-tile-chart > svg").count() == 2
        for chart in model.locator(".forecast-run-tile-chart > svg").all():
            assert chart.bounding_box()["height"] >= 60
        ridge = page.locator('.forecast-run-tile[data-family="ridge"]')
        for chart in ridge.locator(".forecast-run-tile-chart > svg").all():
            assert chart.bounding_box()["height"] == 48
        assert model.locator("summary > span:first-child").all_text_contents() == ["G", "P", "F"]
        assert "test ratio:" in model.inner_text()
        assert "wape · validation" in model.inner_text()
        assert (
            page.locator("#regression-tiles-grid").evaluate(
                "node => getComputedStyle(node).columnWidth"
            )
            == "306px"
        )
        colours = (
            model.locator("svg")
            .first.locator("polyline")
            .evaluate_all("nodes => nodes.map(node => node.style.stroke)")
        )
        assert colours == [
            "var(--muted-on)",
            "var(--cream)",
            "var(--muted-on)",
            "var(--tile-tone)",
        ]
        assert page.locator(".forecast-run-tile polyline").evaluate_all(
            "nodes => nodes.every(node => node.getAttribute('points').split(/[ ,]/).every(value => Number(value) >= 0 && Number(value) <= 1))"
        )
        assert (
            model.locator("polyline").first.evaluate("node => getComputedStyle(node).strokeWidth")
            == "1px"
        )
        model.locator('details[data-menu="param"] summary').click()
        model.locator('details[data-menu="feat"] summary').click()
        page.wait_for_function(
            "document.querySelectorAll('.forecast-run-tile details[open]').length === 1"
        )
        assert model.locator('details[data-menu="feat"]').get_attribute("open") is not None
        page.locator("#regression-tiles-metric").focus()
        page.keyboard.press("Escape")
        assert model.locator("details[open]").count() == 0
        model.locator('details[data-menu="param"] summary').click()
        run = tmp_path / "forecast_root/.forecast-runs" / run_id
        curve_path = run / "families/xgboost/generation_curves.jsonl"
        records = [json.loads(line) for line in curve_path.read_text().splitlines()]
        records.append({**records[-1], "generation": 4})
        curve_path.write_text("\n".join(json.dumps(record) for record in records))
        status = json.loads((run / "status.json").read_text())
        status["families"]["xgboost"]["generation"] = 4
        write_json(run / "status.json", status)
        model.locator('polyline[data-generation="4"]').first.wait_for(timeout=15000)
        assert model.locator('details[data-menu="param"]').get_attribute("open") is not None
        assert model.locator('details[data-menu="param"] summary').evaluate(
            "node => node === document.activeElement"
        )
        model.locator('details[data-menu="param"] summary').click()
        ensemble = page.locator('.forecast-run-tile[data-kind="ensemble"]').first
        assert ensemble.locator("svg, details").count() == 0
        assert "+" in ensemble.locator(".forecast-tile-chart-note").inner_text()
        page.set_viewport_size({"width": 1440, "height": 1800})
        page.locator("#regression-tiles").screenshot(path=tmp_path / "run-tiles-masonry.png")
        before = tiles.evaluate_all("nodes => nodes.map(node => node.dataset.tileId)")
        page.locator("#regression-tiles-metric").click()
        assert {"ape", "mape", "wape", "ae", "mae", "wae", "rmse"} <= set(
            page.locator(".menu-item").evaluate_all("nodes => nodes.map(node => node.dataset.id)")
        )
        page.locator('.menu-item[data-id="mape"]').click()
        after = tiles.evaluate_all("nodes => nodes.map(node => node.dataset.tileId)")
        assert before != after
        page.locator("#regression-tiles-top").click()
        page.locator('.menu-item[data-id="5"]').click()
        assert tiles.count() == 5
        page.reload()
        open_tiles(page, contract=True)
        page.locator(".forecast-run-tile").first.wait_for()
        assert tiles.count() == 5
        assert "mape" in page.locator("#regression-tiles-metric").inner_text()


def test_native_loss_curves_and_results_run_names(tmp_path):
    with session(tmp_path, port=0) as (page, _base):
        run_id = make_saved_tiles(tmp_path / "forecast_root")
        open_tiles(page)
        name = page.locator(f'.forecast-run-row[data-run-id="{run_id}"] .name').inner_text()
        page.click("#regression-config-mode-warm")
        page.click("#regression-run-open")
        head = page.locator("#regression-results-run")
        page.wait_for_function(
            "name => document.querySelector('#regression-results-run').textContent === name",
            arg=name,
        )
        assert head.get_attribute("title") == run_id
        page.click("#regression-results-run")
        item = page.locator(f'.menu-item[data-id="{run_id}"]')
        assert name in item.inner_text()
        assert "done" in item.inner_text()
        assert item.get_attribute("title") == run_id
        item.click()
        assert head.inner_text() == name
        context = page.evaluate("forecastTopics.get('regression').infoContext.textContent")
        assert name in context
        assert run_id in context
        checks = page.evaluate(
            """() => {
              const old = {tile_id: 'old', family: 'lightgbm', best_generation: 1,
                curves: Object.fromEntries([0, 1].map(generation => [generation, {
                  validation: {wape: [null, null, null], l1: [3, 2, 1]}, loss: {name: 'l1'},
                }]))};
              const current = {tile_id: 'current', family: 'xgboost', best_generation: 0,
                curves: {0: {validation: {wape: [0.3, 0.2], l1: [3, 2]}}}};
              const ensemble = {kind: 'ensemble', members: ['old', 'current']};
              return [[old, false], [current, false], [ensemble, false], [old, true]].map(([tile, training]) => {
                const chart = forecastRunTileChart(tile, [old, current], 'wape', training);
                return {label: chart.querySelector('.forecast-tile-chart-label > span').textContent,
                  lines: chart.querySelectorAll('polyline').length, text: chart.textContent};
              });
            }"""
        )
        assert checks[0]["lines"] == 2
        assert checks[0]["label"] == "validation l1 (native loss)"
        assert checks[1]["label"] == "validation wape"
        assert checks[2]["lines"] == 2
        assert "native loss" in checks[2]["label"]
        assert "validation wape" in checks[2]["label"]
        assert checks[3]["lines"] == 0
        assert "no curve saved" in checks[3]["text"]


def test_contract_curve_drawing_and_loading(tmp_path):
    with session(tmp_path, port=0) as (page, _base):
        run_id = make_saved_tiles(tmp_path / "forecast_root")
        run_path = tmp_path / "forecast_root/.forecast-runs" / run_id
        result = json.loads((run_path / "result.json").read_text())
        result["winner"]["verdict"]["accuracy"].pop("baseline_error")
        write_json(run_path / "result.json", result)
        page.wait_for_function("() => typeof forecastRunTileChart === 'function'")
        for count in (500, 2000):
            result = page.evaluate(
                """count => {
                  const values = Array.from({length: count}, (_, i) => i % 17 + i / count);
                  const tile = {family: 'ridge', curves: {0: {validation: {wape: values}}}};
                  const chart = forecastRunTileChart(tile, [tile], 'wape', false);
                  const wrap = document.createElement('div');
                  wrap.className = 'forecast-tile';
                  wrap.append(chart); document.body.append(wrap);
                  const line = chart.querySelector('polyline');
                  const points = line.getAttribute('points').split(' ');
                  const result = {lines: chart.querySelectorAll('polyline').length,
                    circles: chart.querySelectorAll('circle').length, points: points.length,
                    first: Number(points[0].split(',')[0]), last: Number(points.at(-1).split(',')[0]),
                    effect: line.getAttribute('vector-effect'), width: getComputedStyle(line).strokeWidth};
                  wrap.remove(); return result;
                }""",
                count,
            )
            assert result == {
                "lines": 1,
                "circles": 0,
                "points": result["points"],
                "first": 0,
                "last": 1,
                "effect": "non-scaling-stroke",
                "width": "1px",
            }
            assert result["points"] <= 150
        model = {
            "tile_id": "ridge",
            "family": "ridge",
            "kind": "model",
            "state": "done",
            "metrics": {"wape": {"test": 0.2}},
            "error_metrics": {"test_ratio": {"value": 0.5, "scope": "test"}},
            "curves": {"0": {"validation": {"wape": [0.1633]}}},
        }
        ensemble = {
            "tile_id": "ensemble",
            "family": "ensemble",
            "kind": "ensemble",
            "members": ["ridge"],
            "metrics": {"wape": {"test": 0.1}},
        }
        empty = {"tile_id": "empty", "family": "empty", "kind": "model", "metrics": {}}
        payload = {"tiles": [model, ensemble, empty]}
        loading_checks = []

        def serve_tiles(route):
            page.wait_for_timeout(300)
            assert page.locator("#regression-tiles-grid").inner_text() == "loading tiles"
            assert page.locator("#regression-tiles-grid .forecast-tile").count() == 0
            loading_checks.append(True)
            route.fulfill(json=payload)

        page.route("**/api/forecast-run-tiles?**", serve_tiles)
        switch_to_regression(page)
        page.select_option("#topic-flavour", "timeseries")
        show_tab(page, "regression-search")
        page.locator(".forecast-run-tile").first.wait_for()
        assert loading_checks
        assert (
            page.locator("#regression-tiles-grid .forecast-tile:not(.forecast-run-tile)").count()
            == 0
        )
        assert page.locator(".forecast-tiles-loading").count() == 0
        tile = page.locator('.forecast-run-tile[data-family="ridge"]')
        dot = tile.locator(".forecast-run-tile-single circle")
        assert dot.count() == 1
        box = dot.bounding_box()
        assert box["width"] == pytest.approx(box["height"], abs=0.1)
        assert box["width"] == pytest.approx(6, abs=0.1)
        assert "wape 0.1633" in tile.inner_text()
        assert (
            tile.locator(
                ".forecast-run-tile-chart-compact svg:not(.forecast-run-tile-single svg)"
            ).count()
            == 0
        )
        assert "no curve saved" in tile.inner_text()
        assert tile.locator(".forecast-run-tile-chart").first.locator("svg").count() == 0
        assert "test ratio: 0.25" in page.locator('[data-tile-id="ensemble"]').inner_text()
        empty_tile = page.locator('[data-tile-id="empty"]')
        assert empty_tile.locator(".forecast-tile-chart-note").all_text_contents() == [
            "one candidate, no curve"
        ]
        assert empty_tile.locator(".forecast-run-tile-chart, svg").count() == 0


def test_ensemble_ratio_states_and_repaint(tmp_path):
    with session(tmp_path, port=0) as (page, _base):
        page.wait_for_function("() => typeof forecastCreateRunTile === 'function'")
        for ratio, baseline in ((0.81, True), (1.2, True), (1.0, True), (None, False)):
            result = page.evaluate(
                """({ratio, baseline}) => {
                  const grid = document.getElementById('regression-tiles-grid');
                  const model = {tile_id: 'model', family: 'ridge', kind: 'model',
                    state: 'done', test_ratio: ratio, metrics: {}};
                  const ensemble = {tile_id: 'ensemble', family: 'ensemble', kind: 'ensemble',
                    state: 'done', members: ['model'], metrics: {wape: {test: ratio ?? 0.81}}};
                  const run = {runTiles: [model, ensemble],
                    verdict: {accuracy: {baseline_error: baseline ? 1 : null}}};
                  const context = {topic: 'regression', tileMetric: 'wape', tileTop: 'all',
                    tileGenerations: new Map(),
                    tileMetricOptions: [], selectedMetrics: () => [{id: 'test_ratio', scope: 'test',
                      label: 'test ratio', unit: 'ratio'}]};
                  ForecastTopic.prototype.paintContractTiles.call(context, run, grid);
                  const tile = grid.querySelector('[data-kind="ensemble"]');
                  const reference = grid.querySelector('[data-kind="model"]');
                  const face = tile.querySelector('.forecast-tile-face');
                  return {state: tile.dataset.state, modelState: reference.dataset.state,
                    ratio: tile.textContent, style: getComputedStyle(face).borderTopStyle,
                    colour: getComputedStyle(face).borderTopColor,
                    modelColour: getComputedStyle(reference.querySelector('.forecast-tile-face')).borderTopColor,
                    tone: getComputedStyle(tile).getPropertyValue('--tile-tone'),
                    modelTone: getComputedStyle(reference).getPropertyValue('--tile-tone')};
                }""",
                {"ratio": ratio, "baseline": baseline},
            )
            expected = {0.81: "better", 1.2: "worse", 1.0: "identical"}
            assert result["state"] == expected.get(ratio, result["modelState"])
            assert result["style"] == "dashed"
            if ratio is not None:
                assert f"test ratio: {ratio:.2f}" in result["ratio"]
                assert result["colour"] == result["modelColour"]
                assert result["tone"] == result["modelTone"]


def test_pruned_run_tiles_show_exit_score_and_pending_colour(tmp_path):
    with session(tmp_path, port=0) as (page, _base):
        page.wait_for_function("() => typeof forecastCreateRunTile === 'function'")
        result = page.evaluate(
            """() => {
              const grid = document.getElementById('regression-tiles-grid');
              const metrics = [{id: 'test_ratio', scope: 'test'}];
              const scored = {tile_id: 'scored', family: 'ridge', kind: 'model',
                state: 'pruned', scored_at_exit: true, metrics: {wape: {validation: 0.3, test: 0.2}},
                error_metrics: {test_ratio: {value: 0.8, scope: 'test'}}};
              const pending = {tile_id: 'pending', family: 'snaive', kind: 'model',
                state: 'pruned', metrics: {}};
              grid.replaceChildren(...[scored, pending].map((tile, index) =>
                forecastCreateRunTile(tile, index, [scored, pending], 'wape', null, metrics)));
              return [...grid.children].map(tile => ({state: tile.dataset.state,
                text: tile.textContent,
                dim: [...tile.querySelectorAll('.forecast-tile-dim')].map(node => node.textContent)}));
            }"""
        )
        assert result[0]["state"] == "better"
        assert "test ratio: 0.80" in result[0]["text"]
        assert "wape · validation 0.3 · test 0.2" in result[0]["text"]
        assert "scored at exit" in result[0]["dim"]
        assert result[1]["state"] == "pending"
        assert "scored at exit" not in result[1]["text"]


def test_ensemble_cards_list_members_without_charts_or_menus(tmp_path):
    with session(tmp_path, port=0) as (page, _base):
        page.wait_for_function("() => typeof forecastCreateRunTile === 'function'")
        page.evaluate(
            """() => {
              const curve = {train_loss: [8, 5, 3], validation: {wape: [0.4, 0.3, 0.2]}};
              const members = ['lightgbm', 'xgboost', 'snaive', 'ridge', 'arima', 'ets', 'theta']
                .map((family, index) => ({tile_id: family, family, kind: 'model', best_generation: 0,
                  curves: {0: index < 2 ? curve : {train_loss: [1685],
                    loss: {name: 'penalized_squared_error'}, validation: {wape: [0.1917]}}}}));
              const ensembles = [members.slice(0, 2), members].map((items, index) => ({
                tile_id: `comparison-${index}`, family: 'ensemble', kind: 'ensemble',
                members: items.map(item => item.tile_id), metrics: {}}));
              const grid = document.createElement('div');
              grid.className = 'forecast-tiles-grid';
              grid.id = 'ensemble-layout-test';
              ensembles.forEach(tile => grid.append(forecastCreateRunTile(
                tile, 0, members, 'wape', null, [])));
              document.body.append(grid);
            }"""
        )
        tiles = page.locator("#ensemble-layout-test .forecast-run-tile")
        mixed = tiles.nth(1)
        assert mixed.locator(".forecast-run-tile-single").count() == 0
        assert mixed.locator("svg, details").count() == 0
        assert mixed.locator(".forecast-tile-chart-note").inner_text() == (
            "lightgbm + xgboost + snaive + ridge + arima + ets + theta"
        )
        assert mixed.bounding_box()["height"] < 160


@pytest.mark.parametrize("tile_state", ["matching", "different", "hidden", "absent"])
def test_results_picker_saved_ensemble_has_one_selected_option(tmp_path, tile_state):
    with session(tmp_path, port=0) as (page, _base):
        write_multifamily_run(tmp_path / "forecast_root")
        open_tiles(page, contract=True)
        page.click("#regression-config-mode-warm")
        page.click("#regression-run-open")
        page.locator("#regression-family-comparison tbody tr").first.wait_for()
        page.evaluate(
            """state => {
              const topic = forecastTopics.get('regression');
              const models = ['ridge', 'xgboost', 'snaive'].map(family => ({
                tile_id: `model:${family}:best`, kind: 'model', family,
              }));
              const tile = {tile_id: 'ensemble:pair', kind: 'ensemble', family: 'ensemble',
                members: [models[0].tile_id, models[state === 'different' ? 2 : 1].tile_id],
                viewable: state !== 'hidden'};
              topic.paintModelResults({...topic.resultsView, model: 'ensemble', family: 'ensemble',
                members: ['xgboost', 'ridge'], ensemble: {members: ['xgboost', 'ridge']},
                run_tiles: state === 'absent' ? [] : [...models, tile]});
            }""",
            tile_state,
        )
        assert (
            "ensemble of xgboost, ridge" in page.locator("#regression-results-model").inner_text()
        )
        page.click("#regression-results-model")
        settle_menu(page)
        legacy = page.locator('.menu-panel .menu-item[data-id="ensemble"]')
        selected = page.locator(".menu-panel .menu-item.on")
        assert selected.count() == 1
        if tile_state == "matching":
            assert legacy.count() == 0
            assert selected.get_attribute("data-id") == "ensemble:pair"
            assert page.locator('.menu-item[data-id="ensemble:pair"]').count() == 1
        else:
            assert legacy.count() == 1
            assert selected.get_attribute("data-id") == "ensemble"


@pytest.mark.parametrize("better_ensemble", [False, True])
def test_results_picker_lists_and_selects_every_saved_tile(tmp_path, better_ensemble):
    from smolsmort.forecast.tables import read_table
    from smolsmort.forecast.tiles import read_tiles

    with session(tmp_path, port=0) as (page, _base):
        run_id = write_multifamily_run(tmp_path / "forecast_root")
        run = tmp_path / "forecast_root/.forecast-runs" / run_id
        if better_ensemble:
            result_path = run / "result.json"
            result = json.loads(result_path.read_text())
            result["winner"]["fitness"] = 0.12
            result["ensemble"]["fitness"] = 0.09
            result_path.write_text(json.dumps(result))
        for index, name in enumerate(("xgboost", "ridge", "snaive")):
            for split in ("validation", "test", "forecast"):
                table = read_table(run / f"winner_{split}.parquet")
                table["prediction"] += index
                write_table(run / f"families/{name}/{split}.parquet", table)
        tiles = read_tiles(run)
        open_tiles(page, contract=True)
        page.click("#regression-config-mode-warm")
        page.click("#regression-run-open")
        page.locator("#regression-family-comparison tbody tr").first.wait_for()
        assert page.locator("#regression-family-comparison tbody tr").count() == 3
        assert "best recipe" in page.locator("#regression-family-comparison").inner_text()
        page.click("#regression-results-model")
        settle_menu(page)
        assert page.locator(".menu-panel .menu-item").count() == len(tiles)
        if better_ensemble:
            assert (
                "ensemble of xgboost, ridge"
                in page.locator("#regression-results-model").inner_text()
            )
            assert page.locator('.menu-item[data-id="ensemble"]').count() == 0
            tile_families = {tile["tile_id"]: tile["family"] for tile in tiles}
            matching_tile = next(
                tile
                for tile in tiles
                if tile["kind"] == "ensemble"
                and {tile_families[member] for member in tile["members"]} == {"xgboost", "ridge"}
            )
            assert (
                page.locator(".menu-panel .menu-item.on").get_attribute("data-id")
                == matching_tile["tile_id"]
            )
            assert page.evaluate("forecastTopics.get('regression').resultsView.model") == "ensemble"
            assert (
                page.evaluate("forecastTopics.get('regression').resultsView.verdict.accuracy.ratio")
                == 1.05
            )
        page.keyboard.press("Escape")
        page.locator(".menu-panel").wait_for(state="detached")
        for tile in tiles:
            page.click("#regression-results-model")
            settle_menu(page)
            assert tile["tile_id"] in page.locator(".menu-panel .menu-item").evaluate_all(
                "nodes => nodes.map(node => node.dataset.id)"
            )
            with page.expect_response(
                lambda response: "/api/forecast-view?" in response.url and "model=" in response.url
            ) as response:
                page.locator(f'.menu-item[data-id="{tile["tile_id"]}"]').click()
            assert response.value.status == 200, response.value.text()
            assert response.value.json()["model"] == tile["tile_id"]
            page.wait_for_function(
                "id => forecastTopics.get('regression').resultsView.model === id",
                arg=tile["tile_id"],
            )
            table_response = page.request.get(
                _base + "/api/forecast-table", params={"id": run_id, "model": tile["tile_id"]}
            )
            assert table_response.status == 200, table_response.text()
            assert table_response.json()["family"] == tile["family"]
            export = page.request.get(
                _base + page.locator("#regression-export").get_attribute("href")
            )
            assert export.status == 200, export.text()


def test_saved_tile_states_scores_and_token_colours(tmp_path):
    cases = [
        ("running", "running", None, "running", "--kingfisher-milk", "…"),
        ("pending", "pending", None, "pending", "--grey-border", "…"),
        ("better", "done", 0.8, "better", "--lichen", "+20%"),
        ("worse", "done", 1.2, "worse", "--stone-red-lift", "−20%"),
        ("identical", "done", 1.0, "identical", "--vanilla", "0%"),
        ("near_lower", "done", 0.996, "identical", "--vanilla", "0%"),
        ("near_upper", "done", 1.004, "identical", "--vanilla", "0%"),
        ("unknown", "done", None, "pending", "--grey-border", "…"),
        ("stopped", "stopped", None, "unknown", "--grey-border", "-"),
        ("cancelled", "cancelled", None, "unknown", "--grey-border", "-"),
        ("starting", "starting", None, "running", "--kingfisher-milk", "…"),
        ("queued", "queued", None, "running", "--kingfisher-milk", "…"),
        ("paused", "paused", None, "running", "--kingfisher-milk", "…"),
        ("pruned", "pruned", None, "pending", "--grey-border", "…"),
        ("pruned_better", "pruned", 0.8, "better", "--lichen", "+20%"),
        ("failed", "failed", None, "unknown", "--grey-border", "-"),
    ]
    with session(tmp_path, port=8098) as (page, _base):
        root = tmp_path / "forecast_root"
        run_id = make_saved_tiles(root)
        run = root / ".forecast-runs" / run_id
        request = json.loads((run / "request.json").read_text())
        request["families"] = [{"name": name} for name, *_ in cases]
        write_json(run / "request.json", request)
        families = {
            name: {"state": state, "verdict": {"accuracy": {"ratio": ratio}}}
            for name, state, ratio, *_ in cases
        }
        write_json(run / "status.json", {"state": "done", "families": families})
        write_json(run / "result.json", {"families": families})
        write_json(run / "recipe.json", {})
        open_tiles(page)
        assert page.locator('link[href="/ui/base.css"]').count() == 1
        for name, _, _, state, token, score in cases:
            tile = find_tile(page, name)
            assert tile.get_attribute("data-state") == state
            scores = tile.locator(".forecast-tile-score")
            assert scores.all_text_contents() == [score, score]
            if state == "identical":
                assert scores.first.get_attribute("title") == "same as naive"
            assert tile.evaluate(
                """(tile, token) => {
                    const root = getComputedStyle(document.documentElement);
                    const value = root.getPropertyValue(token).trim();
                    const probe = document.createElement('span');
                    probe.style.color = value;
                    document.body.append(probe);
                    const colour = getComputedStyle(probe).color;
                    probe.remove();
                    return value.length > 0 && getComputedStyle(tile).getPropertyValue('--tile-tone').trim() === value
                      && getComputedStyle(tile.querySelector('.forecast-tile-face')).borderTopColor === colour;
                }""",
                token,
            )
        pending = find_tile(page, "pending").locator(".forecast-tile-score").first
        assert pending.evaluate(
            "node => getComputedStyle(node).color === getComputedStyle(document.querySelector('.forecast-tile-dim')).color"
        )
        legend = page.locator("#regression-tiles .forecast-tiles-legend i")
        assert legend.evaluate_all("nodes => nodes.map(node => node.dataset.state)") == [
            "better",
            "identical",
            "running",
            "worse",
            "pending",
        ]
        assert legend.evaluate_all(
            "nodes => nodes.map(node => node.parentElement.textContent)"
        ) == ["better", "identical", "running", "worse", "pending or no test score"]
        for square in legend.all():
            state = square.get_attribute("data-state")
            assert square.evaluate("node => getComputedStyle(node).backgroundColor") == (
                find_tile(page, state)
                .locator(".forecast-tile-face")
                .first.evaluate("node => getComputedStyle(node).borderTopColor")
            )


def test_saved_run_start_times(tmp_path):
    with session(tmp_path, port=8098) as (page, _base):
        run_id = make_saved_tiles(tmp_path / "forecast_root")
        runs = [
            {"id": run_id, "state": "done", "started_at": 1793627178.5},
            {"id": "set-20261102-144637-8-weeks-client-groups-0", "state": "done"},
            {"id": "combined-group-20261030-090825", "state": "running"},
            {"id": "unknown-time", "state": "done"},
        ]
        run_root = tmp_path / "forecast_root" / ".forecast-runs"
        running_dir = run_root / runs[2]["id"]
        shutil.copytree(run_root / run_id, running_dir)

        def serve_running_state(route):
            response = route.fetch()
            route.fulfill(response=response, json={**response.json(), "state": "running"})

        page.route("**/api/forecast-run?**", serve_running_state)
        page.route("**/api/forecast-runs", lambda route: route.fulfill(json={"runs": runs}))
        open_tiles(page)
        expected = page.evaluate(
            """seconds => {
                const date = new Date(seconds * 1000);
                const pad = value => String(value).padStart(2, '0');
                return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
            }""",
            runs[0]["started_at"],
        )
        for run, full_time in zip(
            runs[:3],
            [expected, "2026-11-02 14:46:37", "2026-10-30 09:08:25"],
            strict=True,
        ):
            row = page.locator(f'.forecast-run-row[data-run-id="{run["id"]}"]')
            timestamp = row.locator(".forecast-run-subtitle")
            assert timestamp.inner_text() == full_time[2:-3]
            assert row.get_attribute("title") == f"{run['id']} | {full_time}"
            assert timestamp.bounding_box()["y"] > row.locator(".name").bounding_box()["y"]
            assert timestamp.evaluate("node => getComputedStyle(node).fontVariantNumeric") == (
                "tabular-nums"
            )
            assert row.evaluate("row => row.scrollWidth <= row.clientWidth")
            assert page.evaluate("run => forecastRunTime(run)", run) == full_time[2:-3]
        assert page.locator("#regression-running-list .forecast-run-subtitle").count() == 1
        assert page.locator('[data-run-id="unknown-time"] .forecast-run-subtitle').count() == 0
        for run in ({"id": "unknown-time"}, {"started_at": "invalid"}, {"id": "20269999-999999"}):
            assert page.evaluate("run => forecastRunTime(run)", run) == ""


def test_saved_run_short_titles_stable_across_paints(tmp_path):
    examples = [
        ({"id": "label-groups", "label": "8 weeks - client groups"}, "8w groups"),
        (
            {"id": "label-family", "label": "8 weeks - client groups - lightgbm"},
            "8w groups lgbm",
        ),
        ({"id": "label-total", "label": "12 weeks - total - XGBoost"}, "12w total xgb"),
        ({"id": "combined-h8_group-20261102-120000"}, "8w groups"),
        ({"id": "combined-h12_total-20261102-120000"}, "12w total"),
        ({"id": "combined-group-20261102-120000"}, "group"),
        ({"id": "combined-total-20261102-120000"}, "total"),
        (
            {"id": "warm-example", "families": {"lightgbm": {}}, "kind": "warm start"},
            "lgbm warm",
        ),
        (
            {"id": "many-models", "families": {"ridge": {}, "xgboost": {}}, "kind": "refit"},
            "2 models refit",
        ),
        ({"id": "search-example", "families": {"ridge": {}}, "kind": "search"}, "ridge search"),
        ({"id": "empty-example", "label": "   "}, "run"),
    ]
    with session(tmp_path, port=8098) as (page, _base):
        make_saved_tiles(tmp_path / "forecast_root")
        open_tiles(page)
        page.evaluate("clearInterval(forecastTopics.get('regression').pollTimer)")
        runs = [{**run, "state": "done"} for run, _name in examples]
        page.evaluate(
            """runs => {
                const topic = forecastTopics.get('regression');
                topic.runCache = new Map(runs.map(run => [run.id, run]));
                topic.paintRuns();
            }""",
            runs,
        )
        titles = []
        for run, name in examples:
            row = page.locator(f'.forecast-run-row[data-run-id="{run["id"]}"]')
            title = row.locator(".name").inner_text()
            code, actual_name = title.split(" | ")
            assert re.fullmatch("[0-9a-z]{5}", code)
            assert actual_name == name
            assert row.get_attribute("title").startswith(run["id"])
            assert page.evaluate("id => forecastRunCode(id)", run["id"]) == code
            titles.append(title)
        assert len({title.split(" | ")[0] for title in titles}) == len(examples)
        page.evaluate("forecastTopics.get('regression').paintRuns()")
        for (run, _name), title in zip(examples, titles, strict=True):
            row = page.locator(f'.forecast-run-row[data-run-id="{run["id"]}"]')
            assert row.locator(".name").inner_text() == title
        assert page.evaluate("forecastRunShortName({})") == "run"


def test_saved_run_labels_parts_stopped_and_view_caption(tmp_path):
    with session(tmp_path, port=8098) as (page, _base):
        root = tmp_path / "forecast_root"
        run_id = make_saved_tiles(root)
        part_id = "20261029-120000-part"
        run_dir = root / ".forecast-runs" / run_id
        shutil.copytree(run_dir, root / ".forecast-runs" / part_id)
        (run_dir / "families/xgboost").mkdir(parents=True, exist_ok=True)
        (run_dir / "families/xgboost/generation_curves.jsonl").write_text(
            json.dumps({"generation": 0, "metric": "rmse", "validation": [0.4, 0.3]}),
            encoding="utf-8",
        )
        status = json.loads((run_dir / "status.json").read_text())
        status["state"] = "stopped"
        status["families"]["xgboost"]["state"] = "stopped"
        write_json(run_dir / "status.json", status)
        page.route(
            "**/api/forecast-runs",
            lambda route: route.fulfill(
                json={
                    "runs": [
                        {
                            "id": run_id,
                            "label": "8 weeks - client groups - lightgbm",
                            "kind": "set",
                            "state": "stopped",
                        },
                        {"id": part_id, "part_of": run_id, "state": "done"},
                    ]
                }
            ),
        )
        open_tiles(page)
        rows = page.locator("#regression-runs-list .forecast-run-row")
        assert rows.count() == 1
        assert rows.first.locator(".name").inner_text().endswith(" | 8w groups lgbm")
        assert rows.first.locator(".forecast-run-status").inner_text() == "stopped"
        assert page.locator("#regression-search-status").inner_text() == "stopped"
        assert page.locator("#regression-run-stop").inner_text() == "stopped"
        assert find_tile(page, "xgboost").get_attribute("data-state") == "unknown"
        assert page.evaluate("forecastTileState({state: 'stopped'})") == "unknown"
        assert (
            find_tile(page, "xgboost")
            .locator(".forecast-tile-rounds .forecast-tile-chart-label > span")
            .first.inner_text()
            == "learning curve per round - validation error"
        )
        caption = page.locator("#regression-run-view-caption")
        for view in ("tables", "tiles"):
            page.locator(f"#regression-run-view-{view}").click()
            assert caption.is_visible()
            assert caption.inner_text() == (
                "tiles: one card per model family. tables: every family as a row, "
                "plus the best ten recipes."
            )
        removed_button_id = "-".join(("regression", "show", "parts"))
        assert page.locator(f"#{removed_button_id}").count() == 0
        part = page.locator(f'[data-run-id="{part_id}"]')
        assert part.count() == 0
        page.evaluate("id => forecastTopics.get('regression').selectRun(id)", part_id)
        assert rows.count() == 2
        assert part.locator(".name").inner_text().endswith(" | 3 models")
        assert part.is_visible()
        page.locator(f'[data-run-id="{run_id}"]').click()
        assert rows.count() == 1
        assert part.count() == 0


def make_search_curve(values: list[float]) -> list[dict]:
    return [
        {
            "generation": index + 1,
            "validation_error": value,
            "wape": value / 2,
            "mae": value * (index + 2) * 10,
            "rmse": value * (index + 3) * 10,
        }
        for index, value in enumerate(values)
    ]


def make_round_curves(ends: list[float], rounds: int = 6, recorded: bool = True) -> list[dict]:
    """per-round curves falling to each generation's end value; mae is wape times 100"""
    curves = []
    for index, end in enumerate(ends):
        wape = [end * (1 + (rounds - step) / rounds) for step in range(1, rounds + 1)]
        curves.append(
            {
                "generation": index + 1,
                "key": f"g{index}",
                "metric": "rmse",
                "validation": [value * 300 for value in wape],
                "curves": {"wape": wape, "mae": [value * 100 for value in wape]}
                if recorded
                else {},
            }
        )
    return curves


def make_live_tiles() -> dict:
    ridge = {
        "state": "done",
        "best_error": 0.2,
        "tile": {
            "validation_error": 0.2,
            "test_ratio": 0.71,
            "test_error": 0.182,
            "test_error_metric": "wape",
            "test_naive_wape": 0.256,
            "search_curve": make_search_curve([0.5, 0.3, 0.2]),
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
            "generation": 3,
        },
    }
    worse = copy.deepcopy(ridge)
    worse["best_error"] = 0.35
    worse["tile"].update(
        {
            "test_ratio": 1.12,
            "validation_error": 0.35,
            "genome": {"params": {"season": 52}},
            "search_curve": make_search_curve([0.35]),
            "search_single": True,
            "generation": None,
        }
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
                "tile": {
                    "search_curve": make_search_curve([0.8, 0.7, 0.6, 0.5, 0.4]),
                    "round_curves": make_round_curves([0.4, 0.3, 0.2]),
                    "folds": [],
                    "elapsed": 420,
                    "generation": 5,
                },
            },
            "snaive": worse,
            "lightgbm": {
                "state": "pruned",
                "pruned_reason": "paired validation error behind ridge",
                "tile": {
                    "search_curve": make_search_curve([0.7, 0.55]),
                    "round_curves": make_round_curves([0.3, 0.25], recorded=False),
                    "folds": [],
                    "elapsed": 120,
                    "generation": 2,
                },
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
        response = with_error_metrics(snapshot)
        if snapshot.get("metric_registry"):
            response["metric_registry"] = snapshot["metric_registry"]
        route.fulfill(json=response)

    page.route("**/api/forecast-run?**", read_run)
    return calls


def find_tile(page, family: str):
    return page.locator(f'#regression-tiles .forecast-tile[data-family="{family}"]')


def save_tiles(page, name: str) -> None:
    pass


def select_chart_metrics(page, selected: list[str]) -> None:
    page.locator("#regression-error-metrics").click()
    menu = page.locator(".menu-panel")
    for metric in selected:
        item = menu.locator(f'.menu-item[data-id="{metric}"]')
        if "on" not in item.get_attribute("class").split():
            item.click()
    for metric in menu.locator(".menu-item.on").evaluate_all(
        "items => items.map(item => item.dataset.id)"
    ):
        if metric not in selected:
            menu.locator(f'.menu-item[data-id="{metric}"]').click()
    page.keyboard.press("Escape")


def select_tile_metric(page, metric: str) -> None:
    page.locator("#regression-tiles-metric").click()
    page.locator(f'.menu-panel .menu-item[data-id="{metric}"]').click()


def read_chart_scale(chart) -> tuple[float, float]:
    return tuple(float(chart.get_attribute(f"data-scale-{edge}")) for edge in ("min", "max"))


def assert_tall_plots(page) -> None:
    plots = page.locator(".forecast-tile-front svg")
    assert plots.count() >= 2
    for plot in plots.all():
        dimensions = plot.evaluate(
            """svg => {
                const axis = svg.querySelector('.forecast-tile-axis');
                const scale = svg.getScreenCTM().d;
                const box = svg.getBoundingClientRect();
                return {height: (Number(axis.getAttribute('y1')) - 6) * scale,
                    labelsFit: [...svg.querySelectorAll('text')].every(label => {
                        const rect = label.getBoundingClientRect();
                        return rect.top >= box.top && rect.bottom <= box.bottom;
                    })};
            }"""
        )
        assert dimensions["height"] >= 120 - 0.01
        assert dimensions["labelsFit"]
    for tile in page.locator(".forecast-tile").all():
        box = tile.bounding_box()
        assert box["height"] > box["width"]
        for side in ("front", "back"):
            face = tile.locator(f".forecast-tile-{side}")
            assert face.evaluate("node => node.offsetHeight") == pytest.approx(box["height"], abs=1)
            assert face.evaluate("node => node.scrollHeight <= node.clientHeight")


def test_chart_faces_compare_running_done_pruned_and_one_shot_generations(tmp_path):
    snapshot = make_live_tiles()
    with session(tmp_path, port=8098) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        assert_tall_plots(page)
        charts = page.locator("#regression-tiles .forecast-tile-front .forecast-tile-chart")
        scales = charts.evaluate_all(
            "nodes => nodes.map(node => [node.dataset.scaleMin, node.dataset.scaleMax])"
        )
        assert len(scales) == 4 and all(scale == scales[0] for scale in scales)
        assert float(scales[0][0]) <= 0.2 and float(scales[0][1]) >= 0.8
        for name, generation, value in (
            ("xgboost", 5, 0.4),
            ("ridge", 3, 0.2),
            ("lightgbm", 2, 0.55),
            ("snaive", 1, 0.35),
        ):
            chart = find_tile(page, name).locator(".forecast-tile-front .forecast-tile-chart")
            assert chart.get_attribute("data-chart-metric") == "validation_error"
            marker = chart.locator(".forecast-tile-current")
            assert marker.count() == 1
            assert marker.get_attribute("data-generation") == str(generation)
            assert float(marker.get_attribute("data-value")) == value
            assert chart.get_attribute("data-generation-max") == str(generation)
            axis = chart.locator(".forecast-tile-axis")
            edge = "x2" if generation > 1 else "x1"
            assert float(marker.get_attribute("cx")) == pytest.approx(
                float(axis.get_attribute(edge))
            )
            labels = chart.locator("svg text").all_text_contents()
            assert labels == (
                ["1", "generation", str(generation)] if generation > 1 else ["1", "generation"]
            )
            assert "validation error" in chart.locator(".forecast-tile-chart-label").inner_text()
            assert f"{value * 100:.1f}%" in chart.inner_text()
        assert find_tile(page, "lightgbm").locator(".forecast-tile-pruned").count() == 1
        assert find_tile(page, "xgboost").locator(".forecast-tile-search-line").count() == 1
        assert find_tile(page, "ridge").locator(".forecast-tile-search-line").count() == 1
        assert (
            "test forecast not saved for this family"
            not in page.locator("#regression-tiles").inner_text()
        )
        save_tiles(page, "four-family-generations")


def test_y_metric_controls_both_charts_independently_of_error_metric_picker(tmp_path):
    snapshot = make_live_tiles()
    with session(tmp_path, port=8098) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        tile = find_tile(page, "xgboost")
        chart = tile.locator(".forecast-tile-front .forecast-tile-chart")
        rounds = tile.locator(".forecast-tile-front .forecast-tile-rounds")
        picker = page.locator("#regression-tiles-metric")
        assert "validation error" in picker.inner_text()
        assert picker.evaluate("node => node.closest('.forecast-tiles-bar') !== null")
        picker_box = picker.bounding_box()
        flip_box = page.locator("#regression-tiles-flip").bounding_box()
        assert picker_box["x"] + picker_box["width"] <= flip_box["x"]
        assert abs(picker_box["y"] - flip_box["y"]) < 1
        picker.click()
        settle_menu(page)
        menu = page.locator(".menu-panel")
        assert menu.locator(".menu-item").evaluate_all(
            "items => items.map(item => item.dataset.id)"
        ) == ["validation_error", "wape", "mae"]
        assert menu.locator('input[type="checkbox"]').count() == 0
        page.keyboard.press("Escape")
        assert rounds.get_attribute("data-chart-metric") == "rmse"
        assert rounds.get_attribute("data-recorded") == "true"
        assert "validation error" in rounds.locator(".forecast-tile-chart-label").inner_text()
        assert rounds.locator(".forecast-tile-chart-label > span").first.inner_text() == (
            "learning curve per round - validation error"
        )
        assert rounds.locator(".forecast-tile-round-current").count() == 1
        assert rounds.locator(".forecast-tile-round-line").count() == 2
        assert (
            rounds.locator(".forecast-tile-round-current").get_attribute("data-generation") == "3"
        )
        for shown, value, round_value in (("wape", 0.2, "20.0%"), ("mae", 24, "20")):
            select_tile_metric(page, shown)
            assert rounds.get_attribute("data-chart-metric") == shown
            assert chart.get_attribute("data-chart-metric") == shown
            assert float(
                chart.locator(".forecast-tile-current").get_attribute("data-value")
            ) == pytest.approx(value)
            assert f"gen 3 · round 6: {round_value}\n" in rounds.inner_text()
        select_chart_metrics(page, ["mape"])
        assert chart.get_attribute("data-chart-metric") == "mae"
        assert rounds.get_attribute("data-chart-metric") == "mae"
        select_tile_metric(page, "wape")
        fallback = find_tile(page, "lightgbm").locator(".forecast-tile-rounds")
        assert fallback.get_attribute("data-chart-metric") == "rmse"
        assert fallback.get_attribute("data-recorded") == "false"
        assert "own loss, wape not recorded" in fallback.inner_text()
        ridge = find_tile(page, "ridge").locator(".forecast-tile-rounds")
        assert ridge.inner_text().endswith("no round curve saved for this family")
        save_tiles(page, "round-curves")


def test_y_metric_shared_recorded_ranges_and_separate_fallback_ranges(tmp_path):
    snapshot = make_live_tiles()
    families = snapshot["families"]
    families["ridge"]["tile"]["round_curves"] = make_round_curves([0.8, 0.6], rounds=4)
    families["snaive"]["tile"]["round_curves"] = make_round_curves([2, 1], rounds=3, recorded=False)
    for curve in families["snaive"]["tile"]["round_curves"]:
        curve["metric"] = "mae"
    for name, values in (("lightgbm", [30, 20]), ("snaive", [4])):
        families[name]["tile"]["search_curve"] = [
            {"generation": index + 1, "validation_error": value}
            for index, value in enumerate(values)
        ]
    with session(tmp_path, port=8098) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        native_scales = [
            read_chart_scale(find_tile(page, name).locator(".forecast-tile-rounds"))
            for name in ("xgboost", "ridge", "lightgbm")
        ]
        assert native_scales[0] == native_scales[1] == native_scales[2]
        other_native = find_tile(page, "snaive").locator(".forecast-tile-rounds")
        assert other_native.get_attribute("data-chart-metric") == "mae"
        assert read_chart_scale(other_native) != native_scales[0]
        select_tile_metric(page, "wape")
        for selector in (".forecast-tile-chart", ".forecast-tile-rounds"):
            recorded = [find_tile(page, name).locator(selector) for name in ("xgboost", "ridge")]
            scales = [read_chart_scale(chart) for chart in recorded]
            assert scales[0] == scales[1]
            assert all(chart.get_attribute("data-chart-metric") == "wape" for chart in recorded)
            fallback = [find_tile(page, name).locator(selector) for name in ("lightgbm", "snaive")]
            fallback_scales = [read_chart_scale(chart) for chart in fallback]
            assert fallback_scales[0] != fallback_scales[1]
            assert all(scale != scales[0] for scale in fallback_scales)
            for name, chart in zip(("lightgbm", "snaive"), fallback, strict=True):
                assert "own loss, wape not recorded" in chart.inner_text()
                expected = "rmse" if name == "lightgbm" else "mae"
                if selector == ".forecast-tile-chart":
                    expected = "validation_error"
                assert chart.get_attribute("data-chart-metric") == expected
        for name, rounds in (("xgboost", 6), ("ridge", 4), ("snaive", 3)):
            chart = find_tile(page, name).locator(".forecast-tile-rounds")
            assert chart.locator("svg text").all_text_contents() == ["1", "round", str(rounds)]
            points = chart.locator(".forecast-tile-round-current").get_attribute("points")
            last_x = float(points.split()[-1].split(",")[0])
            axis_end = float(chart.locator(".forecast-tile-axis").get_attribute("x2"))
            assert last_x == pytest.approx(axis_end)


def test_y_metric_persisted_for_session_across_reload(tmp_path):
    snapshot = make_live_tiles()
    key = "smolsmort:forecast:regression:tile-metric"
    with session(tmp_path, port=8098) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        select_tile_metric(page, "mae")
        assert page.evaluate("key => sessionStorage.getItem(key)", key) == "mae"
        page.reload()
        open_tiles(page)
        assert "mae" in page.locator("#regression-tiles-metric").inner_text()
        chart = find_tile(page, "xgboost").locator(".forecast-tile-chart")
        assert chart.get_attribute("data-chart-metric") == "mae"
        page.evaluate("key => sessionStorage.removeItem(key)", key)
        page.reload()
        open_tiles(page)
        assert "validation error" in page.locator("#regression-tiles-metric").inner_text()
        assert chart.get_attribute("data-chart-metric") == "validation_error"


def test_y_metric_open_menu_and_focus_survived_polling(tmp_path):
    snapshot = make_live_tiles()
    with session(tmp_path, port=8098) as (page, _base):
        calls = mock_live_tiles(page, snapshot)
        open_tiles(page)
        select_tile_metric(page, "wape")
        picker = page.locator("#regression-tiles-metric")
        picker.focus()
        page.keyboard.press("Enter")
        settle_menu(page)
        menu = page.locator(".menu-panel")
        option = menu.locator('.menu-item[data-id="mae"]')
        for _ in range(3):
            page.keyboard.press("ArrowDown")
        assert option.evaluate("node => node === document.activeElement")
        before = len(calls)
        snapshot["families"]["xgboost"]["tile"]["generation"] = 6
        snapshot["families"]["xgboost"]["tile"]["search_curve"].append(
            {"generation": 6, "validation_error": 0.3, "wape": 0.15, "mae": 21}
        )
        page.wait_for_function(
            "() => document.querySelector('[data-family=\"xgboost\"] .forecast-tile-current')"
            "?.dataset.generation === '6'"
        )
        assert len(calls) > before
        assert menu.is_visible()
        assert option.evaluate("node => node === document.activeElement")
        assert "wape" in picker.inner_text()
        page.keyboard.press("Enter")
        assert not menu.count()
        assert "mae" in picker.inner_text()
        assert picker.evaluate("node => node === document.activeElement")


def test_y_metric_still_worked_when_storage_was_blocked(tmp_path):
    with session(tmp_path, port=8098) as (page, _base):
        mock_live_tiles(page, make_live_tiles())
        page.add_init_script(
            """(() => {
                for (const method of ['getItem', 'setItem']) {
                    const original = Storage.prototype[method];
                    Storage.prototype[method] = function(key, ...values) {
                        if (key === 'smolsmort:forecast:regression:tile-metric') {
                            throw new DOMException('storage blocked', 'SecurityError');
                        }
                        return original.call(this, key, ...values);
                    };
                }
            })()"""
        )
        page.reload()
        open_tiles(page)
        select_tile_metric(page, "wape")
        chart = find_tile(page, "xgboost").locator(".forecast-tile-chart")
        assert chart.get_attribute("data-chart-metric") == "wape"


def test_chart_broad_phase_single_value_and_missing_curve_message(tmp_path):
    snapshot = make_live_tiles()
    snapshot["families"]["xgboost"]["tile"].update(
        search_curve=make_search_curve([0.45]), search_single=True, generation=None
    )
    snapshot["families"]["ridge"]["tile"]["search_curve"] = []
    with session(tmp_path, port=8098) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        running = find_tile(page, "xgboost").locator(".forecast-tile-chart")
        assert running.locator(".forecast-tile-current").count() == 1
        assert "45.0%" in running.inner_text()
        missing = find_tile(page, "ridge").locator(".forecast-tile-chart")
        assert "no search curve saved for this family" in missing.inner_text()
        assert missing.locator("svg").count() == 0


def make_live_legacy_tiles() -> dict:
    return {
        "id": "20261030-130000-legacy-live",
        "version": 1,
        "kind": "search",
        "state": "running",
        "family": "xgboost",
        "families": {},
        "budget": {"time_cap": 1200, "max_generations": 10, "plateau": 5},
        "progress": {"generation": 1, "elapsed": 42},
        "generations": [
            {"generation": 0, "best": 0.5, "metrics": {"wape": 0.5, "mae": 12}},
            {"generation": 1, "best": 0.25, "metrics": {"wape": 0.25, "mae": 8}},
        ],
        "generation_curves": [
            {"generation": 0, "key": "legacy-0", "metric": "rmse", "validation": [20, 15]},
            {"generation": 1, "key": "legacy-1", "metric": "rmse", "validation": [15, 10]},
        ],
        "leaderboard": [
            {
                "key": "legacy-1",
                "fitness": 0.25,
                "generation": 1,
                "genome": {"params": {"max_depth": 2}},
                "metrics": {"wape": 0.25, "mae": 8},
            }
        ],
        "leaderboard_generation": 1,
    }


def test_live_legacy_chart_polled_generations_and_kept_menu_focus(tmp_path):
    snapshot = make_live_legacy_tiles()
    with session(tmp_path, port=8098) as (page, _base):
        calls = mock_live_tiles(page, snapshot)
        open_tiles(page)
        tile = find_tile(page, "xgboost")
        chart = tile.locator(".forecast-tile-chart")
        marker = chart.locator(".forecast-tile-current")
        assert marker.get_attribute("data-generation") == "2"
        assert marker.get_attribute("data-value") == "0.25"
        assert chart.locator(".forecast-tile-search-line").count() == 1
        rounds = tile.locator(".forecast-tile-rounds")
        assert rounds.get_attribute("data-chart-metric") == "rmse"
        assert (
            rounds.locator(".forecast-tile-round-current").get_attribute("data-generation") == "2"
        )
        menu = tile.locator('.forecast-tile-front details[data-menu="param"]')
        summary = menu.locator("summary")
        summary.focus()
        page.keyboard.press("Enter")
        assert menu.get_attribute("open") is not None
        before = len(calls)
        snapshot["progress"]["generation"] = 2
        snapshot["generations"].append(
            {"generation": 2, "best": 0.2, "metrics": {"wape": 0.2, "mae": 6}}
        )
        snapshot["generation_curves"].append(
            {"generation": 2, "key": "legacy-2", "metric": "rmse", "validation": [10, 7]}
        )
        page.wait_for_function(
            "() => document.querySelector('#regression-tiles .forecast-tile-current')"
            "?.dataset.generation === '3'"
        )
        assert len(calls) > before
        assert marker.get_attribute("data-value") == "0.2"
        assert (
            rounds.locator(".forecast-tile-round-current").get_attribute("data-generation") == "3"
        )
        assert menu.get_attribute("open") is not None
        assert summary.evaluate("node => node === document.activeElement")


def test_live_legacy_chart_without_generations_showed_single_saved_value(tmp_path):
    snapshot = make_live_legacy_tiles()
    snapshot.update(generations=[], generation_curves=[], progress={"elapsed": 10})
    with session(tmp_path, port=8098) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        chart = find_tile(page, "xgboost").locator(".forecast-tile-chart")
        assert chart.locator(".forecast-tile-current").get_attribute("data-value") == "0.25"
        assert chart.locator(".forecast-tile-current-value").text_content() == "single value: 25.0%"
        assert chart.locator(".forecast-tile-search-line").count() == 0
        rounds = find_tile(page, "xgboost").locator(".forecast-tile-rounds")
        assert rounds.inner_text().endswith("no round curve saved for this family")


@pytest.mark.parametrize("metric", ["validation_error", "wape"])
def test_chart_missing_latest_metric_did_not_highlight_older_generation(tmp_path, metric):
    snapshot = make_live_tiles()
    for name in ("ridge", "lightgbm"):
        snapshot["families"][name]["tile"]["search_curve"][-1][metric] = None
    with session(tmp_path, port=8098) as (page, _base):
        mock_live_tiles(page, snapshot)
        open_tiles(page)
        select_tile_metric(page, metric)
        for name, label in (
            ("ridge", "gen 3: not saved"),
            ("lightgbm", "pruned · gen 2: not saved"),
        ):
            chart = find_tile(page, name).locator(".forecast-tile-chart")
            assert chart.locator(".forecast-tile-current").count() == 0
            assert chart.locator(".forecast-tile-pruned").count() == 0
            assert chart.locator(".forecast-tile-current-value").text_content() == label
            assert chart.locator(".forecast-tile-search-point").count() > 0


def test_run_panels_fit_config_and_keep_layout_across_views(tmp_path):
    with session(tmp_path, width=1600, port=0) as (page, _base):
        make_saved_tiles(tmp_path / "forecast_root")
        open_tiles(page)
        progress = page.locator(".forecast-run-progress-panel:has(#regression-run-title)")
        config = page.locator("#regression-config")
        config.wait_for()
        family_table = page.locator("#regression-run-tables #regression-family-progress")
        assert not family_table.is_visible()
        assert progress.locator("table").count() == 0
        assert config.locator("table").count() == 0
        assert family_table.locator("tbody tr").count() == 3
        progress_box = progress.bounding_box()
        config_box = config.bounding_box()
        assert abs(progress_box["y"] - config_box["y"]) <= 2
        assert config_box["x"] >= progress_box["x"] + progress_box["width"]
        heads = config.locator(".dropdown-head")
        assert heads.count() == 6
        assert len({round(head.bounding_box()["y"]) for head in heads.all()}) == 2
        for view in ("tables", "tiles"):
            page.locator(f"#regression-run-view-{view}").click()
            assert family_table.is_visible() == (view == "tables")
            if view == "tables":
                toggle_box = page.locator("#regression-run-view").bounding_box()
                assert family_table.bounding_box()["y"] > toggle_box["y"] + toggle_box["height"]
            for panel, before in ((progress, progress_box), (config, config_box)):
                after = panel.bounding_box()
                for dimension in ("x", "y", "width", "height"):
                    assert after[dimension] == pytest.approx(before[dimension], abs=2)
        page.set_viewport_size({"width": 800, "height": 1000})
        progress_box = progress.bounding_box()
        config_box = config.bounding_box()
        assert config_box["y"] >= progress_box["y"] + progress_box["height"]
        assert config_box["x"] == pytest.approx(progress_box["x"], abs=2)
        page.evaluate("forecastTopics.get('regression').paintRunDetail(null)")
        assert not page.locator("#regression-run-actions").is_visible()
        assert config.is_visible()
        assert page.locator("#regression-config-mode-new").get_attribute("aria-pressed") == "true"
        page.set_viewport_size({"width": 1600, "height": 1000})
        assert config.bounding_box()["x"] >= progress.bounding_box()["x"]


def test_tiles_read_saved_winner_artifacts_and_kept_tables(tmp_path):
    with session(tmp_path, port=8098) as (page, _base):
        make_saved_tiles(tmp_path / "forecast_root")
        with page.expect_response("**/api/forecast-run?**") as response:
            open_tiles(page)
        data = response.value.json()
        tile_data = data["families"]["ridge"]["tile"]
        assert tile_data["search_curve"][-1]["validation_error"] == 0.2
        assert "chart" not in tile_data
        assert page.locator("#regression-run-view-tiles").get_attribute("aria-pressed") == "true"
        assert page.locator("#regression-tiles .forecast-tile").count() == 3
        assert not page.locator("#regression-family-progress").is_visible()
        save_tiles(page, "saved-charts")
        ridge = find_tile(page, "ridge")
        front = ridge.locator(".forecast-tile-front")
        assert "test ratio: 0.71" in front.inner_text()
        assert front.locator(".forecast-tile-current").get_attribute("data-value") == "0.2"
        assert front.locator(".forecast-tile-split").count() == 0
        assert "no search curve saved for this family" in find_tile(page, "snaive").inner_text()
        assert (
            "test forecast not saved for this family"
            not in page.locator("#regression-tiles").inner_text()
        )
        assert find_tile(page, "xgboost").get_attribute("data-state") == "pending"
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
        assert "wape per validation window; mark = naive on test" in back.inner_text()
        fold = back.locator('.forecast-tile-bar[data-label="fold 1"]')
        assert fold.get_attribute("aria-label") == "fold 1: wape 19.0%; lower is better"
        assert float(
            fold.locator(".forecast-tile-fill").evaluate("node => parseFloat(node.style.width)")
        ) == pytest.approx(0.19 / 0.26 * 100)
        save_tiles(page, "saved-folds")
        page.locator("#regression-run-view-tables").click()
        assert page.locator("#regression-family-progress").is_visible()
        assert page.locator("#regression-family-progress tbody tr").count() == 3
        assert not page.locator("#regression-tiles").is_visible()
        page.locator("#regression-run-view-tiles").click()
        assert page.locator("#regression-tiles").is_visible()


def test_tiles_polled_states_without_losing_flip_menu_or_focus(tmp_path):
    snapshot = make_live_tiles()
    with session(tmp_path, port=8098) as (page, _base):
        calls = mock_live_tiles(page, snapshot)
        open_tiles(page)
        families = page.locator("#regression-tiles-grid .forecast-tile").evaluate_all(
            "tiles => tiles.map(tile => tile.dataset.family)"
        )
        assert families[:2] == ["ridge", "snaive"]
        save_tiles(page, "live-charts")
        for name, score, color in (
            ("ridge", "test ratio: 0.71", "rgb(199, 237, 95)"),
            ("snaive", "test ratio: 1.12", "rgb(173, 121, 111)"),
            ("xgboost", "test ratio: -", "rgb(119, 183, 199)"),
            ("lightgbm", "test ratio: -", None),
        ):
            front = find_tile(page, name).locator(".forecast-tile-front")
            assert score in front.inner_text()
            if color is not None:
                assert front.evaluate("node => getComputedStyle(node).borderTopColor") == color
            else:
                assert find_tile(page, name).get_attribute("data-state") == "pending"
        running = find_tile(page, "xgboost").locator(".forecast-tile-front")
        assert running.locator(".forecast-tile-search-line").count() == 1
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
        assert fold.get_attribute("aria-label") == "fold 1: wape 19.0%; lower is better"
        assert float(
            fold.locator(".forecast-tile-fill").evaluate("node => parseFloat(node.style.width)")
        ) == pytest.approx(0.19 / 0.26 * 100)
        origin = back.locator(
            '.forecast-tile-bar[data-label="2025-01-20"] .forecast-tile-bar-label'
        )
        assert origin.evaluate("node => node.scrollWidth <= node.clientWidth")
        assert "wape per validation window; mark = naive on test" in back.inner_text()
        save_tiles(page, "live-folds")
        menu = back.locator('details[data-menu="feat"]')
        summary = menu.locator("summary")
        summary.focus()
        page.keyboard.press("Enter")
        assert menu.get_attribute("open") is not None
        before = len(calls)
        snapshot["families"]["xgboost"]["tile"]["generation"] = 6
        snapshot["families"]["xgboost"]["tile"]["search_curve"].append(
            {"generation": 6, "validation_error": 0.3, "wape": 0.15, "mae": 21}
        )
        snapshot["families"]["xgboost"]["tile"]["folds"] = [{"label": "fold 1", "error": 0.633}]
        snapshot["families"]["ridge"]["tile"]["test_ratio"] = 0.69
        page.wait_for_function(
            "() => document.querySelector('[data-family=\"xgboost\"] .forecast-tile-front')"
            ".textContent.includes('gen 6')"
        )
        assert len(calls) > before
        assert running.locator(".forecast-tile-current").get_attribute("data-generation") == "6"
        assert running.locator(".forecast-tile-current").get_attribute("data-value") == "0.3"
        assert (
            ridge.locator('.forecast-tile-front [data-metric-id="test_ratio"]').inner_text()
            == "test ratio: 0.69"
        )
        assert flip.get_attribute("aria-pressed") == "true"
        assert menu.get_attribute("open") is not None
        assert summary.evaluate("node => node === document.activeElement")
        assert back.locator(".forecast-tile-bars").get_attribute("data-scale-max") == "0.64"
        assert float(
            back.locator('.forecast-tile-bar[data-label="fold 1"] .forecast-tile-fill').evaluate(
                "node => parseFloat(node.style.width)"
            )
        ) == pytest.approx(0.19 / 0.64 * 100)
        page.keyboard.press("Enter")
        assert menu.get_attribute("open") is None
        flip.click()
        assert flip.get_attribute("aria-pressed") == "false"
        assert ridge.locator(".forecast-tile-back").evaluate("node => node.inert")


@pytest.mark.parametrize("seasonal", [False, True])
def test_back_folds_fill_each_family_on_one_wape_scale(tmp_path, seasonal):
    with session(tmp_path, port=8098) as (page, _base):
        root = tmp_path / "forecast_root"
        run_id = make_saved_tiles(root)
        run = root / ".forecast-runs" / run_id
        scores = {
            "ridge": [0.193, 0.21, 0.24, *([0.266] if seasonal else [])],
            "xgboost": [0.35, 0.29],
            "snaive": [0.42],
        }
        origins = ["2023-12-04", "2024-01-01", "2024-02-05", "2024-03-04"]
        result = json.loads((run / "result.json").read_text())
        if seasonal:
            result["holdout"] = {"validation_origins": origins}
        result["winner"]["verdict"]["accuracy"]["baseline_error"] = 0.88
        result["families"]["xgboost"]["state"] = "running"
        write_json(run / "result.json", result)
        status = json.loads((run / "status.json").read_text())
        status["state"] = status["families"]["xgboost"]["state"] = "running"
        write_json(run / "status.json", status)
        for name, values in scores.items():
            write_json(
                run / "families" / name / "leaderboard.json",
                [
                    {"fitness": 0.9, "metrics": {"fold_scores": [0.9] * 4}},
                    {
                        "fitness": sum(values) / len(values),
                        "genome": {"family": name, "params": {}},
                        "metrics": {"fold_scores": values, "fold_count": len(values)},
                    },
                ],
            )
        open_tiles(page)
        select_chart_metrics(page, ["validation_error"])
        page.locator("#regression-tiles-flip").click()
        for name, values in scores.items():
            back = find_tile(page, name).locator(".forecast-tile-back")
            bars = back.locator(".forecast-tile-bars")
            assert bars.get_attribute("data-scale-max") == "0.88"
            assert bars.locator(".forecast-tile-bar").count() == len(values) + 1
            assert bars.evaluate(
                "node => [...node.children].every(row => "
                "row.getBoundingClientRect().bottom <= node.getBoundingClientRect().bottom + 1)"
            )
            for index, value in enumerate(values):
                label = origins[index] if seasonal else f"fold {index + 1}"
                bar = bars.locator(f'.forecast-tile-bar[data-label="{label}"]')
                assert bar.locator(".forecast-tile-bar-value").inner_text() == f"{value * 100:.1f}%"
                assert float(
                    bar.locator(".forecast-tile-fill").evaluate(
                        "node => parseFloat(node.style.width)"
                    )
                ) == pytest.approx(value / 0.88 * 100, abs=0.001)
                assert bar.locator(".forecast-tile-mark").count() == 0
            assert "wape per validation window; mark = naive on test" in back.inner_text()
        ridge = find_tile(page, "ridge")
        test = ridge.locator('.forecast-tile-back .forecast-tile-bar[data-label="test"]')
        assert test.locator(".forecast-tile-bar-value").inner_text() == "18.2%"
        assert test.locator(".forecast-tile-mark").evaluate("node => node.style.left") == "100%"
        assert test.locator(".forecast-tile-mark").get_attribute("title") == "naive test wape 88.0%"
        assert page.locator("#regression-tiles .forecast-tile-mark").count() == 1
        assert ridge.locator(".forecast-tile-back .forecast-tile-score").inner_text() == "+29%"
        save_tiles(page, "seasonal-wape-folds" if seasonal else "block-wape-folds")


def test_tiles_mobile_layout_motion_and_palette_tokens(tmp_path):
    with session(tmp_path, width=390, port=8098) as (page, _base):
        mock_live_tiles(page, make_live_tiles())
        open_tiles(page)
        tiles = page.locator("#regression-tiles-grid .forecast-tile")
        boxes = [tile.bounding_box() for tile in tiles.all()]
        assert all(box and abs(box["width"] - 306) < 1 for box in boxes)
        assert all(box["height"] > box["width"] for box in boxes)
        assert_tall_plots(page)
        assert all(box["x"] >= 0 and box["x"] + box["width"] <= 390 for box in boxes)
        assert all(a["y"] + a["height"] <= b["y"] for a, b in pairwise(boxes))
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#regression-tiles-metric").click()
        menu_box = page.locator(".menu-panel").bounding_box()
        assert menu_box["x"] >= 0 and menu_box["x"] + menu_box["width"] <= 390
        page.locator('.menu-panel .menu-item[data-id="wape"]').click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert all(abs(tile.bounding_box()["width"] - 306) < 1 for tile in tiles.all())
        assert_tall_plots(page)
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
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        save_tiles(page, "mobile-light-tokens")
        page.locator("#regression-tiles-flip").click()
        assert tiles.first.locator(".forecast-tile-front").evaluate("node => node.inert")


def test_legacy_tiles_loaded_saved_search_curves_and_kept_live_metadata(tmp_path):
    with session(tmp_path, port=8098) as (page, _base):
        write_multifamily_run(tmp_path / "forecast_root", version=1)
        with page.expect_response("**/api/forecast-view?**") as response:
            open_tiles(page)
        assert list(response.value.json()["tiles"]) == ["xgboost"]
        tile = find_tile(page, "xgboost")
        tile.locator(".forecast-tile-front .forecast-tile-search").wait_for()
        assert (
            tile.locator('.forecast-tile-front [data-metric-id="test_ratio"]').inner_text()
            == "test ratio: 0.95"
        )
        assert tile.locator(".forecast-tile-front .forecast-tile-current").count() == 1
        assert tile.locator(".forecast-tile-search-line").count() == 1
        assert [
            point["generation"]
            for point in response.value.json()["tiles"]["xgboost"]["search_curve"]
        ] == [1, 2]
        value = page.evaluate(
            """() => forecastTileEntries({version: 1, state: 'running',
              families: {xgboost: {state: 'running'}}, progress: {elapsed: 42, generation: 5},
              leaderboard: [{genome: {params: {max_depth: 2}}, fitness: 0.3}]})[0]"""
        )
        assert value["elapsed"] == 42 and value["generation"] == 5
        assert value["genome"]["params"] == {"max_depth": 2}
        assert page.evaluate("forecastTileState({state: 'paused'})") == "running"
