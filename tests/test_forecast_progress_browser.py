"""candidate progress and saved loss history in headless chromium"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("playwright")
pytest.importorskip("xgboost")
pytest.importorskip("duckdb")

from forecast_ui_audit import assert_family_layout
from test_forecast_ui_browser import session, show_tab, switch_to_regression

from smolsmort.forecast.tables import write_table


def write_multifamily_run(root, *, version=2):
    """persist a small synthetic result so browser checks exercise the real routes"""
    import numpy as np

    run_id = "20261016-120000-synthetic" if version == 2 else "20261015-120000-legacy"
    folder = root / ".forecast-runs" / run_id
    prepared = root / ".forecast-cache" / run_id
    folder.mkdir(parents=True)
    prepared.mkdir(parents=True)

    def save_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    spec = {
        "mode": "timeseries",
        "task": "regression",
        "step": "week",
        "horizon": 2,
        "columns": [
            {"name": "day", "role": "time"},
            {"name": "units", "role": "target", "aggregation": "sum"},
        ],
    }
    names = ("xgboost", "ridge", "snaive")
    fitness = {"xgboost": 0.12, "ridge": 0.18, "snaive": 0.31}
    verdict = {"accuracy": {"ratio": 0.95}}
    ensemble_verdict = {"accuracy": {"ratio": 1.05}}
    families = {
        name: {
            "state": "done",
            "phase": "finished",
            "candidates_done": 4 - index,
            "candidates_expected": 0,
            "best_error": fitness[name],
            "pruned_reason": None,
        }
        for index, name in enumerate(names)
    }
    genome = {
        "families": ["lag"],
        "objective": "squared",
        "transform": "none",
        "params": {"max_depth": 2, "eta": 0.1},
        "family": "xgboost",
    }
    request = {
        "spec": spec,
        "prepared": str(prepared),
        "budget": {"max_generations": 2, "plateau": 2, "time_cap": 60},
    }
    if version == 2:
        request.update(
            version=2,
            families=[{"name": name} for name in names],
            time_budget_s=60,
        )
    else:
        genome.pop("family")
    save_json(folder / "request.json", request)
    save_json(
        folder / "status.json",
        {
            "state": "done",
            "stopped": "families complete" if version == 2 else "max_generations",
            "verdict": verdict,
            **({"families": families} if version == 2 else {}),
        },
    )
    save_json(folder / "recipe.json", {"genome": genome})
    result = {"verdict": verdict, "model_error": 0.14}
    if version == 2:
        result.update(
            families={
                name: {
                    "state": "done",
                    "fitness": fitness[name],
                    "candidates": families[name]["candidates_done"],
                    "pruned_reason": None,
                }
                for name in names
            },
            winner={"family": "xgboost", "verdict": verdict},
            ensemble={
                "members": ["xgboost", "ridge"],
                "fitness": 0.15,
                "verdict": ensemble_verdict,
            },
        )
    save_json(folder / "result.json", result)
    for index, name in enumerate(names if version == 2 else ("xgboost",)):
        target = folder / "families" / name if version == 2 else folder
        if version == 2:
            save_json(target / "status.json", families[name])
        save_json(
            target / "leaderboard.json",
            [
                {
                    "key": f"candidate-{index}",
                    "fitness": fitness[name],
                    "families": "lag",
                    "genome": {**genome, **({"family": name} if version == 2 else {})},
                    "generation": 1,
                }
            ],
        )
        events = [
            {"event": "generation", "generation": generation, "best": fitness[name], "elapsed": 1}
            for generation in range(2 if name == "xgboost" else 1)
        ]
        (target / "events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
        curves = [
            {
                "generation": generation,
                "key": f"{name}-{generation}",
                "metric": "rmse" if name == "xgboost" else "mae",
                "validation": [0.4 + index * 0.1, 0.2 + index * 0.1],
            }
            for generation in range(2 if name == "xgboost" else 1)
        ]
        (target / "generation_curves.jsonl").write_text(
            "".join(json.dumps(curve) + "\n" for curve in curves), encoding="utf-8"
        )
        if version == 2 and name != "snaive":
            save_json(
                target / "importance.json",
                {
                    "method": "gain" if name == "xgboost" else "absolute_standardized_coefficient",
                    "features": [
                        {"feature": "lag_1", "importance": 0.75},
                        {"feature": "lag_2", "importance": 0.25},
                    ],
                },
            )
    dates = np.array(
        ["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26"], dtype="datetime64[D]"
    )
    write_table(
        prepared / "panel.parquet",
        {
            "__step": dates,
            "__series": np.array(["all"] * 4),
            "units": np.array([10.0, 12.0, 14.0, 16.0]),
        },
    )
    for model, offset in (("", 0.0), ("winner_", 0.0), ("ensemble_", 1.0)):
        for kind, index in (("validation", 2), ("test", 3)):
            write_table(
                folder / f"{model}{kind}.parquet",
                {
                    "series": np.array(["all"]),
                    "step": dates[index : index + 1],
                    "bucket": np.array(["1-2"]),
                    "actual": np.array([14.0 if index == 2 else 16.0]),
                    "prediction": np.array([13.0 + offset if index == 2 else 15.0 + offset]),
                },
            )
        write_table(
            folder / f"{model}forecast.parquet",
            {
                "series": np.array(["all", "all"]),
                "step": np.array(["2026-02-02", "2026-02-09"], dtype="datetime64[D]"),
                "bucket": np.array(["1-2", "1-2"]),
                "prediction": np.array([18.0 + offset, 20.0 + offset]),
                "lower": np.array([17.0 + offset, 19.0 + offset]),
                "upper": np.array([19.0 + offset, 21.0 + offset]),
            },
        )
    return run_id


def make_run(*, state="running", progress=None, curves=None):
    return {
        "id": "20261009-120000-progress",
        "kind": "search",
        "state": state,
        "budget": {"time_cap": 1200, "max_generations": 10, "plateau": 5},
        "progress": progress
        or {
            "generation": 0,
            "candidate": 1,
            "population": 10,
            "elapsed": 10,
            "flat_generations": 0,
        },
        "generations": [],
        "generation_curves": curves or [],
        "leaderboard": [],
    }


def mock_saved_run(page, run, polls=None):
    def read_run(route):
        if polls is not None:
            polls.append(time.monotonic())
        cursor = int(parse_qs(urlparse(route.request.url).query).get("since_generation", ["-1"])[0])
        payload = {
            **run,
            "generation_curves": [
                curve for curve in run["generation_curves"] if curve["generation"] > cursor
            ],
        }
        route.fulfill(json=payload)

    page.route("**/api/forecast-runs", lambda route: route.fulfill(json={"runs": [run]}))
    page.route("**/api/forecast-run?**", read_run)
    switch_to_regression(page)
    show_tab(page, "regression-search")
    page.wait_for_function(
        "document.getElementById('regression-run-title').innerText.includes('progress')"
    )


@pytest.mark.parametrize(
    ("progress", "budget", "expected"),
    [
        ({"elapsed": 600}, {}, 0.5),
        ({"generation": 2, "candidate": 5}, {}, 0.25),
        ({"flat_generations": 3}, {}, 0.6),
        ({"elapsed": 600, "generation": 8, "candidate": 5, "flat_generations": 3}, {}, 0.85),
        ({"elapsed": 2000}, {}, 1),
        ({"elapsed": -10, "candidate": -1}, {}, 0),
        ({"elapsed": 30}, {"time_cap": 60}, 0.5),
    ],
)
def test_search_progress_used_nearest_stop_condition(tmp_path, progress, budget, expected):
    with session(tmp_path) as (page, _):
        run = make_run()
        run["progress"].update(elapsed=0, candidate=0)
        run["progress"].update(progress)
        run["budget"].update(budget)
        value = page.evaluate("run => forecastSearchProgress(run)", run)
        assert value["fill"] == pytest.approx(expected)
        assert value["done"] is False


@pytest.mark.parametrize(
    ("state", "stopped", "fill", "done", "suffix"),
    [
        ("running", None, 0.4, False, ""),
        ("done", "plateau", 1, True, "done"),
        ("done", "cancelled", 0.4, False, "cancelled"),
        ("cancelled", "cancelled", 0.4, False, "cancelled"),
        ("failed", None, 0.4, False, "failed"),
    ],
)
def test_search_progress_label_and_terminal_state(tmp_path, state, stopped, fill, done, suffix):
    with session(tmp_path) as (page, _):
        run = make_run(
            state=state,
            progress={
                "generation": 3,
                "candidate": 7,
                "population": 10,
                "elapsed": 54,
                "flat_generations": 2,
            },
        )
        if stopped:
            run["stopped"] = stopped
        value = page.evaluate("run => forecastSearchProgress(run)", run)
        assert value["fill"] == pytest.approx(fill)
        assert value["done"] is done
        label = "generation 3 - candidate 7 of 10 - 54s of 20 min - 2 of 5 flat generations"
        assert value["label"] == label + (f" - {suffix}" if suffix else "")


def test_failed_search_kept_progress_after_an_older_generation_summary(tmp_path):
    with session(tmp_path) as (page, _):
        run = make_run(state="failed")
        run["elapsed"] = 10
        run["budget"]["time_cap"] = 120
        run["progress"]["elapsed"] = 90
        value = page.evaluate("run => forecastSearchProgress(run)", run)
        assert value["fill"] == pytest.approx(0.75)
        assert "90s of 2 min" in value["label"]
        assert value["label"].endswith(" - failed")
        assert value["done"] is False


def test_running_search_bar_grew_between_one_second_polls(tmp_path):
    with session(tmp_path) as (page, _):
        run = make_run()
        polls = []
        mock_saved_run(page, run, polls)
        fill = page.locator("#regression-run-bar-fill")
        rail_fill = page.locator(".forecast-run-row .bar-fill")
        page.wait_for_function(
            "document.getElementById('regression-run-bar-fill').getBoundingClientRect().width > 0"
        )
        before = fill.bounding_box()["width"]
        rail_before = rail_fill.bounding_box()["width"]
        run["progress"].update(candidate=8, elapsed=54)
        page.wait_for_function(
            "width => document.getElementById('regression-run-bar-fill').getBoundingClientRect().width > width",
            arg=before,
            timeout=1800,
        )
        assert rail_fill.bounding_box()["width"] > rail_before
        assert "candidate 8 of 10" in page.locator("#regression-run-progress-label").inner_text()
        assert len(polls) >= 2
        assert 0.75 <= polls[-1] - polls[-2] <= 1.6
        run.update(state="done", stopped="max_generations")
        page.wait_for_function(
            "document.getElementById('regression-run-title').innerText.endsWith('· done')"
        )
        assert fill.bounding_box()["width"] == pytest.approx(
            fill.locator("..").bounding_box()["width"], abs=1
        )
        assert "good" in fill.get_attribute("class").split()


def test_saved_loss_chart_filtered_metrics_and_collapsed_repeated_keys(tmp_path):
    with session(tmp_path) as (page, _):
        curves = [
            {"generation": 0, "key": "a", "metric": "mae", "validation": [4, 3, 2]},
            {"generation": 1, "key": "b", "metric": "rmse", "validation": [7, 6, 5]},
            {"generation": 2, "key": "c", "metric": "mae", "validation": [3, 2, 1]},
            {"generation": 3, "key": "d", "metric": "rmse", "validation": [6, 5, 4]},
            {"generation": 4, "key": "e", "metric": "mae", "validation": [2, 1, 0.5]},
            {"generation": 5, "key": "e", "metric": "mae", "validation": [2, 1, 0.5]},
        ]
        run = make_run(state="done", curves=curves)
        mock_saved_run(page, run)
        loss = page.locator("#regression-search-loss")
        chart = page.locator("#regression-loss-chart")
        page.wait_for_function(
            "document.querySelectorAll('#regression-loss-chart .chart-line').length === 3"
        )
        assert page.locator("#regression-loss-metric").inner_text() == "mae (lower is better)"
        assert loss.locator(".chart-legend-item:visible").all_text_contents() == [
            "current generation",
            "previous generations",
        ]
        assert page.locator("#regression-loss-hidden").inner_text() == (
            "2 earlier generations used a different metric and are hidden"
        )
        colors = page.evaluate(
            "() => { const css = getComputedStyle(document.documentElement);"
            " return ['--kingfisher', '--cream'].map(token => css.getPropertyValue(token).trim()); }"
        )
        assert loss.locator(".chart-legend-swatch:visible").evaluate_all(
            "nodes => nodes.map(node => getComputedStyle(node).borderTopColor)"
        ) == page.evaluate(
            "colors => colors.map(color => { const node = document.createElement('span');"
            " node.style.color = color; document.body.append(node);"
            " const resolved = getComputedStyle(node).color; node.remove(); return resolved; })",
            colors,
        )
        assert chart.locator(".chart-line").evaluate_all(
            "(lines, cream) => lines.every(line => line.getAttribute('stroke') === cream)",
            colors[1],
        )
        show_tab(page, "regression-data")
        show_tab(page, "regression-search")
        assert chart.locator(".chart-line").count() == 3
        assert loss.locator(".chart-legend-item:visible").count() == 2


def test_multifamily_progress_updated_counts_and_terminal_reasons(tmp_path):
    with session(tmp_path, width=1440) as (page, _):
        run = make_run()
        run["families"] = {
            name: {
                "state": "running",
                "phase": "broad",
                "candidates_done": 1,
                "candidates_expected": 6,
                "best_error": 0.4,
            }
            for name in ("xgboost", "ridge", "snaive")
        }
        mock_saved_run(page, run)
        progress = page.locator("#regression-family-progress")
        page.wait_for_function(
            "document.querySelectorAll('#regression-family-progress tbody tr').length === 3"
        )
        assert progress.locator("tbody tr").first.locator("td").all_text_contents() == [
            "xgboost",
            "running",
            "broad",
            "1",
            "6",
            "0.4",
            "-",
        ]
        run["families"]["xgboost"].update(
            phase="intensive", candidates_done=5, candidates_expected=2, best_error=0.2
        )
        run["families"]["ridge"].update(
            state="pruned", pruned_reason="validation error exceeds current best"
        )
        run["families"]["snaive"].update(
            state="failed", terminated=True, reason="synthetic worker stopped"
        )
        page.wait_for_function(
            "document.getElementById('regression-family-progress').innerText.includes('synthetic worker stopped')",
            timeout=4000,
        )
        rows = progress.locator("tbody tr")
        assert rows.nth(0).locator("td").all_text_contents() == [
            "xgboost",
            "running",
            "intensive",
            "5",
            "2",
            "0.2",
            "-",
        ]
        assert rows.nth(1).locator("td").nth(1).inner_text() == "pruned"
        assert (
            rows.nth(1).locator("td").last.inner_text() == "validation error exceeds current best"
        )
        assert rows.nth(2).locator("td").nth(1).inner_text() == "failed (terminated)"
        assert rows.nth(2).locator("td").last.inner_text() == "synthetic worker stopped"
        assert_family_layout(page)


def test_multifamily_saved_run_named_models_and_explained_importance(tmp_path):
    with session(tmp_path, width=1440) as (page, _):
        page.set_viewport_size({"width": 1440, "height": 1500})
        run_id = write_multifamily_run(tmp_path / "forecast_root")
        switch_to_regression(page)
        page.select_option("#topic-flavour", "timeseries")
        show_tab(page, "regression-search")
        page.locator(f'#regression-runs-list [data-run-id="{run_id}"]').click()
        board = page.locator("#regression-leaderboard-table")
        page.wait_for_function(
            "document.querySelectorAll('#regression-leaderboard-table tbody tr').length === 3"
        )
        headers = board.locator("thead th").all_text_contents()
        family_index = headers.index("family")
        assert board.locator("tbody tr").evaluate_all(
            "(rows, index) => rows.map(row => row.children[index].textContent)", family_index
        ) == ["xgboost", "ridge", "snaive"]
        assert page.locator("#regression-family-progress tbody tr").count() == 3
        assert page.locator("#regression-loss-metric").inner_text() == "rmse (lower is better)"
        assert "generations 2" in page.locator("#regression-run-stats").inner_text()
        page.click("#regression-generation-family")
        page.locator('.menu-panel .menu-item[data-id="ridge"]').click()
        assert page.locator("#regression-loss-metric").inner_text() == "mae (lower is better)"
        assert "generations 1" in page.locator("#regression-run-stats").inner_text()
        assert page.locator("#regression-loss-chart .chart-line").count() == 1
        page.click("#regression-generation-family")
        page.locator('.menu-panel .menu-item[data-id="xgboost"]').click()
        assert page.locator("#regression-loss-chart .chart-line").count() == 2
        assert_family_layout(page)
        artifacts = Path(__file__).resolve().parents[1] / "outputs" / "u10"
        artifacts.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=artifacts / "training.png", full_page=True)
        page.click("#regression-run-open")
        page.wait_for_function(
            "document.querySelectorAll('#regression-family-comparison tbody tr').length === 4"
        )
        comparison = page.locator("#regression-family-comparison tbody tr")
        assert [row.locator("td").all_text_contents() for row in comparison.all()] == [
            ["xgboost", "done", "0.12", "4", "-", "ratio 0.95, better than naive"],
            ["ridge", "done", "0.18", "3", "-", "-"],
            ["snaive", "done", "0.31", "2", "-", "-"],
            [
                "ensemble of xgboost, ridge",
                "done",
                "0.15",
                "-",
                "-",
                "ratio 1.05, worse than naive",
            ],
        ]
        assert page.locator("#regression-verdict").inner_text() == (
            "xgboost: ratio 0.95, better than naive"
        )
        assert page.locator("#regression-importance-status").inner_text().startswith("gain")
        assert page.locator("#regression-importance tbody tr").all_text_contents() == [
            "lag_175.00%",
            "lag_225.00%",
        ]
        assert (
            page.locator("#regression-results-table tr").nth(1).locator("td").last.inner_text()
            == "xgboost"
        )
        assert_family_layout(page)
        legend_box = page.locator("#regression-chart .chart-legend").bounding_box()
        table_box = page.locator("#regression-results-table").bounding_box()
        assert legend_box["y"] + legend_box["height"] <= table_box["y"]
        assert (
            page.locator("#regression-results-table tr").nth(1).locator("td").nth(1).inner_text()
            == "18"
        )
        page.screenshot(path=artifacts / "results.png", full_page=True)
        for family, expected in (
            ("ridge", "absolute standardised coefficient"),
            ("snaive", "no feature importance for this family"),
        ):
            page.click("#regression-importance-family")
            page.locator(f'.menu-panel .menu-item[data-id="{family}"]').click()
            page.wait_for_function(
                "text => document.getElementById('regression-importance-status').innerText.startsWith(text)",
                arg=expected,
            )
            assert page.locator("#regression-importance").is_visible() == (family == "ridge")
            if family == "ridge":
                assert page.locator("#regression-importance tbody tr").count() == 2
        page.click("#regression-results-model")
        page.locator('.menu-panel .menu-item[data-id="ensemble"]').click()
        page.wait_for_function(
            "document.getElementById('regression-verdict').innerText === 'ensemble of xgboost, ridge: ratio 1.05, worse than naive'"
        )
        page.wait_for_function(
            "document.querySelector('#regression-results-table tr:nth-child(2) td:last-child').innerText === 'ensemble'"
        )
        assert (
            page.locator("#regression-results-table tr").nth(1).locator("td").nth(1).inner_text()
            == "19"
        )
        assert page.locator("#regression-export").get_attribute("href").endswith("/ensemble.csv")
        page.screenshot(path=artifacts / "results-ensemble.png", full_page=True)


def test_legacy_saved_run_defaulted_to_xgboost_without_comparison(tmp_path):
    with session(tmp_path) as (page, _):
        run_id = write_multifamily_run(tmp_path / "forecast_root", version=1)
        switch_to_regression(page)
        page.select_option("#topic-flavour", "timeseries")
        show_tab(page, "regression-search")
        page.locator(f'#regression-runs-list [data-run-id="{run_id}"]').click()
        board = page.locator("#regression-leaderboard-table")
        board.locator("tbody tr").wait_for()
        family_index = board.locator("thead th").all_text_contents().index("family")
        assert board.locator("tbody tr td").nth(family_index).inner_text() == "xgboost"
        page.click("#regression-run-open")
        page.locator("#regression-verdict").wait_for()
        assert page.locator("#regression-verdict").inner_text() == (
            "xgboost: ratio 0.95, better than naive"
        )
        assert not page.locator("#regression-family-comparison").is_visible()
        assert (
            page.locator("#regression-importance-status").inner_text()
            == "no saved feature importance for this run"
        )


def test_result_and_run_banners_compared_ratios_and_kept_plain_facts(tmp_path):
    with session(tmp_path) as (page, _):
        run_id = write_multifamily_run(tmp_path / "forecast_root")
        switch_to_regression(page)
        page.select_option("#topic-flavour", "timeseries")
        show_tab(page, "regression-search")
        page.locator(f'#regression-runs-list [data-run-id="{run_id}"]').click()
        page.locator("#regression-run-verdict").wait_for()
        page.click("#regression-run-open")
        page.locator("#regression-verdict").wait_for()
        facts = [
            {"check": "band", "group": "reliability", "detail": "test coverage 80%"},
            {"check": "history", "group": "reliability", "detail": "60 periods"},
        ]
        for ratio, comparison, colour in (
            (0.95, "better", "rgb(199, 237, 95)"),
            (1, "worse", "rgb(173, 121, 111)"),
            (1.05, "worse", "rgb(173, 121, 111)"),
            (None, None, None),
        ):
            verdict = {"accuracy": {"ratio": ratio}, "checks": facts}
            page.evaluate(
                "run => { const ctl = forecastTopics.get('regression');"
                " ctl.paintRunDetail(run); ctl.paintVerdict(run.verdict, 'xgboost'); }",
                {**make_run(state="done"), "verdict": verdict},
            )
            expected = (
                f"ratio {ratio}, {comparison} than naive" if comparison else "ratio unavailable"
            )
            assert page.locator("#regression-run-verdict").inner_text() == expected
            assert page.locator("#regression-verdict .forecast-verdict-summary").inner_text() == (
                f"xgboost: {expected}"
            )
            assert page.locator(
                "#regression-verdict .forecast-verdict-check"
            ).all_text_contents() == ["band: test coverage 80%", "history: 60 periods"]
            for selector in ("#regression-run-verdict", "#regression-verdict"):
                banner = page.locator(selector)
                classes = banner.get_attribute("class").split()
                assert ("better" in classes) == (comparison == "better")
                assert ("worse" in classes) == (comparison == "worse")
                if colour:
                    assert banner.evaluate("node => getComputedStyle(node).color") == colour
                    assert (
                        banner.evaluate("node => getComputedStyle(node).borderTopColor") == colour
                    )
        page.evaluate(
            "() => forecastTopics.get('regression').paintVerdict("
            "{summary: 'failed to load', checks: []})"
        )
        banner = page.locator("#regression-verdict")
        assert banner.inner_text() == "failed to load"
        assert not {"better", "worse"}.intersection(banner.get_attribute("class").split())


def test_legacy_progress_kept_generation_budget_with_family_summary(tmp_path):
    with session(tmp_path) as (page, _):
        run = make_run()
        run["families"] = {"xgboost": {"state": "running"}}
        run["progress"].update(generation=8, candidate=5, elapsed=600, flat_generations=3)
        value = page.evaluate("run => forecastSearchProgress(run)", run)
        assert value["fill"] == pytest.approx(0.85)
        assert "generation 8 - candidate 5 of 10" in value["label"]


def test_results_kept_latest_model_importance_and_table_after_delayed_responses(tmp_path):
    with session(tmp_path, width=1440) as (page, _):
        run_id = write_multifamily_run(tmp_path / "forecast_root")
        switch_to_regression(page)
        page.select_option("#topic-flavour", "timeseries")
        show_tab(page, "regression-search")
        page.locator(f'#regression-runs-list [data-run-id="{run_id}"]').click()
        page.click("#regression-run-open")
        page.wait_for_function(
            "document.querySelector('#regression-results-table tr:nth-child(2) td:last-child')?.innerText === 'xgboost'"
        )
        pending = []

        def hold_response(route):
            pending.append((route, route.fetch().json()))
            page.evaluate("count => window.delayedForecastRequests = count", len(pending))

        def select_item(picker, name, count):
            page.click(f"#regression-{picker}")
            page.locator(f'.menu-panel .menu-item[data-id="{name}"]').click()
            page.wait_for_function("count => window.delayedForecastRequests === count", arg=count)

        def release_response(index):
            route, payload = pending[index]
            with page.expect_response(lambda response: response.url == route.request.url):
                route.fulfill(json=payload)
            page.wait_for_timeout(100)

        def assert_ensemble_table():
            row = page.locator("#regression-results-table tr").nth(1).locator("td")
            assert row.last.inner_text() == "ensemble"
            assert row.nth(1).inner_text() == "19"

        page.route("**/api/forecast-view?**", hold_response)
        select_item("results-model", "winner", 1)
        select_item("results-model", "ensemble", 2)
        release_response(1)
        page.wait_for_function(
            "document.querySelector('#regression-results-table tr:nth-child(2) td:last-child').innerText === 'ensemble'"
        )
        release_response(0)
        assert page.locator("#regression-verdict").inner_text() == (
            "ensemble of xgboost, ridge: ratio 1.05, worse than naive"
        )
        assert_ensemble_table()
        page.unroute("**/api/forecast-view?**", hold_response)

        pending.clear()
        page.evaluate("window.delayedForecastRequests = 0")
        page.route("**/api/forecast-table?**", hold_response)
        select_item("results-model", "winner", 1)
        select_item("results-model", "ensemble", 2)
        release_response(1)
        release_response(0)
        assert_ensemble_table()
        page.unroute("**/api/forecast-table?**", hold_response)

        pending.clear()
        page.evaluate("window.delayedForecastRequests = 0")
        page.route("**/api/forecast-view?**", hold_response)
        select_item("importance-family", "xgboost", 1)
        select_item("importance-family", "ridge", 2)
        release_response(1)
        release_response(0)
        assert page.locator("#regression-importance-family").inner_text() == "ridge"
        assert (
            page.locator("#regression-importance-status")
            .inner_text()
            .startswith("absolute standardised coefficient")
        )
        assert page.locator("#regression-verdict").inner_text() == (
            "ensemble of xgboost, ridge: ratio 1.05, worse than naive"
        )
        assert_ensemble_table()
