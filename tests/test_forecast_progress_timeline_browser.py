"""timeline status rows and neutral progress ticks in the real app"""

import json
import re

import pytest
from shots import shot_dir
from test_forecast_progress_browser import make_run, mock_saved_run
from test_forecast_ui_browser import session


def make_timeline_run(phase="pruning", state="running"):
    run = make_run(state=state, progress={"elapsed": 86400 if state == "done" else 4680})
    run.update(
        version=2,
        phase=phase,
        time_budget_s=86400,
        timeline={
            "started_at": 100,
            "deadline": 86500,
            "search_deadline": 86200,
            "finish_reserve_s": 300,
            "phase_history": [
                {"phase": "broad", "started_at": 100},
                {"phase": "pruning", "started_at": 160},
            ],
        },
    )
    return run


@pytest.mark.parametrize(
    ("phase", "state", "label", "next_step"),
    [
        (
            "broad",
            "running",
            "broad · running",
            "next: pruning when every family has its first score",
        ),
        ("pruning", "running", "pruning · running", "next: deep, by 22h 12m"),
        ("deep", "running", "deep · running", "next: finish at 23h 55m"),
        ("finish", "running", "finish · running", "next: test scoring and ensembles, then done"),
        ("finish", "done", "done", ""),
        ("deep", "failed", "failed", ""),
        ("deep", "cancelled", "stopped", ""),
    ],
)
def test_timeline_status_rows(tmp_path, phase, state, label, next_step):
    with session(tmp_path) as (page, _):
        mock_saved_run(page, make_timeline_run(phase, state))
        assert page.locator("#regression-run-progress-label").inner_text() == label
        time_row = page.locator("#regression-run-progress-time").inner_text()
        assert time_row == ("24h of 24h" if state == "done" else "1h 18m of 24h")
        assert not re.search(r"\d+s\b", time_row)
        assert page.locator("#regression-run-progress-next").inner_text() == next_step
        if state in ("running", "done") and phase in ("pruning", "finish"):
            image_dir = shot_dir("progress-screenshots")
            page.locator(".forecast-run-progress-panel").filter(
                has=page.locator("#regression-run-progress")
            ).screenshot(path=str(image_dir / f"{phase}-{state}.png"))


def test_ticks_use_recorded_starts_and_planned_bounds(tmp_path):
    with session(tmp_path) as (page, _):
        run = make_timeline_run()
        mock_saved_run(page, run)
        ticks = page.locator("#regression-run-progress-axis .forecast-progress-tick")
        assert ticks.count() == 4
        assert ticks.locator(".now").count() == 0
        values = ticks.evaluate_all(
            "nodes => nodes.map(n => ({left: n.style.left, passed: n.classList.contains('done'), tone: getComputedStyle(n).borderLeftColor}))"
        )
        assert values[0]["passed"]
        assert not values[1]["passed"]
        assert values[0]["tone"] != values[1]["tone"]
        colors = page.evaluate("""() => {
          const style = getComputedStyle(document.documentElement);
          return ['--text', '--grey-border'].map(key => style.getPropertyValue(key).trim());
        }""")
        rules = page.evaluate("""() => Array.from(document.styleSheets).flatMap(sheet =>
          Array.from(sheet.cssRules).map(rule => rule.cssText)).join(' ')""")
        assert ".forecast-progress-tick.done { border-left-color: var(--text); }" in rules
        assert "border-left: 1px solid var(--grey-border)" in rules
        assert all(colors)
        positions = [float(value["left"].removesuffix("%")) for value in values]
        assert positions == pytest.approx(
            [60 / 86400 * 100, 86100 * 6.5 / 7 / 86400 * 100, 86100 / 86400 * 100, 100]
        )
        assert all(position != pytest.approx(4680 / 86400 * 100) for position in positions)
        assert (
            ticks.last.evaluate("n => getComputedStyle(n).transform") == "matrix(1, 0, 0, 1, -1, 0)"
        )
        run["timeline"]["phase_history"].append({"phase": "deep", "started_at": 4000})
        value = page.evaluate("run => forecastSearchProgress(run)", run)
        assert value["ticks"][1]["position"] == pytest.approx(3900 / 86400)


def test_old_run_has_no_axis_and_short_caps_use_minutes(tmp_path):
    with session(tmp_path) as (page, _):
        run = make_timeline_run()
        del run["timeline"]
        run["time_budget_s"] = 7200
        mock_saved_run(page, run)
        assert page.locator("#regression-run-progress-axis").is_hidden()
        assert page.locator(".forecast-progress-tick").count() == 0
        assert page.locator("#regression-run-progress-time").inner_text() == "78 min of 120 min"


def test_real_runs_route_passes_timeline(tmp_path):
    from test_forecast_progress_browser import write_multifamily_run

    with session(tmp_path) as (page, base):
        run_id = write_multifamily_run(tmp_path / "forecast_root")
        folder = tmp_path / "forecast_root" / ".forecast-runs" / run_id
        path = folder / "status.json"
        status = json.loads(path.read_text())
        status["timeline"] = make_timeline_run()["timeline"]
        path.write_text(json.dumps(status))
        payload = page.request.get(base + "/api/forecast-runs").json()
        run = next(run for run in payload["runs"] if run["id"] == run_id)
        assert run["timeline"] == status["timeline"]
