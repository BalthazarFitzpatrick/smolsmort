"""machine defaults and calibration decisions without native fits or real timing"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_forecast_coordinator_controls import tunable_race as tunable_race
from test_forecast_requests_pruning import mock_run as mock_run

from smolsmort.forecast import calibration, coordinator, machine_config, runs, tab
from smolsmort.forecast.requests import parse_request

TIMINGS = {1: 8, 2: 6, 3: 5.5, 5: 5.4, 10: 23}


@pytest.fixture
def config_path():
    return machine_config.forecast_config_path()


@pytest.fixture(autouse=True)
def stub_machine_probe(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "describe_machine",
        lambda threads, cap: {"nthread": threads, "max_concurrent_families": cap},
    )


def make_request(**changes):
    return {
        "version": 2,
        "time_budget_s": 90,
        "families": [{"name": "xgboost"}],
        **changes,
    }


@pytest.mark.parametrize("system", ["Darwin", "Linux", "Windows"])
def test_config_path_override_wins(tmp_path, monkeypatch, system):
    override = tmp_path / "custom" / "settings.toml"
    monkeypatch.setenv("SMOLSMORT_FORECAST_CONFIG", str(override))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(machine_config.platform, "system", lambda: system)
    assert machine_config.forecast_config_path() == override


@pytest.mark.parametrize("system", ["Darwin", "Linux"])
@pytest.mark.parametrize("xdg", [False, True])
def test_unix_config_path(tmp_path, monkeypatch, system, xdg):
    monkeypatch.delenv("SMOLSMORT_FORECAST_CONFIG")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(machine_config.platform, "system", lambda: system)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    root = tmp_path / "xdg" if xdg else tmp_path / "home" / ".config"
    if xdg:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(root))
    assert machine_config.forecast_config_path() == root / "smolsmort" / "forecast.toml"


@pytest.mark.parametrize("appdata", [False, True])
def test_windows_config_path(tmp_path, monkeypatch, appdata):
    monkeypatch.delenv("SMOLSMORT_FORECAST_CONFIG")
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "ignored"))
    monkeypatch.setattr(machine_config.platform, "system", lambda: "Windows")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    root = tmp_path / "appdata" if appdata else tmp_path / "home" / "AppData" / "Roaming"
    if appdata:
        monkeypatch.setenv("APPDATA", str(root))
    assert machine_config.forecast_config_path() == root / "smolsmort" / "forecast.toml"


def test_missing_config_uses_defaults(config_path):
    assert machine_config.load_machine_config() == {
        "max_concurrent_families": 1,
        "threads_per_family": None,
    }
    assert not config_path.exists()


@pytest.mark.parametrize(
    "text, key",
    [
        ("max_concurrent_families = 0", "max_concurrent_families"),
        ("max_concurrent_families = true", "max_concurrent_families"),
        ('max_concurrent_families = "3"', "max_concurrent_families"),
        ("threads_per_family = -1", "threads_per_family"),
        ("threads_per_family = 2.5", "threads_per_family"),
        ("threads_per_family = false", "threads_per_family"),
        ("calibrated_at = 79", "calibrated_at"),
        ("calibration = []", "calibration"),
        ('[calibration]\n"zero" = 8', "calibration.zero"),
        ('[calibration]\n"0" = 8', "calibration.0"),
        ('[calibration]\n"01" = 8', "calibration.01"),
        ('[calibration]\n"1" = -8', "calibration.1"),
        ('[calibration]\n"1" = true', "calibration.1"),
        ('[calibration]\n"1" = nan', "calibration.1"),
        ('[calibration]\n"1" = inf', "calibration.1"),
    ],
)
def test_invalid_config_names_path_and_key(config_path, text, key):
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        machine_config.load_machine_config()
    assert str(config_path) in str(caught.value)
    assert key in str(caught.value)


@pytest.mark.parametrize("content", [b"threads_per_family = [", b"\xff"])
def test_unreadable_config_names_path(config_path, content):
    config_path.write_bytes(content)
    with pytest.raises(ValueError) as caught:
        machine_config.load_machine_config()
    assert str(config_path) in str(caught.value)


@pytest.mark.parametrize(
    "explicit, configured, expected", [(4, 3, 4), (None, 3, 3), (None, None, 1)]
)
def test_request_config_precedence(config_path, explicit, configured, expected):
    if configured is not None:
        config_path.write_text(f"max_concurrent_families = {configured}\n")
    request = make_request(**({"max_concurrent_families": explicit} if explicit else {}))
    parsed = parse_request(request)
    assert parsed["max_concurrent_families"] == expected
    assert parsed["_max_concurrent_families_explicit"] is (explicit is not None)
    assert parse_request(json.loads(json.dumps(parsed))) == parsed
    if explicit is None:
        assert "max_concurrent_families" not in request


def test_implicit_request_resolves_latest_config_when_coordinator_starts(config_path, tmp_path):
    parsed = parse_request(make_request())
    config_path.write_text("max_concurrent_families = 3\n")
    restored = parse_request(json.loads(json.dumps(parsed)))
    assert restored == parsed
    state = coordinator.initialize_state(tmp_path, restored)
    assert state["machine"]["config_values"]["max_concurrent_families"] == 3


@pytest.mark.parametrize("marker", [None, 1, "false", [], {}])
def test_request_rejects_invalid_explicit_marker(marker):
    with pytest.raises(ValueError, match="_max_concurrent_families_explicit"):
        parse_request(
            make_request(max_concurrent_families=3, _max_concurrent_families_explicit=marker)
        )


@pytest.mark.parametrize("marker", [False, True])
def test_request_rejects_explicit_marker_without_resolved_cap(marker):
    with pytest.raises(ValueError, match="_max_concurrent_families_explicit"):
        parse_request(make_request(_max_concurrent_families_explicit=marker))


@pytest.mark.parametrize("explicit, expected", [(None, 3), (4, 4)])
def test_start_run_preserves_cap_origin(mock_run, config_path, explicit, expected):
    config_path.write_text("max_concurrent_families = 3\n")
    root, spec, prepared, starts = mock_run
    request = make_request(**({"max_concurrent_families": explicit} if explicit else {}))
    run_id = runs.start_run(root, spec, prepared, **request)
    saved = json.loads((root / run_id / "request.json").read_text())
    assert saved["max_concurrent_families"] == expected
    assert saved["_max_concurrent_families_explicit"] is (explicit is not None)
    assert len(starts) == 1


@pytest.mark.parametrize("explicit", [None, 4])
def test_route_keeps_omitted_cap_implicit(mock_run, config_path, monkeypatch, explicit):
    config_path.write_text("max_concurrent_families = 3\n")
    root, spec, prepared, _ = mock_run
    prepared.sql, prepared.reused = "", False
    monkeypatch.setattr(tab, "_parse_spec", lambda *args: spec)
    monkeypatch.setattr(tab, "_prepare_spec", lambda *args: prepared)
    monkeypatch.setattr(tab, "_runs_root", lambda app: root)
    monkeypatch.setattr(tab, "_validate_selections", lambda payload, spec: parse_request(payload))
    monkeypatch.setattr(tab.views, "save_prepared_spec", lambda *args: None)
    request = make_request(**({"max_concurrent_families": explicit} if explicit else {}))
    result = tab._start(None, request)
    saved = json.loads((root / result["run_id"] / "request.json").read_text())
    assert saved["max_concurrent_families"] == (explicit or 3)
    assert saved["_max_concurrent_families_explicit"] is (explicit is not None)


@pytest.mark.parametrize(
    "cap, requested_threads, total_threads, expected_cap, expected_threads",
    [(3, 3, 10, 3, 3), (3, 8, 10, 3, 3), (3, 8, 6, 3, 2), (12, 3, 10, 10, 1)],
)
def test_machine_records_effective_config(
    tmp_path,
    config_path,
    monkeypatch,
    cap,
    requested_threads,
    total_threads,
    expected_cap,
    expected_threads,
):
    config_path.write_text(
        f"max_concurrent_families = {cap}\nthreads_per_family = {requested_threads}\n"
    )
    monkeypatch.setattr(coordinator.os, "cpu_count", lambda: 10)
    state = coordinator.initialize_state(tmp_path, make_request(nthread=total_threads))
    assert state["nthread"] == total_threads
    assert state["machine"]["config_path"] == str(config_path)
    assert state["machine"]["config_values"] == {
        "max_concurrent_families": expected_cap,
        "threads_per_family": expected_threads,
    }
    assert expected_cap * expected_threads <= min(total_threads, 10)


def test_resume_keeps_recorded_config_and_total_threads(tmp_path, config_path, monkeypatch):
    config_path.write_text("max_concurrent_families = 3\nthreads_per_family = 3\n")
    monkeypatch.setattr(coordinator.os, "cpu_count", lambda: 10)
    request = parse_request(make_request())
    state = coordinator.initialize_state(tmp_path, request, started_at=100)
    (tmp_path / "coordinator.json").write_text(json.dumps(state))
    config_path.write_text("threads_per_family = invalid\n")
    restored = parse_request(json.loads(json.dumps(request)))
    resumed = coordinator.initialize_state(tmp_path, restored, started_at=200)
    assert resumed == state


@pytest.mark.parametrize("threads", [3, 8])
def test_configured_threads_stay_fixed_through_worker_launch_and_redistribution(
    tunable_race,
    config_path,
    monkeypatch,
    threads,
):
    config_path.write_text(f"max_concurrent_families = 3\nthreads_per_family = {threads}\n")
    monkeypatch.setattr(coordinator.os, "cpu_count", lambda: 10)
    tunable_race.plateaus.update(ridge=1, lightgbm=2, xgboost=3)
    result = tunable_race.run(["ridge", "lightgbm", "xgboost"], cap=3, nthread=10)
    assert tunable_race.launches
    assert all(process.allocation["nthread"] == 3 for process in tunable_race.launches)
    for snapshot in tunable_race.snapshots:
        for state in snapshot["families"].values():
            if "nthread" in state:
                assert state["nthread"] == 3
    assert all(state["nthread"] == 3 for state in result["families"].values())


def test_calibration_caps_search_at_three_families():
    # c=1: 1/23; c=2: 2/5.4; c=3: 3/5.5; default bound makes c=3 best
    # c=5: 5/6; c=10: 10/8; allowing ten families instead makes c=10 best
    assert calibration.choose_parallelism(TIMINGS, 10) == {
        "max_concurrent_families": 3,
        "threads_per_family": 3,
    }


def test_calibration_can_raise_the_family_bound():
    assert calibration.choose_parallelism(TIMINGS, 10, max_families=10) == {
        "max_concurrent_families": 10,
        "threads_per_family": 1,
    }


def test_calibration_uses_nearest_lower_threads_and_prefers_fewer_families():
    assert calibration.choose_parallelism({1: 8, 3: 4, 10: 2}, 10, max_families=2) == {
        "max_concurrent_families": 1,
        "threads_per_family": 10,
    }
    assert calibration.choose_parallelism({1: 8}, 2, max_families=10) == {
        "max_concurrent_families": 2,
        "threads_per_family": 1,
    }


@pytest.mark.parametrize(
    "timings, cores, max_families",
    [
        ({}, 10, 3),
        ({1: 0}, 10, 3),
        ({1: float("nan")}, 10, 3),
        ({1: True}, 10, 3),
        ({0: 1}, 10, 3),
        ({2: 1}, 10, 10),
        (TIMINGS, 0, 3),
        (TIMINGS, 10, 0),
        (TIMINGS, 10, True),
    ],
)
def test_calibration_rejects_invalid_decision_inputs(timings, cores, max_families):
    with pytest.raises(ValueError):
        calibration.choose_parallelism(timings, cores, max_families)


def test_calibration_panel_is_fixed_and_small():
    panel = calibration.build_calibration_panel()
    assert [item.shape for item in panel] == [(2000, 60), (2000,), (240, 60)]
    for actual, repeated in zip(panel, calibration.build_calibration_panel(), strict=True):
        np.testing.assert_array_equal(actual, repeated)
        assert np.isfinite(actual).all()


def test_candidate_timing_excludes_warmup_and_uses_median(monkeypatch):
    calls = []
    times = iter([0, 8, 10, 12, 20, 24])
    monkeypatch.setattr(calibration, "perf_counter", lambda: next(times))
    panel = (np.zeros((2, 1)), np.zeros(2), np.ones((1, 1)))

    def fit(x, **kwargs):
        assert x is panel[0] and kwargs["y"] is panel[1]
        assert kwargs["nthread"] == 3 and kwargs["seed"] == 79
        calls.append("fit")

        def predict(x):
            assert x is panel[2]
            calls.append("predict")

        return SimpleNamespace(predict=predict)

    elapsed = calibration._measure_candidate(
        SimpleNamespace(fit=fit), "squared", {}, panel, ["q"], 3
    )
    assert elapsed == 4
    assert calls == ["fit", "predict"] * 4


@pytest.mark.parametrize("has_lightgbm", [False, True])
def test_measurement_uses_default_candidates_and_skips_optional_library(
    monkeypatch,
    capsys,
    has_lightgbm,
):
    from smolsmort.forecast import families
    from smolsmort.forecast.search import DEFAULTS

    candidates = {
        name: SimpleNamespace(
            name=name,
            available=lambda name=name: (name == "xgboost" or has_lightgbm, "not installed"),
            defaults=lambda: {"num_leaves": 31},
        )
        for name in ("xgboost", "lightgbm")
    }
    monkeypatch.setattr(families, "get_family", candidates.__getitem__)
    monkeypatch.setattr(
        calibration, "build_calibration_panel", lambda: (np.zeros((2, 60)), None, None)
    )
    calls = []

    def measure(family, objective, params, panel, feature_types, count):
        calls.append((family.name, count))
        assert feature_types == ["q"] * 60
        assert objective == ("squared" if family.name == "xgboost" else "regression_l1")
        assert params == (DEFAULTS if family.name == "xgboost" else {"num_leaves": 31})
        return count * (2 if family.name == "xgboost" else 3)

    monkeypatch.setattr(calibration, "_measure_candidate", measure)
    assert calibration.measure_candidates([1, 3]) == (
        {1: 5, 3: 15} if has_lightgbm else {1: 2, 3: 6}
    )
    assert len(calls) == (4 if has_lightgbm else 2)
    assert ("skipping lightgbm" in capsys.readouterr().out) is (not has_lightgbm)


def test_missing_xgboost_fails_before_measuring(monkeypatch):
    from smolsmort.forecast import families

    monkeypatch.setattr(
        families,
        "get_family",
        lambda name: SimpleNamespace(available=lambda: (False, "install it")),
    )
    monkeypatch.setattr(
        calibration, "build_calibration_panel", lambda: pytest.fail("unexpected panel")
    )
    with pytest.raises(ValueError, match="calibration needs xgboost"):
        calibration.measure_candidates([1])


@pytest.mark.parametrize("write", [False, True])
def test_cli_calibration_round_trips_config_and_preserves_unknown_keys(
    config_path,
    monkeypatch,
    capsys,
    write,
):
    from smolsmort.cli import main

    text = (
        'owner = "fabs"\nupdated = 2026-10-21\n'
        'escaped = "\\u007f\\U0001D11E"\n'
        '[future]\nlabels = ["a", "b"]\nnested = { enabled = true }\n'
        '[[future.items]]\nname = "first"\n'
    )
    config_path.write_text(text)
    original = tomllib.loads(text)
    monkeypatch.setattr(calibration.os, "cpu_count", lambda: 10)

    def measure(threads):
        assert threads == [1, 2, 3, 5, 10]
        return TIMINGS

    monkeypatch.setattr(calibration, "measure_candidates", measure)
    args = ["forecast", "calibrate", "--threads", "1,2,3,5,10", "--max-families", "3"]
    assert main(args + (["--write"] if write else [])) == 0
    output = capsys.readouterr().out
    assert "threads  median seconds" in output
    assert "max_concurrent_families=3 threads_per_family=3" in output
    if write:
        saved = machine_config.load_machine_config()
        assert saved["max_concurrent_families"] == saved["threads_per_family"] == 3
        assert saved["calibration"] == {str(key): value for key, value in TIMINGS.items()}
        assert isinstance(saved["calibrated_at"], str)
        assert all(saved[key] == value for key, value in original.items())
    else:
        assert config_path.read_text() == text


def test_cli_honors_larger_family_bound(config_path, monkeypatch):
    from smolsmort.cli import main

    monkeypatch.setattr(calibration.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(calibration, "measure_candidates", lambda threads: TIMINGS)
    assert main(["forecast", "calibrate", "--max-families", "10", "--write"]) == 0
    saved = machine_config.load_machine_config()
    assert saved["max_concurrent_families"] == 10 and saved["threads_per_family"] == 1


def test_calibration_always_samples_one_and_clips_to_cores(monkeypatch):
    monkeypatch.setattr(calibration.os, "cpu_count", lambda: 4)

    def measure(threads):
        assert threads == [1, 3, 4]
        return {1: 8, 3: 4, 4: 3}

    monkeypatch.setattr(calibration, "measure_candidates", measure)
    assert calibration.run_calibration([3, 10, 10]) == 0


@pytest.mark.parametrize(
    "args", [["--threads", "0,2"], ["--threads", "a"], ["--max-families", "0"]]
)
def test_cli_rejects_bad_options_before_measuring(monkeypatch, args):
    from smolsmort.cli import main

    monkeypatch.setattr(calibration, "measure_candidates", lambda _: pytest.fail("unexpected fit"))
    with pytest.raises(SystemExit) as caught:
        main(["forecast", "calibrate", *args])
    assert caught.value.code == 2
