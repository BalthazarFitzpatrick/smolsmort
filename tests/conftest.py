"""suite-wide pytest setup.

SET BEFORE ANY TEST MODULE IMPORTS. torch (the heatmap/box backends) bundles its own openmp
runtime; xgboost's homebrew libomp aborts or segfaults the process on macos if it finds one
already initialised. conftest.py loads before every test module in the session, which is the
only point early enough to matter - setting this inside smolsmort/tabular/backend.py was too
late, because torch tests earlier in the same run had already initialised their openmp first.
"""

import os
import sys

import pytest

pytest_plugins = ["saved_runs"]

# macos only: the clash is torch's bundled openmp against homebrew's libomp. linux (the ci image)
# has one runtime, and single-threading every torch op there just made the suite slower
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # single-threaded: allowing torch's openmp pool and xgboost's to both spin up threads
    # segfaults the process even with the duplicate-lib check disabled above
    os.environ.setdefault("OMP_NUM_THREADS", "1")


# test modules that import torch while pytest collects them. a process that has torch loaded cannot
# fit xgboost or lightgbm (guard_process_family, omp error #179 in production), and every xdist
# worker collects every module, so the suite runs as two process groups: --process-group plain and
# --process-group torch (scripts/run_tests.py runs both). explicit file arguments always collect
TORCH_MODULES = frozenset(
    {
        "test_backend_evaluate.py",
        "test_classes.py",
        "test_coreml_guard.py",
        "test_coreml_runtime.py",
        "test_forecast_family_lstm.py",
        "test_forecast_family_mlp.py",
        "test_forecast_new_families_run.py",
        "test_hyperparams.py",
        "test_lr_schedule.py",
        "test_model.py",
        "test_optim.py",
        "test_review_train.py",
        "test_review_ui_logic.py",
        "test_review_ui_server.py",
        "test_train.py",
        "test_variable_boxes.py",
        "test_window_sampling.py",
    }
)


def pytest_addoption(parser):
    parser.addoption(
        "--process-group",
        choices=("plain", "torch"),
        help="collect only the modules that do not (plain) or do (torch) import torch",
    )


def pytest_ignore_collect(collection_path, config):
    group = config.getoption("--process-group")
    if group is None or not collection_path.name.startswith("test_"):
        return None
    return (collection_path.name in TORCH_MODULES) != (group == "torch") or None


def pytest_collection_finish(session):
    # a new torch-importing module must be added to TORCH_MODULES, or the plain group breaks
    if session.config.getoption("--process-group") == "plain" and "torch" in sys.modules:
        raise pytest.UsageError("a plain-group module imported torch; add it to TORCH_MODULES")


@pytest.fixture(autouse=True)
def isolate_forecast_run_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOLSMORT_RUN_LOCK", str(tmp_path / "forecast.lock"))


@pytest.fixture(autouse=True)
def isolate_forecast_machine_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOLSMORT_FORECAST_CONFIG", str(tmp_path / "forecast.toml"))


@pytest.fixture(scope="session", autouse=True)
def screenshot_root(tmp_path_factory):
    from shots import set_shot_root

    set_shot_root(tmp_path_factory.mktemp("shots"))
