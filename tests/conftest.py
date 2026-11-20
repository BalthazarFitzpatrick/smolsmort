"""suite-wide pytest setup.

SET BEFORE ANY TEST MODULE IMPORTS. torch (the heatmap/box backends) bundles its own openmp
runtime; xgboost's homebrew libomp aborts or segfaults the process on macos if it finds one
already initialised. conftest.py loads before every test module in the session, which is the
only point early enough to matter - setting this inside smolsmort/tabular/backend.py was too
late, because torch tests earlier in the same run had already initialised their openmp first.
"""

import os
import sys
from importlib.util import find_spec
from types import ModuleType

import pytest

pytest_plugins = ["saved_runs"]

# macos only: the clash is torch's bundled openmp against homebrew's libomp. linux (the ci image)
# has one runtime, and single-threading every torch op there just made the suite slower
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # single-threaded: allowing torch's openmp pool and xgboost's to both spin up threads
    # segfaults the process even with the duplicate-lib check disabled above
    os.environ.setdefault("OMP_NUM_THREADS", "1")


# test modules that import torch, at collection or inside a test. a process that has torch loaded cannot
# fit xgboost or lightgbm (guard_process_family, omp error #179 in production), and every xdist
# worker collects every module, so the suite runs as two process groups: --process-group plain and
# --process-group torch (scripts/run_tests.py runs both). explicit file arguments always collect
TORCH_MODULES = frozenset(
    {
        "test_backend_evaluate.py",
        "test_boxes.py",
        "test_capture_size.py",
        "test_classes.py",
        "test_cli.py",
        "test_coreml_guard.py",
        "test_coreml_runtime.py",
        "test_crops.py",
        "test_downscale.py",
        "test_forecast_family_lstm.py",
        "test_forecast_family_mlp.py",
        "test_frame_store.py",
        "test_hyperparams.py",
        "test_lr_schedule.py",
        "test_model.py",
        "test_optim.py",
        "test_review_routes.py",
        "test_review_train.py",
        "test_review_train_options.py",
        "test_review_ui_logic.py",
        "test_review_ui_server.py",
        "test_train.py",
        "test_variable_boxes.py",
        "test_window_sampling.py",
    }
)


# modules that run real worker processes against wall-clock budgets (slice, candidate and final
# fit limits). under -n auto the workers fight for cores and those limits fire ("too slow",
# "exceeded deadline grace"); they pass one at a time on an idle host, so they run with -n0
TIMED_MODULES = frozenset(
    {
        "test_forecast_coordinator.py",
        "test_forecast_coordinator_controls.py",
        "test_forecast_family_ridge.py",
        "test_forecast_routes_v2.py",
        "test_forecast_runs_v2.py",
        "test_forecast_transform.py",
    }
)
TORCH_TIMED_MODULES = frozenset({"test_forecast_new_families_run.py"})

GROUPS = ("plain", "timed", "torch", "torch-timed")
NO_TORCH_GROUPS = ("plain", "timed")


def group_of(name: str) -> str:
    if name in TORCH_TIMED_MODULES:
        return "torch-timed"
    if name in TORCH_MODULES:
        return "torch"
    return "timed" if name in TIMED_MODULES else "plain"


def pytest_addoption(parser):
    parser.addoption(
        "--process-group",
        choices=GROUPS,
        help="collect one group: plain and timed never import torch, the torch groups do; "
        "timed groups are meant for -n0",
    )


def pytest_ignore_collect(collection_path, config):
    group = config.getoption("--process-group")
    if group is None or not collection_path.name.startswith("test_"):
        return None
    return group_of(collection_path.name) != group or None


def pytest_collection_finish(session):
    # a new torch-importing module must be added to TORCH_MODULES, or the plain group breaks
    if session.config.getoption("--process-group") in NO_TORCH_GROUPS and "torch" in sys.modules:
        raise pytest.UsageError("a plain-group module imported torch; add it to TORCH_MODULES")


# one report per process: the first importer is the cause, later tests only inherit it.
# without xgboost or lightgbm installed there is nothing for torch to clash with
_torch_reports: list[str] = []


@pytest.fixture(autouse=True)
def keep_torch_out_of_the_plain_group(request):
    yield
    # torch loaded by one test breaks every later xgboost or lightgbm fit in this process
    if (
        request.config.getoption("--process-group") in NO_TORCH_GROUPS
        # a real module: some tests park a stand-in under the name on purpose
        and isinstance(sys.modules.get("torch"), ModuleType)
        and any(find_spec(name) for name in ("xgboost", "lightgbm"))
        and not _torch_reports
    ):
        _torch_reports.append(request.node.nodeid)
        pytest.fail(f"{request.node.nodeid} imported torch in the plain group", pytrace=False)


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
