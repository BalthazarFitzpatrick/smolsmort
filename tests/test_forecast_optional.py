"""the forecast extra is optional: without duckdb or xgboost the cli, the page and the vision half
still import, and only the forecast routes fail - by name - when used"""

from __future__ import annotations

import subprocess
import sys

BLOCK = "import sys; sys.modules['duckdb'] = None; sys.modules['xgboost'] = None; "


def _python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCK + code], capture_output=True, text=True)


def test_a_plain_install_imports_the_cli_and_the_forecast_tab():
    done = _python(
        "import smolsmort.cli, smolsmort.forecast.tab, smolsmort.forecast.views, "
        "smolsmort.forecast.prep, smolsmort.forecast.runs; print('ok')"
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_a_forecast_route_without_the_extra_fails_on_first_use_not_at_import():
    done = _python("from smolsmort.forecast.tables import connect; connect()")
    assert done.returncode != 0 and "duckdb" in done.stderr
