"""saved and source-aware feature families retain readable recipe groups"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "smolsmort" / "review_ui" / "static" / "tab_forecast.js"
RUN_HELPER = """
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const context = {};
vm.runInNewContext(source.slice(
  source.indexOf('function forecastSplitFamilyName('),
  source.indexOf('function forecastNumber('),
), context);
const result = context[process.argv[2]](JSON.parse(process.argv[3]));
process.stdout.write(JSON.stringify(result));
"""


@pytest.fixture(scope="module")
def evaluate_helper():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")

    def evaluate(function, value):
        result = subprocess.run(
            [node, "-e", RUN_HELPER, str(SCRIPT), function, json.dumps(value)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return json.loads(result.stdout)

    return evaluate


@pytest.mark.parametrize(
    "name,expected",
    [
        ("unit@need_by_date:lag+0", ["unit", "need_by_date", "lag+0"]),
        ("unit@need_by_date:roll:13:q10", ["unit", "need_by_date", "roll:13:q10"]),
        ("need_by_date:cal:week", [None, "need_by_date", "cal:week"]),
        ("site:dim", ["site", None, "dim"]),
        ("size:log", ["size", None, "log"]),
        ("measure:size", ["size", None, "measure"]),
        ("date:trend", [None, None, "cal:trend"]),
        ("part%3Aid%40x%25@due%3Adate:lag:2", ["part:id@x%", "due:date", "lag:2"]),
        ("measure:dim", ["dim", None, "measure"]),
        ("%64im:measure", ["dim", None, "measure"]),
        ("dim:measure", ["measure", None, "dim"]),
        ("%6Deasure:dim", ["measure", None, "dim"]),
    ],
)
def test_family_sources_and_operations_parse(evaluate_helper, name, expected):
    assert evaluate_helper("forecastSplitFamilyName", name) == expected


@pytest.mark.parametrize(
    "families,expected",
    [
        (
            [
                "lag+3",
                "lag+0",
                "lag+1",
                "roll:52",
                "roll:8",
                "roll:13",
                "ewm:0.5",
                "ewm:0.1",
                "ewm:0.3",
                "cal:quarter",
                "cal:week",
                "dim:project",
            ],
            [
                ["lags", "0,1,3"],
                ["rolling", "8,13,52"],
                ["ewm", "0.1,0.3,0.5"],
                ["calendar", "week,quarter"],
                ["dims", "project"],
            ],
        ),
        (
            [
                "unit@need_by_date:lag+0",
                "unit@need_by_date:roll:13:q10",
                "need_by_date:cal:week",
                "site:dim",
                "size@need_by_date:measure",
                "size:log",
            ],
            [
                ["lags", "unit@need_by_date:0"],
                ["rolling", "unit@need_by_date:13:q10"],
                ["calendar", "need_by_date:week"],
                ["dims", "site"],
                ["measures", "size@need_by_date"],
                ["log measures", "size"],
            ],
        ),
        (
            ["date:quarter", "date:week", "date:trend"],
            [["calendar", "week,quarter,trend"]],
        ),
        (
            ["measure:dim", "%64im:measure", "dim:measure", "%6Deasure:dim"],
            [["dims", "measure"], ["measures", "dim"]],
        ),
    ],
    ids=["legacy", "source-aware", "legacy-row-calendar", "reserved-source-names"],
)
def test_recipe_groups_keep_sources_and_legacy_labels(evaluate_helper, families, expected):
    assert evaluate_helper("forecastFamilyGroups", families) == expected
