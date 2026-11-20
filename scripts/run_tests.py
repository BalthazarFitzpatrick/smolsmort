"""run the suite as the process groups conftest.py defines, one pytest process each.

torch and xgboost/lightgbm cannot share a process, so the torch-importing modules run apart,
and the wall-clock budget tests run serially:
    uv run python scripts/run_tests.py -q
extra arguments go to both runs; the exit code is the first failing group's."""

import subprocess
import sys

NO_TESTS_COLLECTED = 5


# timed groups hold wall-clock budget tests, which need the cores to themselves
GROUPS = {"plain": [], "timed": ["-n0"], "torch": [], "torch-timed": ["-n0"]}


def run_group(group: str, args: list[str]) -> int:
    command = [sys.executable, "-m", "pytest", "--process-group", group, *GROUPS[group], *args]
    print(f"==> pytest --process-group {group}", flush=True)
    code = subprocess.run(command, check=False).returncode
    # a path filter can leave one group empty
    return 0 if code == NO_TESTS_COLLECTED else code


def main(args: list[str]) -> int:
    codes = [run_group(group, args) for group in GROUPS]
    return next((code for code in codes if code), 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
