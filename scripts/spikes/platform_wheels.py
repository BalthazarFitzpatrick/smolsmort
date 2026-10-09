"""resolve binary-only dependency plans for the supported desktop platforms"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

PLATFORMS = ("x86_64-pc-windows-msvc", "aarch64-apple-darwin")
PYTHONS = ("3.11", "3.12", "3.13")
PACKAGES = ("statsforecast", "lightgbm", "numba", "llvmlite")
IMPORTS = ("numpy", "duckdb", "xgboost", "lightgbm", "statsforecast", "pandas", "numba", "llvmlite")


def measure_imports(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    results = {
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "import_protocol": "one fresh process per library; os caches not flushed",
        }
    }
    for library in IMPORTS:
        code = (
            "import importlib, json, time; started = time.perf_counter(); "
            f"module = importlib.import_module({library!r}); "
            "print(json.dumps({'seconds': time.perf_counter() - started, "
            "'version': getattr(module, '__version__', None)}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd="/private/tmp",
            env=environment,
            timeout=60,
            check=False,
        )
        results[library] = (
            json.loads(completed.stdout)
            if completed.returncode == 0
            else {"exit_code": completed.returncode, "error": completed.stderr}
        )
    size = subprocess.run(
        ["du", "-sk", str(Path(sys.executable).parent.parent)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    results["venv_kib"] = int(size.stdout.split()[0])
    (output_dir / "imports.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)
    return results


def resolve_wheels(output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, UV_CACHE_DIR="/private/tmp/multifamily-uv-cache")
    results = []
    for target_platform in PLATFORMS:
        for python in PYTHONS:
            command = [
                "uv",
                "pip",
                "install",
                "--dry-run",
                "--only-binary",
                ":all:",
                "--python-platform",
                target_platform,
                "--python-version",
                python,
                "--target",
                str(output_dir / "target"),
                "--no-config",
                *PACKAGES,
            ]
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=environment,
                cwd="/private/tmp",
                timeout=90,
                check=False,
            )
            output = completed.stdout + completed.stderr
            log_path = output_dir / f"{target_platform}-{python}.txt"
            log_path.write_text(output)
            result = {
                "platform": target_platform,
                "python": python,
                "exit_code": completed.returncode,
                "seconds": round(time.perf_counter() - started, 4),
                "packages": [
                    line.strip()[2:]
                    for line in output.splitlines()
                    if line.strip().startswith("+ ")
                ],
                "command": command,
                "log": str(log_path),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/multifamily-wheels"))
    parser.add_argument("--imports-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.output.resolve().is_relative_to(Path("/private/tmp")):
        parser.error("output must be inside /private/tmp")
    if arguments.imports_only:
        measure_imports(arguments.output)
    else:
        resolve_wheels(arguments.output)


if __name__ == "__main__":
    main()
