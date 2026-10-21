"""per-machine forecast defaults and portable toml persistence"""

from __future__ import annotations

import json
import math
import os
import platform
import tomllib
from datetime import date, datetime, time
from pathlib import Path


def forecast_config_path() -> Path:
    """keep platform-specific user config directories behind one seam"""
    override = os.environ.get("SMOLSMORT_FORECAST_CONFIG")
    if override:
        return Path(override).expanduser()
    if platform.system() == "Windows":
        root = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "smolsmort" / "forecast.toml"


def _read_config(path: Path) -> dict:
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: could not read forecast config: {exc}") from exc


def _validate_config(values: dict, path: Path) -> None:
    for key in ("max_concurrent_families", "threads_per_family"):
        if key in values and (type(values[key]) is not int or values[key] < 1):
            raise ValueError(f"{path}: {key} must be a positive integer")
    if "calibrated_at" in values and not isinstance(values["calibrated_at"], (str, datetime)):
        raise ValueError(f"{path}: calibrated_at must be a timestamp string or datetime")
    calibration = values.get("calibration", {})
    if not isinstance(calibration, dict):
        raise ValueError(f"{path}: calibration must be a table of seconds by thread count")
    for key, seconds in calibration.items():
        if not key.isascii() or not key.isdecimal() or int(key) < 1 or str(int(key)) != key:
            raise ValueError(f"{path}: calibration.{key} must name a positive thread count")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(f"{path}: calibration.{key} must be positive finite seconds")


def load_machine_config(path: Path | None = None) -> dict:
    path = forecast_config_path() if path is None else Path(path)
    values = _read_config(path)
    _validate_config(values, path)
    return {"max_concurrent_families": 1, "threads_per_family": None, **values}


def resolve_concurrency(request: dict, config: dict) -> int:
    explicit = request.get(
        "_max_concurrent_families_explicit", "max_concurrent_families" in request
    )
    return request["max_concurrent_families"] if explicit else config["max_concurrent_families"]


def resolve_parallelism(request: dict, config: dict, total_threads: int, cores: int) -> dict:
    cap = resolve_concurrency(request, config)
    threads = config["threads_per_family"]
    if threads is not None:
        budget = min(total_threads, cores)
        cap = min(cap, budget)
        threads = min(threads, max(1, budget // cap))
    return {"max_concurrent_families": cap, "threads_per_family": threads}


def _toml_value(value) -> str:
    """inline tables preserve unknown settings without a writer dependency"""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = (f"{_toml_value(key)} = {_toml_value(item)}" for key, item in value.items())
        return "{ " + ", ".join(pairs) + " }"
    if isinstance(value, (int, float)):
        return repr(value)
    raise ValueError(f"unsupported toml value: {type(value).__name__}")


def write_machine_config(values: dict, path: Path | None = None) -> Path:
    path = forecast_config_path() if path is None else Path(path)
    existing = _read_config(path)
    merged = {**existing, **values}
    _validate_config(merged, path)
    lines = [
        f"{_toml_value(key)} = {_toml_value(value)}"
        for key, value in merged.items()
        if key != "calibration"
    ]
    if "calibration" in merged:
        lines.extend(["", "[calibration]"])
        lines.extend(
            f"{_toml_value(key)} = {_toml_value(value)}"
            for key, value in merged["calibration"].items()
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
