"""family contracts and a registry that keeps optional libraries lazy"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class Param:
    name: str
    kind: Literal["float", "int", "log", "choice"]
    lo: float | int | None = None
    hi: float | int | None = None
    choices: tuple[object, ...] = ()
    default: object = None


class Fitted(Protocol):
    @property
    def eval_history(self) -> dict | None: ...

    def predict(self, x) -> np.ndarray: ...


class Family(Protocol):
    """optional isolated=True requires its own process; absent means False"""

    name: str
    label: str
    needs: Literal["lag_features", "raw_series"]
    pip_extra: str

    def available(self) -> tuple[bool, str]: ...

    def space(self) -> list[Param]: ...

    def defaults(self) -> dict: ...

    def objectives(self, task: str) -> tuple[str, ...]: ...

    def cost(self, shape: tuple[int, int], params: dict) -> float: ...

    def fit(
        self,
        x,
        *,
        objective: str,
        y=None,
        feature_types: list[str] | None = None,
        params: dict | None = None,
        seed: int = 0,
        nthread: int = 1,
        **options,
    ) -> Fitted: ...


def guard_process_family(name: str) -> None:
    """reject incompatible native runtimes before importing or doing thread work"""
    neural = name in {"mlp", "lstm", "torch"}
    conflict = (
        any(module in sys.modules for module in ("xgboost", "lightgbm"))
        if neural
        else name in {"xgboost", "lightgbm"} and "torch" in sys.modules
    )
    if conflict:
        from smolsmort.forecast.model import ModelError

        raise ModelError(f"family {name!r}: torch families must run in their own process")


def load_torch():
    """import torch; it must never share a process with xgboost or lightgbm [two libomp copies abort]"""
    guard_process_family("torch")
    return import_module("torch")


def probe_import(module: str, pip_extra: str) -> tuple[bool, str]:
    """probe only when asked; shared libraries can fail even when the package is installed"""
    try:
        from smolsmort.forecast.model import ModelError

        try:
            guard_process_family(module)
        except ModelError:
            # availability must not load a conflicting runtime into its caller
            result = subprocess.run(
                [sys.executable, "-c", f"import importlib; importlib.import_module({module!r})"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return (
                (True, "")
                if result.returncode == 0
                else (False, f"uv sync --extra {pip_extra} ({result.stderr.strip()})")
            )
        load_torch() if module == "torch" else import_module(module)
    except (ImportError, OSError, subprocess.SubprocessError) as exc:
        return False, f"uv sync --extra {pip_extra} ({exc})"
    return True, ""


_REGISTRY: dict[str, Family] = {}


def register(family: Family) -> None:
    if family.name in _REGISTRY:
        raise ValueError(f"family {family.name!r} is already registered")
    _REGISTRY[family.name] = family


def get_family(name: str) -> Family:
    return _REGISTRY[name]


def available_families() -> list[Family]:
    return [family for family in _REGISTRY.values() if family.available()[0]]


def registered_families() -> list[Family]:
    return list(_REGISTRY.values())
