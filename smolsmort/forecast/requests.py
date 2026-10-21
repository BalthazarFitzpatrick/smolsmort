"""validate the new request while leaving unversioned requests untouched"""

from __future__ import annotations

import math
import re

from smolsmort.forecast.machine_config import load_machine_config


def parse_request(request: dict) -> dict:
    if "version" not in request:
        return request
    if request["version"] != 2:
        raise ValueError("unsupported forecast request version")
    transform = request.get("transform", "none")
    if transform not in ("none", "log1p", "logdiff"):
        raise ValueError("transform must be none, log1p or logdiff")
    duration = request.get("time_budget_s")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("time_budget_s must be positive and finite")
    families = request.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("select at least one family")
    names = set()
    for selection in families:
        name = selection.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise ValueError("family names must be unique safe names")
        names.add(name)
        if selection.get("method", "genetic") not in ("genetic", "grid"):
            raise ValueError("family method must be genetic or grid")
        if not isinstance(selection.get("space", {}), dict):
            raise ValueError("family space must be an object")
    if request.get("nthread") is not None and (
        type(request["nthread"]) is not int or request["nthread"] < 1
    ):
        raise ValueError("nthread must be a positive integer")
    explicit_concurrency = request.get(
        "_max_concurrent_families_explicit", "max_concurrent_families" in request
    )
    if type(explicit_concurrency) is not bool or (
        "_max_concurrent_families_explicit" in request and "max_concurrent_families" not in request
    ):
        raise ValueError("_max_concurrent_families_explicit must be a boolean with a resolved cap")
    # persisted requests keep their resolved cap; workers need no live config to validate them
    concurrency = (
        request["max_concurrent_families"]
        if "max_concurrent_families" in request
        else load_machine_config()["max_concurrent_families"]
    )
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("max_concurrent_families must be a positive integer")
    fraction = request.get("candidate_limit_fraction", 0.25)
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise ValueError("candidate_limit_fraction must be finite and between zero and one")
    plateau = request.get("plateau", 5)
    min_generations = request.get("min_generations", 0)
    if type(plateau) is not int or plateau < 1:
        raise ValueError("plateau must be a positive integer")
    if type(min_generations) is not int or min_generations < 0:
        raise ValueError("min_generations must be a non-negative integer")
    prune_sigma = request.get("prune_sigma", 1.0)
    if (
        isinstance(prune_sigma, bool)
        or not isinstance(prune_sigma, (int, float))
        or not math.isfinite(prune_sigma)
        or prune_sigma <= 0
    ):
        raise ValueError("prune_sigma must be positive and finite")
    ensemble = request.get("ensemble", {"enabled": True, "top": 3})
    if (
        not isinstance(ensemble, dict)
        or type(ensemble.get("top", 3)) is not int
        or ensemble.get("top", 3) < 1
    ):
        raise ValueError("ensemble.top must be a positive integer")
    if type(ensemble.get("enabled", True)) is not bool:
        raise ValueError("ensemble.enabled must be a boolean")
    return {
        **request,
        "transform": transform,
        "ensemble": ensemble,
        "max_concurrent_families": concurrency,
        "_max_concurrent_families_explicit": explicit_concurrency,
        "candidate_limit_fraction": fraction,
        "plateau": plateau,
        "min_generations": min_generations,
        "prune_sigma": prune_sigma,
    }
