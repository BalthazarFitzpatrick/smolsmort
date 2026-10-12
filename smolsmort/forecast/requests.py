"""validate the new request while leaving unversioned requests untouched"""

from __future__ import annotations

import math
import re


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
    return {
        **request,
        "transform": transform,
        "ensemble": request.get("ensemble", {"enabled": False, "top": 3}),
    }
