"""target transform values are per-run v2 settings"""

from __future__ import annotations

import pytest

from smolsmort.forecast.requests import parse_request


def make_request(**changes):
    return {
        "version": 2,
        "time_budget_s": 2,
        "families": [{"name": "snaive"}],
        **changes,
    }


@pytest.mark.parametrize("transform", ["none", "log1p", "logdiff"])
def test_v2_preserves_supported_transform(transform):
    request = make_request(transform=transform)

    parsed = parse_request(request)

    assert parsed["transform"] == transform
    assert request["transform"] == transform


def test_v2_defaults_to_none_without_mutating_request():
    request = make_request()

    parsed = parse_request(request)

    assert parsed["transform"] == "none"
    assert "transform" not in request


@pytest.mark.parametrize("transform", ["log", "LOG1P", "", None, 0, False, [], {}])
def test_v2_rejects_unknown_transform_with_supported_values(transform):
    with pytest.raises(ValueError, match="transform must be none, log1p or logdiff"):
        parse_request(make_request(transform=transform))


def test_unversioned_request_stays_unchanged():
    request = {"budget": {"population": 3}}

    assert parse_request(request) is request
    assert "transform" not in request
