"""canonical forecast modes preserve saved specs, prepared caches and run readers"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from smolsmort.forecast import prep, views
from smolsmort.forecast.features import family_aliases
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.spec import FILES, Column, PrepSpec, SpecError, spec_from_dict
from smolsmort.forecast.tab import _columns, _parse_spec
from smolsmort.review.routes import RequestError

MODES = [("series", "timeseries"), ("row", "entity")]


def make_spec(mode, source="data.csv"):
    clock = "time" if mode == "timeseries" else "anchor"
    return PrepSpec(
        str(source),
        mode,
        "regression",
        (Column("day", clock), Column("value", "target", aggregation="sum")),
        step="day" if mode == "timeseries" else None,
    )


def make_app(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("day,value\n2026-01-01,2\n2026-01-02,3\n2026-01-03,\n")
    return SimpleNamespace(state=SimpleNamespace(bases={"forecast": str(tmp_path)})), source


def legacy_digest(spec, legacy):
    data = asdict(spec)
    data["mode"] = legacy
    if data["holidays_country"] is None:
        del data["holidays_country"]
    # pre-flag specs never carried skip_missing
    for column in data["columns"]:
        column.pop("skip_missing")
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]


@pytest.mark.parametrize(("legacy", "canonical"), MODES)
@pytest.mark.parametrize("country", [None, "DE"])
def test_saved_spec_keeps_its_digest_and_serializes_the_new_name(legacy, canonical, country):
    spec = replace(make_spec(canonical), holidays_country=country)
    saved = {**asdict(spec), "mode": legacy}
    restored = spec_from_dict(saved)

    assert restored == spec
    assert restored.digest() == spec.digest() == legacy_digest(spec, legacy)
    assert json.loads(json.dumps(asdict(restored)))["mode"] == canonical
    assert spec_from_dict(json.loads(json.dumps(asdict(spec)))) == spec
    assert saved["mode"] == legacy


@pytest.mark.parametrize(("legacy", "canonical"), MODES)
def test_existing_prepared_cache_is_reused_without_rewriting_it(
    tmp_path, monkeypatch, legacy, canonical
):
    pytest.importorskip("duckdb")
    _, source = make_app(tmp_path)
    spec = make_spec(canonical, source)
    saved = {**asdict(spec), "mode": legacy}
    first = prep.prepare(spec_from_dict(saved), tmp_path / "cache")
    key = f"{legacy_digest(spec, legacy)}-{hashlib.sha256(source.read_bytes()).hexdigest()[:12]}"
    assert first.folder.name == key
    assert not first.reused
    assert first.summary["mode"] == canonical

    # old caches carry old metadata as well as the digest; parquet names stay stable
    summary_path = first.folder / "prep.json"
    summary_path.write_text(json.dumps({**first.summary, "mode": legacy}))
    (first.folder / "spec.json").write_text(json.dumps(saved))
    files_before = {path.name: path.read_bytes() for path in first.folder.iterdir()}

    def refuse_rebuild():
        pytest.fail("cache hit must not open a new preparation connection")

    monkeypatch.setattr(prep, "connect", refuse_rebuild)
    second = prep.prepare(spec, tmp_path / "cache")

    assert second.reused
    assert second.folder == first.folder
    assert second.sql == first.sql
    assert second.summary["mode"] == canonical
    assert {path.name: path.read_bytes() for path in first.folder.iterdir()} == files_before
    assert views.read_prepared_spec(first.folder, tmp_path / "runs") == spec
    assert all((second.folder / name).is_file() for name in FILES[canonical])


@pytest.mark.parametrize(("legacy", "canonical"), MODES)
@pytest.mark.parametrize("use_legacy", [False, True])
def test_route_parser_accepts_both_spellings(tmp_path, legacy, canonical, use_legacy):
    app, source = make_app(tmp_path)
    spec = make_spec(canonical, source)
    payload = {"spec": {**asdict(spec), "mode": legacy if use_legacy else canonical}}
    assert _parse_spec(app, payload) == spec


@pytest.mark.parametrize("mode", ["series", "timeseries"])
def test_legacy_spelling_cannot_bypass_route_validation(tmp_path, mode):
    app, source = make_app(tmp_path)
    data = {**asdict(make_spec("timeseries", source)), "mode": mode, "task": "classification"}
    with pytest.raises(RequestError, match="require regression"):
        _parse_spec(app, {"spec": data})


@pytest.mark.parametrize("mode", ["row", "entity"])
def test_entity_route_still_supports_classification(tmp_path, mode):
    app, source = make_app(tmp_path)
    data = {**asdict(make_spec("entity", source)), "mode": mode, "task": "classification"}
    parsed = _parse_spec(app, {"spec": data})
    assert parsed.mode == "entity"
    assert parsed.task == "classification"


@pytest.mark.parametrize("mode", ["unknown", "", None, []])
def test_unknown_modes_are_rejected_by_specs_and_routes(tmp_path, mode):
    app, source = make_app(tmp_path)
    data = {**asdict(make_spec("entity", source)), "mode": mode}
    with pytest.raises(SpecError, match="timeseries or entity"):
        spec_from_dict(data)
    with pytest.raises(RequestError, match="timeseries or entity"):
        _parse_spec(app, {"spec": data})


@pytest.mark.parametrize(("legacy", "canonical"), MODES)
def test_column_suggestions_accept_saved_mode_names(tmp_path, legacy, canonical):
    pytest.importorskip("duckdb")
    app, source = make_app(tmp_path)
    payload = _columns(app, {"source": source.name, "mode": legacy})
    clock = next(column for column in payload["columns"] if column["name"] == "day")
    assert clock["suggested_role"] == ("time" if canonical == "timeseries" else "anchor")


@pytest.mark.parametrize(("legacy", "canonical"), MODES)
def test_saved_requests_results_and_recipes_load_canonical_modes(tmp_path, legacy, canonical):
    spec = make_spec(canonical)
    saved_spec = {**asdict(spec), "mode": legacy}
    saved = {
        "spec": saved_spec,
        "summary": {"mode": legacy},
        "recipe": {"spec": saved_spec},
        "unchanged": "kept",
    }
    expected = {
        "spec": asdict(spec),
        "summary": {"mode": canonical},
        "recipe": {"spec": asdict(spec)},
        "unchanged": "kept",
    }
    expected = json.loads(json.dumps(expected))
    for name in ("request.json", "result.json", "recipe.json"):
        (tmp_path / name).write_text(json.dumps(saved))

    request = views.read_request(tmp_path)
    result = views.read_result(tmp_path)
    recipe = views._read_json(tmp_path / "recipe.json")
    assert request == result == recipe == expected
    assert parse_request(json.loads(json.dumps(saved))) == expected
    assert spec_from_dict(recipe["spec"]) == spec
    assert json.loads((tmp_path / "request.json").read_text())["spec"]["mode"] == legacy


@pytest.mark.parametrize(("legacy", "canonical"), MODES)
def test_legacy_requests_select_the_correct_view_and_table(
    tmp_path, monkeypatch, legacy, canonical
):
    request = {"spec": {**asdict(make_spec(canonical)), "mode": legacy}}
    selected = {"model": "winner", "family": "xgboost", "members": []}
    monkeypatch.setattr(views, "result_details", lambda *args: selected)
    monkeypatch.setattr(views, "_require_done", lambda *args: {})
    monkeypatch.setattr(views, "_select_model", lambda *args: selected)
    monkeypatch.setattr(views, "entity_view", lambda *args: {"mode": "entity", "series": []})
    monkeypatch.setattr(
        views, "timeseries_view", lambda *args: {"mode": "timeseries", "series": []}
    )
    monkeypatch.setattr(views, "entity_table", lambda *args: {"columns": ["entity"], "rows": []})
    monkeypatch.setattr(
        views, "timeseries_table", lambda *args: {"columns": ["timeseries"], "rows": []}
    )

    assert views.view(tmp_path, request, {})["mode"] == canonical
    assert views.table(tmp_path, request, {})["columns"][0] == canonical


@pytest.mark.parametrize(("legacy", "canonical"), MODES)
def test_saved_calendar_families_keep_the_spec_clock(legacy, canonical):
    spec = spec_from_dict({**asdict(make_spec(canonical)), "mode": legacy})
    calendar = "cal:month" if canonical == "timeseries" else "date:month"
    assert family_aliases(spec, [calendar])[calendar] == "day:cal:month"


@pytest.mark.parametrize(("legacy", "canonical"), MODES)
def test_versioned_requests_normalize_saved_modes(legacy, canonical):
    spec = make_spec(canonical)
    saved = {
        "version": 2,
        "time_budget_s": 60,
        "families": [{"name": "xgboost"}],
        "spec": {**asdict(spec), "mode": legacy},
        "summary": {"mode": legacy},
    }
    parsed = parse_request(saved)
    assert parsed["spec"]["mode"] == parsed["summary"]["mode"] == canonical
    assert parsed["transform"] == "none"
    assert saved["spec"]["mode"] == saved["summary"]["mode"] == legacy


@pytest.mark.parametrize(("legacy", "canonical"), MODES)
def test_finish_estimate_keeps_the_mode_specific_fit_count(
    tmp_path, monkeypatch, legacy, canonical
):
    from smolsmort.forecast import coordinator

    monkeypatch.setattr(coordinator, "best_entry", lambda *args: {"elapsed_s": 10})
    expected = 30 if canonical == "entity" else 20
    for mode in (legacy, canonical):
        request = {"spec": {"mode": mode}}
        assert coordinator.estimate_finish_seconds(tmp_path, ["xgboost"], {}, request) == expected


@pytest.mark.parametrize("relative", ["README.md", "docs/GUIDE.md"])
def test_docs_have_no_old_mode_labels(relative):
    path = Path(__file__).resolve().parents[1] / relative
    assert re.search(r"\b(?:row|series)[ -]mode\b", path.read_text(), re.IGNORECASE) is None
