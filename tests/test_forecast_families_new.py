"""catboost and extra trees are registered, show install hints and use the names effort scales"""

from __future__ import annotations

from smolsmort.forecast.effort import COSTLY
from smolsmort.forecast.families import get_family, registered_families

NEW = {"catboost": ("lag_features", "catboost"), "extratrees": ("lag_features", "trees")}


def test_new_families_are_registered_with_their_extras():
    names = [family.name for family in registered_families()]
    for name, (needs, extra) in NEW.items():
        assert names.count(name) == 1
        family = get_family(name)
        assert family.needs == needs
        assert family.pip_extra == extra


def test_missing_library_gives_the_install_hint(monkeypatch):
    from smolsmort.forecast.families import base

    def refuse(name):
        raise ImportError(f"no {name}")

    monkeypatch.setattr(base, "import_module", refuse)
    for name, (_needs, extra) in NEW.items():
        available, hint = get_family(name).available()
        assert not available
        assert f"uv sync --extra {extra}" in hint


def test_cost_params_use_the_names_effort_scales():
    for name in NEW:
        space = {param.name for param in get_family(name).space()}
        assert space & set(COSTLY.get(name, {})), name
