"""the class-scheme seam: a class is a person's own cross-product, not a fixed list baked into code.

Ported from the parent project's classdefs module (see docs/REVIEW_TOOL_DESIGN.md, seam 3a/3b) with
two things dropped rather than carried across: the profile-rooted definitions directory, and the
colour-based guesser that used to choose a class for you. Both are enforced here, not just asserted
in prose, so a later edit that quietly reintroduces either fails the build.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest

from smolsmort.review import classscheme
from smolsmort.review.classscheme import UNSET, ClassDef, ClassDefError, Dimension

_PACKAGE_DIR = Path(classscheme.__file__).resolve().parent


# ---------------------------------------------------------------- the seam shape, test-enforced


@runtime_checkable
class _ClassSchemeShape(Protocol):
    """a local stand-in for the seam's `ClassScheme` Protocol (docs/REVIEW_TOOL_DESIGN.md, seam 3a).
    The real Protocol lives in tests/test_loop.py until smolsmort/review/seams.py exists; this
    package's lease does not cover authoring that module, so the shape is pinned independently here
    rather than importing across two test files that may land in either order."""

    def classes(self) -> list[str]: ...
    def label_for(self, picked) -> str: ...


def test_a_class_def_satisfies_the_class_scheme_seam_by_shape_alone():
    """no adapter class needed: ClassDef already has classes() and label_for(picked)"""
    definition = ClassDef(
        name="colours", dimensions=[Dimension(name="hue", members=["red", "blue"])]
    )
    assert isinstance(definition, _ClassSchemeShape)


def test_nothing_here_imports_the_parent_project():
    """THE DEPENDENCY THIS PORT CUTS. `parent.config.profile.PROFILES_ROOT` rooted every definitions
    directory under one game's screen/character profiles; a class scheme now stands on its own."""
    offenders = []
    for path in _PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            module = getattr(node, "module", None) if isinstance(node, ast.ImportFrom) else None
            names = [module] if module else []
            names += (
                [alias.name for alias in getattr(node, "names", [])]
                if isinstance(node, ast.Import)
                else []
            )
            if any(n and (n == "parent" or n.startswith("parent.")) for n in names):
                offenders.append(f"{path.relative_to(_PACKAGE_DIR)}:{node.lineno}")
    assert not offenders, f"reaches back into the parent project: {offenders}"


def test_nothing_here_offers_a_guesser():
    """the colour-based class guesser stays out of scope for this port - if it ever returns, it
    belongs behind its own seam implementation, not folded back into the scheme."""
    assert not any("guess" in name.lower() for name in classscheme.__all__)


def test_the_definitions_directory_is_not_rooted_in_a_profile():
    """no WoW profile root: the old default lived under a per-game profiles directory, keyed by
    screen. this one sits beside the rest of what the loop produces instead."""
    parts = {p.lower() for p in classscheme.DEFS_DIR.parts}
    assert "profiles" not in parts
    assert classscheme.DEFS_DIR.is_absolute()


# ---------------------------------------------------------------- the cross-product


def test_classes_is_the_sorted_cross_product_of_every_dimension():
    definition = ClassDef(
        name="test",
        dimensions=[
            Dimension(name="primitive", members=["hostile", "friendly"]),
            Dimension(name="state", members=["normal", "dim"]),
        ],
    )
    assert definition.classes() == sorted(
        [
            "hostile / normal",
            "hostile / dim",
            "friendly / normal",
            "friendly / dim",
        ]
    )


def test_a_dimension_with_no_members_yet_contributes_nothing():
    definition = ClassDef(
        name="test",
        dimensions=[
            Dimension(name="primitive", members=["hostile"]),
            Dimension(name="state", members=[]),
        ],
    )
    assert definition.classes() == ["hostile"]


def test_no_dimensions_means_no_classes():
    assert ClassDef(name="empty").classes() == []


# ---------------------------------------------------------------- label_for


def test_label_for_joins_one_member_per_dimension_in_definition_order():
    definition = ClassDef(
        name="test",
        dimensions=[
            Dimension(name="primitive", members=["hostile", "friendly"]),
            Dimension(name="state", members=["normal", "dim"]),
        ],
    )
    label = definition.label_for({"primitive": "friendly", "state": "dim"})
    assert label == "friendly / dim"


def test_label_for_writes_unset_for_an_unanswered_dimension_but_keeps_its_position():
    definition = ClassDef(
        name="test",
        dimensions=[
            Dimension(name="primitive", members=["hostile", "friendly"]),
            Dimension(name="state", members=["normal", "dim"]),
        ],
    )
    label = definition.label_for({"primitive": "hostile"})
    assert label == f"hostile{classscheme.JOIN}{UNSET}"


def test_label_for_refuses_a_label_that_answers_nothing():
    definition = ClassDef(
        name="test", dimensions=[Dimension(name="primitive", members=["hostile"])]
    )
    with pytest.raises(ClassDefError, match="nothing was picked"):
        definition.label_for({})


def test_label_for_refuses_a_member_not_in_its_dimension():
    definition = ClassDef(
        name="test", dimensions=[Dimension(name="primitive", members=["hostile"])]
    )
    with pytest.raises(ClassDefError, match="not a member"):
        definition.label_for({"primitive": "friendly"})


def test_label_for_refuses_a_definition_with_no_dimensions_at_all():
    with pytest.raises(ClassDefError, match="no dimensions"):
        ClassDef(name="empty").label_for({})


# ---------------------------------------------------------------- parsing


def test_parse_reads_name_and_dimensions_from_toml():
    definition = classscheme.parse(
        """
        name = "colours"
        [[dimension]]
        name = "hue"
        members = ["red", "blue"]
        """
    )
    assert definition.name == "colours"
    assert definition.dimensions == [Dimension(name="hue", members=["red", "blue"])]


def test_parse_refuses_unreadable_toml():
    with pytest.raises(ClassDefError, match="not readable as toml"):
        classscheme.parse("not = [valid")


def test_parse_refuses_a_missing_name():
    with pytest.raises(ClassDefError, match="needs a name"):
        classscheme.parse('[[dimension]]\nname = "hue"\nmembers = ["red"]\n')


def test_parse_refuses_more_than_the_readable_number_of_dimensions():
    toml = 'name = "test"\n' + "\n".join(
        f'[[dimension]]\nname = "d{i}"\nmembers = ["a"]\n'
        for i in range(classscheme.MAX_DIMENSIONS + 1)
    )
    with pytest.raises(ClassDefError, match="dimensions"):
        classscheme.parse(toml)


def test_parse_refuses_two_dimensions_sharing_a_name():
    toml = """
    name = "test"
    [[dimension]]
    name = "hue"
    members = ["red"]
    [[dimension]]
    name = "hue"
    members = ["blue"]
    """
    with pytest.raises(ClassDefError, match="share a name"):
        classscheme.parse(toml)


def test_parse_refuses_a_dimension_listing_the_same_member_twice():
    toml = """
    name = "test"
    [[dimension]]
    name = "hue"
    members = ["red", "red"]
    """
    with pytest.raises(ClassDefError, match="twice"):
        classscheme.parse(toml)


# ---------------------------------------------------------------- save / load / available


def test_save_then_load_round_trips_a_definition(tmp_path):
    definition = ClassDef(
        name="Test Scheme",
        dimensions=[Dimension(name="hue", members=["red", "blue"])],
    )
    path = classscheme.save(definition, tmp_path)
    assert path == tmp_path / "test-scheme.toml"

    loaded = classscheme.load("Test Scheme", tmp_path)
    assert loaded == definition


def test_load_is_case_and_punctuation_insensitive_via_the_slug(tmp_path):
    # save() and load() both slugify the name, so a definition saved under one spelling is found
    # under any spelling that slugifies the same way
    classscheme.save(ClassDef(name="Test Scheme"), tmp_path)
    assert classscheme.load("test---scheme", tmp_path).name == "Test Scheme"


def test_available_lists_slugs_sorted(tmp_path):
    classscheme.save(ClassDef(name="Zebra"), tmp_path)
    classscheme.save(ClassDef(name="Apple"), tmp_path)
    assert classscheme.available(tmp_path) == ["apple", "zebra"]


def test_available_on_a_missing_directory_is_empty(tmp_path):
    assert classscheme.available(tmp_path / "does-not-exist") == []


def test_load_refuses_a_name_with_no_matching_file(tmp_path):
    with pytest.raises(ClassDefError, match="no definition called"):
        classscheme.load("nope", tmp_path)


def test_load_refuses_a_name_that_would_escape_the_definitions_directory(tmp_path):
    # RESOLVE THEN CONTAIN: a name reaching load() from an http route must never read outside its
    # directory just because it is spelled with a path separator or "..". _slug already strips both
    # to "-", so this proves the containment holds even if a future caller stopped slugifying first.
    outside = tmp_path.parent / "escaped.toml"
    outside.write_text('name = "escaped"\n')
    with pytest.raises(ClassDefError):
        classscheme.load("../escaped", tmp_path)


def test_as_toml_round_trips_through_parse():
    definition = ClassDef(
        name="Test",
        dimensions=[
            Dimension(name="primitive", members=["hostile", "friendly"]),
            Dimension(name="state", members=["normal"]),
        ],
    )
    reparsed = classscheme.parse(definition.as_toml())
    assert reparsed == definition


def test_as_json_reports_name_slug_dimensions_and_classes():
    definition = ClassDef(name="Test", dimensions=[Dimension(name="hue", members=["red"])])
    payload = definition.as_json()
    assert payload == {
        "name": "Test",
        "slug": "test",
        "dimensions": [{"name": "hue", "members": ["red"]}],
        "classes": ["red"],
    }
