"""the review tool's mutable paths must be read through the module, never imported by value.

THE FAILURE THIS PREVENTS, and it is not a style rule. `SESSIONS_DIR` and `LABELS_DIR` are defaults
that the settings popup repoints at runtime, and that six test fixtures monkeypatch so a test writes
into its own tmp_path. A module doing `from ...paths import SESSIONS_DIR` binds a COPY at import
time: the test patches one name, the code reads the other, and nothing raises. The suite stays green
while the code reads and writes the REAL sessions/ and labels/ directories - a data-loss bug wearing
a passing test run.

Reading `paths.SESSIONS_DIR` at call time has no such gap, so that is the only allowed form.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from snapshot.review import paths

# repointed at runtime by set_bases, and monkeypatched by tests - a copy of any goes stale.
# INTERFACE_SHOTS joined them after the suite was found writing into the real working set: it is
# not repointed by the settings popup, but conftest redirects it for every test, which needs the
# same attribute access to work
MUTABLE = {"SESSIONS_DIR", "LABELS_DIR", "INTERFACE_SHOTS"}

PACKAGE = Path(paths.__file__).resolve().parents[2]


def _python_files() -> list[Path]:
    return [p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts]


def test_nothing_imports_the_mutable_paths_by_value():
    offenders = []
    for path in _python_files():
        if path.resolve() == Path(paths.__file__).resolve():
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.endswith("review.paths"):
                continue
            taken = {alias.name for alias in node.names} & MUTABLE
            if taken:
                offenders.append(
                    f"{path.relative_to(PACKAGE)}:{node.lineno} imports {sorted(taken)}"
                )
    assert not offenders, (
        "these bind a copy that a runtime repoint and every test monkeypatch will miss - "
        "read them as paths.<NAME> instead:\n  " + "\n  ".join(offenders)
    )


def test_the_paths_module_is_the_one_that_moves():
    """a repoint has to be visible to every reader, which is what attribute access buys"""
    original = paths.SESSIONS_DIR
    try:
        paths.SESSIONS_DIR = Path("/tmp/somewhere-else")
        from snapshot.review import paths as seen_elsewhere

        assert Path("/tmp/somewhere-else") == seen_elsewhere.SESSIONS_DIR
    finally:
        paths.SESSIONS_DIR = original


@pytest.mark.parametrize("name", sorted(MUTABLE))
def test_the_defaults_are_absolute(name):
    """relative defaults only worked when the tool was launched from the repo root, and silently
    found nothing from anywhere else"""
    assert getattr(paths, name).is_absolute()


def test_the_bases_point_at_the_repo_not_the_package():
    base = Path(paths.DEFAULT_BASES["sessions"])
    assert base.name == "sessions"
    assert (base.parent / "pyproject.toml").exists()


# ---------------------------------------------------------------- the data flow is named once


def test_every_directory_the_loop_touches_is_named_here():
    """SEVEN MODULES EACH BUILT THESE THEMSELVES, so "where do the weights go" had seven answers
    and moving anything meant finding all of them. This is the list; it is what makes task 100 a
    change in one file rather than a hunt.
    """
    for name in (
        "RECORDINGS",
        "LABELS_DIR",
        "TILES_DIR",
        "TILES_ARCHIVE",
        "LIBRARY_DIR",
        "DATASETS_DIR",
        "MODEL_PATH",
        "CHECKPOINTS_DIR",
        "WEIGHTS_DIR",
        "TRAINING_DIR",
    ):
        assert hasattr(paths, name), (
            f"paths.{name} is gone - the flow is no longer named in one place"
        )


def test_drawn_and_swept_boxes_are_told_apart_by_name(tmp_path):
    """THE DISTINCTION THE WHOLE LOOP TURNS ON. A human's boxes and a model's proposals live in the
    same directory and differ only by infix, and confusing them has cost real work: promote once
    resolved hand-drawn tiles against an empty sweep file because ".cnn-" sorts before ".drawn-".
    """
    (tmp_path / "rec.drawn-20260903-0114.candidates.jsonl").write_text("")
    (tmp_path / "rec.drawn-20260901-0018.candidates.jsonl").write_text("")
    (tmp_path / "rec.cnn-20260903-015830.candidates.jsonl").write_text("")

    drawn = paths.drawn_candidates(tmp_path, "rec")
    swept = paths.swept_candidates(tmp_path, "rec")
    assert [p.name for p in drawn] == [
        "rec.drawn-20260901-0018.candidates.jsonl",
        "rec.drawn-20260903-0114.candidates.jsonl",
    ], "drawn files must come back oldest first, and a sweep must never appear among them"
    assert [p.name for p in swept] == ["rec.cnn-20260903-015830.candidates.jsonl"]


def test_no_data_path_is_relative_to_the_working_directory():
    """A RELATIVE DATA PATH ONLY WORKS FROM THE REPO ROOT and fails silently anywhere else - it does
    not raise, it writes a second model, or matches an empty library, somewhere nobody looks.
    TrainState.MODEL_PATH was Path("vision/plate_model.pt").
    """
    from snapshot.review.paths import LIBRARY_DIR
    from snapshot.review.train import TrainState

    assert TrainState.MODEL_PATH.is_absolute()
    assert LIBRARY_DIR.is_absolute()
    for name in (
        "TILES_DIR",
        "TILES_ARCHIVE",
        "LIBRARY_DIR",
        "DATASETS_DIR",
        "MODEL_PATH",
        "CHECKPOINTS_DIR",
        "WEIGHTS_DIR",
        "TRAINING_DIR",
        "INTERFACE_SHOTS",
    ):
        assert getattr(paths, name).is_absolute(), f"paths.{name} is relative to the cwd"
