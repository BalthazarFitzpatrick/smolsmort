"""the housekeeping tab's server side: recording discovery, orphan detection, tag peeling, and
the two destructive actions (archive, delete) - never against real data, always a tmp_path tree
the fixture below builds and paths.py is monkeypatched onto.
"""

from __future__ import annotations

import json

import pytest

from smolsmort.review import housekeeping as hk
from smolsmort.review import paths


@pytest.fixture
def bases(tmp_path, monkeypatch):
    """a full training/ tree under tmp_path, with paths.* repointed at it"""
    sessions = tmp_path / "sessions"
    training = tmp_path / "training"
    labels = training / "boxes"
    tiles = training / "tiles"
    tiles_synth = training / "tiles-synth"
    sets = training / "sets"
    checkpoints = training / "weights" / "checkpoints"
    for d in (sessions, labels, tiles, tiles_synth, sets, checkpoints):
        d.mkdir(parents=True)

    monkeypatch.setattr(paths, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(paths, "LABELS_DIR", labels)
    monkeypatch.setattr(paths, "TILES_DIR", tiles)
    monkeypatch.setattr(paths, "TILES_SYNTH_DIR", tiles_synth)
    monkeypatch.setattr(paths, "DATASETS_DIR", sets)
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", checkpoints)
    return {
        "sessions": sessions,
        "labels": labels,
        "tiles": tiles,
        "tiles_synth": tiles_synth,
        "sets": sets,
        "checkpoints": checkpoints,
    }


# ---------------------------------------------------------------- tag peeling


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("some_test__1_cnn-20260908-090744_k00797", "some_test__1"),
        ("some_test__3.drawn-20260901-0148", "some_test__3"),
        ("some_test__3.cnn-20260907-011512", "some_test__3"),
        # two stamps stacked - a re-swept set's source name
        (
            "target_bearing_distance_more_zooms_cnn-20260903-222136",
            "target_bearing_distance_more_zooms",
        ),
        ("calibration_run__2026-09-06_k00207", "calibration_run__2026-09-06"),
    ],
)
def test_tag_to_recording_peels_stamps_and_tile_index(tag, expected):
    assert hk.tag_to_recording(tag) == expected


def test_unflatten_restores_the_nested_session_path():
    assert hk.unflatten("some_test__1") == "some_test/1"


# ---------------------------------------------------------------- recording discovery


def test_live_recording_is_found_and_grouped_by_its_directory(bases):
    (bases["sessions"] / "some_test" / "1" / "frames").mkdir(parents=True)
    recordings = hk.all_recordings()
    assert len(recordings) == 1
    rec = recordings[0]
    assert rec.tag == "some_test__1"
    assert rec.path == "some_test/1"
    assert rec.group == "some_test"
    assert rec.missing is False


def test_flat_recording_is_its_own_top_level_group(bases):
    (bases["sessions"] / "solo_run" / "frames").mkdir(parents=True)
    rec = hk.all_recordings()[0]
    assert rec.group == "solo_run"
    assert rec.missing is False


def test_a_deleted_recording_with_leftover_boxes_shows_up_as_an_orphan(bases):
    # nothing under sessions/ at all - the recording is gone
    (bases["labels"] / "ghost_run.drawn-20260901-0018.candidates.jsonl").write_text("{}")
    recordings = hk.all_recordings()
    assert len(recordings) == 1
    assert recordings[0].tag == "ghost_run"
    assert recordings[0].missing is True


def test_a_live_recording_with_no_assets_is_not_flagged_missing(bases):
    (bases["sessions"] / "still_here" / "frames").mkdir(parents=True)
    recordings = hk.all_recordings()
    assert len(recordings) == 1
    assert recordings[0].missing is False


# ---------------------------------------------------------------- set orphaning needs ALL sources gone


def test_a_set_with_one_live_source_is_not_orphaned(bases):
    (bases["sessions"] / "still_here" / "frames").mkdir(parents=True)
    meta = {
        "name": "mixed_set",
        "sources": [{"source": "still_here"}, {"source": "ghost_run"}],
    }
    (bases["sets"] / "mixed_set.meta.json").write_text(json.dumps(meta))
    assert "mixed_set" not in hk.orphaned_sets()


def test_a_set_with_every_source_gone_is_orphaned(bases):
    meta = {"name": "dead_set", "sources": [{"source": "ghost_one"}, {"source": "ghost_two"}]}
    (bases["sets"] / "dead_set.meta.json").write_text(json.dumps(meta))
    assert "dead_set" in hk.orphaned_sets()


# ---------------------------------------------------------------- assets_for, grouped by kind


def test_assets_for_finds_boxes_tiles_and_their_sidecars(bases):
    tag = "some_test__1"
    jsonl = bases["labels"] / f"{tag}.drawn-20260901-0018.candidates.jsonl"
    jsonl.write_text("{}")
    decisions = bases["labels"] / f"{tag}.drawn-20260901-0018.candidates.decisions.json"
    decisions.write_text("{}")
    (bases["tiles"] / f"{tag}_k00001.npz").write_bytes(b"x")
    (bases["tiles"] / f"{tag}_k00002.npz").write_bytes(b"xx")
    # a different recording's tile must never bleed into this one's list
    (bases["tiles"] / "other_recording_k00001.npz").write_bytes(b"y")

    groups = hk.assets_for(tag)
    assert len(groups["boxes"]) == 1
    assert set(groups["boxes"][0]["paths"]) == {str(jsonl.resolve()), str(decisions.resolve())}
    assert len(groups["tiles"]) == 1
    assert len(groups["tiles"][0]["paths"]) == 2
    assert groups["tiles_synth"] == []


def test_assets_for_finds_a_set_from_any_of_its_several_sources(bases):
    meta = {
        "name": "combo",
        "sources": [{"source": "rec_a"}, {"source": "rec_b_cnn-20260901-010101"}],
    }
    (bases["sets"] / "combo.meta.json").write_text(json.dumps(meta))
    (bases["sets"] / "combo.jsonl").write_text("{}")

    assert len(hk.assets_for("rec_a")["sets"]) == 1
    assert len(hk.assets_for("rec_b")["sets"]) == 1
    assert hk.assets_for("rec_c")["sets"] == []


def test_assets_for_finds_a_checkpoint_via_its_training_sets_meta(bases):
    meta = {"name": "combo", "sources": [{"source": "rec_a"}]}
    (bases["sets"] / "combo.meta.json").write_text(json.dumps(meta))
    classes = bases["checkpoints"] / "combo-20260901-000000.classes.json"
    classes.write_text(json.dumps({"training_set": "combo", "classes": ["x"]}))
    (bases["checkpoints"] / "combo-20260901-000000.pt").write_bytes(b"weights")

    assert len(hk.assets_for("rec_a")["checkpoints"]) == 1
    assert hk.assets_for("rec_z")["checkpoints"] == []


# ---------------------------------------------------------------- containment refusal


def test_delete_refuses_a_path_outside_the_configured_bases(bases, tmp_path):
    outside = tmp_path / "somewhere_else" / "passwd"
    outside.parent.mkdir()
    outside.write_text("do not touch")
    with pytest.raises(hk.HousekeepingError):
        hk.delete_paths([str(outside)])
    assert outside.exists()


def test_delete_refuses_a_traversal_that_escapes_the_labels_dir(bases):
    escaping = str(bases["labels"] / ".." / ".." / "etc" / "passwd")
    with pytest.raises(hk.HousekeepingError):
        hk.delete_paths([escaping])


def test_a_refused_batch_touches_nothing_even_when_part_of_it_is_valid(bases):
    """one bad path in a batch must not let the good ones through - a partial delete is exactly
    the kind of half-consistent state this whole module exists to avoid"""
    good = bases["labels"] / "keep_me.drawn-20260901-0018.candidates.jsonl"
    good.write_text("{}")
    outside = "/etc/passwd"
    with pytest.raises(hk.HousekeepingError):
        hk.delete_paths([str(good), outside])
    assert good.exists()


# ---------------------------------------------------------------- archive is reversible


def test_archive_moves_a_box_file_into_a_sibling_archive_dir_with_content_intact(bases):
    jsonl = bases["labels"] / "rec.drawn-20260901-0018.candidates.jsonl"
    jsonl.write_text('{"kept": true}')
    result = hk.archive_paths([str(jsonl)])
    assert result["ok"] is True
    archived = bases["labels"] / "_archive" / jsonl.name
    assert archived.exists()
    assert archived.read_text() == '{"kept": true}'
    assert not jsonl.exists()


def test_delete_actually_removes_the_file(bases):
    jsonl = bases["labels"] / "rec.drawn-20260901-0018.candidates.jsonl"
    jsonl.write_text("{}")
    hk.delete_paths([str(jsonl)])
    assert not jsonl.exists()


# ---------------------------------------------------------------- _labels.json stays consistent


def test_removing_a_tile_updates_the_labels_index_in_the_same_operation(bases):
    tile = bases["tiles"] / "rec_k00001.npz"
    tile.write_bytes(b"x")
    other_tile = bases["tiles"] / "rec_k00002.npz"
    other_tile.write_bytes(b"y")
    index_path = bases["tiles"] / "_labels.json"
    index_path.write_text(
        json.dumps(
            {
                "rec_k00001": {"label": "friendly"},
                "rec_k00002": {"label": "hostile"},
            }
        )
    )

    hk.delete_paths([str(tile)])

    index = json.loads(index_path.read_text())
    assert "rec_k00001" not in index, "a corrupted index still names the tile that was removed"
    assert "rec_k00002" in index, "removing one tile must not touch another's entry"


def test_archiving_a_tile_also_updates_the_labels_index(bases):
    tile = bases["tiles"] / "rec_k00001.npz"
    tile.write_bytes(b"x")
    index_path = bases["tiles"] / "_labels.json"
    index_path.write_text(json.dumps({"rec_k00001": {"label": "friendly"}}))

    hk.archive_paths([str(tile)])

    index = json.loads(index_path.read_text())
    assert "rec_k00001" not in index
    assert (bases["tiles"] / "_archive" / "rec_k00001.npz").exists()
