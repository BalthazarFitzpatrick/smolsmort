"""train / val / test is assigned per frame, reproducibly"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from smolsmort.review import splits

FRAMES = [f"{i:07d}.jpg" for i in range(1000)]


def test_seventy_twenty_ten_lands_exactly():
    got = splits.assign(FRAMES, (70, 20, 10), seed=1)
    counts = {name: list(got.values()).count(name) for name in splits.SPLITS}
    assert counts == {"train": 700, "val": 200, "test": 100}


def test_the_counts_always_sum_to_the_frames():
    """a round() per split loses or invents a frame at most sizes"""
    for n in range(1, 60):
        got = splits.assign(FRAMES[:n], (70, 20, 10), seed=0)
        assert len(got) == n


def test_a_split_asked_for_is_never_empty_when_there_are_frames_to_give():
    """a holdout of zero frames reports a perfect score"""
    got = splits.assign(FRAMES[:4], (70, 20, 10), seed=0)
    assert set(got.values()) == set(splits.SPLITS)


def test_the_same_seed_splits_the_same_way_in_any_order():
    forward = splits.assign(FRAMES, seed=5)
    backward = splits.assign(reversed(FRAMES), seed=5)
    assert forward == backward


def test_a_different_seed_draws_a_different_holdout():
    a = {f for f, s in splits.assign(FRAMES, seed=1).items() if s == "test"}
    b = {f for f, s in splits.assign(FRAMES, seed=2).items() if s == "test"}
    assert a != b


def test_the_holdout_does_not_track_generation_order():
    """test must not simply be the last tenth of the frames"""
    got = splits.assign(FRAMES, seed=0)
    test = [f for f in FRAMES if got[f] == "test"]
    assert test != FRAMES[-100:]
    assert any(f in test for f in FRAMES[:500])


@pytest.mark.parametrize("text", ["70/20", "70/20/10/0", "a/b/c", "0/0/0", "-1/50/51"])
def test_a_bad_ratio_is_refused(text):
    with pytest.raises(ValueError):
        splits.parse_ratio(text)


def test_a_ratio_reads_with_slashes_or_commas():
    assert splits.parse_ratio("70/20/10") == (70, 20, 10)
    assert splits.parse_ratio("80,10,10") == (80, 10, 10)


def test_keep_matches_examples_back_through_the_frame_path(tmp_path):
    sessions = tmp_path / "sessions"
    rows = [
        {"recording": "synthetic/run", "frame": "a.jpg", "split": "train"},
        {"recording": "synthetic/run", "frame": "b.jpg", "split": "val"},
        {"recording": "synthetic/run", "frame": "c.jpg", "split": "test"},
    ]
    examples = [
        SimpleNamespace(path=sessions / "synthetic" / "run" / "frames" / name)
        for name in ("a.jpg", "b.jpg", "c.jpg")
    ]
    for split, frame in (("train", "a.jpg"), ("val", "b.jpg"), ("test", "c.jpg")):
        kept = splits.keep(examples, rows, sessions, split)
        assert [e.path.name for e in kept] == [frame]


def test_a_set_from_before_splits_is_all_train(tmp_path):
    sessions = tmp_path / "sessions"
    rows = [{"recording": "old", "frame": "a.jpg"}]
    examples = [SimpleNamespace(path=sessions / "old" / "frames" / "a.jpg")]
    assert splits.keep(examples, rows, sessions, "train") == examples
    assert splits.keep(examples, rows, sessions, "test") == []


def test_an_unknown_split_is_refused():
    with pytest.raises(ValueError):
        splits.keep([], [], Path("."), "holdout")


def _old_set(tmp_path, frames=10):
    path = tmp_path / "old.jsonl"
    rows = [
        {"recording": "synthetic/run", "frame": f"{i}.jpg", "negative": False}
        for i in range(frames)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows * 3))
    path.with_suffix(".meta.json").write_text(json.dumps({"name": "old", "rows": len(rows) * 3}))
    return path


def test_an_old_set_is_split_in_place_and_its_meta_records_it(tmp_path):
    path = _old_set(tmp_path)
    counts = splits.write_into(path, (70, 20, 10))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert counts["frames"] == {"train": 7, "val": 2, "test": 1}
    assert counts["rows"] == {"train": 21, "val": 6, "test": 3}
    assert json.loads(path.with_suffix(".meta.json").read_text())["split"] == counts
    seen: dict[str, str] = {}
    for row in rows:
        assert seen.setdefault(row["frame"], row["split"]) == row["split"]


def test_a_set_that_already_has_a_split_is_never_redrawn(tmp_path):
    """a checkpoint may already have been judged against its holdout"""
    path = _old_set(tmp_path)
    splits.write_into(path)
    before = path.read_text()
    with pytest.raises(ValueError):
        splits.write_into(path, seed=9)
    assert path.read_text() == before


# ---------------------------------------------------------------- partition into train/val/test


def _items(n=100):
    return [{"frame": f"rec/f{i:03d}", "row": i} for i in range(n)] * 2  # two rows per frame


def test_partition_sizes_follow_the_ratio_and_default_has_a_test_split():
    from smolsmort.review import splits

    out = splits.partition(_items(), key=lambda r: r["frame"])
    frames = {s: len({r["frame"] for r in rows}) for s, rows in out.items()}
    assert frames == {"train": 70, "val": 20, "test": 10}
    custom = splits.partition(_items(), lambda r: r["frame"], ratio=(50, 25, 25))
    assert {len({r["frame"] for r in v}) for v in custom.values()} == {50, 25}


def test_partition_never_leaks_a_frame_between_splits_and_keeps_every_row():
    from smolsmort.review import splits

    out = splits.partition(_items(), key=lambda r: r["frame"])
    seen = [{r["frame"] for r in rows} for rows in out.values()]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    assert sum(len(v) for v in out.values()) == 200


def test_partition_is_deterministic_by_seed_and_can_skip_test():
    from smolsmort.review import splits

    key = lambda r: r["frame"]  # noqa: E731
    first = splits.partition(_items(), key, seed=3)
    assert first == splits.partition(_items(), key, seed=3)
    assert first != splits.partition(_items(), key, seed=4)
    assert splits.partition(_items(), key, ratio=(80, 20, 0))["test"] == []
