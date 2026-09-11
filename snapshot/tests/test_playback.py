"""replaying a recorded session for the overlay tab.

the tab promised "nothing playing back yet" for months. This is the part that answers what was true
at a moment - and INTERPRETS it rather than dumping rows, because the point is to watch the pipeline
label real play: the same windows autolabel builds, the same gaps, the same modes.
"""

from __future__ import annotations

import json

import pytest

from snapshot.tools import playback


def _session(root, name, states, events=(), mouse=()):
    """a recording in the layout resolve_session_paths expects: <name>/<name>.jsonl"""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"kind": "state", **s} for s in states]
    rows += [{"kind": "key", **e} for e in events]
    rows += list(mouse)
    rows.sort(key=lambda r: r["t"])
    (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return folder


def _walk(count=40, start=0.0, hop=0.1):
    """states at 10 Hz, moving, so autolabel has something to call roaming.

    `m` MATTERS: GameState.position_known requires x, y, facing AND map_id, and Window.moved counts
    only located states - so a walk without a map id reads as perfectly stationary and every window
    comes back unlabelled. That is a real trap for anyone building a recording by hand.
    """
    return [
        {"t": start + i * hop, "x": 10.0 + i * 0.05, "y": 20.0, "f": 1.0, "m": 1955, "ph": 100.0}
        for i in range(count)
    ]


def test_only_real_session_logs_are_offered(tmp_path):
    """a session log is named for its own folder - anything else in there is not a recording"""
    _session(tmp_path, "rec", _walk())
    (tmp_path / "rec" / "notes.jsonl").write_text("{}\n")
    (tmp_path / "loose.jsonl").write_text("{}\n")
    assert [s["name"] for s in playback.sessions_under(tmp_path)] == ["rec"]


def test_a_missing_session_is_an_error_not_an_empty_playback(tmp_path):
    with pytest.raises((playback.PlaybackError, FileNotFoundError)):
        playback.load(tmp_path, "nope")


def test_a_log_with_no_states_is_refused(tmp_path):
    """there is nothing to scrub through, and a silent empty player looks like a broken tab"""
    folder = tmp_path / "rec"
    folder.mkdir()
    (folder / "rec.jsonl").write_text(json.dumps({"kind": "key", "t": 1.0, "down": True}) + "\n")
    with pytest.raises(playback.PlaybackError, match="no state rows"):
        playback.load(tmp_path, "rec")


def test_duration_spans_the_states(tmp_path):
    _session(tmp_path, "rec", _walk(count=41))  # 0.0 .. 4.0
    loaded = playback.load(tmp_path, "rec")
    assert loaded.duration == pytest.approx(4.0)


def test_a_moment_reports_the_last_state_at_or_before_it(tmp_path):
    """what was true THEN, never what came next - a scrub must not show the future"""
    _session(tmp_path, "rec", _walk(count=20))
    loaded = playback.load(tmp_path, "rec")
    frame = playback.frame_at(loaded, loaded.start + 0.55)
    assert frame["state"]["x"] == pytest.approx(10.0 + 5 * 0.05)


def test_keys_held_at_a_moment_come_from_their_down_up_pair(tmp_path):
    _session(
        tmp_path,
        "rec",
        _walk(count=40),
        events=[
            {"t": 0.5, "key": "q", "down": True},
            {"t": 1.5, "key": "q", "down": False},
            {"t": 1.0, "key": "w", "down": True},
        ],
    )
    loaded = playback.load(tmp_path, "rec")
    assert playback.keys_down_at(loaded, loaded.start + 1.2) == ["q", "w"]
    assert playback.keys_down_at(loaded, loaded.start + 2.0) == ["w"]
    assert playback.keys_down_at(loaded, loaded.start + 0.2) == []


def test_the_mouse_vector_sums_a_window_not_one_row(tmp_path):
    """a single delta row is jitter; turning is a vector over a moment"""
    mouse = [
        {"kind": "mouse_delta", "t": 1.0, "dx": 10, "dy": -4},
        {"kind": "mouse_delta", "t": 1.1, "dx": 12, "dy": -6},
        {"kind": "mouse_delta", "t": 3.0, "dx": 99, "dy": 99},
    ]
    _session(tmp_path, "rec", _walk(count=40), mouse=mouse)
    loaded = playback.load(tmp_path, "rec")
    assert playback.mouse_at(loaded, loaded.start + 1.2, window=0.5) == {"dx": 22, "dy": -10}
    # the far-away row must not leak in
    assert playback.mouse_at(loaded, loaded.start + 1.2, window=0.5)["dx"] == 22


def test_a_moment_carries_the_autolabelled_mode(tmp_path):
    """THE REASON THIS IS INTERPRETATION AND NOT A LOG VIEWER: the tab shows what the pipeline makes
    of the play, so a window autolabel cannot label is visible as a gap rather than inferred from a
    coverage percentage."""
    _session(tmp_path, "rec", _walk(count=60))
    loaded = playback.load(tmp_path, "rec")
    modes = {playback.label_at(loaded, loaded.start + t) for t in (1.0, 2.0, 3.0)}
    assert modes == {"roaming"}, f"a moving player should read as roaming, got {modes}"


def test_an_unlabelled_window_reports_none_rather_than_guessing(tmp_path):
    """a wrong label is worse than a missing one - the rule autolabel itself follows"""
    _session(tmp_path, "rec", [{"t": i * 0.1, "ph": 100.0} for i in range(30)])
    loaded = playback.load(tmp_path, "rec")
    assert playback.label_at(loaded, loaded.start + 1.0) is None


def test_the_frame_carries_the_fields_that_used_to_be_dropped(tmp_path):
    """combo points, stealth and target health reach the overlay only because the shared session
    reader maps them - nothing read cp, rip or sth before it existed"""
    states = _walk(count=20)
    states[10].update({"cp": 4, "sth": True, "te": True, "th": 61.0})
    _session(tmp_path, "rec", states)
    loaded = playback.load(tmp_path, "rec")
    frame = playback.frame_at(loaded, loaded.start + 1.05)
    assert frame["state"]["combo_points"] == 4
    assert frame["state"]["stealthed"] is True
    assert frame["state"]["target"] is True
    assert frame["state"]["target_health"] == 61.0


def test_elapsed_is_measured_from_the_first_state(tmp_path):
    """recordings carry a monotonic clock with an arbitrary origin, so absolute t means nothing to
    a scrub bar - the page works in seconds from the start"""
    _session(tmp_path, "rec", _walk(count=20, start=538109.6))
    loaded = playback.load(tmp_path, "rec")
    frame = playback.frame_at(loaded, loaded.start + 1.0)
    assert frame["elapsed"] == pytest.approx(1.0)
