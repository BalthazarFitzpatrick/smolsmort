"""replaying a recorded session for the overlay tab.

the tab promised "nothing playing back yet" for months. This is the part that answers what was true
at a moment - and INTERPRETS it rather than dumping rows, because the point is to watch the pipeline
label real play: the same windows autolabel builds, the same gaps, the same modes.
"""

from __future__ import annotations

import json

import pytest

from parent.tools import playback


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


def test_a_moment_shows_the_newest_saved_frame_and_its_plates(tmp_path):
    """what the left half draws: the frame at or before t, and its plate rows if it has any"""
    rows = [
        {"kind": "frame", "t": 0.5, "path": "0000001.jpg"},
        {"kind": "frame", "t": 1.5, "path": "0000002.jpg"},
        {
            "kind": "plates",
            "t": 1.5,
            "frame": "0000002.jpg",
            "width": 1440,
            "plates": [{"box": [10, 20, 100, 12], "d": 8.0, "r": -12.0, "b": 348.0, "f": 90.0}],
        },
    ]
    _session(tmp_path, "rec", _walk(count=30), mouse=rows)
    loaded = playback.load(tmp_path, "rec")
    assert playback.frame_at(loaded, loaded.start + 0.2)["frame"] is None
    early = playback.frame_at(loaded, loaded.start + 1.0)["frame"]
    assert early["path"] == "0000001.jpg" and early["plates"] == []
    late = playback.frame_at(loaded, loaded.start + 2.0)["frame"]
    assert late["path"] == "0000002.jpg" and late["width"] == 1440
    assert late["plates"][0]["d"] == 8.0


def test_the_mode_machine_runs_over_the_recording(tmp_path):
    """the strip under the slider: the bot's own mode machine fed the recorded states"""
    states = _walk(count=30)
    for state in states[10:20]:
        state["ph"] = 10.0  # under the flee threshold
    _session(tmp_path, "rec", states)
    loaded = playback.load(tmp_path, "rec")
    assert playback.frame_at(loaded, loaded.start + 1.5)["machine_mode"] == "FLEE"
    runs = playback.timeline(loaded)
    assert "FLEE" in {run["mode"] for run in runs}
    assert runs[0]["start"] == 0.0 and runs[-1]["end"] == pytest.approx(loaded.duration)


def test_the_sink_lights_a_held_key_as_the_output_sink_would(tmp_path):
    """the right half's top panel: OverlayModel's own snapshot, so a key flashes past its release"""
    events = [{"t": 1.0, "key": "w", "down": True}, {"t": 2.0, "key": "w", "down": False}]
    _session(tmp_path, "rec", _walk(count=40), events=events)
    loaded = playback.load(tmp_path, "rec")

    def lit(at):
        sink = playback.frame_at(loaded, loaded.start + at)["sink"]
        return {key["key"] for key in sink["keys"] if key["lit"]}

    assert "w" in lit(1.5)
    assert "w" in lit(2.1)  # within the linger after release
    assert "w" not in lit(2.6)
    sink = playback.frame_at(loaded, loaded.start + 1.5)["sink"]
    # f is recorded clockwise, 1.0 rad; the dial shows the compass heading
    assert sink["facing_degrees"] == pytest.approx(57.2958, abs=0.01)
    assert sink["mode"]


def test_a_frame_name_reaching_outside_the_recording_is_refused(tmp_path, monkeypatch):
    from snapshot.review import paths
    from snapshot.review.playback_state import PlaybackState

    rows = [{"kind": "frame", "t": 0.5, "path": "0000001.jpg"}]
    folder = _session(tmp_path, "rec", _walk(count=10), mouse=rows)
    (folder / "frames").mkdir()
    (folder / "frames" / "0000001.jpg").write_bytes(b"jpeg")
    (folder / "secret.txt").write_text("no")
    monkeypatch.setattr(paths, "SESSIONS_DIR", tmp_path)
    state = PlaybackState()
    opened = state.open("rec")
    assert opened["frames"] == 1 and opened["timeline"]
    assert state.frame_bytes("0000001.jpg") == b"jpeg"
    assert state.frame_bytes("../secret.txt") is None
    assert state.frame_bytes("../rec.jsonl") is None


def test_elapsed_is_measured_from_the_first_state(tmp_path):
    """recordings carry a monotonic clock with an arbitrary origin, so absolute t means nothing to
    a scrub bar - the page works in seconds from the start"""
    _session(tmp_path, "rec", _walk(count=20, start=538109.6))
    loaded = playback.load(tmp_path, "rec")
    frame = playback.frame_at(loaded, loaded.start + 1.0)
    assert frame["elapsed"] == pytest.approx(1.0)
