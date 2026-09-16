"""PlaybackState: the one loaded recording the overlay tab is scrubbing through.

Exercised against a FAKE reader, never a real one - what a recording actually contains (a game's
own state schema, an autolabelled mode, whatever a consumer records) is domain work that stays
behind a `PlaybackReader`; this module only caches the one loaded parse. See
smolsmort/review/playback_state.py's docstring for why the reader is injected rather than imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from smolsmort.review import paths
from smolsmort.review.playback_state import PlaybackError, PlaybackReader, PlaybackState


@dataclass
class FakeLoaded:
    """a stand-in for one parsed recording - just enough shape for PlaybackState to summarise."""

    name: str
    start: float = 0.0
    duration: float = 4.0
    states: list = field(default_factory=lambda: [1, 2, 3])
    events: list = field(default_factory=lambda: [1])
    labels: list = field(default_factory=lambda: [("w0", "roaming"), ("w1", None)])


@dataclass
class FakeReader:
    """seam-shaped fake: a fixed session list, a loadable name, and a frame_at that just echoes
    what it was asked for - a real reader interprets a recording; this one only proves the wiring.
    """

    known: dict
    refuses: str | None = None
    frame_calls: list = field(default_factory=list)

    def sessions_under(self, root: Path) -> list[dict]:
        return [{"name": name, "size": 1} for name in sorted(self.known)]

    def load(self, root: Path, name: str):
        if name == self.refuses:
            raise PlaybackError(f"no session log under {name}")
        if name not in self.known:
            raise FileNotFoundError(name)
        return self.known[name]

    def frame_at(self, loaded, when: float) -> dict:
        self.frame_calls.append(when)
        return {"t": when, "name": loaded.name}


def test_fake_reader_satisfies_the_playback_reader_seam():
    assert isinstance(FakeReader(known={}), PlaybackReader)


def test_sessions_delegates_to_the_reader(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "SESSIONS_DIR", tmp_path)
    reader = FakeReader(known={"rec": FakeLoaded(name="rec")})
    state = PlaybackState(reader)
    assert state.sessions() == {"sessions": [{"name": "rec", "size": 1}]}


def test_opening_a_known_recording_summarises_it():
    loaded = FakeLoaded(name="rec", duration=4.0, states=[1, 2, 3], events=[1, 2])
    state = PlaybackState(FakeReader(known={"rec": loaded}))
    summary = state.open("rec")
    assert summary == {
        "name": "rec",
        "duration": 4.0,
        "states": 3,
        "events": 2,
        "windows": 2,
        "labelled": 1,  # only the window with a non-None segment counts
    }
    assert state.loaded is loaded


def test_opening_an_unknown_recording_reports_an_error_not_a_crash():
    state = PlaybackState(FakeReader(known={}))
    result = state.open("nope")
    assert "error" in result
    assert state.loaded is None


def test_a_reader_that_refuses_a_recording_is_caught_as_an_error():
    """PlaybackError is the reader's own way of saying no - it must surface as {"error": ...},
    never propagate and crash the request"""
    state = PlaybackState(FakeReader(known={}, refuses="rec"))
    result = state.open("rec")
    assert result == {"error": "no session log under rec"}


def test_frame_before_anything_is_open_is_an_error():
    state = PlaybackState(FakeReader(known={}))
    assert state.frame(1.0) == {"error": "no recording open"}


def test_frame_asks_the_reader_relative_to_the_recordings_own_start():
    loaded = FakeLoaded(name="rec", start=100.0)
    reader = FakeReader(known={"rec": loaded})
    state = PlaybackState(reader)
    state.open("rec")
    frame = state.frame(2.5)
    assert frame == {"t": 102.5, "name": "rec"}
    assert reader.frame_calls == [102.5]


def test_a_negative_elapsed_clamps_to_the_start_rather_than_reading_before_it():
    loaded = FakeLoaded(name="rec", start=50.0)
    reader = FakeReader(known={"rec": loaded})
    state = PlaybackState(reader)
    state.open("rec")
    state.frame(-10.0)
    assert reader.frame_calls == [50.0]


def test_playback_state_requires_a_reader():
    with pytest.raises(TypeError):
        PlaybackState()
