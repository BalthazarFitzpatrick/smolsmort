"""replaying a recorded session: what was true at time t, for the overlay tab.

WHAT THIS IS FOR. the overlay tab has always been a stub - a canvas, a pin button, and the words
"nothing playing back yet". This is the part that makes it real: scrub through a recording and watch
the state, the keys, the mouse and the labelled mode as they were.

INTERPRETATION, NOT JUST REPLAY. drawing the raw rows back would be a log viewer. The point is to
show what the pipeline MAKES of them - the same window labelling autolabel does, over the window
containing t - so the mode machine's whole story (windows, runs, gaps) can be watched happening on
real play instead of read about. That is also what makes it a debugger: a window labelled `None`,
or a combat rule that never fires, is visible here rather than inferred from a coverage percentage.

STATELESS PER REQUEST, DELIBERATELY. the page owns the clock and asks for a moment; nothing here
remembers where playback "is". A scrub then has nothing to fight - the answer for t is the same
whether it arrived from a timer tick or a dragged slider - and two viewers cannot desync each other.
The session is parsed once and cached, because the parse is the only expensive part.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from parent.imitation.autolabel import InputEvent, label_session
from parent.imitation.record import resolve_session_paths
from parent.imitation.session import events_of, mouse_of, states_of

# how far either side of t a key may be held and still be drawn as "down now"
HOLD_LOOKBACK = 5.0


@dataclass
class Loaded:
    """one parsed recording, kept so a scrub does not re-read the file every 100 ms"""

    name: str
    path: Path
    states: list
    events: list
    mouse: list
    labels: list  # (window, segment | None) from autolabel

    @property
    def start(self) -> float:
        return self.states[0].timestamp if self.states else 0.0

    @property
    def end(self) -> float:
        return self.states[-1].timestamp if self.states else 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


class PlaybackError(Exception):
    pass


def sessions_under(root: Path) -> list[dict]:
    """every recording with a log to play, newest-looking last.

    a recording is a directory holding a jsonl of its own name - resolve_session_paths knows both
    layouts, so this does not need to guess at either.
    """
    found = []
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("*.jsonl")):
        if path.stem != path.parent.name:  # a session log is named for its own folder
            continue
        try:
            name = str(path.parent.relative_to(root))
        except ValueError:
            continue
        found.append({"name": name, "size": path.stat().st_size})
    return found


def load(root: Path, name: str) -> Loaded:
    records_path, _ = resolve_session_paths(root / name)
    if not records_path.is_file():
        raise PlaybackError(f"no session log under {name}")
    states = states_of(records_path)
    if not states:
        raise PlaybackError(f"{name} has no state rows - nothing to play back")
    events = events_of(records_path)
    labels = label_session(
        states,
        [InputEvent(e.timestamp, e.key, e.down) for e in events],
    )
    return Loaded(
        name=name,
        path=records_path,
        states=states,
        events=events,
        mouse=mouse_of(records_path),
        labels=labels,
    )


def _at(values: list, when: float, key) -> object | None:
    """the last entry at or before `when` - what was true then, not what came next"""
    best = None
    for value in values:
        if key(value) > when:
            break
        best = value
    return best


def keys_down_at(loaded: Loaded, when: float) -> list[str]:
    """which keys were held at that moment, from the down/up pairs around it"""
    held: dict[str, bool] = {}
    for event in loaded.events:
        if event.timestamp > when:
            break
        if when - event.timestamp > HOLD_LOOKBACK and not event.down:
            held.pop(event.key, None)
            continue
        held[event.key] = event.down
    return sorted(key for key, down in held.items() if down)


def mouse_at(loaded: Loaded, when: float, window: float = 0.5) -> dict:
    """summed mouse motion over the last `window` seconds - a vector, which is what turning looks
    like. one delta row on its own is a jitter, not a movement."""
    dx = dy = 0
    for row in loaded.mouse:
        if row.get("kind") != "mouse_delta":
            continue
        stamp = row.get("t", 0.0)
        if stamp > when:
            break
        if when - stamp <= window:
            dx += row.get("dx", 0) or 0
            dy += row.get("dy", 0) or 0
    return {"dx": dx, "dy": dy}


def label_at(loaded: Loaded, when: float) -> str | None:
    """the autolabel segment for the window containing t.

    windows OVERLAP by half, so a moment usually sits in two of them. the later one is taken: it is
    the one with the most evidence about what is happening NOW rather than what was still finishing.
    """
    found = None
    for window, segment in loaded.labels:
        if window.start <= when < window.end:
            found = segment
    return str(found) if found else None


def frame_at(loaded: Loaded, when: float) -> dict:
    """everything the overlay draws for one moment"""
    state = _at(loaded.states, when, lambda s: s.timestamp)
    if state is None:
        state = loaded.states[0]
    return {
        "t": when,
        "elapsed": when - loaded.start,
        "state": {
            "x": state.x,
            "y": state.y,
            "facing": state.facing,
            "map_id": state.map_id,
            "health": state.player_health_pct,
            "resource": state.player_resource_pct,
            "target": bool(state.target_exists),
            "target_health": state.target_health_pct,
            "combo_points": state.combo_points,
            "stealthed": state.stealthed,
        },
        "keys": keys_down_at(loaded, when),
        "mouse": mouse_at(loaded, when),
        "mode": label_at(loaded, when),
    }
