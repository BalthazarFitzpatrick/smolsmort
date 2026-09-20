"""replaying a recorded session: what was true at time t, for the sessions tab.

WHAT THIS IS FOR. scrub through a recording and watch the frame, the state, the keys, the mouse, the
mode machine and the labelled mode as they were - and the same moment as the output sink would have
drawn it, so a recording can be read the way the bot will one day produce one.

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

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from parent.execute.overlay import LINGER_SECONDS, OverlayModel
from parent.extract.arrow import to_geometry_facing
from parent.imitation.autolabel import InputEvent, label_session
from parent.imitation.keynames import MOUSE_PREFIX, is_mouse
from parent.imitation.record import resolve_session_paths
from parent.imitation.session import events_of, mouse_of, rows_of, states_of
from parent.policy.fsm import ModeMachine

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
    frames: list = field(default_factory=list)  # (t, frame file) in time order
    frames_dir: Path | None = None
    plates: dict = field(default_factory=dict)  # frame file -> its plates row
    starts: dict = field(default_factory=dict)  # frame file -> the character's (x, y) on it
    modes: list = field(default_factory=list)  # the mode machine's mode after each state

    def __post_init__(self):
        # sorted stamps, so a moment in a ten-hour recording is a bisect, not a scan
        self._state_t = [s.timestamp for s in self.states]
        self._frame_t = [t for t, _ in self.frames]
        self._deltas = [r for r in self.mouse if r.get("kind") == "mouse_delta"]
        self._delta_t = [r.get("t", 0.0) for r in self._deltas]

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


def _plate_rows(records_path: Path) -> dict[str, dict]:
    """plate rows by frame: the recording's own (wt-record --weights), else wt-annotate-plates'"""
    rows = rows_of(records_path, "plates")
    if not rows:
        from snapshot.vision.plate_log import derived_path

        sidecar = derived_path(records_path)
        if sidecar.is_file():
            lines = sidecar.read_text().splitlines()
            rows = [json.loads(line) for line in lines if line.strip()]
    return {row["frame"]: row for row in rows if row.get("kind") == "plates" and "frame" in row}


def _arrow_starts(records_path: Path, frames_dir: Path, plates: dict) -> dict[str, tuple]:
    """where each frame's plate arrows start: the character, projected through the fitted camera.

    boxes are in the saved frame's own pixels, so the camera is resolved at that frame's size too.
    no fitted camera means no starts, and the page draws boxes and cards without arrows
    """
    if not plates:
        return {}
    from PIL import Image

    from parent.camera import pinhole
    from parent.camera.reckon import camera_states
    from snapshot.tools.bearing_overlay import CAMERA_PATH, OverlayError, load_camera, pose_at

    try:
        relative, per_notch, at_zero = load_camera(CAMERA_PATH)
    except OverlayError:
        return {}
    cameras, _ = camera_states(records_path)
    starts = {}
    for name in plates:
        path = frames_dir / name
        if name not in cameras or not path.is_file():
            continue
        with Image.open(path) as image:
            camera = relative.at(image.width, image.height)
        me = pinhole.project(camera, pose_at(cameras[name], per_notch, at_zero), 0.0, 0.0)
        starts[name] = (float(me[0]), float(me[1]))
    return starts


def load(root: Path, name: str) -> Loaded:
    records_path, frames_dir = resolve_session_paths(root / name)
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
    frames = sorted((row["t"], row["path"]) for row in rows_of(records_path, "frame"))
    plates = _plate_rows(records_path)
    machine = ModeMachine()
    return Loaded(
        name=name,
        path=records_path,
        states=states,
        events=events,
        mouse=mouse_of(records_path),
        labels=labels,
        frames=frames,
        frames_dir=frames_dir,
        plates=plates,
        starts=_arrow_starts(records_path, frames_dir, plates),
        modes=[str(machine.update(state)) for state in states],
    )


def _index_at(stamps: list[float], when: float) -> int | None:
    """the index of the last stamp at or before `when` - what was true then, not what came next"""
    index = bisect_right(stamps, when) - 1
    return index if index >= 0 else None


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
    first = bisect_left(loaded._delta_t, when - window)
    last = bisect_right(loaded._delta_t, when)
    rows = loaded._deltas[first:last]
    return {
        "dx": sum(r.get("dx", 0) or 0 for r in rows),
        "dy": sum(r.get("dy", 0) or 0 for r in rows),
    }


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


def _press(model: OverlayModel, key: str, down: bool) -> None:
    # the merged stream names buttons mouse_left; the model takes them on its button channel
    if is_mouse(key):
        model.mouse_button(key[len(MOUSE_PREFIX) :], down)
    else:
        (model.key_down if down else model.key_up)(key)


def sink_at(loaded: Loaded, when: float, state, mode: str | None, label: str | None) -> dict:
    """the moment as the output sink would draw it: OverlayModel's own snapshot, so the page's one
    renderer draws a recording and a live shadow run off the same shape.

    a key released within the model's linger still flashes, as it would have live
    """
    model = OverlayModel(mode=mode or "", note=f"labelled {label}" if label else "unlabelled")
    since = when - LINGER_SECONDS
    model.tick(since)
    for key in keys_down_at(loaded, since):
        _press(model, key, True)
    for event in loaded.events:
        if event.timestamp <= since:
            continue
        if event.timestamp > when:
            break
        model.tick(event.timestamp)
        _press(model, event.key, event.down)
    model.tick(when)
    motion = mouse_at(loaded, when)
    model.on_mouse_delta(motion["dx"], motion["dy"])
    if state.facing is not None:
        model.set_bearing(to_geometry_facing(state.facing), None)
    return model.snapshot()


def frame_at(loaded: Loaded, when: float) -> dict:
    """everything the sessions tab draws for one moment"""
    index = _index_at(loaded._state_t, when) or 0
    state = loaded.states[index]
    label = label_at(loaded, when)
    mode = loaded.modes[index] if loaded.modes else None
    shown = _index_at(loaded._frame_t, when)
    frame = None
    if shown is not None:
        t, name = loaded.frames[shown]
        row = loaded.plates.get(name)
        frame = {
            "path": name,
            "t": t,
            "width": row.get("width") if row else None,
            "plates": row["plates"] if row else [],
            "start": loaded.starts.get(name),
        }
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
            "target_hostility": state.target_hostility,
            "in_combat": state.in_combat,
            "combo_points": state.combo_points,
            "stealthed": state.stealthed,
            "cast": state.cast_state,
            "xp": state.xp_fraction,
            "error": state.error_showing,
        },
        "keys": keys_down_at(loaded, when),
        "mouse": mouse_at(loaded, when),
        "mode": label,
        "machine_mode": mode,
        "frame": frame,
        "sink": sink_at(loaded, when, state, mode, label),
    }


def timeline(loaded: Loaded) -> list[dict]:
    """the mode machine's runs over the recording, in seconds from its start"""
    runs: list[dict] = []
    for state, mode in zip(loaded.states, loaded.modes, strict=False):
        at = state.timestamp - loaded.start
        if runs and runs[-1]["mode"] == mode:
            runs[-1]["end"] = at
        else:
            runs.append({"mode": mode, "start": at, "end": at})
    return runs
