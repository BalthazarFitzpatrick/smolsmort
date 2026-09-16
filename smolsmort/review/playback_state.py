"""the one recording the overlay tab is scrubbing through.

Holds a PARSE, not a position: the page owns the clock and asks what was true at a moment,
so a scrub has nothing to fight.

READING A RECORDING IS DOMAIN WORK, AND STAYS BEHIND IT. what a "state" or "frame" even contains -
position, health, whatever a consumer's own session format records - is not something the loop
knows, so this module never imports a reader itself. A `PlaybackReader` is handed in at construction:
the parent project's own recording format is one implementation, a synthetic one for tests is
another. This class only caches the one loaded parse so a scrub does not re-read the file ten times
a second, which is the one thing that is true regardless of what a recording contains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from smolsmort.review import paths


class PlaybackError(Exception):
    """a reader could not answer - e.g. no session log where one was asked for"""


@runtime_checkable
class PlaybackReader(Protocol):
    """what PlaybackState needs from a recording reader: parse once, answer per-moment after
    that. `load` may raise `PlaybackError`, `FileNotFoundError` or `OSError` for a recording that
    cannot be opened; anything else is left to propagate."""

    def sessions_under(self, root: Path) -> list[dict]: ...
    def load(self, root: Path, name: str) -> Any: ...
    def frame_at(self, loaded: Any, when: float) -> dict: ...


class PlaybackState:
    """the one loaded recording the overlay tab is scrubbing through.

    HOLDS A PARSE, NOT A POSITION. the page owns the clock; this only remembers which session was
    opened so a scrub does not re-read the file ten times a second.
    """

    def __init__(self, reader: PlaybackReader):
        self.loaded = None
        self._reader = reader

    def sessions(self) -> dict:
        return {"sessions": self._reader.sessions_under(paths.SESSIONS_DIR)}

    def open(self, name: str) -> dict:
        try:
            self.loaded = self._reader.load(paths.SESSIONS_DIR, name)
        except (PlaybackError, FileNotFoundError, OSError) as exc:
            return {"error": str(exc)}
        loaded = self.loaded
        labelled = sum(1 for _, segment in loaded.labels if segment)
        return {
            "name": loaded.name,
            "duration": loaded.duration,
            "states": len(loaded.states),
            "events": len(loaded.events),
            "windows": len(loaded.labels),
            # coverage is the honest headline: an unlabelled window is a gap, and a recording that
            # is mostly gaps has little to say about behaviour
            "labelled": labelled,
        }

    def frame(self, elapsed: float) -> dict:
        if self.loaded is None:
            return {"error": "no recording open"}
        return self._reader.frame_at(self.loaded, self.loaded.start + max(0.0, elapsed))
