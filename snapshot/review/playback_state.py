"""the one recording the overlay tab is scrubbing through.

Holds a PARSE, not a position: the page owns the clock and asks what was true at a moment,
so a scrub has nothing to fight. See tools/playback.py for the per-moment answer.
"""

from __future__ import annotations

from snapshot.review import paths


class PlaybackState:
    """the one loaded recording the overlay tab is scrubbing through.

    HOLDS A PARSE, NOT A POSITION. the page owns the clock; this only remembers which session was
    opened so a scrub does not re-read the file ten times a second. See tools/playback.py for why
    the per-moment answer is stateless.
    """

    def __init__(self):
        self.loaded = None

    def sessions(self) -> dict:
        from snapshot.tools import playback

        return {"sessions": playback.sessions_under(paths.SESSIONS_DIR)}

    def open(self, name: str) -> dict:
        from snapshot.tools import playback

        try:
            self.loaded = playback.load(paths.SESSIONS_DIR, name)
        except (playback.PlaybackError, FileNotFoundError, OSError) as exc:
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
        from snapshot.tools import playback

        if self.loaded is None:
            return {"error": "no recording open"}
        return playback.frame_at(self.loaded, self.loaded.start + max(0.0, elapsed))
