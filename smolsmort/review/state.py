"""the review tool's own state: what is bound, what is in the pool, and what has been judged.

the one object every tab talks to. it owns no http - routes.py calls into it and serialises what
comes back, which is what lets all of this be tested without a socket. split by concern:

    find_state.py    the bound recording, drawn boxes, the crop rule, cutting tiles, paging
    pool_state.py    the tile pool, its judgements, the unsaved label buffer, the grid
    promote.py       pool -> a named training set
    bases.py         the pickers' base directories and the directory tree

the domain lives behind seams passed in at construction: a `renderer` (frames and crops; the
image one is the default) and an optional `guesser` that only ever prefills a class.

PATHS ARE READ THROUGH `paths`, never imported by value - they are repointed at runtime by
set_bases and monkeypatched by tests, and a bound copy would silently keep pointing at the real
sessions/ and labels/. see review/paths.py.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from smolsmort.review import paths
from smolsmort.review.bases import BasesMixin
from smolsmort.review.find_state import FindMixin
from smolsmort.review.pool_state import PoolMixin
from smolsmort.review.promote import PromoteMixin
from smolsmort.review.render import ImageRenderer

# a neutral starting box for uniform tiles, from the dataset layer's own defaults (64x14); the
# find tab's tile-size control moves the height from here
DEFAULT_TILE_SIZE = (64, 14)


class ReviewState(FindMixin, PoolMixin, PromoteMixin, BasesMixin):
    def __init__(
        self,
        frames_dir: Path,
        candidates: list[dict],
        decisions_path: Path,
        session_tag: str = "",
        pool: Path | None = None,
        *,
        renderer: Any = None,
        guesser: Any = None,
        bases_file: Path | None = None,
        crop_file: Path | None = None,
    ):
        self.frames_dir = frames_dir
        self.candidates = candidates
        self.decisions_path = decisions_path
        self.decisions: dict[str, dict] = self._load_decisions()
        # every session shares one staging pool, so a tile is named after its recording - two
        # sessions both saving candidate index 5 must not overwrite each other's tile
        self.session_tag = session_tag
        self.renderer = renderer if renderer is not None else ImageRenderer()
        self.guesser = guesser
        # where the pickers point. built from the CURRENT paths (not DEFAULT_BASES), so a state
        # built around a test's tmp_path reads and writes there rather than the real directories
        self.bases: dict = {
            **paths.DEFAULT_BASES,
            "sessions": str(paths.SESSIONS_DIR),
            "labels": str(paths.LABELS_DIR),
        }
        # SEVERAL POOL DIRECTORIES, primary first: the grid lists every tile across all of them,
        # but a new tile is only ever written to the primary
        self.pools: list[Path] = [pool if pool is not None else paths.TILES_DIR]
        self.bases["pool"] = [str(self.pools[0])]
        self.bases_file = bases_file
        self.crop_file = crop_file
        self._dataset_cache: dict = {}
        # where a record was last read from, so a write goes back to the pool it came from
        self._record_origin: dict[str, Path] = {}
        # unsaved class assignments: dataset tag -> {tile name: (label, definition)}
        self._label_buffer: dict[str, dict[str, tuple[str, str]]] = {}
        self.lock = threading.RLock()
        # how much ground a cut tile keeps around its box; see FindMixin.crop_pads
        self.pad_x = paths.PAD_X_DEFAULT
        self.pad_y = paths.PAD_Y_DEFAULT
        self.crop_mode = "percent"
        self.crop_w = 0
        self.crop_h = 0
        self.aspect = "free"
        # ONE size for every candidate's tile, not each guessing its own. width is treated as
        # known and fixed; height is nudged live from the find tab
        self.uniform_width, self.uniform_height = DEFAULT_TILE_SIZE
        self.load_crop_settings()
