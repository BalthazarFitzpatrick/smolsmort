"""the directories the pickers browse: which folder holds which kind of data, and how to walk them.

a mixin over `ReviewState` (state.py). `set_bases` repoints the live paths AS ATTRIBUTES on the
`paths` module (see paths.py for why by-value imports are forbidden), so every reader sees the
change at once.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from smolsmort.review import paths
from smolsmort.review.find_state import write_json_atomic
from smolsmort.review.recordings import session_frames_dirs

# the browsable bases dir-tree can be rooted at, by name. "tiles" is the primary pool
TREE_ROOTS = ("tiles", "checkpoints", "recordings")


class BaseRootError(ValueError):
    """dir-tree was rooted at a name that is not one of the configured bases"""


class BasesMixin:
    def dir_list(self) -> dict:
        """what the pickers offer, each scoped to its own base so a path cannot be typed wrong:
        sessions are directories holding frames, labels are candidate files"""
        sessions_root, labels_root = Path(self.bases["sessions"]), Path(self.bases["labels"])
        labels = []
        if labels_root.is_dir():
            labels = [p.as_posix() for p in sorted(labels_root.glob("*.candidates.jsonl"))]
        return {
            "bases": self.bases,
            "sessions": [d.parent.as_posix() for d in session_frames_dirs(sessions_root)],
            "labels": labels,
        }

    def tree_root(self, key: str | None) -> Path:
        """the directory a tree is rooted at: a configured base by name, or the settings root"""
        if not key:
            return Path(self.bases["root"])
        if key == "tiles":
            return self.pools[0]
        if key == "recordings":
            return Path(self.bases["sessions"])
        if key == "checkpoints":
            return paths.CHECKPOINTS_DIR
        raise BaseRootError(f"root must be one of {', '.join(TREE_ROOTS)}, got {key!r}")

    def dir_tree(self, under: str | None = None, root: str | None = None) -> dict:
        """the directories one level under a path, for a picker to walk.

        ONE LEVEL AT A TIME: a recordings folder can hold dozens and a tile pool thousands, so a
        recursive listing would be enormous and almost entirely uninteresting. `has_children` lets
        a row show it can be opened without paying to find out.

        `root` names one of the configured bases (tiles, checkpoints, recordings) so two pickers
        can be rooted at different places. REFUSES TO LEAVE THE ROOT: a path outside it comes back
        as the root's own listing with `refused` set, because the honest answer to "show me /etc"
        is what this browser is allowed to show.
        """
        base = self.tree_root(root).resolve()
        refused = False
        try:
            here = Path(under).resolve() if under else base
        except (OSError, ValueError):
            here, refused = base, True
        if not (here == base or base in here.parents) or not here.is_dir():
            refused = refused or bool(under)
            here = base

        entries = []
        with contextlib.suppress(OSError):
            for child in sorted(here.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                has_children = False
                with contextlib.suppress(OSError):
                    has_children = any(
                        g.is_dir() and not g.name.startswith(".") for g in child.iterdir()
                    )
                entries.append(
                    {"name": child.name, "path": str(child), "has_children": has_children}
                )
        return {
            "root": str(base),
            "base": root or "",
            "here": str(here),
            "parent": str(here.parent) if here != base else None,
            "entries": entries,
            "refused": refused,
        }

    def set_bases(self, bases: dict) -> dict:
        """repoint the pickers. only known keys are accepted, and nothing is validated into
        existence: a base that does not exist simply offers nothing, visible in the picker rather
        than failing later at bind time."""
        for key in paths.BASE_KEYS:
            value = bases.get(key)
            if key in paths.MULTI_BASES:
                # several paths, and a bare string is still accepted so a file written before
                # this, or a caller with only one, keeps working
                wanted = [value] if isinstance(value, str) else value
                if isinstance(wanted, list):
                    cleaned = [v.strip() for v in wanted if isinstance(v, str) and v.strip()]
                    if cleaned:
                        self.bases[key] = cleaned
            elif isinstance(value, str) and value.strip():
                self.bases[key] = value.strip()

        # the pool is a LIVE handle, not just a string: repointing it has to move the directories
        # derived from it, or the setting only takes effect after a restart
        wanted_pools = [Path(v) for v in self.bases["pool"]]
        if wanted_pools and wanted_pools != self.pools:
            self.pools = wanted_pools
            self._dataset_cache.clear()
            self._record_origin.clear()
        # ASSIGNED ON THE MODULE, which is the whole reason paths.py exists: every reader reaches
        # through `paths.`, so this is seen everywhere at once
        paths.SESSIONS_DIR = Path(self.bases["sessions"])
        paths.LABELS_DIR = Path(self.bases["labels"])
        self._save_bases()
        return self.bases

    def _save_bases(self) -> None:
        """remember a repoint across restarts. a repoint that cannot be remembered still works,
        so a write failure must not break the request that made it."""
        if self.bases_file is None:
            return
        with contextlib.suppress(OSError):
            write_json_atomic(self.bases_file, self.bases, indent=2)

    def load_saved_bases(self) -> None:
        """apply the remembered repoint, if there is one. called once at startup."""
        if self.bases_file is None:
            return
        try:
            saved = json.loads(self.bases_file.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(saved, dict):
            self.set_bases(saved)  # through set_bases, so the module paths move with it
