"""find and remove the downstream assets a recording left behind - including the ones whose
recording is already gone.

recordings get deleted by hand from sessions/, but nothing downstream of them (boxes, tiles,
sets, checkpoints) knows that happened, so it sits under training/ forever pointing at nothing.
This module answers two questions: what does a recording (live or gone) own, and how to remove
that safely - archived (moved to a sibling `_archive/`, recoverable) or deleted (unlinked, not).

TAG PEELING IS THE CORE TRICK. an asset filename is never just the recording's name: a sweep or
draw pass appends a `.cnn-<stamp>` / `.drawn-<stamp>` (boxes) or `_cnn-<stamp>` (tiles, sets), and
a tile appends `_k<index>` on top of that, sometimes with two stamps stacked (a re-swept set). All
of it has to be peeled off before comparing against a live recording's flattened name.

READS PATHS.* AT CALL TIME, never by value - see paths.py's own warning. `_labels.json` is a
tile pool's index, kept in the SAME OPERATION a tile removal happens in: a stale entry pointing at
a tile that no longer exists is corruption, not an edge case.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from smolsmort.review import paths

# a stamp is "-<yyyymmdd>-<hhmm[ss]>", joined to what came before it by either separator - boxes
# use a dot ("<recording>.cnn-20260907-010714"), tiles and set sources use an underscore
# ("<recording>_cnn-20260907-010714") - so both must be accepted rather than assuming one
_STAMP_RE = re.compile(r"[._](?:cnn|drawn)-\d{6,8}-\d{3,8}$")

# a saved tile's own index, always last: "<tag>_k00005"
_TILE_INDEX_RE = re.compile(r"_k\d+$")


def _peel_stamp(tag: str) -> str:
    """strip every trailing stamp, not just one - a re-swept set can carry two stacked
    ("..._cnn-20260903-222136" on top of an already-stamped name)"""
    while True:
        stripped = _STAMP_RE.sub("", tag)
        if stripped == tag:
            return tag
        tag = stripped


def tag_to_recording(tag: str) -> str:
    """a box/tile/set-source tag, peeled down to the flat recording name it was minted from"""
    return _peel_stamp(_TILE_INDEX_RE.sub("", tag))


def unflatten(tag: str) -> str:
    """the flat filename form of a recording back to its sessions/ path - see naming._flat"""
    return tag.replace("__", "/")


def _group_of(relative_path: str) -> str:
    """the left column's grouping key: the directory a recording sits under, or itself when it
    sits directly under sessions/ - a flat recording is its own top-level group"""
    parts = relative_path.split("/")
    return parts[0]


class HousekeepingError(ValueError):
    """a request named a path outside the configured bases, or something else refused to run"""


@dataclass
class Recording:
    tag: str  # flat form, e.g. "some_test__1"
    path: str  # sessions/-relative, e.g. "some_test/1"
    group: str
    missing: bool  # True when nothing lives at `path` any more - an orphan


@dataclass
class AssetGroup:
    kind: str  # boxes | tiles | tiles_synth | sets | checkpoints
    label: str  # a human name for this one item within the kind
    paths: list[str] = field(default_factory=list)  # absolute, resolved
    size_bytes: int = 0


def _session_frame_dirs(root: Path) -> list[Path]:
    """every frames/ directory under `root`, however deep a recording is nested."""
    if not root.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        direct = child / "frames"
        if direct.is_dir():
            found.append(direct)
        else:
            found.extend(_session_frame_dirs(child))
    return found


def live_recordings() -> dict[str, Recording]:
    """every recording that still exists on disk, keyed by its flat tag"""
    out: dict[str, Recording] = {}
    for frames_dir in _session_frame_dirs(paths.SESSIONS_DIR):
        rec_dir = frames_dir.parent
        rel = rec_dir.relative_to(paths.SESSIONS_DIR).as_posix()
        tag = rel.replace("/", "__")
        out[tag] = Recording(tag=tag, path=rel, group=_group_of(rel), missing=False)
    return out


def _box_tags() -> set[str]:
    if not paths.LABELS_DIR.is_dir():
        return set()
    tags = set()
    for p in paths.LABELS_DIR.glob("*.candidates.jsonl"):
        stem = p.name.removesuffix(".candidates.jsonl")
        tags.add(tag_to_recording(stem))
    return tags


def _tile_tags(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {tag_to_recording(p.stem) for p in root.glob("*.npz") if p.parent.name != "_archive"}


def _set_source_tags() -> dict[str, list[str]]:
    """set name -> the recording tags its meta.json names as sources"""
    out: dict[str, list[str]] = {}
    if not paths.DATASETS_DIR.is_dir():
        return out
    for meta_path in paths.DATASETS_DIR.glob("*.meta.json"):
        name = meta_path.name.removesuffix(".meta.json")
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            out[name] = []
            continue
        sources = []
        for source in meta.get("sources", []):
            src = source.get("source") if isinstance(source, dict) else None
            if src and src != "synthetic":
                sources.append(tag_to_recording(src))
        out[name] = sources
    return out


def all_recordings() -> list[Recording]:
    """every recording the left column shows: live ones plus orphans - a tag with downstream
    assets but no directory left under sessions/"""
    live = live_recordings()
    referenced = _box_tags() | _tile_tags(paths.TILES_DIR) | _tile_tags(paths.TILES_SYNTH_DIR)
    for sources in _set_source_tags().values():
        referenced.update(sources)

    out = list(live.values())
    for tag in sorted(referenced - live.keys()):
        rel = unflatten(tag)
        out.append(Recording(tag=tag, path=rel, group=_group_of(rel), missing=True))
    return out


def _files_for_tile_index(root: Path) -> dict[str, list[Path]]:
    """recording tag -> every tile file under `root` that peels back to it"""
    out: dict[str, list[Path]] = {}
    if not root.is_dir():
        return out
    for p in root.glob("*.npz"):
        if p.parent.name == "_archive":
            continue
        out.setdefault(tag_to_recording(p.stem), []).append(p)
    return out


def assets_for(tag: str) -> dict[str, list[dict]]:
    """everything a recording (live or orphaned) owns, grouped by asset type - what the right
    column renders once a recording is picked"""
    groups: dict[str, list[dict]] = {
        "boxes": [],
        "tiles": [],
        "tiles_synth": [],
        "sets": [],
        "checkpoints": [],
    }

    if paths.LABELS_DIR.is_dir():
        for jsonl in sorted(paths.LABELS_DIR.glob(f"{tag}.*.candidates.jsonl")):
            stem = jsonl.name.removesuffix(".jsonl")
            if tag_to_recording(jsonl.name.removesuffix(".candidates.jsonl")) != tag:
                continue
            decisions = jsonl.with_name(f"{stem}.decisions.json")
            files = [jsonl] + ([decisions] if decisions.exists() else [])
            groups["boxes"].append(_asset_row(jsonl.name.removesuffix(".candidates.jsonl"), files))

    for key, root in (("tiles", paths.TILES_DIR), ("tiles_synth", paths.TILES_SYNTH_DIR)):
        by_tag = _files_for_tile_index(root)
        files = sorted(by_tag.get(tag, []))
        if files:
            groups[key].append(_asset_row(f"{len(files)} tile(s)", files))

    sources_by_set = _set_source_tags()
    if paths.DATASETS_DIR.is_dir():
        for name, sources in sorted(sources_by_set.items()):
            if tag not in sources:
                continue
            jsonl = paths.DATASETS_DIR / f"{name}.jsonl"
            meta = paths.DATASETS_DIR / f"{name}.meta.json"
            files = [p for p in (jsonl, meta) if p.exists()]
            if files:
                groups["sets"].append(_asset_row(name, files))

    if paths.CHECKPOINTS_DIR.is_dir():
        for classes_path in sorted(paths.CHECKPOINTS_DIR.glob("*.classes.json")):
            try:
                classes = json.loads(classes_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            set_name = classes.get("training_set")
            if tag not in sources_by_set.get(set_name, []):
                continue
            stem = classes_path.name.removesuffix(".classes.json")
            weights = classes_path.with_name(f"{stem}.pt")
            files = [p for p in (weights, classes_path) if p.exists()]
            if files:
                groups["checkpoints"].append(_asset_row(stem, files))

    return groups


def _asset_row(label: str, files: list[Path]) -> dict:
    return {
        "label": label,
        "paths": [str(p.resolve()) for p in files],
        "size_bytes": sum(p.stat().st_size for p in files if p.is_file()),
    }


def orphaned_sets() -> set[str]:
    """set names whose meta.json names no recording still live - NONE of the sources, not just
    the first, since one set can be promoted from several recordings"""
    live = live_recordings().keys()
    return {
        name
        for name, sources in _set_source_tags().items()
        if sources and not (set(sources) & live)
    }


# ---------------------------------------------------------------- containment and destruction


def _allowed_roots() -> list[Path]:
    """every directory housekeeping is allowed to touch, resolved fresh every call - these are
    read off `paths` at call time, the same rule the rest of the module follows, so a settings
    repoint of labels/sessions is honoured rather than a value captured once at import"""
    roots = [
        paths.LABELS_DIR,
        paths.TILES_DIR,
        paths.TILES_SYNTH_DIR,
        paths.DATASETS_DIR,
        paths.CHECKPOINTS_DIR,
    ]
    return [r.resolve() for r in roots if str(r)]


def _resolve_within_bases(raw: str) -> Path:
    """a path the browser sent, resolved and checked against the allowed roots - refuses rather
    than silently no-opping, since a silent no-op reads as success to a caller that never sees it
    fail"""
    resolved = Path(raw).resolve()
    for root in _allowed_roots():
        if resolved == root or root in resolved.parents:
            return resolved
    raise HousekeepingError(f"refusing to touch a path outside the configured bases: {raw}")


def _rewrite_labels_index(directory: Path, removed_stems: set[str]) -> None:
    """drop every removed tile from `_labels.json`, in the same operation as the removal - a
    stale entry pointing at a tile that no longer exists is a corrupted index, not a shrug"""
    index_path = directory / "_labels.json"
    if not index_path.is_file() or not removed_stems:
        return
    try:
        index = json.loads(index_path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    kept = {k: v for k, v in index.items() if k not in removed_stems}
    if len(kept) == len(index):
        return
    # write-then-replace: a crash mid-write leaves the OLD index intact, never a half-written one
    tmp = index_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(kept, indent=1))
    os.replace(tmp, index_path)


def _apply(raw_paths: list[str], mover) -> dict:
    """resolve and validate every path first, THEN act - a request that names one bad path
    touches nothing rather than acting on the good ones and refusing the rest partway through"""
    resolved = [_resolve_within_bases(p) for p in raw_paths]
    touched_dirs: dict[Path, set[str]] = {}
    moved = []
    for path in resolved:
        if not path.exists():
            continue
        if path.suffix == ".npz":
            touched_dirs.setdefault(path.parent, set()).add(path.stem)
        mover(path)
        moved.append(str(path))
    for directory, stems in touched_dirs.items():
        _rewrite_labels_index(directory, stems)
    return {"ok": True, "moved": moved}


def archive_paths(raw_paths: list[str]) -> dict:
    """moves every path into a sibling `_archive/` beside it - the same recoverable move an
    un-kept pool tile already gets"""

    def _move(path: Path) -> None:
        archive_dir = path.parent / "_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(archive_dir / path.name))

    return _apply(raw_paths, _move)


def delete_paths(raw_paths: list[str]) -> dict:
    """unlinks every path for good - training/, sessions/ and labels/ are gitignored working
    data with no second copy, so this is the one action the ui gates behind a confirm step"""

    def _unlink(path: Path) -> None:
        path.unlink()

    return _apply(raw_paths, _unlink)
