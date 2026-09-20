"""where a recording's frames live, and how a dataset tag finds its recording.

a recording is a directory under `paths.SESSIONS_DIR` holding a `frames/` folder of images. a
dataset tag (the name a candidates file or a tile carries) is that recording's name flattened by
naming._flat, sometimes with a run suffix - these helpers undo both.
"""

from __future__ import annotations

from pathlib import Path

from smolsmort.review import paths
from smolsmort.review.naming import _flat

FRAME_SUFFIXES = (".jpg", ".png")


def frames_dir_of(session_dir: Path) -> Path:
    return session_dir / "frames"


def frame_files(frames_dir: Path) -> list[Path]:
    """every frame image in a frames directory, jpgs first then pngs, each sorted by name"""
    if not frames_dir.is_dir():
        return []
    found: list[Path] = []
    for suffix in FRAME_SUFFIXES:
        found.extend(sorted(frames_dir.glob(f"*{suffix}")))
    return found


def recording_name(frames_dir: Path) -> str:
    """the recording a frames directory belongs to: its path from sessions/ when it is under it,
    else just the folder's own name"""
    sessions = paths.SESSIONS_DIR.resolve()
    if sessions in frames_dir.resolve().parents:
        return str(frames_dir.parent.resolve().relative_to(sessions))
    return frames_dir.parent.name


def session_frames_dirs(root: Path, depth: int = 3) -> list[Path]:
    """every frames/ directory at or under `root`, nearest first.

    a set gathered for one purpose is a folder OF recordings (sessions/set/{1,2,3}/frames), so
    requiring frames/ directly inside each child would list nothing for it.
    """
    if not root.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        frames_dir = frames_dir_of(child)
        if frames_dir.is_dir():
            found.append(frames_dir)
        elif depth > 1:
            found.extend(session_frames_dirs(child, depth - 1))
    return found


def session_frames_for(tag: str) -> Path | None:
    """the frames dir a candidates file or tile tag was cut from, allowing variant names.

    a tag is not always exactly its recording: re-mining gives "<recording>.band", and a model
    sweep gives "<recording>_cnn-<stamp>". drop one suffix at a time until a real recording turns
    up. "__" is _flat's stand-in for the "/" of a nested recording's name.
    """
    candidate = tag
    while candidate:
        for name in (candidate, candidate.replace("__", "/")):
            frames_dir = frames_dir_of(paths.SESSIONS_DIR / name)
            if frames_dir.is_dir():
                return frames_dir
        # a sweep tag separates with an underscore, not a dot, so a dot-only strip never reaches
        # its recording and every proposal from it would be dropped as unresolved
        for infix in ("_cnn-", "_drawn-"):
            if infix in candidate:
                candidate = candidate.rsplit(infix, 1)[0]
                break
        else:
            if "." not in candidate:
                return None
            candidate = candidate.rsplit(".", 1)[0]
    return None


def flat_recording(name: str) -> str:
    return _flat(name)


def dataset_tag(filename: str) -> tuple[str, str]:
    """a candidates file's pool tag, and whether a hand or a model produced it: (tag, kind).

    the one place this mapping lives - opening a dataset, listing what is openable and finding a
    tile's source all derive the tag identically.
    """
    if ".drawn-" in filename:
        return filename.split(".drawn-")[0], "find"
    if ".cnn-" in filename:
        base, rest = filename.split(".cnn-", 1)
        return f"{base}_cnn-{rest.split('.')[0]}", "sweep"
    return filename.split(".")[0], "find"


def candidates_file_for(tag: str) -> Path | None:
    """the candidates file a pool tag came from, or None when it cannot be named for certain.

    a tile's tag is not its file's stem: the file keeps the run suffix ("rec.drawn-2026...") while
    the tile drops it, and a sweep flattens its dot. try the literal name, then a flattened or
    suffix-dropped match. ambiguity is refused - two runs are two different lists and picking the
    wrong one writes a rect onto an unrelated box.
    """
    if not tag:
        return None
    exact = paths.LABELS_DIR / f"{tag}.candidates.jsonl"
    if exact.is_file():
        return exact
    matches = []
    for path in paths.LABELS_DIR.glob("*.candidates.jsonl"):
        stem = path.name[: -len(".candidates.jsonl")]
        if stem.replace(".", "_") == tag or stem.split(".", 1)[0] == tag:
            matches.append(path)
    return matches[0] if len(matches) == 1 else None
