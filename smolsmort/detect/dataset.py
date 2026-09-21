"""training examples for the heatmap cnn, assembled from the review tool's own decisions.

ONE CLASS, DELIBERATELY. The label is "there is an object centred here" and nothing else. Which
class it belongs to is a separate question the consumer's own matcher already answers, and folding
it in here would mean inventing labels for
combinations the captures have never contained.

THE LABELS ARE INCOMPLETE, AND THAT IS THE INTERESTING PART. A kept candidate is an object. A frame
having no other kept candidate does NOT mean it holds no other - the band datasets are a
FILTERED subset of what was mined (an NCC score band), so a frame can hold objects that were never
offered for review at all. Training those pixels as background teaches the model to suppress exactly
what it is meant to find.

So every candidate that was mined but is not a confirmed positive becomes an IGNORE region rather
than background: the loss simply does not look there.

A DISCARD IS NOT A NEGATIVE, and this was got wrong first time. The review tool's Keep means "a
real, well-aligned tile, worth having", so Discard covers misaligned, clipped and redundant tiles
just as much as not-an-object. Drawing the labels on a real frame settled it: a discarded
candidate sat squarely on a perfectly legible object. Training that as background
teaches the model to suppress the very thing it is for. Discards are therefore ignored too, and this
dataset carries NO explicit negatives - the background is everything the mining never proposed,
which is almost all of every frame and is plenty.

THIS IS THE RULE FOR build/build_from, a candidates queue with keep and discard decisions. the
review tool's promoted sets differ: there a discard ("not a class") is one verdict, "this box is a
negative", and promotion writes it as a negative row for swept and drawn boxes alike.

THAT REASONING HOLDS ONLY WHERE THE MINING WAS A FILTERED SUBSET. A human can also declare a frame
EXHAUSTIVE - every object in it has been proposed and reviewed, nothing else is out there. On such
a frame an unkept candidate is not an unknown, it is a confirmed absence, so it becomes a real
negative instead of an ignore region. See build_from and Example.exhaustive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# the review tool pads each candidate crop by this much before the user aligns a rect inside it;
# rect coordinates are relative to that padded crop, so undoing it recovers frame coordinates
MARGIN_X = 30
MARGIN_Y = 12


class DatasetError(Exception):
    pass


def frame_width(path: Path) -> int:
    """a frame's width in px, read from its header without decoding it"""
    from PIL import Image

    with Image.open(path) as handle:
        return handle.width


@dataclass
class Example:
    """one frame, everything known about where the objects in it are"""

    path: Path
    centres: list[tuple[float, float]] = field(default_factory=list)
    # the class each centre belongs to, parallel to `centres` and the same length. None where the
    # the object is confirmed but unlabelled, which is the honest state of a drawn box before it has
    # been through cluster - a single-channel model ignores this entirely, so it stays optional
    labels: list[str | None] = field(default_factory=list)
    ignore: list[tuple[float, float, float, float]] = field(default_factory=list)
    negatives: list[tuple[float, float]] = field(default_factory=list)
    # (width, height) in frame px, parallel to `centres`. optional - the fixed-size backend
    # never reads it, since its whole premise is one box size for every object
    sizes: list[tuple[float, float]] = field(default_factory=list)
    # true when a human declared this frame's mining complete - see the module docstring
    exhaustive: bool = False

    @property
    def object_count(self) -> int:
        return len(self.centres)


def centre_of(candidate: dict, decision: dict | None, uniform_width: int, uniform_height: int):
    """where the object actually sits in FRAME coordinates.

    the candidate's own box is the detector's guess; if the reviewer dragged an alignment rect the
    rect is the corrected position and wins. rect coordinates are crop-local, so the crop's origin
    (the candidate box less the padding) is added back.
    """
    rect = (decision or {}).get("rect")
    if rect:
        origin_left = candidate["left"] - MARGIN_X
        origin_top = candidate["top"] - MARGIN_Y
        height = max(4, uniform_height + (decision or {}).get("height_delta", 0))
        left = origin_left + float(rect["left"])
        top = origin_top + float(rect["top"])
        return left + uniform_width / 2.0, top + height / 2.0
    return (
        candidate["left"] + candidate["width"] / 2.0,
        candidate["top"] + candidate["height"] / 2.0,
    )


def build(
    candidates_path: Path,
    frames_dir: Path,
    *,
    uniform_width: int = 64,
    uniform_height: int = 14,
) -> list[Example]:
    """group one dataset's reviewed candidates into per-frame examples, reading from disk"""
    if not candidates_path.is_file():
        raise DatasetError(f"no candidates file at {candidates_path}")
    candidates = [
        json.loads(line) for line in candidates_path.read_text().splitlines() if line.strip()
    ]
    decisions_path = candidates_path.with_suffix(".decisions.json")
    decisions = json.loads(decisions_path.read_text()) if decisions_path.is_file() else {}
    return build_from(
        candidates,
        decisions,
        frames_dir,
        uniform_width=uniform_width,
        uniform_height=uniform_height,
    )


_REQUIRED_ROW_KEYS = ("recording", "frame", "left", "top", "width", "height")
_NUMERIC_ROW_KEYS = ("left", "top", "width", "height")


class ClassMap(dict):
    """the label->channel map build_training_set returns - still a plain dict, so every
    existing caller (bind(), dict(classes), len(classes)) keeps working unchanged.

    `skipped` rides along on the same object: reason string -> row count, for whichever rows
    build_training_set had to drop. it lives here rather than a third return value or a module
    global because it is only ever meaningful alongside the map it was computed with.
    """

    skipped: dict[str, int]


def _row_skip_reason(row: dict) -> str | None:
    """why this row can't become an Example, or None if it's usable.

    fail-loud on provenance, but per-row rather than per-file: one bad row should not refuse
    a whole promoted set, so the caller counts and skips instead of raising.
    """
    for key in _REQUIRED_ROW_KEYS:
        if key not in row:
            return key
    for key in _NUMERIC_ROW_KEYS:
        try:
            float(row[key])
        except (TypeError, ValueError):
            return f"{key}: non-numeric"
    if not row.get("negative") and "label" not in row:
        return "label"
    return None


def build_training_set(
    path: Path, sessions_dir: Path, *, refuse_mixed_resolutions: bool = True
) -> tuple[list[Example], ClassMap]:
    """the promoted training set on disk -> examples, plus the class->channel map to train with.

    ONE FILE, MANY RECORDINGS. each row names its own recording, so a set assembled from several
    drawn sessions and several cnn sweeps loads as one thing - which is the point of promoting
    rather than training off whichever candidates queue happened to be bound.

    the channel map is derived from the labels PRESENT, sorted, so it is stable across runs on the
    same file. it is returned rather than stored because a checkpoint is only meaningful with the
    map it was trained under - see detect.train.load's classes argument.

    refuse_mixed_resolutions is on by default because the fixed-size backend has no single box
    size that is right across two capture resolutions - see _refuse_mixed_resolutions. a
    size-aware backend rescales its input instead, so that premise does not hold for it and it
    passes False.
    """
    if not path.is_file():
        raise DatasetError(f"no training set at {path}")
    raw_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not raw_rows:
        raise DatasetError(f"{path} is empty - promote some labelled kernels first")

    skipped: dict[str, int] = {}
    rows = []
    for row in raw_rows:
        reason = _row_skip_reason(row)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            rows.append(row)

    labels = sorted({r["label"] for r in rows if r.get("label")})
    classes = ClassMap((label, index) for index, label in enumerate(labels))
    classes.skipped = skipped
    by_frame: dict[tuple[str, str], Example] = {}
    for row in rows:
        key = (row["recording"], row["frame"])
        image = sessions_dir / row["recording"] / "frames" / row["frame"]
        example = by_frame.setdefault(key, Example(path=image))
        if row.get("exhaustive"):
            example.exhaustive = True
        left, top = float(row["left"]), float(row["top"])
        width, height = float(row["width"]), float(row["height"])
        centre = (left + width / 2.0, top + height / 2.0)
        if row.get("negative"):
            # A HARD NEGATIVE: somewhere the model fired and was told no. sampled deliberately
            # during training because a random window almost never contains one, so without them
            # the model never revisits its own mistakes
            example.negatives.append(centre)
        else:
            example.centres.append(centre)
            example.labels.append(row["label"])
            example.sizes.append((width, height))
    examples = [e for e in by_frame.values() if e.path.is_file()]
    if refuse_mixed_resolutions:
        _refuse_mixed_resolutions(examples, path)
    return examples, classes


def _refuse_mixed_resolutions(examples: list[Example], path: Path) -> None:
    """a training set must not mix capture resolutions, and this is where that is caught.

    WHY IT MATTERS MORE THAN IT LOOKS. the object is a fixed number of PIXELS on a given screen -
    226x35 on retina, ~162x18 on a 2560-wide capture - which is the whole reason the model has
    nothing to regress and only has to answer where and which. Mix two resolutions into one set
    and that premise breaks: the fitted box size is right for neither half, the sweep emits
    proposals at a size that fits neither, and the loss is asked to find one object that is two
    different sizes. Nothing downstream can detect it, because every row looks individually fine.

    checked by reading each recording's FIRST frame only - one open per recording, not per row.
    """
    from PIL import Image

    sizes: dict[str, tuple[int, int]] = {}
    for example in examples:
        recording = example.path.parent.parent.name
        if recording in sizes:
            continue
        try:
            with Image.open(example.path) as handle:
                sizes[recording] = handle.size
        except (OSError, ValueError):
            continue
    distinct = set(sizes.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{name} {w}x{h}" for name, (w, h) in sorted(sizes.items()))
        raise DatasetError(
            f"{path.name} mixes capture resolutions ({detail}). The object is a fixed pixel size on "
            "one screen, so a set spanning two has no single box size that is right - promote one "
            "set per screen instead"
        )


def _inside_kept_extent(centre: tuple[float, float], example: Example) -> bool:
    """true if centre sits inside any of this frame's kept boxes, using their own sizes"""
    for (cx, cy), (width, height) in zip(example.centres, example.sizes, strict=True):
        if abs(centre[0] - cx) <= width / 2.0 and abs(centre[1] - cy) <= height / 2.0:
            return True
    return False


def build_from(
    candidates: list[dict],
    decisions: dict,
    frames_dir: Path,
    *,
    uniform_width: int = 64,
    uniform_height: int = 14,
    labels: dict[int, str] | None = None,
    exhaustive_frames: set[str] | None = None,
) -> list[Example]:
    """same, from lists already in memory - the review server holds live decisions that the file
    on disk may not have caught up with yet.

    EXHAUSTIVE MODE. exhaustive_frames names frames a human declared complete - every candidate on
    them was proposed and reviewed, nothing else is out there. On those frames an unkept candidate
    is a confirmed absence, so it becomes a NEGATIVE rather than an ignore region. This needs two
    passes over the frame's candidates, because a kept box that should exempt a nearby discard can
    come later in list order than the discard itself. A discard whose centre falls inside a kept
    box's own extent is dropped entirely - neither ignore nor negative - since that is a
    misaligned or redundant tile of a real object, not evidence of absence. Explicit (non-
    exhaustive) frames are unaffected: the module docstring's "a discard is not a negative" still
    holds there.
    """
    exhaustive_frames = exhaustive_frames or set()
    by_frame: dict[str, Example] = {}
    pending: list[tuple[str, tuple[float, float]]] = []
    for index, candidate in enumerate(candidates):
        decision = decisions.get(str(index))
        image = frames_dir / candidate["path"]
        example = by_frame.setdefault(candidate["path"], Example(path=image))
        if candidate["path"] in exhaustive_frames:
            example.exhaustive = True
        centre = centre_of(candidate, decision, uniform_width, uniform_height)

        if decision and decision.get("keep"):
            example.centres.append(centre)
            # cluster's own naming, looked up by candidate index. it lives POOL-WIDE keyed by
            # kernel name (see ReviewState.pool_labels) because index alone collides across
            # datasets; the caller resolves that to indices before handing it here
            example.labels.append((labels or {}).get(index))
            rect = (decision or {}).get("rect")
            if rect:
                height = max(4, uniform_height + decision.get("height_delta", 0))
                example.sizes.append((uniform_width, height))
            else:
                example.sizes.append((candidate["width"], candidate["height"]))
        else:
            # everything else - discarded OR never ruled on - is unknowable as a label on an
            # explicit frame; on an exhaustive one it is resolved in the second pass below
            pending.append((candidate["path"], centre))

    for path, centre in pending:
        example = by_frame[path]
        if example.exhaustive:
            if _inside_kept_extent(centre, example):
                continue
            example.negatives.append(centre)
        else:
            half = (uniform_width, uniform_height)
            example.ignore.append(
                (centre[0] - half[0], centre[1] - half[1], centre[0] + half[0], centre[1] + half[1])
            )
    return [e for e in by_frame.values() if e.path.is_file()]


def summarise(examples: list[Example]) -> dict:
    """what the caller needs to decide whether this is worth training on"""
    return {
        "frames": len(examples),
        "objects": sum(e.object_count for e in examples),
        "with_objects": sum(1 for e in examples if e.object_count),
        "negatives": sum(len(e.negatives) for e in examples),
        "ignored": sum(len(e.ignore) for e in examples),
        "exhaustive_frames": sum(1 for e in examples if e.exhaustive),
        "max_per_frame": max((e.object_count for e in examples), default=0),
    }
