"""example renderer seam, image implementation: a frame, a padded crop, a thumbnail.

the seam (docs/REVIEW_TOOL_DESIGN.md, seam 2) is `frame(name)` and `crop(candidate)`. this
implementation adds one optional keyword to both, `frames_dir`, because the loop routinely renders
a candidate from a recording other than the one bound: the pool spans datasets. a renderer that
needs no directory (a tabular one) simply ignores it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from smolsmort.review import paths
from smolsmort.review.tiles import png_bytes

# how far a thumbnail reaches left and right of its box, in frame pixels. a look-around, not a
# measured value: wide enough to show what the box sits beside without shrinking the box itself
THUMB_SIDE_MARGIN = 20


def read_frame(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def corrected_box(candidate: dict, decision: dict | None) -> dict:
    """the candidate box with a re-centre or a drag applied, in FRAME coordinates.

    a correction is stored crop-local, relative to the padded crop the browser was shown, whose
    origin is the candidate inset by MARGIN_X/MARGIN_Y - while everything downstream (promotion,
    training) works in frame coordinates. unchanged when there is no correction.
    """
    rect = (decision or {}).get("rect")
    if not rect:
        return candidate
    return {
        **candidate,
        "left": candidate["left"] - paths.MARGIN_X + int(rect["left"]),
        "top": candidate["top"] - paths.MARGIN_Y + int(rect["top"]),
        "width": int(rect["width"]),
        "height": int(rect["height"]),
    }


def padded_canvas(frame: np.ndarray, candidate: dict) -> np.ndarray:
    """the box plus margin, ALWAYS exactly (width+2*MARGIN_X, height+2*MARGIN_Y), black where the
    frame does not reach.

    never clamped smaller: the displayed crop and the extracted rect must agree on where (0,0) is,
    and a clamped crop near a frame edge silently moves it.
    """
    frame_h, frame_w = frame.shape[:2]
    pad_h = candidate["height"] + 2 * paths.MARGIN_Y
    pad_w = candidate["width"] + 2 * paths.MARGIN_X
    origin_top = candidate["top"] - paths.MARGIN_Y
    origin_left = candidate["left"] - paths.MARGIN_X

    canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    src_top, src_left = max(0, origin_top), max(0, origin_left)
    src_bottom, src_right = min(frame_h, origin_top + pad_h), min(frame_w, origin_left + pad_w)
    if src_bottom <= src_top or src_right <= src_left:
        return canvas
    dest_top, dest_left = src_top - origin_top, src_left - origin_left
    canvas[
        dest_top : dest_top + (src_bottom - src_top),
        dest_left : dest_left + (src_right - src_left),
    ] = frame[src_top:src_bottom, src_left:src_right]
    return canvas


class ImageRenderer:
    """renders from frame files on disk. every method takes the recording's frames directory."""

    def frame(self, name: str, *, frames_dir: Path | None = None) -> bytes:
        """one whole frame file, for drawing on. confined to `frames_dir`."""
        if frames_dir is None:
            raise FileNotFoundError(name)
        target = (frames_dir / Path(name).name).resolve()
        if frames_dir.resolve() not in target.parents or not target.is_file():
            raise FileNotFoundError(name)
        return target.read_bytes()

    def crop(self, candidate: dict, *, frames_dir: Path | None = None) -> bytes:
        """the padded crop around a candidate, as png"""
        if frames_dir is None:
            raise FileNotFoundError(candidate.get("path", ""))
        frame = read_frame(frames_dir / candidate["path"])
        return png_bytes(Image.fromarray(padded_canvas(frame, candidate)))

    def thumb(self, box: dict, *, frames_dir: Path) -> bytes:
        """the box with real frame around it, kept centred vertically however little room a frame
        has: the margin above and below is the SAME amount, capped by the smaller side, so a box
        near an edge degrades to a tight crop instead of a lopsided one."""
        frame = read_frame(frames_dir / box["path"])
        frame_h, frame_w = frame.shape[:2]
        top, left, height, width = box["top"], box["left"], box["height"], box["width"]
        margin = max(0, min(height // 2, top, frame_h - (top + height)))
        visible = frame[
            top - margin : top + height + margin,
            max(0, left - THUMB_SIDE_MARGIN) : min(frame_w, left + width + THUMB_SIDE_MARGIN),
        ]
        return png_bytes(Image.fromarray(visible))
