"""a rectangle on a frame, and the two tests for "is this the same object again".

WHY IT LIVES ON THIS SIDE OF THE BOUNDARY. It used to be `PlateBox` in `vision/labels.py`, which
made `detect/scoring.py` import from the consumer to score a detector - the dependency running
backwards through the seam this package exists to draw. A rectangle is a rectangle; nothing here
knows what is inside it.

THE TOLERANCES ARE DEFAULTS, NOT FACTS, and that distinction is the whole reason this move is
honest rather than cosmetic. Every number below was measured on a long thin target bar, ~132x12 -
and a consumer detecting something of a different shape should pass its own. They are parameters
with a provenance, which is why the provenance is written down beside them.
"""

from __future__ import annotations

from dataclasses import dataclass

# two boxes this close in both axes are the same object seen again. measured on real captures: the
# teacher's own left edge jitters a few px between frames and a mob drifts while walking, so an
# exact match finds nothing
SAME_OBJECT_PX = 40
# two boxes in the SAME frame sharing this much of the narrower one's WIDTH, and lying within
# SAME_OBJECT_MAX_ROWS_APART box-heights of each other vertically, are one object seen twice
SAME_OBJECT_MIN_SHARE = 0.5
SAME_OBJECT_MAX_ROWS_APART = 1.5


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    width: int
    height: int
    origin: str = "teacher"  # teacher | repaired | human

    @property
    def centre_x(self) -> int:
        return self.left + self.width // 2

    def near(self, other: Box, tolerance: int = SAME_OBJECT_PX) -> bool:
        """the same object across DIFFERENT frames, by corner distance.

        deliberately conservative: a generous tolerance here would start matching two different
        objects standing near each other, and bridging between those invents one never on screen.
        """
        return abs(self.left - other.left) <= tolerance and abs(self.top - other.top) <= tolerance

    def overlaps(
        self,
        other: Box,
        min_share: float = SAME_OBJECT_MIN_SHARE,
        max_rows_apart: float = SAME_OBJECT_MAX_ROWS_APART,
    ) -> bool:
        """the same object WITHIN one frame - horizontal overlap plus a row-height sanity check.

        TWO WRONG TESTS WERE TRIED FIRST, both on the same real case: frame 0000010 had a teacher
        box at left=1337 and repair inserted a second at 1384, both over one 'Giant Moss Creeper'.
        CORNER DISTANCE missed it because 47 px is outside a 40 px tolerance. IOU missed it too, for
        a subtler reason worth keeping: the object is ~132x12, so a mere 8 px of vertical offset
        drops the intersection to a third of one box's height and IoU to 0.12 - area overlap is
        hopeless on a shape this thin.

        what actually identifies it is that the boxes lie along the same row and cover the same
        span, so that is what this measures. A caller whose objects are not long and thin should
        pass its own thresholds, or use IoU, which is fine for a squarer shape.
        """
        left = max(self.left, other.left)
        right = min(self.left + self.width, other.left + other.width)
        shared = right - left
        if shared <= 0:
            return False
        share = shared / min(self.width, other.width)
        rows_apart = abs((self.top + self.height / 2) - (other.top + other.height / 2)) / max(
            self.height, other.height, 1
        )
        return share >= min_share and rows_apart <= max_rows_apart
