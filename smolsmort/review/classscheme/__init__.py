"""the class-scheme seam: what a class is, pluggable rather than rooted in one domain's profile.

This package is the home for `ClassScheme` (seam 3a in docs/REVIEW_TOOL_DESIGN.md): the vocabulary
of classes, and how a human's picks compose into one label. `ClassDef` (definitions.py, carried over
from the parent project's classdefs module) already has the two methods the seam asks for -
`classes()` and `label_for(picked)` - so it satisfies the seam by shape alone, with no adapter class
needed in between.

`ClassGuesser` (seam 3b) is the seam's other half: an optional companion that proposes a class to
prefill the judge view. The parent project's guesser chose a class by nearest measured colour, which
only ever made sense for its one fixed vocabulary - it is deliberately not ported. A scheme with no
guesser is equally valid; the seam simply offers no prefill, and the human's own pick always wins
regardless.
"""

from __future__ import annotations

from .definitions import (
    DEFS_DIR,
    JOIN,
    MAX_DIMENSIONS,
    NOT_A_CLASS,
    UNSET,
    ClassDef,
    ClassDefError,
    Dimension,
    available,
    load,
    parse,
    save,
)

__all__ = [
    "DEFS_DIR",
    "JOIN",
    "MAX_DIMENSIONS",
    "NOT_A_CLASS",
    "UNSET",
    "ClassDef",
    "ClassDefError",
    "Dimension",
    "available",
    "load",
    "parse",
    "save",
]
