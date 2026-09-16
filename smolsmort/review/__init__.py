"""the review web tool: a human-in-the-loop sorting loop, find -> judge -> train -> predict.

Everything here is the loop itself, not any one domain's facts about what is being found or judged -
those live behind the four seams in docs/REVIEW_TOOL_DESIGN.md (example source, example renderer,
class scheme, model backend). This package fills in behind `snapshot/`, a module at a time; a module
moved here is deleted from `snapshot/` in the same change, so what is left there is always what is
still to do.
"""

from __future__ import annotations
