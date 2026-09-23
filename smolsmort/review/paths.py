"""where the review loop looks, and the tunables that go with it.

**ALWAYS REFERENCE THESE AS ATTRIBUTES** - `paths.SESSIONS_DIR`, never
`from ...paths import SESSIONS_DIR`. The difference is not style, and it is not small.

`SESSIONS_DIR`, `LABELS_DIR` and `INTERFACE_SHOTS` are DEFAULTS that get repointed at runtime (by a
settings popup, in the parent project this was carried from) and are monkeypatched by test fixtures
so a test writes into its own tmp_path instead of into real recordings. An importing module binds a
COPY at import time: the test then patches this module's name while the code goes on reading its own,
and the failure is silent - the suite still passes, having quietly read and written the real
`sessions/` and `training/` directories. That is a data-loss bug wearing a green test suite.

`tests/test_review_paths.py` and `tests/test_loop.py` fail the build if any module imports these by
value.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- where things live

# ABSOLUTE, NOT RELATIVE. these were once relative ("sessions", "labels"), so the tool only listed
# anything when launched from the repo root - from anywhere else it silently found nothing, and the
# obvious repair (typing "sessions" into the settings popup) is then wrong from the root.
# this file lives at smolsmort/review/paths.py, so the repo root is three levels up.
_REPO = Path(__file__).resolve().parents[2]
_ROOT = _REPO if (_REPO / "pyproject.toml").exists() else Path.cwd()

# everything the labelling loop produces, in one place: boxes, tiles, sets, weights, library
_TRAINING = _ROOT / "training"

# WHERE EACH KIND OF DATA LIVES, and the one root they are all browsed from. `root` is not itself
# read from - it is where the settings tree starts, so a folder can be PICKED rather than typed.
# `pool` IS THE TILE DIRECTORY ITSELF. It used to be the library path with "_unsorted" appended,
# which is why looking at a generated set needed a directory literally named
# `plate_templates_data_synth_unsorted`; a pool is now just a folder the picker points at.
DEFAULT_BASES = {
    "root": str(_ROOT),
    "sessions": str(_ROOT / "sessions"),
    "labels": str(_TRAINING / "boxes"),
    "templates": str(_TRAINING / "library"),
    "pool": str(_TRAINING / "tiles"),
    # the forecast tool's data root: sources, its prep cache and its run folders all live under
    # it. default is a sibling of sessions - "the recordings root's parent"
    "forecast": str(_ROOT / "forecast"),
}

# the keys set_bases will accept. anything else in a payload is ignored rather than stored, so a
# stale browser cannot invent a base the server then tries to read
BASE_KEYS = ("root", "sessions", "labels", "templates", "pool", "forecast")

# WHICH BASES TAKE SEVERAL PATHS. Only the tile pool so far, and it is the only one where several
# directories mean something obvious: the grid lists every tile across all of them, so a generated
# set is reviewed beside the real one without switching or copying.
#
# NOT sessions: `SESSIONS_DIR` is a single Path that promote turns into every training row's
# recording field via relative_to, so a list there is a change to the training-set schema. NOT
# templates: the example source it feeds matches against one library and merging several has no
# defined meaning. Both are one path until something actually needs otherwise.
MULTI_BASES = ("pool",)

SESSIONS_DIR = Path(DEFAULT_BASES["sessions"])
LABELS_DIR = Path(DEFAULT_BASES["labels"])

# the shared design system and the page, served under /ui/ - nothing else in that directory is
# reachable through the route (see the handler's _serve_ui_asset)
REVIEW_UI_DIR = Path(__file__).resolve().parent / "review_ui"

# ---------------------------------------------------------------- the data flow, in one place

# EVERY DIRECTORY THE LOOP TOUCHES, NAMED ONCE. Seven modules each built these paths themselves,
# so "where do the weights go" had seven answers and moving anything meant finding all of them.
# The flow, in the order a box travels:
#
#   1. RECORDINGS     a session: frames/ plus a log of inputs and state. Nothing writes here.
#   2. LABELS_DIR     training/boxes. Two kinds, and telling them apart is the whole trick:
#                       <recording>.drawn-<stamp>.candidates.jsonl   a human drew these
#                       <recording>.cnn-<stamp>.candidates.jsonl     a sweep proposed these
#                     plus a .decisions.json sidecar per file - cheap, rewritten often, while the
#                     boxes beside it are expensive and never rewritten.
#   3. TILES_DIR      training/tiles: the crops cut from those boxes, one npz each, named
#                     "<tag>_k<index>" so a crop resolves back to the box, the frame and the
#                     recording it came from. TILES_ARCHIVE is `_archive` beneath it - the same
#                     kind of thing in a different state, and an underscore keeps it out of a
#                     tag glob's way.
#   4. DATASETS_DIR   training/sets: promoted rows, the kept labelled tiles plus their negatives.
#   5. MODEL_PATH     training/weights: what training produces. Overwritten by every run.
#   6. CHECKPOINTS    named snapshots of that, with the class map beside them. Never overwritten.
#
# and then a sweep of the new model writes back into (2) as a cnn- file, which is what closes it.
#
# these are data, and they live under `training/` beside `sessions/` rather than inside the
# importable package tree - so a test run, or a `pip install`, cannot touch curated tiles.

RECORDINGS = (
    SESSIONS_DIR  # an alias that says what it is, since SESSIONS_DIR is repointed at runtime
)

TRAINING_DIR = _TRAINING
TILES_DIR = _TRAINING / "tiles"
TILES_ARCHIVE = TILES_DIR / "_archive"
# a generated pool, reviewed beside the real one (see MULTI_BASES) - never remapped by settings,
# unlike TILES_DIR, since nothing there is a live recording's own output
TILES_SYNTH_DIR = _TRAINING / "tiles-synth"
LIBRARY_DIR = _TRAINING / "library"
WEIGHTS_DIR = _TRAINING / "weights"
MODEL_PATH = WEIGHTS_DIR / "plate_model.pt"
CHECKPOINTS_DIR = WEIGHTS_DIR / "checkpoints"
DATASETS_DIR = _TRAINING / "sets"

# WHERE A SCREEN CAPTURE (or other whole-frame reference shot) THE FIND STEP DRAWS AGAINST LIVES.
# repointed at runtime and monkeypatched by tests the same way SESSIONS_DIR and LABELS_DIR are -
# see the module docstring's warning.
INTERFACE_SHOTS = _ROOT / "profiles" / "interface_shots"


def drawn_candidates(labels_root: Path, tag: str):
    """every hand-drawn candidates file for a recording, newest last"""
    return sorted(labels_root.glob(f"{tag}.drawn-*.candidates.jsonl"))


def swept_candidates(labels_root: Path, tag: str):
    """every model-proposed candidates file for a recording, newest last"""
    return sorted(labels_root.glob(f"{tag}.cnn-*.candidates.jsonl"))


# ---------------------------------------------------------------- tunables

# a sweep keeps everything at or above this, whatever threshold is chosen afterwards - low enough
# that the distribution has a shape to look at, high enough that an undertrained model's noise
# floor does not become the whole answer. MEASURED in the parent project this was carried from.
SWEEP_KEEP_FLOOR = 0.05

# drag slack around a candidate, not compensation for a wrong starting position: candidates start
# centred on the detected box, so this only has to cover a hand adjustment. MEASURED in the parent
# project this was carried from.
MARGIN_X = 30
MARGIN_Y = 12

PAGE_SIZE = 4

# HOW MUCH GROUND AROUND A DRAWN BOX a cut tile keeps, as a fraction of the box on each side.
# Two numbers rather than one: vertical pad used to be derived from a fixed tile aspect, which made
# it depend on the box's SHAPE - MEASURED at 71% of height for a 132x24 box and 144% for a 162x18
# one, for the same setting. Both are settable at runtime from find's crop menu.
#
# defaults chosen to sit near what the tile-derived pad produced for a real object in the parent
# project this was carried from (226x35 -> 89%), so this change does not silently reframe every
# existing crop
PAD_X_DEFAULT = 0.25
PAD_Y_DEFAULT = 0.9

# the cell a tile occupies when uniform tiles are switched ON. off, a tile keeps its own aspect
TILE_WIDTH = 208
TILE_HEIGHT = 60
