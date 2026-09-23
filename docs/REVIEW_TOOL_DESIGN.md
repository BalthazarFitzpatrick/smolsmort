# the review web tool: plugin seams and package layout

This is the design every later port card refactors against. It is authored from the opening brief,
not from the snapshot branch: it says what the ported tool must become, and where each piece lives,
so a port card is a move-and-adapt against a fixed target rather than a fresh decision each time.

The tool is a human-in-the-loop sorting loop, and nothing about the loop is specific to one domain:

    find -> judge -> train -> predict -> judge -> ...

- **find**    an example source proposes candidates to look at
- **judge**   a human keeps or discards each, and (optionally) names its class
- **train**   a model backend learns from the kept, judged examples
- **predict** the same backend proposes on frames nobody has judged; those re-enter judging

The four things that vary between domains are pulled behind explicit seams. Everything else - the
paths, the naming, the splits, the housekeeping, the web server, the tabs - is the loop, and is
carried across nearly as-is. `smolsmort/detect` already exists as the first model backend and states
this same split in its own docstrings: "ONE BACKEND, NOT THE PRODUCT ... anything that would be
equally true of a spreadsheet row belongs in the loop instead". The seams are where that boundary is
drawn in code.

## the four seams

Each seam is a `typing.Protocol` (see `tests/test_loop.py`, where they are authored and exercised
against fakes). A protocol rather than a base class on purpose: an implementation conforms by shape,
so a backend or a source need not import the loop to be usable by it, and the dependency only ever
runs one way - the loop depends on the seam, never on a concrete implementation.

### 1. example source - where candidates come from

    class ExampleSource(Protocol):
        def find(self, recording: str) -> list[Candidate]: ...

`find` returns candidate boxes in the schema the rest of the loop already reads: a dict with
`path, left, top, width, height, matched_template, score`, and an optional `negative` flag. That is
the schema drawn boxes, template matches and model sweeps all already emit, so a source's output
opens in the judging tab unchanged.

**Responsibilities.** Propose *where to look*, and nothing about *what is there*. A source may score
its proposals (a match strength, a peak height, or `1.0` for a box a human drew), but it never
assigns a class and never decides keep/discard - those are the human's, at the judge step.

**Boundary.** A source knows about frames and rectangles. It must not know about the class scheme,
the model, or the renderer. It is the seam the parent project's template matcher sits behind - the
first implementation - and the CNN sweep is a second implementation of the *same* seam: `predict`
turns a trained backend back into a source, which is what closes the loop. A third implementation is
a human drawing boxes on whole frames (the cold start, when no model and no library exist yet).

Known implementations at port time:
- template matching over a fixed-size bar (the parent project's first source)
- human-drawn boxes (the find tab's save path)
- a model sweep (wraps `ModelBackend.predict`; the loop's return edge)

### 2. example renderer - turning a candidate into something a human can judge

    class ExampleRenderer(Protocol):
        def frame(self, name: str, *, frames_dir: Path) -> bytes: ...     # a whole frame, to draw on
        def crop(self, candidate: Candidate, *, frames_dir: Path) -> bytes: ...  # framed for judging
        def thumb(self, box: Candidate, *, frames_dir: Path) -> bytes: ...  # the box in its frame

Every method takes the recording's `frames_dir`, because one pool can span several recordings, and
every read goes through one confinement (`render.frame_file`): a stored path is cut to its bare
file name and must land inside `frames_dir`, so no method trusts what a candidates file says.

**Responsibilities.** Read a source frame and produce the pixels the browser shows: a whole frame for
the drawing surface, and a padded crop for one candidate on the judge tab. The pad around a crop is
the renderer's concern (a bare box stops at the object and leaves nothing to lock onto), and it is
re-derivable - the frames are on disk and so are the rectangles, so the pad is never baked into
anything that cannot be cut again.

**Boundary.** A renderer knows how to read a frame and cut a crop; it knows nothing about classes or
models. It is an image concern for the image backend, and a table concern for a tabular one - a
tabular backend renders a row, not a picture, behind the same seam. This is why it is a seam and not
a shared helper: "render a candidate for judging" means different things per domain, and the loop
must not assume pixels.

### 3. class scheme (+ optional guesser) - what a class is, and a guess to prefill

    class ClassScheme(Protocol):
        def classes(self) -> list[str]: ...              # every class, in a stable sorted order
        def label_for(self, picked: Mapping[str, str]) -> str: ...

    class ClassGuesser(Protocol):   # optional companion; a scheme may have none
        def guess(self, candidate: Candidate) -> str | None: ...

**Responsibilities.** The scheme defines the vocabulary: which classes exist, and how a human's
picks compose into one label. It replaces two things that used to be hardcoded in the parent project
- a fixed class list and a colour-based guesser that *chose* the class for you. In the ported tool
the scheme is data (a definition of dimensions and their members; a class is one member from each,
the full cross-product), and `classes()` is derived from that definition, sorted - so the model's
channel map, built from the labels present sorted, lines up with it by construction.

**The guesser is optional and never authoritative.** It proposes a class to prefill the judge view;
the human's pick always wins, and a scheme with no guesser simply offers no prefill. The parent
project's colour-nearest-template guesser is the first implementation; a scheme that cannot be
guessed at all (returns `None`) is equally valid.

**Boundary.** The scheme knows the vocabulary and the guesser knows how to peek at a candidate to
suggest one word of it. Neither trains, sweeps, renders, or decides keep/discard. Definitions are
versioned and old labels stand: editing a definition never rewrites a judgement already made, and a
training set mixing two definitions is refused at promotion - the same guard that refuses a set
mixing capture resolutions.

### 4. model backend - train on judgements, predict new candidates

    class ModelBackend(Protocol):
        def train(self, examples, *, classes, on_progress=None) -> object: ...  # -> a weights handle
        def predict(self, weights, frames, *, classes) -> list[Candidate]: ...
        def save(self, weights, path) -> Path: ...
        def load(self, path) -> object: ...

**Responsibilities.** Learn from the kept, judged examples and their class map; produce a weights
handle; and sweep frames to propose new candidates in the same `Candidate` schema `find` uses - so
`predict` is itself an example source, and the loop closes without the loop knowing which backend
produced the proposals.

**Boundary.** The backend owns everything domain-specific about *modelling*: the object's fixed
known size (the image backend regresses no box and only answers where/which), the training window,
the loss, the peak decode - and, for a tabular backend, none of those and a different set entirely.
The backend carries the box size it was trained at (from the training set), so `predict` needs only
weights, frames and the class map. `smolsmort/detect` is the first implementation (a ~50k-parameter
heatmap CNN); a tabular one (xgboost, for forecasting) is meant to sit beside it, not replace it.

**Two implementations exist, picked by name.** `smolsmort/boxes` is the second: a size-aware CNN
that predicts each object's own box, for objects that vary in size and frames of different
resolutions. `smolsmort/backends.py` maps a name (`heatmap`, `box`, `xgboost`) to the class that
implements it, imported lazily so listing names never imports torch. The vision names are what the
train tab offers (the `xgboost` row backend stays a mechanics check of the seam), and
`register()` adds an outside model without editing smolsmort. A JSON sidecar saved beside each
checkpoint names the backend that wrote it; a checkpoint without one is a heatmap checkpoint. The
example dicts the loop hands a backend carry per-object `sizes` beside `centres`: a size-aware
backend learns them, and a fixed-size one reads the single width/height.

**Two labelling modes, decided per frame, in the dataset layer rather than in any backend.**
*Explicit* is the default: a box nobody judged is ignored. In the review tool a discard
("not a class") is a negative on every frame, swept or drawn. *Exhaustive* means a human declared the frame complete, so whatever
was proposed and not kept becomes a negative. The exception is a discard centred inside a kept box,
which is a misaligned copy of a real object. Both backends get both modes for free.

**Two optional files.** A training set
names its backend and size mode in `<set>._backend.json`, and a recording keeps per-frame labelling
modes in `<recording>._frames.json` (see the README); both are read through `smolsmort.review.setconfig`
and absent means the old behaviour.

**Torch stays optional.** `detect/model.py` imports torch lazily inside `_torch()`, and the package
declares it as an optional `vision` extra (`pyproject.toml`). The loop itself never imports torch,
so a consumer doing tabular work is not made to install it. Tests that exercise the real CNN keep
the `pytest.importorskip("torch")` pattern; the loop test uses a FAKE backend and therefore runs
with no torch at all.

## topics, and why forecasting is not a fifth seam

The page groups its tabs into topics - vision, regression, classification - each with its own tabs
and a first-row dropdown for backend and flavour. A tab registered without a topic lands in vision,
so a host page that never heard of topics keeps working.

The regression and classification topics do not run through the four seams. The seams exist for a
loop in which **a human judges** each candidate; a forecast has nobody to judge it. Its judge is the
holdout: rows cut by time into train, validation and test, a search that only ever reads validation,
and a test split touched once by the frozen winner. What comes back is a forecast plus a verdict that
says whether it is worth acting on. Forcing that through `ModelBackend` would have meant candidates
with no boxes and a judge step that never happens - the shape would lie about the work.

Two rules the forecast package keeps that the vision loop never needed:

- **Nothing after the origin.** A series feature for step d, predicted from origin d - b, reads only
  values up to the origin. A test changes every value after an origin and asserts no feature at or
  before it moves.
- **xgboost never shares a process with torch.** Both bring an OpenMP runtime and the process aborts
  when both start one; the search runs in a worker process that refuses to start if torch is loaded.

## package layout

Where the review web tool and each seam implementation live once the port cards land. `detect/` and
`tests/` exist today; everything under `review/` is created and filled by later port cards, moved out
of `snapshot/` a module at a time (a module moved out of `snapshot/` is deleted from `snapshot/` in
the same change).

    smolsmort/
      detect/                model backend seam, implementation #1: the heatmap CNN. ALREADY PRESENT.
        model.py            the net, its targets, its loss, the peak decode
        train.py            the training loop and `sweep` (the predict edge)
        dataset.py          judged decisions -> training Examples
        box.py scoring.py track.py   geometry, scoring, single-object tracking
        backend.py          the ModelBackend adapter

      boxes/                 model backend seam, implementation #2: the size-aware box CNN.
        model.py train.py    the net, its targets and decode; training and `sweep`
        synthetic.py         frames with known boxes, for proving it without real data
        backend.py           the ModelBackend adapter

      backends.py           backends by name: what the train tab offers, `register()` for more

      review/               the loop: the domain-free review web tool. NO domain facts live here.
        seams.py            the four Protocols (lifted verbatim from tests/test_loop.py by the
                            first port card; until then they live in the test, see below)
        paths.py            every directory the loop touches, named once; the mutable ones read
                            as attributes only (see the contract below)
        naming.py           tile/tag/index naming, carried as-is
        splits.py           train/val/test splits, carried as-is
        playback_state.py   scrub-a-recording state, carried as-is
        housekeeping.py     disk housekeeping, carried as-is
        train.py            the web tool's view of a training run: threads, abort, progress,
                            checkpoints. drives a ModelBackend; holds no model code itself
        state.py            the tool's own state - the generic half only. the domain half
                            (the fixed class list, the colour guesser) stays behind the seams
        routes.py           request handling - the generic half only
        server.py           the entry point. stdlib http.server, no web framework, no new dep
        review_ui/          app.js, index.html, tabs.css: find, select, train, load, settings,
                            housekeeping tabs (the domain-only tabs stay behind)

        classscheme/        class scheme seam
          __init__.py       ClassScheme + ClassGuesser homes (the generic half of classdefs)
          definitions.py    dimensions/members -> classes, versioned (from snapshot classdefs.py)

        sources/            example source seam implementations
          template.py       template matching (from snapshot vision/plate_templates.py)
          drawn.py          human-drawn boxes (from the find tab's save path)
          sweep.py          a trained ModelBackend, wrapped as a source (the return edge)

        render/             example renderer seam implementations
          crop.py           padded-crop + whole-frame rendering (from state + tools/playback)

      tabular/              model backend seam, implementation #3 (xgboost): a mechanics check that
                            keeps the seam visibly plural, not a forecasting tool

      forecast/             the regression and classification topics - OUTSIDE the four seams
        spec.py             what a prep spec says and the reserved columns every unit reads
        prep.py             spec + source file -> cached parquet, as one duckdb sql script
        splits.py           chronological train / val / test cuts, as-of relabelling
        profile.py          train-only stats: leaks, censoring, season, lead-lag, starting genes
        features.py         row features; series features per horizon bucket, read at the origin
        model.py evaluate.py  xgboost fits, conformal bands, the trust verdict
        pipeline.py search.py  scoring phases and the genetic search
        worker.py runs.py   the search in its own process; run folders
        tab.py views.py     the http routes and the chart / table / export views

    docs/
      REVIEW_TOOL_DESIGN.md   this document
    tests/
      test_loop.py            the find->judge->train->predict loop test against FAKE seams
      test_*.py               the existing detect tests (torch via importorskip)

### why the seams are authored in the test first

The lease for this card is `smolsmort/detect/**`, `docs/**` and `tests/**`; `smolsmort/review/` does
not exist yet and is created by the port cards. So the Protocols are authored where they can run and
be checked today - `tests/test_loop.py` - and the first port card lifts them verbatim into
`smolsmort/review/seams.py`. `test_loop.py` carries an `xfail` that flips to a pass the moment that
module exists, which is the signal the move is done.

## the attribute-only paths contract

**Mutable paths are referenced as attributes - `paths.SESSIONS_DIR`, never
`from ...paths import SESSIONS_DIR`.** This is not a style rule.

`SESSIONS_DIR` and `LABELS_DIR` are defaults the settings popup repoints at runtime, and that test
fixtures monkeypatch so a test writes into its own `tmp_path`. A module doing
`from ...paths import SESSIONS_DIR` binds a *copy* at import time: a runtime repoint or a test
monkeypatch changes the module's name while the importer goes on reading its own stale copy. Nothing
raises. The suite stays green while the code reads and writes the real `sessions/` and `training/`
directories - a data-loss bug wearing a passing test run. Reading `paths.SESSIONS_DIR` at call time
has no such gap, so it is the only allowed form.

**Test-enforceable.** `tests/test_loop.py::test_no_smolsmort_module_imports_mutable_paths_by_value`
walks every `.py` under `smolsmort/` and fails the build if any module imports a mutable path name by
value from a `review.paths` module. It passes trivially today (nothing under `smolsmort/` imports
those names yet) and stays green as `review/` is filled only if every reader uses attribute access -
which is exactly the guard `snapshot/tests/test_review_paths.py` runs on the pre-port copy. The
mutable set is `SESSIONS_DIR`, `LABELS_DIR`, `INTERFACE_SHOTS`; extend it in one place if another
default becomes runtime-repointable.

## provenance of defaults

**A default with a stated provenance is a parameter; one without is a magic number.** Every measured
number carried across a seam keeps the comment that says where it was measured, and it stays a
default a caller can override rather than a constant baked into the loop. The parent project's
numbers are the clearest cases:

- box match tolerances (`SAME_OBJECT_PX`, the overlap share/rows-apart) were measured on a long thin
  target ~132x12; a consumer detecting a different shape passes its own.
- the crop pads (`PAD_X_DEFAULT`, `PAD_Y_DEFAULT`) were chosen to sit near what the older
  tile-derived pad produced for a real object, so the change did not silently reframe every crop.
- the model's resolution and window numbers cite the reference object they were measured against.

The rule for the port: when a seam implementation carries a default in from the parent project, carry
its provenance comment with it, and expose it as an argument. The loop hardcodes none of them - it
either takes them from the active seam implementation or from the data (the training set's own fitted
box size, the class map derived from labels present). A number that arrives at the loop without a
provenance and without a way to override it is the smell this rule exists to catch.
