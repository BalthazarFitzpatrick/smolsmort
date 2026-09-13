"""the loop test: find -> judge -> train -> predict, against FAKE seams.

This is the skeleton every later port card fills in. It authors the four plugin seams as
`typing.Protocol`s and drives one full turn of the loop through FAKE implementations of all of them,
so the SHAPES of the seams are pinned and test-enforceable before any real module has moved out of
`snapshot/`. See docs/REVIEW_TOOL_DESIGN.md for the responsibilities and boundaries of each seam.

NO TORCH, DELIBERATELY. the loop is model-agnostic - only a real model backend needs torch, and the
FAKE backend here trains and predicts with plain arithmetic. so this file must never import torch and
must never `importorskip` it; the real detect backend keeps that pattern in its own tests. a loop
test that needed a gpu would be testing the backend, not the loop.

The Protocols live in this test for now because `smolsmort/review/` does not exist yet; the first
port card lifts them verbatim into `smolsmort/review/seams.py`, at which point
`test_seams_have_moved_into_the_review_package` flips from xfail to pass.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

import pytest

# ---------------------------------------------------------------- the candidate schema


# the one dict every seam speaks in. drawn boxes, template matches and model sweeps all already emit
# exactly this, so a candidate from any source opens in the judge tab unchanged. `negative` is the
# only optional key - a place the model fired and was told no, sampled deliberately at train time.
class Candidate(TypedDict, total=False):
    path: str
    left: int
    top: int
    width: int
    height: int
    matched_template: str
    score: float
    negative: bool


# ---------------------------------------------------------------- the four seams


@runtime_checkable
class ExampleSource(Protocol):
    """seam 1: where candidates come from. proposes WHERE to look, never WHAT is there."""

    def find(self, recording: str) -> list[Candidate]: ...


@runtime_checkable
class ExampleRenderer(Protocol):
    """seam 2: turning a candidate into pixels a human can judge, plus a whole frame to draw on."""

    def frame(self, name: str) -> bytes: ...
    def crop(self, candidate: Candidate) -> bytes: ...


@runtime_checkable
class ClassScheme(Protocol):
    """seam 3a: the class vocabulary. `classes()` is sorted so it lines up with the model's channel
    map, which is built from the labels present, sorted."""

    def classes(self) -> list[str]: ...
    def label_for(self, picked: Mapping[str, str]) -> str: ...


@runtime_checkable
class ClassGuesser(Protocol):
    """seam 3b: optional. proposes a class to prefill the judge view; the human's pick always wins,
    and a scheme with no guesser simply offers no prefill."""

    def guess(self, candidate: Candidate) -> str | None: ...


@runtime_checkable
class ModelBackend(Protocol):
    """seam 4: learn from judged examples, then predict new candidates in the same schema `find`
    speaks - so a trained backend IS an example source, and the loop closes."""

    def train(
        self,
        examples: list[dict],
        *,
        classes: Mapping[str, int],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> object: ...
    def predict(
        self, weights: object, frames: list[str], *, classes: Mapping[str, int]
    ) -> list[Candidate]: ...
    def save(self, weights: object, path: Path) -> Path: ...
    def load(self, path: Path) -> object: ...


# ---------------------------------------------------------------- a judgement


@dataclass
class Judgement:
    """what a human decides about one candidate: keep or not, and (if kept) which class."""

    keep: bool
    label: str | None = None


# ---------------------------------------------------------------- FAKE seam implementations


@dataclass
class FakeSource:
    """seam 1, faked. hands back a fixed candidate list so the loop has something to judge without a
    real frame on disk. a port card replaces this with template matching / drawn boxes / a sweep."""

    candidates: list[Candidate]

    def find(self, recording: str) -> list[Candidate]:
        # the recording only tags the work here; a real source would read its frames
        return [{**c, "path": c.get("path", f"{recording}/0000.jpg")} for c in self.candidates]


@dataclass
class FakeRenderer:
    """seam 2, faked. returns deterministic placeholder bytes rather than a real image, so the loop
    can be exercised with no Pillow decode and no frame on disk."""

    def frame(self, name: str) -> bytes:
        return f"frame:{name}".encode()

    def crop(self, candidate: Candidate) -> bytes:
        return f"crop:{candidate['path']}:{candidate['left']},{candidate['top']}".encode()


@dataclass
class FakeScheme:
    """seam 3a, faked. a flat list of class names; label_for just echoes the single picked member,
    where a real scheme composes one member per dimension."""

    names: list[str]

    def classes(self) -> list[str]:
        return sorted(self.names)

    def label_for(self, picked: Mapping[str, str]) -> str:
        chosen = [v for v in picked.values() if v]
        if not chosen:
            raise ValueError("a label that answers nothing is not a class")
        return " / ".join(chosen)


@dataclass
class FakeGuesser:
    """seam 3b, faked. suggests a class by a candidate's score band, standing in for the colour
    guesser. never authoritative - the fake human below is free to override it."""

    scheme: FakeScheme

    def guess(self, candidate: Candidate) -> str | None:
        names = self.scheme.classes()
        if not names:
            return None
        band = min(int(candidate.get("score", 0.0) * len(names)), len(names) - 1)
        return names[band]


@dataclass
class FakeWeights:
    """what FakeBackend.train hands back and FakeBackend.predict reads: the class map it saw and how
    many examples it learned from. a stand-in for a state dict."""

    classes: dict[str, int]
    trained_on: int
    box: tuple[int, int]


@dataclass
class FakeBackend:
    """seam 4, faked. 'trains' by remembering the class map and example count, and 'predicts' one
    candidate per frame at the box size it was trained on, cycling through the known classes. no
    torch, no gpu, fully deterministic."""

    epochs_seen: int = 0

    def train(
        self,
        examples: list[dict],
        *,
        classes: Mapping[str, int],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> FakeWeights:
        usable = [e for e in examples if e.get("centres")]
        if not usable:
            raise ValueError("no example has a confirmed object - nothing to learn from")
        # a fixed size stands in for the training set's own fitted box; a real backend carries this
        # so predict needs only weights, frames and the class map
        box = (usable[0]["width"], usable[0]["height"])
        for epoch in range(1, 3):
            self.epochs_seen = epoch
            if on_progress:
                on_progress(epoch, 2)
        return FakeWeights(classes=dict(classes), trained_on=len(usable), box=box)

    def predict(
        self, weights: object, frames: list[str], *, classes: Mapping[str, int]
    ) -> list[Candidate]:
        assert isinstance(weights, FakeWeights)
        names = sorted(classes, key=classes.get) or ["object"]
        width, height = weights.box
        out: list[Candidate] = []
        for i, frame in enumerate(frames):
            out.append(
                {
                    "path": frame,
                    "left": 100 + i,
                    "top": 50,
                    "width": width,
                    "height": height,
                    "matched_template": names[i % len(names)],
                    "score": 0.9,
                }
            )
        return out

    def save(self, weights: object, path: Path) -> Path:
        assert isinstance(weights, FakeWeights)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(weights)))
        return path

    def load(self, path: Path) -> object:
        # plain json, never eval: a weights file is data, and a real backend's loader sets the pattern
        data = json.loads(path.read_text())
        return FakeWeights(
            classes=data["classes"], trained_on=data["trained_on"], box=tuple(data["box"])
        )


# ---------------------------------------------------------------- the loop, over the seams


@dataclass
class LoopResult:
    kept: list[dict]
    classes: dict[str, int]
    weights: object
    proposed: list[Candidate]


def judge(
    candidates: list[Candidate],
    renderer: ExampleRenderer,
    guesser: ClassGuesser | None,
    human: Callable[[Candidate, str | None], Judgement],
) -> list[Judgement]:
    """the human step. render each candidate, offer a guess, let the human decide. the guess only
    ever prefills; `human` is free to overrule it, which is what keeps the guesser non-authoritative.
    """
    judgements = []
    for candidate in candidates:
        renderer.crop(candidate)  # the human looks at this
        suggestion = guesser.guess(candidate) if guesser is not None else None
        judgements.append(human(candidate, suggestion))
    return judgements


def _examples_from(candidates: list[Candidate], judgements: list[Judgement]) -> list[dict]:
    """kept candidates grouped into per-frame examples, the shape a backend trains on. a discard
    becomes neither a positive nor a negative here - it is simply dropped, matching how the real
    dataset builder turns an un-kept box into an ignore region rather than background."""
    by_frame: dict[str, dict] = {}
    for candidate, decision in zip(candidates, judgements, strict=True):
        if not decision.keep:
            continue
        example = by_frame.setdefault(
            candidate["path"],
            {
                "path": candidate["path"],
                "centres": [],
                "labels": [],
                "sizes": [],
                "width": 0,
                "height": 0,
            },
        )
        example["centres"].append(
            (candidate["left"] + candidate["width"] / 2, candidate["top"] + candidate["height"] / 2)
        )
        example["labels"].append(decision.label)
        # each object's own size, parallel to centres - a size-aware backend learns it, and a
        # fixed-size one reads the single width/height below instead
        example["sizes"].append((candidate["width"], candidate["height"]))
        example["width"], example["height"] = candidate["width"], candidate["height"]
    return list(by_frame.values())


def run_loop(
    *,
    source: ExampleSource,
    renderer: ExampleRenderer,
    scheme: ClassScheme,
    guesser: ClassGuesser | None,
    backend: ModelBackend,
    recording: str,
    frames: list[str],
    human: Callable[[Candidate, str | None], Judgement],
) -> LoopResult:
    """one full turn: find -> judge -> train -> predict. the predicted candidates are the loop's
    return edge - they are in the same schema `find` produced, so they re-enter judging unchanged."""
    found = source.find(recording)
    judgements = judge(found, renderer, guesser, human)
    kept = _examples_from(found, judgements)
    # the channel map is derived from the labels actually kept, sorted - the same rule the real
    # dataset builder and class scheme both follow, so the two line up by construction
    labels = sorted({label for e in kept for label in e["labels"] if label})
    classes = {label: index for index, label in enumerate(labels)}
    weights = backend.train(kept, classes=classes)
    proposed = backend.predict(weights, frames, classes=classes)
    return LoopResult(kept=kept, classes=classes, weights=weights, proposed=proposed)


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def seams():
    scheme = FakeScheme(names=["hostile", "friendly", "neutral"])
    candidates: list[Candidate] = [
        {"left": 10, "top": 10, "width": 40, "height": 12, "matched_template": "x", "score": 0.2},
        {"left": 80, "top": 60, "width": 40, "height": 12, "matched_template": "x", "score": 0.95},
        {"left": 150, "top": 90, "width": 40, "height": 12, "matched_template": "x", "score": 0.5},
    ]
    return {
        "source": FakeSource(candidates=candidates),
        "renderer": FakeRenderer(),
        "scheme": scheme,
        "guesser": FakeGuesser(scheme=scheme),
        "backend": FakeBackend(),
    }


# ---------------------------------------------------------------- the seams are what the doc says


def test_every_fake_satisfies_its_seam_protocol(seams):
    """the shapes are pinned: a fake that drifts from its protocol fails here, and a real port that
    conforms passes the same check for free (runtime_checkable protocols)."""
    assert isinstance(seams["source"], ExampleSource)
    assert isinstance(seams["renderer"], ExampleRenderer)
    assert isinstance(seams["scheme"], ClassScheme)
    assert isinstance(seams["guesser"], ClassGuesser)
    assert isinstance(seams["backend"], ModelBackend)


def test_the_guesser_is_optional(seams):
    """a scheme with no guesser must still drive the loop - the guess only ever prefills."""
    found = seams["source"].find("rec")
    judgements = judge(found, seams["renderer"], None, lambda c, g: Judgement(keep=True, label="x"))
    assert all(j.keep for j in judgements)


# ---------------------------------------------------------------- the loop closes


def test_find_judge_train_predict_closes_the_loop(seams):
    """one full turn against fakes. keep everything at or above a threshold, take the guess as the
    label, train, and sweep three unjudged frames. the proposals come back in the candidate schema,
    so they are ready to re-enter judging - which is the whole point of the loop."""

    def human(candidate: Candidate, suggestion: str | None) -> Judgement:
        # a real human keeps the good ones and takes the guess unless they disagree; here: keep at
        # or above 0.5, and trust the guess
        keep = candidate.get("score", 0.0) >= 0.5
        return Judgement(keep=keep, label=suggestion if keep else None)

    result = run_loop(
        **seams,
        recording="rec-a",
        frames=["rec-b/0.jpg", "rec-b/1.jpg", "rec-b/2.jpg"],
        human=human,
    )

    assert result.kept, "at least one candidate should have been kept to train on"
    assert result.classes, "kept labels should derive a non-empty class map"
    # the return edge: one proposal per unjudged frame, each a well-formed candidate
    assert len(result.proposed) == 3
    for proposal in result.proposed:
        assert set(proposal) >= {"path", "left", "top", "width", "height", "score"}
        assert proposal["matched_template"] in result.classes

    # and the proposals really do re-enter judging unchanged - the loop is a loop
    second_pass = judge(
        result.proposed,
        seams["renderer"],
        seams["guesser"],
        lambda c, g: Judgement(keep=True, label=g),
    )
    assert len(second_pass) == len(result.proposed)


def test_training_on_nothing_kept_refuses(seams):
    """discard everything and there is nothing to learn from - the backend must say so rather than
    train an empty model, matching detect's own 'no example has a confirmed object' guard."""
    with pytest.raises(ValueError, match="nothing to learn from"):
        run_loop(
            **seams,
            recording="rec-a",
            frames=["f.jpg"],
            human=lambda c, g: Judgement(keep=False),
        )


def test_a_weights_handle_round_trips(seams, tmp_path):
    """save then load must return equivalent weights - a checkpoint the next sweep can read."""
    backend = seams["backend"]
    weights = backend.train(
        [{"path": "f.jpg", "centres": [(1, 2)], "labels": ["hostile"], "width": 40, "height": 12}],
        classes={"hostile": 0},
    )
    path = backend.save(weights, tmp_path / "w.txt")
    assert backend.load(path) == weights


# --------------------------------------------- the attribute-only paths contract, test-enforceable

# repointed at runtime by the settings popup and monkeypatched by tests; a by-value copy goes stale.
# extend this set in one place if another default becomes runtime-repointable.
MUTABLE_PATHS = {"SESSIONS_DIR", "LABELS_DIR", "INTERFACE_SHOTS"}

_SMOLSMORT = Path(__file__).resolve().parents[1] / "smolsmort"


def test_no_smolsmort_module_imports_mutable_paths_by_value():
    """THE DATA-LOSS BUG THIS PREVENTS. `from ...review.paths import SESSIONS_DIR` binds a copy at
    import time; a runtime repoint or a test monkeypatch then changes the module's name while the
    importer reads its own stale one, and nothing raises - the suite passes while reading and writing
    the real sessions/ directory. reading `paths.SESSIONS_DIR` at call time has no such gap.

    passes trivially today (nothing under smolsmort/ imports these yet) and stays green as review/ is
    filled only if every reader uses attribute access - the same guard snapshot runs on the pre-port
    copy.
    """
    offenders = []
    for path in _SMOLSMORT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.endswith("review.paths"):
                continue
            taken = {alias.name for alias in node.names} & MUTABLE_PATHS
            if taken:
                offenders.append(
                    f"{path.relative_to(_SMOLSMORT)}:{node.lineno} imports {sorted(taken)}"
                )
    assert not offenders, (
        "these bind a copy a runtime repoint and every test monkeypatch will miss - "
        "read them as paths.<NAME> instead:\n  " + "\n  ".join(offenders)
    )


def test_attribute_access_sees_a_repoint_but_a_by_value_copy_does_not():
    """the contract's MEANING, demonstrated against a stand-in module so it is enforceable before
    review/paths.py exists. a caller reading through the module sees a repoint; one that bound a copy
    does not - which is exactly the gap that reads and writes the real directory in a green suite."""
    from types import SimpleNamespace

    paths = SimpleNamespace(SESSIONS_DIR=Path("/original"))

    def reads_through_module() -> Path:
        return paths.SESSIONS_DIR  # call-time attribute access - the only allowed form

    bound_copy = paths.SESSIONS_DIR  # the forbidden form: a value captured at import time

    paths.SESSIONS_DIR = Path("/repointed")  # what set_bases or a monkeypatch does

    assert reads_through_module() == Path("/repointed")
    assert bound_copy == Path("/original"), "a by-value copy silently keeps pointing at the old dir"


# --------------------------------------------- the port target, signalled by an xfail that flips


@pytest.mark.xfail(
    reason="smolsmort/review/seams.py is authored by the first port card; until then the seams live "
    "in this test. this flips to a pass (xpass) when that module lands.",
    strict=False,
)
def test_seams_have_moved_into_the_review_package():
    import importlib

    seams = importlib.import_module("smolsmort.review.seams")
    for name in ("ExampleSource", "ExampleRenderer", "ClassScheme", "ClassGuesser", "ModelBackend"):
        assert hasattr(seams, name), f"the port must expose {name} from smolsmort.review.seams"
