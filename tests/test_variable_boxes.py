"""the whole loop with the box backend swapped in: find, judge, train, sweep - on synthetic frames.

THE SWAP IS THE POINT. run_loop is the loop test's own driver, untouched; only the backend handed to
it changes. If this passes, a model is something the loop is given, not something it is built
around. Frames are two resolutions with objects 24-160 px long and none overlapping, so what is
measured is size and resolution, the two things this backend exists for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_loop import FakeRenderer, FakeScheme, Judgement, run_loop

from smolsmort import backends
from smolsmort.boxes.synthetic import write_recording
from smolsmort.detect.box import Box

# none of the imports above need torch; training does
pytest.importorskip("torch")

# class index -> name, in synthetic.py's pattern order
NAMES = ["plain", "banded", "striped"]
EPOCHS = 40


class DrawnSource:
    """the cold-start source: boxes a human drew on whole frames, here the generator's own truth"""

    def __init__(self, frames):
        self.frames = frames

    def find(self, recording):
        return [
            {
                "path": str(frame["path"]),
                "left": round(x0),
                "top": round(y0),
                "width": round(x1 - x0),
                "height": round(y1 - y0),
                "matched_template": NAMES[cls],
                "score": 1.0,
            }
            for frame in self.frames
            for (x0, y0, x1, y1), cls in zip(frame["boxes"], frame["classes"], strict=True)
        ]


@pytest.fixture(scope="module")
def turn(tmp_path_factory):
    frames = write_recording(tmp_path_factory.mktemp("recording"), 30, seed=1, overlap_rate=0.0)
    labelled, unseen = frames[:20], frames[20:]
    result = run_loop(
        source=DrawnSource(labelled),
        renderer=FakeRenderer(),
        scheme=FakeScheme(names=NAMES),
        guesser=None,
        backend=backends.get_backend("box", epochs=EPOCHS, device="cpu"),
        recording="synthetic",
        frames=[str(frame["path"]) for frame in unseen],
        # the human keeps every drawn box and names it - the judging a real reviewer does
        human=lambda candidate, _: Judgement(keep=True, label=candidate["matched_template"]),
    )
    return result, unseen


def _found(result, frame) -> list[Box]:
    name = Path(frame["path"]).name
    return [
        Box(left=c["left"], top=c["top"], width=c["width"], height=c["height"])
        for c in result.proposed
        if c["path"] == name
    ]


def _recall(result, frames) -> float:
    hits = total = 0
    for frame in frames:
        found = _found(result, frame)
        for x0, y0, x1, y1 in frame["boxes"]:
            truth = Box(left=round(x0), top=round(y0), width=round(x1 - x0), height=round(y1 - y0))
            hits += any(box.iou(truth) >= 0.5 for box in found)
            total += 1
    return hits / total


def test_the_proposals_speak_the_candidate_schema(turn):
    result, _ = turn
    assert result.proposed
    for candidate in result.proposed:
        assert {"path", "left", "top", "width", "height", "matched_template", "score"} <= set(
            candidate
        )
        assert candidate["matched_template"] in NAMES


def test_unseen_objects_are_found_at_their_own_size(turn):
    """a floor that says it learned, not a quality bar - random boxes score near 0 at iou 0.5.
    measured 0.52 after 40 cpu epochs (~37 s); the u0 spike reached 0.97 on standing-apart objects
    with 8 minutes of training, so a caller wanting more trains longer"""
    result, unseen = turn
    assert _recall(result, unseen) >= 0.4


def test_both_resolutions_are_found(turn):
    result, unseen = turn
    for width in (640, 1280):
        at_size = [f for f in unseen if _width_of(f) == width]
        assert at_size and _recall(result, at_size) >= 0.5, width


def _width_of(frame) -> int:
    from PIL import Image

    with Image.open(frame["path"]) as handle:
        return handle.width


def test_predicted_sizes_span_the_range(turn):
    """one fixed size is what the heatmap backend gives; this one must not"""
    result, _ = turn
    widths = [c["width"] for c in result.proposed if c["score"] >= 0.5]
    assert max(widths) / min(widths) > 3


def test_weights_round_trip_and_name_their_backend(turn, tmp_path):
    result, _ = turn
    backend = backends.get_backend("box", device="cpu")
    path = backend.save(result.weights, tmp_path / "weights" / "boxes.pt")
    assert backends.backend_of(path) == "box"
    assert backend.load(path).classes == result.classes
