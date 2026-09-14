"""model backends by name - what the train tab picks from, and where an outside model plugs in.

LAZY ON PURPOSE. a backend is named by its module path and imported only when asked for, so listing
the names never imports torch, and a consumer that never picks a vision backend never pays for one.

A NEW MODEL IS ONE register() CALL. anything with the ModelBackend shape - train, predict, save,
load - can sit here beside the two built in, without an edit to smolsmort:

    backends.register("mine", "my_package.backend", "MyBackend")

THE SHARED GLUE LIVES HERE TOO, not in either backend: turning the loop's example dicts into
Examples, and the json beside a checkpoint that names who wrote it. Either backend can then be
removed without the other noticing.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from smolsmort.detect.dataset import Example

_REGISTRY: dict[str, tuple[str, str]] = {
    "heatmap": ("smolsmort.detect.backend", "HeatmapBackend"),
    "box": ("smolsmort.boxes.backend", "BoxBackend"),
    "xgboost": ("smolsmort.tabular.backend", "TabularBackend"),
}
# what a checkpoint with no sidecar is: every checkpoint predating named backends was a heatmap one
LEGACY = "heatmap"


class BackendError(Exception):
    pass


def names() -> list[str]:
    return sorted(_REGISTRY)


def register(name: str, module: str, attribute: str) -> None:
    """add or replace a backend by the module and class that implement it"""
    _REGISTRY[name] = (module, attribute)


def get_backend(name: str, **options):
    """a backend instance by name; options go to its constructor"""
    if name not in _REGISTRY:
        raise BackendError(f"no backend called {name!r} - known: {', '.join(names())}")
    module, attribute = _REGISTRY[name]
    return getattr(importlib.import_module(module), attribute)(**options)


def sidecar(path: Path) -> Path:
    """the json beside a checkpoint that names its backend - weights/a.pt -> weights/a.pt.json"""
    return path.with_name(path.name + ".json")


def backend_of(path: Path) -> str:
    """which backend wrote a checkpoint, from the json saved beside it"""
    meta = sidecar(path)
    if not meta.is_file():
        return LEGACY
    try:
        return json.loads(meta.read_text())["backend"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise BackendError(f"{meta} does not name a backend") from exc


def example_from(item) -> Example:
    """an Example as-is, or the loop's dict shape turned into one.

    per-object `sizes` win; a dict carrying only one width/height (the fixed-size shape) gives every
    object that size, which is honest for a set drawn at one size and wrong for any other
    """
    if isinstance(item, Example):
        return item
    centres = [tuple(c) for c in item.get("centres", [])]
    sizes = item.get("sizes") or [(item["width"], item["height"])] * len(centres)
    return Example(
        path=Path(item["path"]),
        centres=centres,
        labels=list(item.get("labels", [])),
        sizes=[tuple(s) for s in sizes],
        ignore=[tuple(r) for r in item.get("ignore", [])],
        negatives=[tuple(n) for n in item.get("negatives", [])],
        exhaustive=bool(item.get("exhaustive", False)),
    )
