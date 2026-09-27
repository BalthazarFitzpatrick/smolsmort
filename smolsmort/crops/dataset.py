"""labelled fixed-size image crops for the multi-head classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class CropDatasetError(Exception):
    pass


@dataclass(frozen=True)
class Example:
    path: Path
    labels: dict[str, object]
    group: str


def load_set(path: Path) -> list[Example]:
    """read crop rows, resolving relative crop paths beside the jsonl file."""
    if not path.is_file():
        raise CropDatasetError(f"no crop set at {path}")
    examples = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            crop, labels, group = row["crop"], row["labels"], row["group"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise CropDatasetError(f"invalid row {line_number} in {path}") from error
        image = Path(crop)
        if not image.is_absolute():
            image = path.parent / image
        if not image.is_file():
            raise CropDatasetError(f"no crop at {image}")
        if not isinstance(labels, dict) or not isinstance(group, str):
            raise CropDatasetError(f"invalid labels or group on row {line_number} in {path}")
        examples.append(Example(image, labels, group))
    if not examples:
        raise CropDatasetError(f"{path} is empty")
    return examples


def load_heads(path: Path, examples: list[Example]) -> dict[str, dict]:
    """load a head spec and derive stable classes from labelled examples when needed."""
    if not path.is_file():
        raise CropDatasetError(f"no heads spec at {path}")
    try:
        heads = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise CropDatasetError(f"invalid heads spec at {path}") from error
    if not isinstance(heads, dict) or not heads:
        raise CropDatasetError(f"{path} has no heads")
    for name, spec in heads.items():
        if not isinstance(spec, dict) or spec.get("kind") not in {
            "categorical",
            "regression",
            "embedding",
        }:
            raise CropDatasetError(f"head {name} needs categorical, regression, or embedding kind")
        if spec["kind"] in {"categorical", "embedding"}:
            values = spec.get("classes") or sorted(
                {str(example.labels[name]) for example in examples if name in example.labels}
            )
            if not values:
                raise CropDatasetError(f"head {name} has no classes")
            spec["classes"] = list(values)
        elif any(
            not 0.0 <= float(example.labels[name]) <= 1.0
            for example in examples
            if name in example.labels
        ):
            raise CropDatasetError(f"regression head {name} labels must be in 0..1")
    return heads


def split_groups(
    examples: list[Example], fraction: float = 0.2
) -> tuple[list[Example], list[Example]]:
    """hold out complete groups, never random crops from a group."""
    groups = sorted({example.group for example in examples})
    if len(groups) < 2:
        return examples, []
    count = max(1, round(len(groups) * fraction))
    held_out = set(groups[-count:])
    return (
        [example for example in examples if example.group not in held_out],
        [example for example in examples if example.group in held_out],
    )
