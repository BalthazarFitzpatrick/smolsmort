"""train, evaluate, save and load multi-head fixed-size crop classifiers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from smolsmort.crops.dataset import Example
from smolsmort.crops.model import INPUT_SIZE, build_model
from smolsmort.detect.model import _torch
from smolsmort.optim import build_optimizer


def image_input(image: Path | np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """an rgb crop as channels-first float32, resized to the fixed classifier input."""
    from PIL import Image

    if isinstance(image, Path):
        with Image.open(image) as handle:
            picture = handle.convert("RGB").resize((size, size), Image.BILINEAR)
    else:
        picture = (
            Image.fromarray(np.asarray(image)).convert("RGB").resize((size, size), Image.BILINEAR)
        )
    return np.asarray(picture, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _device(device):
    torch = _torch()
    return device or ("mps" if torch.backends.mps.is_available() else "cpu")


def train(
    examples: list[Example],
    heads: dict[str, dict],
    *,
    epochs: int = 20,
    batch: int = 16,
    learning_rate: float = 2e-3,
    device: str | None = None,
):
    """train on labelled crops and return the model plus per-epoch mean loss."""
    torch = _torch()
    device = _device(device)
    model = build_model(heads).to(device)
    optimiser = build_optimizer(
        model.parameters(),
        optimizer="adamw",
        learning_rate=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
    )
    categorical = {
        name: {label: index for index, label in enumerate(spec["classes"])}
        for name, spec in heads.items()
        if spec["kind"] == "categorical"
    }
    cache = {example.path: image_input(example.path) for example in examples}
    history = []
    for _ in range(epochs):
        total, count = 0.0, 0
        for start in range(0, len(examples), batch):
            rows = examples[start : start + batch]
            x = torch.from_numpy(np.stack([cache[row.path] for row in rows])).to(device)
            outputs = model(x)
            losses = []
            for name, spec in heads.items():
                present = [index for index, row in enumerate(rows) if name in row.labels]
                if not present:
                    continue
                output = outputs[name][present]
                values = [rows[index].labels[name] for index in present]
                if spec["kind"] == "categorical":
                    target = torch.tensor(
                        [categorical[name][str(value)] for value in values], device=device
                    )
                    losses.append(torch.nn.functional.cross_entropy(output, target))
                elif spec["kind"] == "regression":
                    target = torch.tensor(values, dtype=torch.float32, device=device)[:, None]
                    losses.append(torch.nn.functional.mse_loss(torch.sigmoid(output), target))
                else:
                    # embedding labels train compact class prototypes; inference can instead use art refs
                    target = torch.tensor(
                        [spec["classes"].index(str(value)) for value in values], device=device
                    )
                    prototypes = torch.eye(len(spec["classes"]), output.shape[1], device=device)
                    losses.append(torch.nn.functional.cross_entropy(output @ prototypes.T, target))
            if not losses:
                continue
            loss = sum(losses)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach().cpu())
            count += 1
        history.append(total / count if count else 0.0)
    return model, history


def predict(
    model, heads: dict[str, dict], image: Path | np.ndarray, *, references=None
) -> dict[str, object]:
    """return per-head values; embedding heads use nearest named reference image when supplied."""
    torch = _torch()
    device = next(model.parameters()).device
    x = torch.from_numpy(image_input(image)[None]).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    result = {}
    for name, spec in heads.items():
        output = outputs[name][0]
        if spec["kind"] == "categorical":
            result[name] = spec["classes"][int(output.argmax())]
        elif spec["kind"] == "regression":
            result[name] = float(torch.sigmoid(output).item())
        elif references and name in references:
            refs = references[name]
            labels, images = zip(*refs.items(), strict=True)
            ref_x = torch.from_numpy(np.stack([image_input(Path(value)) for value in images])).to(
                device
            )
            with torch.no_grad():
                ref_embeddings = model(ref_x)[name]
            scores = torch.nn.functional.cosine_similarity(output[None], ref_embeddings)
            result[name] = labels[int(scores.argmax())]
        else:
            result[name] = spec["classes"][int(output[: len(spec["classes"])].argmax())]
    return result


def evaluate(model, heads: dict[str, dict], examples: list[Example]) -> dict[str, float]:
    """per-head categorical accuracy and regression mae over labelled rows."""
    sums = {name: [0.0, 0] for name in heads}
    for example in examples:
        prediction = predict(model, heads, example.path)
        for name, value in example.labels.items():
            if heads[name]["kind"] == "regression":
                sums[name][0] += abs(float(prediction[name]) - float(value))
            elif heads[name]["kind"] in {"categorical", "embedding"}:
                sums[name][0] += prediction[name] == str(value)
            else:
                continue
            sums[name][1] += 1
    return {name: total / count for name, (total, count) in sums.items() if count}


def save(model, heads: dict[str, dict], path: Path) -> Path:
    """write weights and complete inference metadata atomically."""
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save({"state": model.state_dict(), "heads": heads}, temp)
    os.replace(temp, path)
    return path


def load(path: Path, device: str | None = None):
    """load standalone inference weights, returning `(model, heads)`."""
    torch = _torch()
    checkpoint = torch.load(path, map_location=_device(device))
    heads = checkpoint["heads"]
    model = build_model(heads).to(_device(device))
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, heads
