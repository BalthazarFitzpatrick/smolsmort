import json

import numpy as np
from PIL import Image

from smolsmort.crops.dataset import load_heads, load_set
from smolsmort.crops.train import evaluate, load, predict, save, train


def test_multi_head_crops_learn_colours_and_fill(tmp_path):
    rows = []
    for index in range(24):
        label = "red" if index % 2 else "blue"
        fill = 0.75 if label == "red" else 0.25
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[:, :, 0 if label == "red" else 2] = 255
        pixels[: round(fill * 32), :, 1] = 80
        path = tmp_path / f"{index}.png"
        Image.fromarray(pixels).save(path)
        rows.append(
            {
                "crop": path.name,
                "labels": {"identity": label, "fill": fill},
                "group": str(index // 6),
            }
        )
    set_path = tmp_path / "set.jsonl"
    set_path.write_text("\n".join(json.dumps(row) for row in rows))
    heads_path = tmp_path / "heads.json"
    heads_path.write_text(
        json.dumps({"identity": {"kind": "categorical"}, "fill": {"kind": "regression"}})
    )

    examples = load_set(set_path)
    heads = load_heads(heads_path, examples)
    model, _ = train(examples, heads, epochs=30, batch=8, learning_rate=0.01)
    metrics = evaluate(model, heads, examples)
    assert metrics["identity"] > 0.8
    assert metrics["fill"] < 0.15
    weights = save(model, heads, tmp_path / "crops.pt")
    restored, restored_heads = load(weights)
    assert predict(restored, restored_heads, examples[0].path)["identity"] == "blue"
