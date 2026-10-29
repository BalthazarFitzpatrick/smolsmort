"""the compact cnn used by crops.train."""

from __future__ import annotations

from smolsmort.detect.model import _torch

INPUT_SIZE = 32
EMBEDDING_SIZE = 16


def build_model(heads: dict[str, dict], width: int = 16):
    """build a small shared encoder with one output layer per named head."""
    torch = _torch()
    nn = torch.nn

    class CropModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, width, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(width, width * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(width * 2, width * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.heads = nn.ModuleDict(
                {
                    name: nn.Linear(
                        width * 2,
                        1
                        if spec["kind"] == "regression"
                        else max(EMBEDDING_SIZE, len(spec["classes"]))
                        if spec["kind"] == "embedding"
                        else len(spec["classes"]),
                    )
                    for name, spec in heads.items()
                }
            )

        def forward(self, image):
            features = self.encoder(image).flatten(1)
            return {name: layer(features) for name, layer in self.heads.items()}

    return CropModel()
