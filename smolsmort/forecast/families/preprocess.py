"""training-only imputation, optional scaling and one-hot encoding, shared by all families"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from smolsmort.forecast.model import ModelError


@dataclass
class Preprocessor:
    columns: int
    numeric: np.ndarray
    medians: np.ndarray
    missing: np.ndarray
    categories: tuple[tuple[int, np.ndarray], ...]
    # scales are set only when fitted with scale=True; dtype none keeps float64
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    dtype: type | None = None
    # error prefix, so a family keeps its own message wording
    label: str = field(default="", repr=False)

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        types: list[str],
        *,
        scale: bool = False,
        dtype: type | None = None,
        label: str = "",
    ) -> Preprocessor:
        prefix = f"{label} " if label else ""
        if len(types) != values.shape[1] or any(kind not in {"c", "q"} for kind in types):
            raise ModelError(f"{prefix}needs one c or q type marker per feature")
        numeric = np.array([i for i, kind in enumerate(types) if kind == "q"], dtype=int)
        raw = values[:, numeric]
        # entirely missing columns use zero; they still carry a missing indicator
        medians = np.array(
            [np.median(col[~np.isnan(col)]) if (~np.isnan(col)).any() else 0.0 for col in raw.T]
        )
        means = scales = None
        if scale:
            imputed = np.where(np.isnan(raw), medians, raw)
            means, scales = imputed.mean(axis=0), imputed.std(axis=0)
            scales[scales == 0] = 1.0
        categories = tuple(
            (i, np.unique(values[~np.isnan(values[:, i]), i]))
            for i, kind in enumerate(types)
            if kind == "c"
        )
        missing = np.flatnonzero(np.isnan(values).any(axis=0))
        return cls(
            values.shape[1], numeric, medians, missing, categories, means, scales, dtype, label
        )

    def transform(self, x) -> np.ndarray:
        prefix = f"{self.label} " if self.label else ""
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.columns:
            raise ModelError(f"{prefix}needs a matrix with the training column count")
        if np.isinf(values).any():
            raise ModelError(f"{prefix}features must be finite or nan")
        numeric = values[:, self.numeric]
        block = np.where(np.isnan(numeric), self.medians, numeric)
        if self.scales is not None:
            block = (block - self.means) / self.scales
        blocks = [block, np.isnan(values[:, self.missing]).astype(float)]
        for column, levels in self.categories:
            encoded = values[:, column, None] == levels
            unknown = ~encoded.any(axis=1, keepdims=True)
            blocks.extend((encoded.astype(float), unknown.astype(float)))
        design = np.column_stack(blocks)
        return design if self.dtype is None else design.astype(self.dtype)
