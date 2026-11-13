"""training-only imputation and one-hot encoding for tree families"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smolsmort.forecast.model import ModelError


@dataclass
class Preprocessor:
    columns: int
    numeric: np.ndarray
    medians: np.ndarray
    missing: np.ndarray
    categories: tuple[tuple[int, np.ndarray], ...]

    @classmethod
    def fit(cls, values: np.ndarray, types: list[str]) -> Preprocessor:
        if len(types) != values.shape[1] or any(kind not in {"c", "q"} for kind in types):
            raise ModelError("needs one c or q type marker per feature")
        numeric = np.array([i for i, kind in enumerate(types) if kind == "q"], dtype=int)
        # entirely missing columns use zero; they still carry a missing indicator
        medians = np.array(
            [
                np.median(col[~np.isnan(col)]) if (~np.isnan(col)).any() else 0.0
                for col in values[:, numeric].T
            ]
        )
        categories = tuple(
            (i, np.unique(values[~np.isnan(values[:, i]), i]))
            for i, kind in enumerate(types)
            if kind == "c"
        )
        missing = np.flatnonzero(np.isnan(values).any(axis=0))
        return cls(values.shape[1], numeric, medians, missing, categories)

    def transform(self, x) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.columns:
            raise ModelError("needs a matrix with the training column count")
        if np.isinf(values).any():
            raise ModelError("features must be finite or nan")
        numeric = values[:, self.numeric]
        blocks = [np.where(np.isnan(numeric), self.medians, numeric)]
        blocks.append(np.isnan(values[:, self.missing]).astype(float))
        for column, levels in self.categories:
            encoded = values[:, column, None] == levels
            unknown = ~encoded.any(axis=1, keepdims=True)
            blocks.extend((encoded.astype(float), unknown.astype(float)))
        return np.column_stack(blocks)
