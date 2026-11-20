"""training-only imputation and one-hot encoding for tree families"""

import numpy as np
import pytest

from smolsmort.forecast.families.preprocess import Preprocessor
from smolsmort.forecast.model import ModelError

NAN = np.nan


def test_numeric_nan_takes_the_training_median_and_adds_an_indicator():
    train = np.array([[1.0], [3.0], [NAN], [5.0]])
    prep = Preprocessor.fit(train, ["q"])
    out = prep.transform(np.array([[NAN], [2.0]]))
    # median of 1, 3, 5 is 3; one missing indicator column follows the numeric block
    np.testing.assert_array_equal(out, [[3.0, 1.0], [2.0, 0.0]])


def test_a_column_with_no_training_nan_gets_no_indicator():
    prep = Preprocessor.fit(np.array([[1.0], [2.0]]), ["q"])
    out = prep.transform(np.array([[NAN], [4.0]]))
    np.testing.assert_array_equal(out, [[1.5], [4.0]])


def test_an_all_missing_numeric_column_imputes_zero():
    prep = Preprocessor.fit(np.array([[NAN], [NAN]]), ["q"])
    out = prep.transform(np.array([[7.0], [NAN]]))
    np.testing.assert_array_equal(out, [[7.0, 0.0], [0.0, 1.0]])


def test_categories_one_hot_with_unknown_and_missing_flags():
    train = np.array([[1.0], [2.0], [NAN]])
    prep = Preprocessor.fit(train, ["c"])
    out = prep.transform(np.array([[1.0], [2.0], [9.0], [NAN]]))
    # columns: missing indicator, level 1, level 2, unknown level
    np.testing.assert_array_equal(
        out,
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1]],
    )


def test_numeric_and_category_blocks_keep_a_stable_order():
    train = np.array([[1.0, 5.0], [3.0, 6.0]])
    prep = Preprocessor.fit(train, ["q", "c"])
    out = prep.transform(np.array([[2.0, 6.0]]))
    # numeric first, then the category level columns and the unknown flag
    np.testing.assert_array_equal(out, [[2.0, 0.0, 1.0, 0.0]])


def test_fit_rejects_bad_type_markers():
    values = np.zeros((2, 2))
    with pytest.raises(ModelError, match="c or q type marker"):
        Preprocessor.fit(values, ["q"])
    with pytest.raises(ModelError, match="c or q type marker"):
        Preprocessor.fit(values, ["q", "z"])


def test_transform_rejects_wrong_shape_and_infinite_values():
    prep = Preprocessor.fit(np.array([[1.0, 2.0]]), ["q", "q"])
    with pytest.raises(ModelError, match="training column count"):
        prep.transform(np.zeros((1, 3)))
    with pytest.raises(ModelError, match="training column count"):
        prep.transform(np.zeros(2))
    with pytest.raises(ModelError, match="finite or nan"):
        prep.transform(np.array([[np.inf, 1.0]]))
