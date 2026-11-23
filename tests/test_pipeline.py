"""Example unit tests for model_pipeline.py (run with: pytest)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_pipeline import prepare_data  # noqa: E402


def test_prepare_data():
    X_train, X_test, y_train, y_test = prepare_data("adult.data", "adult.test")
    # no missing values after cleaning
    assert X_train.isna().sum().sum() == 0
    # train and test have exactly the same columns
    assert list(X_train.columns) == list(X_test.columns)
    # the target is binary (0 / 1)
    assert set(y_train.unique()) == {0, 1}
    # dropped columns are really gone
    assert "fnlwgt" not in X_train.columns
