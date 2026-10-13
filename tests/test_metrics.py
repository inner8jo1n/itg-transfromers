import pytest

from itg_transformers.metrics import classification_scores


def test_classification_scores_for_constant_prediction():
    scores = classification_scores([0, 1, 1, 2], [1, 1, 1, 1])

    assert scores["accuracy"] == 0.5
    assert scores["macro_f1"] == pytest.approx((2 / 3) / 3)
