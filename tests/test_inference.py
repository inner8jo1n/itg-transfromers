import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from itg_transformers.inference import (
    FrozenEncoderVectorizer,
    predict_baseline,
    predict_fine_tuned,
)

TEXTS = ["hello", "hello world world", "world", "world hello"]


def test_predict_fine_tuned_accepts_a_single_string(
    word_tokenizer, tiny_classifier
):
    [result] = predict_fine_tuned("hello", tiny_classifier, word_tokenizer)

    assert result["text"] == "hello"
    assert result["prediction"] in {0, 1, 2}
    probabilities = np.asarray(result["probabilities"])
    assert probabilities.shape == (3,)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-6)
    assert result["prediction"] == probabilities.argmax()


def test_predict_fine_tuned_does_not_depend_on_batching(
    word_tokenizer, tiny_classifier
):
    together = predict_fine_tuned(
        TEXTS, tiny_classifier, word_tokenizer, batch_size=4
    )
    one_by_one = predict_fine_tuned(
        TEXTS, tiny_classifier, word_tokenizer, batch_size=1
    )

    assert [p["text"] for p in together] == TEXTS
    for first, second in zip(together, one_by_one, strict=True):
        assert first["prediction"] == second["prediction"]
        np.testing.assert_allclose(
            np.asarray(first["probabilities"]),
            np.asarray(second["probabilities"]),
            atol=1e-5,
        )


def test_predict_fine_tuned_cleans_only_the_model_input(
    word_tokenizer, tiny_classifier
):
    [cleaned] = predict_fine_tuned(
        "HELLO", tiny_classifier, word_tokenizer, clean_func=str.lower
    )
    [plain] = predict_fine_tuned("hello", tiny_classifier, word_tokenizer)

    assert cleaned["text"] == "HELLO"
    np.testing.assert_allclose(
        np.asarray(cleaned["probabilities"]),
        np.asarray(plain["probabilities"]),
    )


class LengthVectorizer:
    """Toy vectorizer: one feature, the number of words."""

    def transform(self, texts):
        return np.array([[len(text.split())] for text in texts], dtype=float)


def make_length_classifier():
    X = np.array([[1.0], [1.0], [3.0], [3.0]])
    return LogisticRegression().fit(X, [0, 0, 2, 2])


def test_frozen_encoder_vectorizer_returns_cls_embeddings(
    word_tokenizer, tiny_model
):
    vectorizer = FrozenEncoderVectorizer(word_tokenizer, tiny_model)

    features = vectorizer.transform(TEXTS)

    assert features.shape == (4, 8)


def test_predict_baseline_predicts_with_probabilities():
    results = predict_baseline(
        ["hello", "a b c"], make_length_classifier(), LengthVectorizer()
    )

    assert [r["prediction"] for r in results] == [0, 2]
    for result in results:
        assert result["probabilities"].sum() == pytest.approx(1.0)


def test_predict_baseline_cleans_only_the_model_input():
    def remove_dashes(text):
        return text.replace("-", " ")

    [result] = predict_baseline(
        "a-b-c",
        make_length_classifier(),
        LengthVectorizer(),
        clean_func=remove_dashes,
    )

    assert result["text"] == "a-b-c"
    assert result["prediction"] == 2
