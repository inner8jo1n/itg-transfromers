import numpy as np
import pytest

from itg_transformers import app
from itg_transformers.app import format_prediction, predict_sentiment


def test_format_prediction_lists_the_label_and_all_probabilities():
    text = format_prediction(
        {
            "text": "great",
            "prediction": 2,
            "probabilities": np.array([0.05, 0.15, 0.8]),
        }
    )

    assert text.splitlines() == [
        "Prediction: Positive",
        "",
        "Probabilities:",
        "Negative: 5.00%",
        "Neutral: 15.00%",
        "Positive: 80.00%",
    ]


def test_predict_sentiment_asks_for_text_without_loading_the_model(
    monkeypatch,
):
    def fail():
        raise AssertionError("the model must not be loaded")

    monkeypatch.setattr(app, "load_fine_tuned", fail)

    assert predict_sentiment("   ") == "Enter a text to analyse."


def test_predict_sentiment_uses_the_fine_tuned_model(
    monkeypatch, word_tokenizer, tiny_classifier
):
    monkeypatch.setattr(
        app, "load_fine_tuned", lambda: (tiny_classifier, word_tokenizer)
    )

    result = predict_sentiment("hello world")

    assert result.startswith("Prediction: ")
    percents = [
        float(line.split(": ")[1].rstrip("%"))
        for line in result.splitlines()[3:]
    ]
    assert sum(percents) == pytest.approx(100, abs=0.05)
