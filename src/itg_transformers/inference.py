from collections.abc import Callable, Sequence
from typing import Protocol, TypedDict

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from itg_transformers.config import MAX_LENGTH
from itg_transformers.embeddings import get_embeddings
from itg_transformers.tokenization import tokenize_texts


class Prediction(TypedDict):
    text: str
    prediction: int
    probabilities: np.ndarray


def predict_fine_tuned(
    texts: str | Sequence[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    clean_func: Callable[[str], str] | None = None,
    batch_size: int = 64,
    max_length: int = MAX_LENGTH,
) -> list[Prediction]:
    """Class and class probabilities of the fine-tuned model per text.

    `text` in the result is the text as given; `clean_func` is applied
    only to what the model sees.
    """
    if isinstance(texts, str):
        texts = [texts]
    inputs = [clean_func(text) for text in texts] if clean_func else texts

    model.eval()
    predictions: list[Prediction] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenize_texts(
            inputs[start : start + batch_size], tokenizer, max_length
        )
        with torch.no_grad():
            outputs = model(**batch.to(model.device))
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        for offset, row in enumerate(probs):
            predictions.append(
                {
                    "text": texts[start + offset],
                    "prediction": int(row.argmax()),
                    "probabilities": row,
                }
            )
    return predictions


class Vectorizer(Protocol):
    def transform(self, texts: Sequence[str]) -> np.ndarray: ...


class Classifier(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class FrozenEncoderVectorizer:
    """Texts to CLS embeddings of a frozen encoder, like a vectorizer.

    Plays the role a TF-IDF vectorizer plays in a classic baseline: it
    turns texts into the feature matrix the classifier was trained on.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 32,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return get_embeddings(
            texts, self.tokenizer, self.model, batch_size=self.batch_size
        )


def predict_baseline(
    texts: str | Sequence[str],
    model: Classifier,
    vectorizer: Vectorizer,
    clean_func: Callable[[str], str] | None = None,
) -> list[Prediction]:
    """Class and class probabilities of the baseline classifier per text."""
    if isinstance(texts, str):
        texts = [texts]
    inputs = [clean_func(text) for text in texts] if clean_func else texts

    X = vectorizer.transform(inputs)
    predictions = model.predict(X)
    probs = model.predict_proba(X)

    return [
        {
            "text": text,
            "prediction": int(predictions[index]),
            "probabilities": probs[index],
        }
        for index, text in enumerate(texts)
    ]
