from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from itg_transformers.config import LABELS, MAX_LENGTH, MODEL_NAME


class SentimentDataset(Dataset[dict[str, torch.Tensor]]):
    """Tweets and labels, tokenized one by one when an item is requested.

    Every item is padded to `max_length`, so the default DataLoader can
    stack items into a batch without a custom collate function.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_LENGTH,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"{len(texts)} texts but {len(labels)} labels")
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


def load_classifier(
    model_name: str = MODEL_NAME, labels: dict[int, str] = LABELS
) -> PreTrainedModel:
    """Pre-trained encoder with a new, randomly initialized class head.

    The label names are stored in the model config, so a saved model
    reports "negative" instead of "LABEL_0".
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=labels,
        label2id={name: label for label, name in labels.items()},
    )


def train_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One pass over the training data; returns the mean batch loss."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def predict(
    model: PreTrainedModel, dataloader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and true labels for every item of the dataloader."""
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].numpy())

    return np.array(predictions), np.array(true_labels)


def evaluate(
    model: PreTrainedModel, dataloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Accuracy and macro F1 of the model on the dataloader."""
    predictions, true_labels = predict(model, dataloader, device)
    accuracy = float(accuracy_score(true_labels, predictions))
    f1 = float(f1_score(true_labels, predictions, average="macro"))
    return accuracy, f1
