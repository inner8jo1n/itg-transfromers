import pytest
import torch
from torch.utils.data import DataLoader

from itg_transformers.finetuning import (
    SentimentDataset,
    evaluate,
    predict,
    train_epoch,
)


def test_sentiment_dataset_item_is_padded_to_max_length(word_tokenizer):
    dataset = SentimentDataset(
        ["hello", "hello world"], [2, 0], word_tokenizer, max_length=4
    )

    item = dataset[0]

    assert len(dataset) == 2
    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].tolist() == [2, 0, 0, 0]
    assert item["attention_mask"].tolist() == [1, 0, 0, 0]
    assert item["labels"].dtype == torch.long
    assert item["labels"].item() == 2


def test_sentiment_dataset_truncates_long_texts(word_tokenizer):
    dataset = SentimentDataset(
        ["hello world hello world"], [1], word_tokenizer, max_length=2
    )

    assert dataset[0]["input_ids"].tolist() == [2, 3]


def test_sentiment_dataset_rejects_mismatched_lengths(word_tokenizer):
    with pytest.raises(ValueError, match="2 texts but 1 labels"):
        SentimentDataset(["hello", "world"], [1], word_tokenizer)


def make_loader(word_tokenizer, batch_size=2):
    texts = ["hello", "world", "hello hello", "world world"]
    labels = [0, 2, 0, 2]
    dataset = SentimentDataset(texts, labels, word_tokenizer, max_length=3)
    return DataLoader(dataset, batch_size=batch_size)


def test_train_epoch_lowers_the_loss(word_tokenizer, tiny_classifier):
    loader = make_loader(word_tokenizer)
    optimizer = torch.optim.AdamW(tiny_classifier.parameters(), lr=1e-2)
    device = torch.device("cpu")

    losses = [
        train_epoch(tiny_classifier, loader, optimizer, device)
        for _ in range(20)
    ]

    assert tiny_classifier.training
    assert losses[-1] < losses[0] - 0.2


def test_predict_returns_a_label_per_item(word_tokenizer, tiny_classifier):
    loader = make_loader(word_tokenizer, batch_size=3)

    predictions, true_labels = predict(
        tiny_classifier, loader, torch.device("cpu")
    )

    assert not tiny_classifier.training
    assert true_labels.tolist() == [0, 2, 0, 2]
    assert len(predictions) == 4
    assert set(predictions.tolist()) <= {0, 1, 2}


def test_evaluate_scores_a_trained_model(word_tokenizer, tiny_classifier):
    loader = make_loader(word_tokenizer)
    optimizer = torch.optim.AdamW(tiny_classifier.parameters(), lr=1e-2)
    device = torch.device("cpu")
    for _ in range(30):
        train_epoch(tiny_classifier, loader, optimizer, device)

    accuracy, f1 = evaluate(tiny_classifier, loader, device)

    assert accuracy == 1.0
    assert f1 == 1.0
