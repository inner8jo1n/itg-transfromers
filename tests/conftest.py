import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertModel,
    PreTrainedTokenizerFast,
)

# Plots in tests must never open a window
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def word_tokenizer() -> PreTrainedTokenizerFast:
    """Whitespace tokenizer with a four-word vocabulary, no download."""
    vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )


@pytest.fixture
def tiny_model() -> DistilBertModel:
    """Randomly initialized one-layer DistilBERT on the CPU, no download."""
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=4, dim=8, n_layers=1, n_heads=2, hidden_dim=16
    )
    return DistilBertModel(config).eval()


@pytest.fixture
def tiny_classifier() -> DistilBertForSequenceClassification:
    """Randomly initialized one-layer DistilBERT with a 3-class head."""
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=4, dim=8, n_layers=1, n_heads=2, hidden_dim=16
    )
    config.num_labels = 3
    return DistilBertForSequenceClassification(config)
