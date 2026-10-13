from collections.abc import Sequence

from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
)

from itg_transformers.config import MAX_LENGTH, MODEL_NAME


def load_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer is None:
        raise ValueError(f"No tokenizer found for {model_name}")
    return tokenizer


def count_tokens(
    tokenizer: PreTrainedTokenizerBase, texts: Sequence[str]
) -> list[int]:
    """Token count per text, special tokens included."""
    encoded = tokenizer(list(texts))
    return [len(ids) for ids in encoded["input_ids"]]


def tokenize_texts(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_LENGTH,
) -> BatchEncoding:
    """Tokenizes a batch into PyTorch tensors of shape (batch, seq_len).

    Texts are padded to the longest one in the batch and cut at
    `max_length`. `token_type_ids` are dropped: DistilBERT has no segment
    embeddings and silently ignores them anyway.
    """
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )


def explain_tokenization(
    text: str, tokenizer: PreTrainedTokenizerBase
) -> None:
    """Prints how the text is split into tokens and their ids."""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    print(f"Count: {len(tokens)}")
