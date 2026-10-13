from itg_transformers.tokenization import (
    count_tokens,
    explain_tokenization,
    tokenize_texts,
)


def test_count_tokens_returns_length_per_text(word_tokenizer):
    counts = count_tokens(word_tokenizer, ["hello world", "hello", "a b c"])

    assert counts == [2, 1, 3]


def test_tokenize_texts_pads_to_longest_text(word_tokenizer):
    batch = tokenize_texts(["hello", "hello world world"], word_tokenizer)

    assert batch["input_ids"].tolist() == [[2, 0, 0], [2, 3, 3]]
    assert batch["attention_mask"].tolist() == [[1, 0, 0], [1, 1, 1]]


def test_tokenize_texts_truncates_and_drops_token_type_ids(word_tokenizer):
    batch = tokenize_texts(
        ["hello world hello world"], word_tokenizer, max_length=2
    )

    assert batch["input_ids"].shape == (1, 2)
    assert "token_type_ids" not in batch


def test_explain_tokenization_prints_tokens_ids_and_count(
    capsys, word_tokenizer
):
    explain_tokenization("hello foo world", word_tokenizer)

    assert capsys.readouterr().out.splitlines() == [
        "Text: hello foo world",
        "Tokens: ['hello', '[UNK]', 'world']",
        "IDs: [2, 1, 3]",
        "Count: 3",
    ]
