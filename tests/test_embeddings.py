import numpy as np
import pytest

from itg_transformers.embeddings import (
    cached_embeddings,
    get_embeddings,
    similarity,
)

TEXTS = ["hello", "hello world world", "world", "world hello"]


def test_get_embeddings_returns_one_vector_per_text(
    word_tokenizer, tiny_model
):
    embeddings = get_embeddings(TEXTS, word_tokenizer, tiny_model)

    assert embeddings.shape == (4, 8)
    assert embeddings.dtype == np.float32


def test_get_embeddings_does_not_depend_on_batching(
    word_tokenizer, tiny_model
):
    one_batch = get_embeddings(TEXTS, word_tokenizer, tiny_model, batch_size=4)
    by_one = get_embeddings(TEXTS, word_tokenizer, tiny_model, batch_size=1)
    uneven = get_embeddings(TEXTS, word_tokenizer, tiny_model, batch_size=3)

    np.testing.assert_allclose(one_batch, by_one, atol=1e-5)
    np.testing.assert_allclose(one_batch, uneven, atol=1e-5)


def test_get_embeddings_of_no_texts(word_tokenizer, tiny_model):
    assert get_embeddings([], word_tokenizer, tiny_model).shape == (0, 8)


def test_similarity_of_a_text_with_itself_is_one(word_tokenizer, tiny_model):
    score = similarity(
        "hello world", "hello world", word_tokenizer, tiny_model
    )

    assert score == pytest.approx(1.0, abs=1e-6)


def test_similarity_is_symmetric_and_bounded(word_tokenizer, tiny_model):
    forward = similarity("hello", "world world", word_tokenizer, tiny_model)
    backward = similarity("world world", "hello", word_tokenizer, tiny_model)

    assert forward == pytest.approx(backward, abs=1e-6)
    assert -1.0 <= forward < 1.0


def test_cached_embeddings_computes_once_then_loads(
    word_tokenizer, tiny_model, tmp_path
):
    path = tmp_path / "cache" / "train.npy"

    first = cached_embeddings(TEXTS, path, word_tokenizer, tiny_model)
    np.save(path, np.zeros_like(first))  # a reload must not recompute
    second = cached_embeddings(TEXTS, path, word_tokenizer, tiny_model)

    assert first.shape == (4, 8)
    assert not second.any()


def test_cached_embeddings_rejects_a_stale_file(
    word_tokenizer, tiny_model, tmp_path
):
    path = tmp_path / "train.npy"
    np.save(path, np.zeros((3, 8), np.float32))

    with pytest.raises(ValueError, match="3 rows for 4 texts"):
        cached_embeddings(TEXTS, path, word_tokenizer, tiny_model)
