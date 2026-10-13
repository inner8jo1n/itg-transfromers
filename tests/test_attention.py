import math

import matplotlib.pyplot as plt
import pytest
import torch

from itg_transformers.attention import (
    head_patterns,
    layer_summary,
    plot_heads,
    visualize_attention,
    word_attention,
)
from itg_transformers.tokenization import tokenize_texts


def make_attentions(layers, heads, length):
    torch.manual_seed(0)
    scores = torch.rand(layers, 1, heads, length, length)
    return tuple(torch.softmax(scores, dim=-1))


def test_visualize_attention_saves_named_png(word_tokenizer, tmp_path):
    tokens = tokenize_texts(["hello world"], word_tokenizer)
    attentions = make_attentions(layers=2, heads=3, length=2)

    visualize_attention(
        tokens, attentions, layer=1, head=2, name="demo", save_dir=tmp_path
    )

    assert (tmp_path / "demo_layer1_head2.png").exists()


def test_visualize_attention_cuts_off_padding(word_tokenizer):
    tokens = tokenize_texts(["hello", "hello world world"], word_tokenizer)
    attentions = make_attentions(layers=1, heads=1, length=3)

    visualize_attention(tokens, attentions, save_dir=None)

    labels = [label.get_text() for label in plt.gca().get_xticklabels()]
    assert labels == ["hello"]


def test_layer_summary_of_uniform_and_one_hot_layers():
    uniform = torch.full((1, 2, 4, 4), 0.25)
    to_last = torch.zeros(1, 2, 4, 4)
    to_last[..., 3] = 1.0
    mask = torch.ones(1, 4)

    summary = layer_summary((uniform, to_last), mask)

    assert summary.loc[0, "entropy"] == pytest.approx(math.log(4))
    assert summary.loc[0, "max_weight"] == pytest.approx(0.25)
    assert summary.loc[1, "entropy"] == pytest.approx(0.0, abs=1e-6)
    assert summary.loc[1, "to_sep"] == pytest.approx(1.0)
    assert summary.loc[1, "to_cls"] == pytest.approx(0.0)


def test_layer_summary_ignores_padding_rows_and_finds_sep():
    attention = torch.zeros(2, 1, 3, 3)
    attention[0, 0, :2, 1] = 1.0  # text 0: 2 real tokens, [SEP] at 1
    attention[0, 0, 2, 0] = 1.0  # padding row, must be ignored
    attention[1, 0, :, 2] = 1.0  # text 1: 3 real tokens, [SEP] at 2
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    summary = layer_summary((attention,), mask)

    assert summary.loc[0, "to_sep"] == pytest.approx(1.0)
    assert summary.loc[0, "to_cls"] == pytest.approx(0.0)


def test_head_patterns_detects_previous_and_next_token_heads():
    length = 4
    previous = torch.zeros(1, 1, length, length)
    previous[0, 0, 0, 0] = 1.0  # the first token has no previous one
    for query in range(1, length):
        previous[0, 0, query, query - 1] = 1.0
    following = torch.zeros(1, 1, length, length)
    following[0, 0, length - 1, length - 1] = 1.0
    for query in range(length - 1):
        following[0, 0, query, query + 1] = 1.0
    attention = torch.cat([previous, following], dim=1)  # 2 heads

    patterns = head_patterns((attention,), torch.ones(1, length))

    assert patterns.loc[(0, 0), "to_previous"] == pytest.approx(1.0)
    assert patterns.loc[(0, 0), "to_next"] == pytest.approx(0.0)
    assert patterns.loc[(0, 1), "to_next"] == pytest.approx(1.0)
    assert patterns.loc[(0, 1), "to_previous"] == pytest.approx(0.0)
    assert list(patterns.index.names) == ["layer", "head"]


def test_plot_heads_draws_one_panel_per_head(word_tokenizer, tmp_path):
    tokens = tokenize_texts(["hello world"], word_tokenizer)
    attentions = make_attentions(layers=1, heads=5, length=2)

    plot_heads(tokens, attentions, columns=4, name="demo", save_dir=tmp_path)

    visible = [ax for ax in plt.gcf().axes if ax.axison]
    assert len(visible) == 5
    assert (tmp_path / "demo_layer0.png").exists()


def test_word_attention_reads_row_or_column_of_the_word(word_tokenizer):
    tokens = tokenize_texts(["hello world"], word_tokenizer)
    attention = torch.tensor([[0.9, 0.1], [0.3, 0.7]]).expand(1, 2, 2, 2)

    reads = word_attention(tokens, (attention,), "world")
    read_by = word_attention(tokens, (attention,), "world", reads_from=False)

    assert reads.columns.tolist() == ["hello", "world"]
    assert reads.to_numpy()[0].tolist() == pytest.approx([0.3, 0.7])
    assert read_by.to_numpy()[0].tolist() == pytest.approx([0.1, 0.7])
