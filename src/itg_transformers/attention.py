from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from transformers import BatchEncoding

from itg_transformers.config import FIGURES_DIR


def visualize_attention(
    tokens: BatchEncoding,
    attentions: tuple[torch.Tensor, ...],
    layer: int = 0,
    head: int = 0,
    name: str = "attention",
    save_dir: Path | None = FIGURES_DIR,
) -> None:
    """Heatmap of one head of one layer for the first text of the batch.

    Rows are queries (the token being updated), columns are keys (the
    tokens it reads from). Padding is cut off. The plot is saved as
    `{name}_layer{layer}_head{head}.png` unless `save_dir` is None.
    Notebooks display the figure on their own; call `plt.show()` in a
    script.
    """
    length = int(tokens["attention_mask"][0].sum())
    attn = attentions[layer][0, head, :length, :length]
    token_list = tokens.tokens(0)[:length]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="viridis",
        cbar=True,
    )
    plt.title(f"Attention - Layer {layer}, Head {head}")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{name}_layer{layer}_head{head}.png")


def plot_heads(
    tokens: BatchEncoding,
    attentions: tuple[torch.Tensor, ...],
    layer: int = 0,
    columns: int = 4,
    name: str = "heads",
    save_dir: Path | None = FIGURES_DIR,
) -> None:
    """All heads of one layer side by side, on one shared color scale."""
    length = int(tokens["attention_mask"][0].sum())
    token_list = tokens.tokens(0)[:length]
    heads = attentions[layer].shape[1]
    rows = -(-heads // columns)

    fig, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False
    )
    for head, ax in enumerate(axes.flat):
        if head >= heads:
            ax.axis("off")
            continue
        attn = attentions[layer][0, head, :length, :length]
        sns.heatmap(
            attn.cpu().numpy(),
            xticklabels=token_list,
            yticklabels=token_list,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"Head {head}")
    fig.suptitle(f"Attention - Layer {layer}, all heads")
    fig.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{name}_layer{layer}.png")


def _to_sep(
    attention: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Weight every query gives to the `[SEP]` of its own text."""
    batch, heads, length, _ = attention.shape
    sep_positions = attention_mask.long().sum(dim=1) - 1
    index = sep_positions.view(batch, 1, 1, 1).expand(batch, heads, length, 1)
    return attention.gather(-1, index).squeeze(-1)


def _mean_per_head(
    values: torch.Tensor, query_mask: torch.Tensor
) -> torch.Tensor:
    """Average of (batch, heads, queries) values over real queries."""
    weights = query_mask[:, None, :].float()
    return (values * weights).sum(dim=(0, 2)) / weights.sum()


def layer_summary(
    attentions: tuple[torch.Tensor, ...], attention_mask: torch.Tensor
) -> pd.DataFrame:
    """How focused attention is in every layer.

    Averaged over heads, texts and real (non-padding) query tokens:
    - entropy: spread of a row in nats, lower means more focused;
    - max_weight: the largest weight of a row;
    - to_cls / to_sep: weight that goes to the `[CLS]` / `[SEP]` token.
    """
    rows = []
    for attention in attentions:
        metrics = {
            "entropy": -torch.special.xlogy(attention, attention).sum(-1),
            "max_weight": attention.max(dim=-1).values,
            "to_cls": attention[..., 0],
            "to_sep": _to_sep(attention, attention_mask),
        }
        rows.append(
            {
                name: float(_mean_per_head(values, attention_mask).mean())
                for name, values in metrics.items()
            }
        )
    return pd.DataFrame(rows).rename_axis("layer")


def head_patterns(
    attentions: tuple[torch.Tensor, ...], attention_mask: torch.Tensor
) -> pd.DataFrame:
    """Average weight every head puts on typical targets.

    One row per (layer, head), averaged over texts and real query tokens:
    the previous token, the token itself, the next token, `[CLS]` and
    `[SEP]`. The rest of the weight goes to all other tokens.
    """
    query_mask = attention_mask.bool()
    # Query i reads from i - 1 (or i + 1) only if both tokens are real
    shifted_mask = query_mask[:, 1:]
    rows = {}
    for layer, attention in enumerate(attentions):
        patterns = {
            "to_previous": _mean_per_head(
                attention.diagonal(offset=-1, dim1=-2, dim2=-1), shifted_mask
            ),
            "to_self": _mean_per_head(
                attention.diagonal(dim1=-2, dim2=-1), query_mask
            ),
            "to_next": _mean_per_head(
                attention.diagonal(offset=1, dim1=-2, dim2=-1), shifted_mask
            ),
            "to_cls": _mean_per_head(attention[..., 0], query_mask),
            "to_sep": _mean_per_head(
                _to_sep(attention, attention_mask), query_mask
            ),
        }
        for head in range(attention.shape[1]):
            rows[(layer, head)] = {
                name: float(values[head]) for name, values in patterns.items()
            }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.names = ["layer", "head"]
    return frame


def word_attention(
    tokens: BatchEncoding,
    attentions: tuple[torch.Tensor, ...],
    word: str,
    reads_from: bool = True,
) -> pd.DataFrame:
    """Attention of one token of the first text, averaged over heads.

    One row per layer, one column per token. With `reads_from=True` a row
    shows where `word` takes information from (its query row), otherwise
    which tokens take information from `word` (its key column).
    """
    length = int(tokens["attention_mask"][0].sum())
    token_list = tokens.tokens(0)[:length]
    position = token_list.index(word)
    rows = []
    for attention in attentions:
        heads = attention[0, :, :length, :length]
        weights = (
            heads[:, position, :] if reads_from else heads[:, :, position]
        )
        rows.append(weights.mean(dim=0).cpu().numpy())
    return pd.DataFrame(rows, columns=token_list).rename_axis("layer")
