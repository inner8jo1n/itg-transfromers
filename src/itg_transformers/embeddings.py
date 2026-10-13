from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizerBase

from itg_transformers.config import MAX_LENGTH, MODEL_NAME
from itg_transformers.tokenization import tokenize_texts


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_name: str = MODEL_NAME,
    device: torch.device | None = None,
    output_attentions: bool = False,
) -> PreTrainedModel:
    """Loads the bare encoder in eval mode, without a task head.

    With `output_attentions=True` every forward pass also returns the
    attention weights. Transformers then switches from the fused SDPA
    kernel to the slower "eager" attention, the only one that exposes the
    weights, so keep it off when only embeddings are needed.
    """
    model = AutoModel.from_pretrained(
        model_name, output_attentions=output_attentions
    )
    model.eval()
    return model.to(device or pick_device())


def get_embeddings(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    batch_size: int = 32,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """CLS embeddings of the last layer, shape (len(texts), hidden_size)."""
    all_embeddings = [np.empty((0, model.config.hidden_size), np.float32)]
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenize_texts(batch_texts, tokenizer, max_length)
        with torch.no_grad():
            outputs = model(**tokens.to(model.device))
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


def similarity(
    text1: str,
    text2: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> float:
    """Cosine similarity of the CLS embeddings of two texts."""
    embeddings = get_embeddings([text1, text2], tokenizer, model)
    return float(cosine_similarity(embeddings[0:1], embeddings[1:2])[0, 0])


def cached_embeddings(
    texts: Sequence[str],
    path: Path,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> np.ndarray:
    """`get_embeddings`, computed once and then loaded from `path`.

    The file does not know which texts or model produced it; delete it
    after changing either, e.g. `clean_text` or `MODEL_NAME`.
    """
    if path.exists():
        embeddings = np.load(path)
        if len(embeddings) != len(texts):
            raise ValueError(
                f"{path} holds {len(embeddings)} rows for {len(texts)} texts"
            )
        return embeddings
    embeddings = get_embeddings(texts, tokenizer, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    return embeddings
