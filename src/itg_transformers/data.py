import html
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from itg_transformers.config import (
    DATA_DIR,
    DATASET_CONFIG,
    DATASET_NAME,
    SPLITS,
)

UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
REPEATED_QUOTES = re.compile(r'"{2,}')


def split_path(split: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"{split}.csv"


def download_dataset(data_dir: Path = DATA_DIR) -> None:
    """Saves every split as a CSV with `text` and `label` columns."""
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    for split in SPLITS:
        part = dataset[split]
        frame = pd.DataFrame({"text": part["text"], "label": part["label"]})
        frame.to_csv(split_path(split, data_dir), index=False)


def clean_text(text: str) -> str:
    """Undoes the export artifacts of train and validation.

    Train and validation store `’` and `,` as literal `\\u2019` and
    `\\u002c`, keep HTML entities like `&amp;`, wrap many tweets in quotes
    and escape inner quotes as `\\"` or `""`. Test has none of that, so
    without cleaning the model would learn the artifacts.
    """
    text = UNICODE_ESCAPE.sub(lambda match: chr(int(match[1], 16)), text)
    # Some entities are encoded twice, like `&amp;quot;`
    text = html.unescape(html.unescape(text))
    if len(text) > 1 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = text.replace('\\"', '"')
    text = REPEATED_QUOTES.sub('"', text)
    return text.strip()


def load_split(
    split: str, data_dir: Path = DATA_DIR, clean: bool = True
) -> pd.DataFrame:
    frame = pd.read_csv(split_path(split, data_dir))
    if clean:
        frame["text"] = frame["text"].map(clean_text)
    return frame
