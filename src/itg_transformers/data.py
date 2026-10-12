from pathlib import Path

import pandas as pd
from datasets import load_dataset

from itg_transformers.config import (
    DATA_DIR,
    DATASET_CONFIG,
    DATASET_NAME,
    SPLITS,
)


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


def load_split(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(split_path(split, data_dir))
