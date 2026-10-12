from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MODEL_NAME = "cardiffnlp/twitter-roberta-base"

DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"
SPLITS = ("train", "validation", "test")

LABELS = {0: "negative", 1: "neutral", 2: "positive"}
