from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MODEL_NAME = "distilbert-base-uncased"

DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"
SPLITS = ("train", "validation", "test")

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

# Truncates 0.01% of train tweets, see notebooks/00_eda.ipynb
MAX_LENGTH = 64

FIGURES_DIR = PROJECT_ROOT / "figures"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
