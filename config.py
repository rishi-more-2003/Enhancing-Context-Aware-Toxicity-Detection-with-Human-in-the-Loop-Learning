"""Centralized configuration for all hyperparameters and paths."""

from pathlib import Path

# ──────────────────────────── Paths ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
GOLDEN_STANDARD_DIR = DATA_DIR / "golden_standard"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = DATA_DIR / "figures"

# ──────────────────────────── Text preprocessing ────────────────────────────
MAX_FEATURES = 20_000
OUTPUT_SEQUENCE_LENGTH = 1800
BATCH_SIZE = 32

# ──────────────────────────── BiLSTM architecture ────────────────────────────
EMBEDDING_DIM = 32
LSTM_UNITS = 32
DENSE_UNITS = [128, 256, 128]

# ──────────────────────────── Training ────────────────────────────
EPOCHS = 5
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 2

# ──────────────────────────── Behavior Cloning ────────────────────────────
BC_TOXICITY_THRESHOLD = 0.5  # upvote_ratio below this → toxic

# ──────────────────────────── DPO ────────────────────────────
DPO_MIN_RATIO_DIFF = 0.2  # minimum upvote-ratio gap for preference pairs

# ──────────────────────────── Data collection ────────────────────────────
SUBREDDIT = "politics"
SEARCH_QUERIES = ["trump", "biden", "kamala", "guns", "ukraine", "israel"]
COMMENTS_PER_QUERY = 2500

# ──────────────────────────── Dataset split ratios ────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
# test = 1 - train - val
