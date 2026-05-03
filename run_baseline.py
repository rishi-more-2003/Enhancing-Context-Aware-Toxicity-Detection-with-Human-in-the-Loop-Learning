"""Train the baseline BiLSTM on the Kaggle Toxic Comment dataset.

Usage::

    python run_baseline.py --kaggle-csv data/kaggle/train.csv
"""

import argparse

import pandas as pd

from config import MODELS_DIR, EPOCHS
from src.models.bilstm import build_bilstm
from src.data.preprocessing import preprocess_dataset
from src.training.callbacks import BatchHistoryCallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline BiLSTM")
    parser.add_argument(
        "--kaggle-csv",
        default="data/kaggle/train.csv",
        help="Path to Kaggle toxic comment train.csv",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    print("Loading Kaggle dataset...")
    df = pd.read_csv(args.kaggle_csv)
    X, Y = df["comment_text"], df["toxic"]

    print("Preprocessing...")
    train, val, test = preprocess_dataset(X, Y)

    print("Building BiLSTM model...")
    model = build_bilstm()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    batch_history = BatchHistoryCallback()
    print(f"\nTraining for {args.epochs} epochs...")
    model.fit(train, epochs=args.epochs, validation_data=val, callbacks=[batch_history])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "pretrained_bilstm.keras"
    model.save(save_path)
    print(f"\nBaseline model saved to {save_path}")


if __name__ == "__main__":
    main()
