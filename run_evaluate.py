"""Evaluate fine-tuned models on the golden-standard dataset.

Usage::

    python run_evaluate.py \
        --bc-model data/models/bc_bilstm.keras \
        --dpo-model data/models/dpo_bilstm.keras \
        --golden-csv data/golden_standard/removed_subreddit_comments.csv
"""

import argparse
import json

import pandas as pd
import tensorflow as tf

from config import RESULTS_DIR
from src.evaluation.evaluate import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BC and DPO models")
    parser.add_argument("--bc-model", default="data/models/bc_bilstm.keras")
    parser.add_argument("--dpo-model", default="data/models/dpo_bilstm.keras")
    parser.add_argument(
        "--golden-csv",
        default="data/golden_standard/removed_subreddit_comments.csv",
    )
    args = parser.parse_args()

    print("Loading golden-standard dataset...")
    golden_df = pd.read_csv(args.golden_csv)
    if "toxic" not in golden_df.columns:
        golden_df["toxic"] = 1
    if "body" not in golden_df.columns and "text" in golden_df.columns:
        golden_df = golden_df.rename(columns={"text": "body"})

    results: dict[str, dict] = {}

    for label, path in [("bc", args.bc_model), ("dpo", args.dpo_model)]:
        print(f"\n── Evaluating {label.upper()} model ──")
        model = tf.keras.models.load_model(path)
        metrics = evaluate_model(model, golden_df)
        results[label] = metrics
        print(f"  Loss:      {metrics['loss']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "evaluation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
