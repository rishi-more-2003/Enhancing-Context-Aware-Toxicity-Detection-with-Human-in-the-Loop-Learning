"""Fine-tune the pretrained BiLSTM using Behavior Cloning or DPO.

Usage::

    # Behavior Cloning
    python run_finetune.py bc \
        --model data/models/pretrained_bilstm.keras \
        --reddit-csv data/subreddit_comments.csv

    # Direct Preference Optimization
    python run_finetune.py dpo \
        --model data/models/pretrained_bilstm.keras \
        --reddit-csv data/subreddit_comments.csv
"""

import argparse

import pandas as pd
import tensorflow as tf

from config import MODELS_DIR, RESULTS_DIR
from src.training.behavior_cloning import prepare_bc_labels, train_behavior_cloning
from src.training.dpo import create_preference_pairs, train_dpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BiLSTM with BC or DPO")
    parser.add_argument("method", choices=["bc", "dpo"], help="Fine-tuning method")
    parser.add_argument(
        "--model",
        default="data/models/pretrained_bilstm.keras",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--reddit-csv",
        default="data/subreddit_comments.csv",
        help="Path to scraped Reddit comments CSV",
    )
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading pretrained model from {args.model} ...")
    base_model = tf.keras.models.load_model(args.model)
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())

    print(f"Loading Reddit comments from {args.reddit_csv} ...")
    reddit_df = pd.read_csv(args.reddit_csv)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.method == "bc":
        print("\n── Behavior Cloning ──")
        bc_df = prepare_bc_labels(reddit_df)
        model, history = train_behavior_cloning(model, bc_df, epochs=args.epochs)

        model.save(MODELS_DIR / "bc_bilstm.keras")
        pd.DataFrame(history.history).to_csv(RESULTS_DIR / "bc_history.csv", index=False)
        print(f"BC model saved to {MODELS_DIR / 'bc_bilstm.keras'}")

    else:
        print("\n── Direct Preference Optimization ──")
        pairs_df = create_preference_pairs(reddit_df[["body", "upvote_ratio"]])
        print(f"Created {len(pairs_df)} preference pairs")

        pairs_df.to_csv(RESULTS_DIR / "dpo_pairs.csv", index=False)
        model, history = train_dpo(model, pairs_df, epochs=args.epochs)

        model.save(MODELS_DIR / "dpo_bilstm.keras")
        pd.DataFrame(history.history).to_csv(RESULTS_DIR / "dpo_history.csv", index=False)
        print(f"DPO model saved to {MODELS_DIR / 'dpo_bilstm.keras'}")


if __name__ == "__main__":
    main()
