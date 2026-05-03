"""Behavior Cloning fine-tuning for toxicity detection.

Labels are derived from Reddit upvote/downvote ratios:
comments with an upvote ratio below a threshold are labeled toxic.
"""

import tensorflow as tf

from config import EPOCHS, EARLY_STOPPING_PATIENCE, BC_TOXICITY_THRESHOLD
from src.data.preprocessing import preprocess_dataset


def prepare_bc_labels(subreddit_df, threshold: float = BC_TOXICITY_THRESHOLD):
    """Convert upvote ratios into binary toxicity labels.

    Parameters
    ----------
    subreddit_df : pd.DataFrame
        Must contain ``body`` and ``upvote_ratio`` columns.
    threshold : float
        Upvote ratio below this value is labeled toxic (1).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``body`` and ``toxic`` columns.
    """
    df = subreddit_df[["body", "upvote_ratio"]].copy()
    df["toxic"] = (df["upvote_ratio"] < threshold).astype(int)
    df = df.drop(columns=["upvote_ratio"])
    return df


def train_behavior_cloning(model, comments_df, epochs: int = EPOCHS):
    """Fine-tune *model* on BC-labeled Reddit comments.

    Parameters
    ----------
    model : tf.keras.Model
        Pre-trained BiLSTM model.
    comments_df : pd.DataFrame
        Must contain ``body`` and ``toxic`` columns.
    epochs : int
        Maximum training epochs.

    Returns
    -------
    model, history
    """
    train_data, val_data, _ = preprocess_dataset(
        comments_df["body"], comments_df["toxic"]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )
        ],
    )
    return model, history
