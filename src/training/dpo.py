"""Direct Preference Optimization fine-tuning for toxicity detection.

Preference pairs are constructed from comment upvote ratios: a comment
with a significantly higher ratio is *preferred* over one with a lower
ratio.  The model is then trained to score preferred comments higher.
"""

import pandas as pd
import tensorflow as tf
from keras.layers import TextVectorization

from config import (
    DPO_MIN_RATIO_DIFF,
    MAX_FEATURES,
    OUTPUT_SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    TRAIN_RATIO,
    VAL_RATIO,
)


def create_preference_pairs(
    df: pd.DataFrame,
    min_ratio_diff: float = DPO_MIN_RATIO_DIFF,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Build DPO preference pairs from upvote ratios.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``body`` and ``upvote_ratio`` columns.
    min_ratio_diff : float
        Minimum upvote-ratio gap to form a valid pair.
    max_pairs : int, optional
        Cap on the total number of pairs.

    Returns
    -------
    pd.DataFrame
        Columns: ``preferred_text``, ``preferred_score``,
        ``non_preferred_text``, ``non_preferred_score``, ``preference_gap``.
    """
    df = df.sort_values("upvote_ratio", ascending=False).reset_index(drop=True)
    pairs: list[dict] = []
    used: set[int] = set()

    for i in range(len(df)):
        if i in used:
            continue
        preferred = df.iloc[i]
        candidates = df[
            (df["upvote_ratio"] < preferred["upvote_ratio"] - min_ratio_diff)
            & (~df.index.isin(used))
            & (df.index > i)
        ]
        if candidates.empty:
            continue

        non_preferred = candidates.iloc[0]
        pairs.append(
            {
                "preferred_text": preferred["body"],
                "preferred_score": preferred["upvote_ratio"],
                "non_preferred_text": non_preferred["body"],
                "non_preferred_score": non_preferred["upvote_ratio"],
                "preference_gap": preferred["upvote_ratio"]
                - non_preferred["upvote_ratio"],
            }
        )
        used.update({i, candidates.index[0]})

        if max_pairs and len(pairs) >= max_pairs:
            break

    return pd.DataFrame(pairs)


def train_dpo(
    model,
    dpo_dataset: pd.DataFrame,
    max_features: int = MAX_FEATURES,
    output_sequence_length: int = OUTPUT_SEQUENCE_LENGTH,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
):
    """Fine-tune *model* using DPO preference pairs.

    Parameters
    ----------
    model : tf.keras.Model
        Pre-trained BiLSTM model.
    dpo_dataset : pd.DataFrame
        Output of :func:`create_preference_pairs`.
    epochs : int
        Maximum training epochs.

    Returns
    -------
    model, history
    """
    preferred_texts = dpo_dataset["preferred_text"].values
    non_preferred_texts = dpo_dataset["non_preferred_text"].values
    preference_gaps = dpo_dataset["preference_gap"].values

    vectorizer = TextVectorization(
        max_tokens=max_features,
        output_sequence_length=output_sequence_length,
        output_mode="int",
    )
    vectorizer.adapt(preferred_texts)

    vec_preferred = vectorizer(preferred_texts)
    vec_non_preferred = vectorizer(non_preferred_texts)

    dataset = tf.data.Dataset.from_tensor_slices(
        ((vec_preferred, vec_non_preferred), preference_gaps)
    )
    dataset = (
        dataset.cache()
        .shuffle(buffer_size=len(dpo_dataset))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    total = len(dataset)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)

    train = dataset.take(train_size)
    val = dataset.skip(train_size).take(val_size)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        train,
        validation_data=val,
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
