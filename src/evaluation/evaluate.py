"""Evaluation utilities for toxicity detection models."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data.preprocessing import preprocess_dataset


def evaluate_model(model, comments_df, batch_size: int = 32) -> dict:
    """Evaluate a compiled model on a labelled DataFrame.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled toxicity model.
    comments_df : pd.DataFrame
        Must contain ``body`` and ``toxic`` columns.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    dict
        Keys: ``loss``, ``accuracy``, ``f1``, ``precision``, ``recall``.
    """
    if not model.compiled_loss:
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    X = comments_df["body"]
    Y = comments_df["toxic"]
    test_data, _, _ = preprocess_dataset(X, Y, batch_size=batch_size)

    loss, accuracy = model.evaluate(test_data, verbose=1)

    y_pred_prob = model.predict(test_data)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = np.concatenate([y for _, y in test_data], axis=0)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
