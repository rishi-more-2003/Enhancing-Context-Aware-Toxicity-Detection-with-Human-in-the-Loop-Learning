"""Text preprocessing utilities for toxicity detection."""

import tensorflow as tf
from keras.layers import TextVectorization

from config import (
    MAX_FEATURES,
    OUTPUT_SEQUENCE_LENGTH,
    BATCH_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
)


def preprocess_dataset(
    X,
    Y,
    max_features: int = MAX_FEATURES,
    output_sequence_length: int = OUTPUT_SEQUENCE_LENGTH,
    batch_size: int = BATCH_SIZE,
):
    """Vectorize text and split into train / val / test ``tf.data.Dataset``s.

    Parameters
    ----------
    X : array-like of str
        Raw comment texts.
    Y : array-like of int
        Binary labels (0 = non-toxic, 1 = toxic).
    max_features : int
        Vocabulary size for ``TextVectorization``.
    output_sequence_length : int
        Fixed-length output of the vectorizer.
    batch_size : int
        Batch size for all splits.

    Returns
    -------
    train, val, test : tf.data.Dataset
    """
    vectorizer = TextVectorization(
        max_tokens=max_features,
        output_sequence_length=output_sequence_length,
        output_mode="int",
    )
    vectorizer.adapt(X.values)
    vectorized_text = vectorizer(X.values)

    dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, Y))
    dataset = dataset.cache().shuffle(buffer_size=len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    total = len(dataset)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)

    train = dataset.take(train_size)
    val = dataset.skip(train_size).take(val_size)
    test = dataset.skip(train_size + val_size)

    return train, val, test
