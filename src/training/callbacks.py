"""Custom Keras callbacks."""

import tensorflow as tf


class BatchHistoryCallback(tf.keras.callbacks.Callback):
    """Record per-batch training and validation losses."""

    def __init__(self):
        super().__init__()
        self.batch_losses: list[float] = []
        self.batch_val_losses: list[float] = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get("loss"))

    def on_test_batch_end(self, batch, logs=None):
        self.batch_val_losses.append(logs.get("loss"))
