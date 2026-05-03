"""BiLSTM model for toxicity detection."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

from config import MAX_FEATURES, EMBEDDING_DIM, LSTM_UNITS, DENSE_UNITS


def build_bilstm(
    max_features: int = MAX_FEATURES,
    embedding_dim: int = EMBEDDING_DIM,
    lstm_units: int = LSTM_UNITS,
    dense_units: list[int] | None = None,
) -> Sequential:
    """Construct a BiLSTM binary classifier.

    Architecture::

        Embedding → BiLSTM → Dense(128, relu) → Dense(256, relu)
                            → Dense(128, relu) → Dense(1, sigmoid)

    Parameters
    ----------
    max_features : int
        Vocabulary size (+1 for OOV).
    embedding_dim : int
        Dimension of the token embeddings.
    lstm_units : int
        Hidden units in each LSTM direction.
    dense_units : list[int]
        Sizes of the fully-connected layers before the output head.

    Returns
    -------
    tensorflow.keras.Model
    """
    if dense_units is None:
        dense_units = list(DENSE_UNITS)

    model = Sequential()
    model.add(Embedding(max_features + 1, embedding_dim))
    model.add(Bidirectional(LSTM(lstm_units, activation="tanh")))
    for units in dense_units:
        model.add(Dense(units, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model
