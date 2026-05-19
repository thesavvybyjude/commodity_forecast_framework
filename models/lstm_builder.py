"""
P3-T2: LSTM Builder
Standalone factory that builds and compiles a fresh LSTM model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_lstm(n_features: int):
    """Build and compile an LSTM model using config hyperparameters.

    Architecture:
        LSTM(50, return_sequences=True) -> Dropout(0.2)
        LSTM(50, return_sequences=False) -> Dropout(0.2)
        Dense(25, relu) -> Dense(1)
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            config.LSTM_UNITS,
            return_sequences=True,
            input_shape=(config.LSTM_SEQUENCE, n_features),
        ),
        tf.keras.layers.Dropout(config.LSTM_DROPOUT),
        tf.keras.layers.LSTM(
            config.LSTM_UNITS,
            return_sequences=False,
        ),
        tf.keras.layers.Dropout(config.LSTM_DROPOUT),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LSTM_LR_INIT),
        loss='mse',
        metrics=['mae'],
    )

    return model
