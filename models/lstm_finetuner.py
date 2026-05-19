"""
P3-T4: LSTM Fine-Tuner
Encapsulates Group B fine-tune logic as a reusable module.
"""
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def finetune(model, X_new: np.ndarray, y_new: np.ndarray):
    """Fine-tune an existing LSTM model on new data.

    1. Set learning rate to LSTM_LR_FINETUNE (0.0001)
    2. Build sequences if len(X_new) > LSTM_SEQUENCE
    3. Fit for LSTM_EPOCHS_FINETUNE epochs
    4. Return updated model
    """
    import tensorflow as tf

    # Reduce learning rate for fine-tuning
    model.optimizer.learning_rate.assign(config.LSTM_LR_FINETUNE)
    logger.info("Set learning rate to %f for fine-tuning", config.LSTM_LR_FINETUNE)

    # Build sequences
    if len(X_new) > config.LSTM_SEQUENCE:
        Xs, ys = [], []
        for i in range(config.LSTM_SEQUENCE, len(X_new)):
            Xs.append(X_new[i - config.LSTM_SEQUENCE:i])
            ys.append(y_new[i])
        X_seq = np.array(Xs)
        y_seq = np.array(ys)
    else:
        logger.warning(
            "Not enough new data for sequences (%d < %d)",
            len(X_new), config.LSTM_SEQUENCE,
        )
        return model

    logger.info("Fine-tuning LSTM on %d sequences for %d epochs",
                len(X_seq), config.LSTM_EPOCHS_FINETUNE)

    model.fit(
        X_seq, y_seq,
        epochs=config.LSTM_EPOCHS_FINETUNE,
        batch_size=32,
        verbose=0,
    )

    return model
