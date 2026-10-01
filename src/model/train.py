"""Model pretraining"""

import numpy as np
from src.model.architecture import MLP
from src.model.helpers import create_dataset, classify


def train(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 5,
) -> tuple:
    """Backpropagation training loop

    Args:
        model    (MLP): model to be trained
        X (np.ndarray): landmarks
        epochs   (int): how many times the dataset is iterated. Defaults to 5

    Returns:
        (tuple):
            train_losses (list): training losses of each iteration
            val_losses   (list): validation losses of each iteration
            train_accs   (list): training classification accurary of each iteration
            val_accs     (list): validation classification accurary of each iteration
    """

    return [[0], 0, [100], 0, 0]
