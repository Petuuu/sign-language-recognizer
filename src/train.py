"""Model pretraining"""

import numpy as np
from src.model import MLP


def binary_cross_entropy(inputi: np.ndarray, target: np.ndarray) -> float:
    """Calculates binary-cross entropy loss

    Args:
        inputi (np.ndarray): input tensor
        target (np.ndarray): target tensor

    Returns:
        calculated loss (float)
    """
    return -float(np.mean(target * np.log(inputi) + (1 - target) * np.log1p(-inputi)))


def train(model: MLP) -> tuple:
    """Backpropagation training loop

    Args:
        model (MLP): model to be trained
        ...

    Returns:
        (tuple):
            train_losses (list): training losses of each iteration
            val_losses   (list): validation losses of each iteration
            train_accs   (list): training classification accurary of each iteration
            val_accs     (list): validation classification accurary of each iteration
    """

    return [[0], 0, [100], 0, 0]
