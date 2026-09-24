"""Model pretraining"""

import numpy as np
from src.model.architecture import MLP


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
