"""Model pretraining"""

import os
import sys
import numpy as np
from src.model.architecture import MLP
from src.model.helpers import create_dataset, classify, adam


def train(
    model: MLP,
    train_dataset: tuple[np.ndarray, np.ndarray],
    val_dataset: tuple[np.ndarray, np.ndarray],
    n_epochs: int = 5,
    eval_freq: int = 5,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Backpropagation training loop

    Args:
        model           (MLP): model to be trained
        train_dataset (tuple): dataset used for training
            (np.ndarray): landmarks and handedness
            (np.ndarray): correct labels

        val_dataset   (tuple): dataset used for training
            (np.ndarray): landmarks and handedness
            (np.ndarray): correct labels

        n_epochs        (int): how many times the dataset is iterated. Defaults to 5
        eval_freq       (int): how many training iterations until model evaluated. Defaults to 5

    Returns:
        (tuple):
            train_losses (list[float]): training losses of each iteration
            val_losses   (list[float]): validation losses of each iteration
            train_accs   (list[float]): training classification accurary of each iteration
            val_accs     (list[float]): validation classification accurary of each iteration
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(n_epochs):
        pass

    return train_losses, val_losses, train_accs, val_accs


def plot_training(
    path: str,
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
) -> None:
    """Plots pretraining losses and accuracies and saves plot to given PNG file

    Args:
        path                 (str): Path to which the plot is to be saved. Must be PNG or JPG
        train_losses (list[float]): training losses of each iteration
        val_losses   (list[float]): validation losses of each iteration
        train_accs   (list[float]): training classification accurary of each iteration
        val_accs     (list[float]): validation classification accurary of each iteration
    """
    if not os.path.exists(path):
        print("Incorrect path. Exiting...")
        sys.exit()
    if path[-4:] != ".png":
        print("File must be PNG or JPG")
        sys.exit()
