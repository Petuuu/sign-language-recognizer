"""Model pretraining"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.model.architecture import MLP, save_model
from src.model.helpers import create_dataset, classify, cross_entropy, adam


def handle_training() -> None:
    """Handles training initialization"""
    landmarks_path = input("Path to landmarks (default 'dataset/landmarks.csv'): ")
    plot_path = input(
        "Path to which plot is to be saved (default 'model_results/pretraining.png')"
    )
    model_path = input(
        "Path to which model is to be saved (default 'models/model.json')"
    )

    if landmarks_path == "":
        landmarks_path = "dataset/landmarks.csv"
    if plot_path == "":
        plot_path = "model_results/pretraining.png"
    if model_path == "":
        model_path = "models/model.json"

    model = MLP()
    X_train, y_train, X_val, y_val = create_dataset(landmarks_path)
    res = train(model, X_train, y_train, X_val, y_val)
    print("\nPlotting...")
    plot_training(*res, plot_path)
    print("\nSaving...")
    save_model(model, model_path)


def train(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 5,
    eval_freq: int = 100,
    verbose: bool = True,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Backpropagation training loop

    Args:
        model          (MLP): model to be trained
        X_train (np.ndarray): landmarks and handedness for training
        y_train (np.ndarray): correct labels for training
        X_val   (np.ndarray): landmarks and handedness for validation
        y_val   (np.ndarray): correct labels for validation
        n_epochs       (int): how many times the dataset is iterated. Defaults to 5
        eval_freq      (int): how many training iterations until model evaluated.
                              Set value to 0 to disable evaluation. Defaults to 5
        verbose       (bool): are evaluation results printed

    Returns:
        (tuple):
            train_losses (list[float]): training losses of each iteration
            val_losses   (list[float]): validation losses of each iteration
            train_accs   (list[float]): training classification accurary of each iteration
            val_accs     (list[float]): validation classification accurary of each iteration
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    step = -1

    for epoch in range(n_epochs):
        for X, y in zip(X_train, y_train):
            logits = model.forward(X.astype(float), training=True)
            loss, grad = cross_entropy(logits, int(y) - 1)
            model.backward(grad)
            step += 1

            if eval_freq != 0 and step % eval_freq == 0:
                # training stats
                train_losses.append(loss)

                train_acc = 100 * np.mean(
                    [
                        classify(model.forward(x.astype(float)))[1] == int(label)
                        for x, label in zip(X_train, y_train)
                    ]
                )
                train_accs.append(train_acc)

                # validation stats
                val_results = [
                    cross_entropy(model.forward(sample.astype(float)), int(label) - 1)
                    for sample, label in zip(X_val, y_val)
                ]
                val_loss = float(np.mean([result[0] for result in val_results]))
                val_losses.append(val_loss)

                val_acc = 100 * np.mean(
                    [
                        classify(model.forward(sample.astype(float)))[1] == int(label)
                        for sample, label in zip(X_val, y_val)
                    ]
                )
                val_accs.append(val_acc)

                if verbose:
                    print(
                        f"{f'epoch {epoch + 1:03d} (step {step:05d})':<23} | "
                        f"{f'train_loss={loss:.3f}':<16} | {f'train_acc={train_acc:.3f}':<16} | "
                        f"{f'val_loss={val_loss:.3f}':<16} | {f'val_acc={val_acc:.3f}':<16}"
                    )

    return train_losses, val_losses, train_accs, val_accs


def plot_training(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    output_path: str = "model_results/pretraining.png",
) -> None:
    """Plots pretraining losses and accuracies and saves plot to given PNG file

    Args:
        train_losses (list[float]): training losses of each iteration
        val_losses   (list[float]): validation losses of each iteration
        train_accs   (list[float]): training classification accurary of each iteration
        val_accs     (list[float]): validation classification accurary of each iteration
        output_path          (str): Path to which the plot is to be saved. Must be PNG or JPG
    """
    if len(output_path) < 5 or output_path[-4:] != ".png":
        print("File must be PNG or JPG")
        sys.exit()
    if os.path.exists(output_path):
        confirm = input("File already exists. Overide? [Y/n] ")
        if confirm not in ("Y", "y"):
            return

    epochs = range(1, len(train_losses) + 1)
    if not (len(train_losses) == len(val_losses) == len(train_accs) == len(val_accs)):
        raise ValueError("All training and validation lists must have the same length.")

    try:
        fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 5))

        loss_ax.plot(epochs, train_losses, label="Training loss")
        loss_ax.plot(epochs, val_losses, label="Validation loss")
        loss_ax.set_title("Loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.legend()
        loss_ax.grid(True)

        acc_ax.plot(epochs, train_accs, label="Training accuracy")
        acc_ax.plot(epochs, val_accs, label="Validation accuracy")
        acc_ax.set_title("Accuracy")
        acc_ax.set_xlabel("Epoch")
        acc_ax.set_ylabel("Accuracy")
        acc_ax.legend()
        acc_ax.grid(True)

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    except:
        print(f"File '{output_path}' not found or could not be opened. Returning...")
