"""Model pretraining"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.model.architecture import MLP
from src.model.helpers import create_dataset, classify, cross_entropy, adam


def train(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 5,
    eval_freq: int = 100,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Backpropagation training loop

    Args:
        model          (MLP): model to be trained
        X_train (np.ndarray): landmarks and handedness for training
        y_train (np.ndarray): correct labels for training
        X_val   (np.ndarray): landmarks and handedness for validation
        y_val   (np.ndarray): correct labels for validation
        n_epochs       (int): how many times the dataset is iterated. Defaults to 5
        eval_freq      (int): how many training iterations until model evaluated. Defaults to 5

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

            if step % eval_freq == 0:
                # training stats
                train_losses.append(loss)

                train_acc = 100 * np.mean(
                    [
                        classify(model.forward(x.astype(float))) == int(label)
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
                        classify(model.forward(sample.astype(float))) == int(label)
                        for sample, label in zip(X_val, y_val)
                    ]
                )
                val_accs.append(val_acc)

                print(
                    f"{f'epoch {epoch + 1:03d} (step {step:05d})':<23} | "
                    f"{f'train_loss={loss:.3f}':<16} | {f'train_acc={train_acc:.3f}':<16} | "
                    f"{f'val_loss={val_loss:.3f}':<16} | {f'val_acc={val_acc:.3f}':<16}"
                )

    return train_losses, val_losses, train_accs, val_accs


def plot_training(
    output_path: str,
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
) -> None:
    """Plots pretraining losses and accuracies and saves plot to given PNG file

    Args:
        output_path          (str): Path to which the plot is to be saved. Must be PNG or JPG
        train_losses (list[float]): training losses of each iteration
        val_losses   (list[float]): validation losses of each iteration
        train_accs   (list[float]): training classification accurary of each iteration
        val_accs     (list[float]): validation classification accurary of each iteration
    """
    if output_path[-4:] != ".png":
        print("File must be PNG or JPG")
        sys.exit()
    if os.path.exists(output_path):
        confirm = input("File already exists. Overide? [Y/n] ")
        if confirm not in ("Y", "y"):
            print("Exiting...")
            sys.exit()

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
        print(f"File '{output_path}' not found or could not be opened. Exiting...")


if __name__ == "__main__":
    model = MLP()
    X_train, y_train, X_val, y_val = create_dataset("dataset/landmarks.csv")
    plot_training(
        "model_results/pretraining.png", *train(model, X_train, y_train, X_val, y_val)
    )
