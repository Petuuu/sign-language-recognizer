"""Model helper methods"""

import os
import sys
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function

    Args:
        x (np.ndarray): input tensor

    Returns:
        (np.ndarray): output tensor
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Derivative of ReLU activation function for backward pass

    Args:
        x    (np.ndarray): input tensor
        grad (np.ndarray): calculated gradient

    Returns:
        (np.ndarray): output tensor
    """
    return grad * (x > 0)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax activation function

    Args:
        logits (np.ndarray): input logits

    Returns:
        (np.ndarray): output probabilities
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    expi = np.exp(shifted)
    return expi / np.sum(expi, axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, label: int) -> tuple[float, np.ndarray]:
    """Cross-entropy loss

    Args:
        logits (np.ndarray): input logits
        label  (np.ndarray): index of target label

    Returns:
        (tuple):
            loss      (float): calculated loss
            grad (np.ndarray): gradient with respect to logits
    """
    loss = float(np.logaddexp.reduce(logits) - logits[label])

    grad = softmax(logits)
    grad[label] -= 1

    return loss, grad


def create_dataset(
    path: str, train_ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts all landmarks and handedness as X and corresponding labels as y

    Args:
        path (str): path to CSV dataset file

    Returns:
        X_train (np.ndarray): 2d tensor containing landmarks and handedness for training
        y_train (np.ndarray): 1d tensor containing correct labels for training
        X_val   (np.ndarray): 2d tensor containing landmarks and handedness for validation
        y_val   (np.ndarray): 1d tensor containing correct labels for validation
    """
    if not os.path.exists(path):
        print("Incorrect path. Exiting...")
        sys.exit()

    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip().split(",")
            X.append(clean[1:])
            y.append(clean[0])

    train_portion = int(train_ratio * len(y))
    X_train = np.array(X[:train_portion])
    y_train = np.array(y[:train_portion])
    X_val = np.array(X[train_portion:])
    y_val = np.array(y[train_portion:])

    return X_train, y_train, X_val, y_val


def classify(logits: np.ndarray) -> int:
    """Select most probable letter (index, label) from model output logits.
    If probability is less than 0.X, return 0 (unknown)

    Args:
        logits (np.ndarray): model output tensor

    Returns:
        (int): most probable letter (label)
    """
    probas = softmax(logits)
    idx = np.argmax(logits)

    return idx + 1 if probas[idx] > 0.5 else 0


def adam():
    """Adam optimizer for training algorithm"""
