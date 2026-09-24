"""Model helper methods"""

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
        x (np.ndarray): input tensor
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
    """Calculates cross-entropy loss

    Args:
        logits (np.ndarray): input logits
        target (np.ndarray): target tensor

    Returns:
        (tuple):
            loss (float): calculated loss
            grad (np.ndarray): gradient with respect to logits
    """
    loss = float(np.logaddexp.reduce(logits) - logits[label])

    grad = softmax(logits)
    grad[label] -= 1

    return loss, grad
