"""Model architecture and training helper methods"""

import numpy as np

NUM_CLASSES = 22  # letters A-I and K-Y + unknown
NUM_LANDMARKS = 63  # 21 × (x, y, z)


class MLP:
    """Multi-Layer Perceptron model architecture

    Attributes:
        size (int): ????????
    """

    def __init__(self):
        """Class constructor, creates layers"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the neural network's forward pass

        Args:
            x (np.ndarray): input landmarks

        Returns:
            probas (np.ndarray): probabilities of each class
        """
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs the neural network's backward pass

        Args:
            grad (np.ndarray): calculated gradients

        Returns:
            probas (np.ndarray): probabilities of each class
        """


class Dense:
    """Fully connected layer"""

    def __init__(self):
        """Class constructor, initializes weights and biases"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the layer's forward pass

        Args:
            x (np.ndarray): input landmarks

        Returns:
            probas (np.ndarray): probabilities of each class
        """
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs the layer's backward pass

        Args:
            grad (np.ndarray): calculated gradients

        Returns:
            probas (np.ndarray): probabilities of each class
        """


class Dropout:
    """Dropout layer"""

    def __init__(self):
        """Class constructor, initializes weights and biases"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the layer's forward pass

        Args:
            x (np.ndarray): input landmarks

        Returns:
            probas (np.ndarray): probabilities of each class
        """
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs the layer's backward pass

        Args:
            grad (np.ndarray): calculated gradients

        Returns:
            probas (np.ndarray): probabilities of each class
        """

    pass


def relu(x: np.ndarray) -> np.ndarray:
    """ReLu activation function

    Args:
        x (np.ndarray): input tensor

    Returns:
        (np.ndarray): output tensor
    """
    return np.maximum(0, x)


def relu_back(x: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """ReLu activation function for backward pass

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
    return np.exp(logits) / np.sum(np.exp(logits))
