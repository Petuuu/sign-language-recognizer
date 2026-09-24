"""Model architecture and training helper methods"""

import numpy as np

NUM_LANDMARKS = 63  # 21 × (x, y, z)
NUM_CLASSES = 22  # letters A-I and K-Y + unknown
LEARNING_RATE = 0.01


class MLP:
    """Multi-Layer Perceptron model architecture

    Attributes:
        input_size (int): dimension of input layer
        hidden_sizes (tuple[int, int]): dimensions of hidden layers
        output_size (int): dimension of output layer
        dropout_rate (float): percentage of neurons to be dropped
    """

    def __init__(
        self,
        input_size: int = NUM_LANDMARKS,
        hidden_sizes: tuple[int, int] = (128, 64),
        output_size: int = NUM_CLASSES,
    ):
        """Class constructor, creates layers"""
        self.dense_1 = Dense(input_size, hidden_sizes[0])
        self.dropout_1 = Dropout(0.2)
        self.dense_2 = Dense(hidden_sizes[0], hidden_sizes[1])
        self.dropout_2 = Dropout(0.2)
        self.dense_3 = Dense(hidden_sizes[1], output_size)

    def forward(self, x: np.ndarray, training=False) -> np.ndarray:
        """Performs the neural network's forward pass

        Args:
            x (np.ndarray): input landmarks
            training (bool): tells whether the model is in training or inference mode

        Returns:
            logits (np.ndarray): prediction scores for each class
        """
        x = self.dense_1(x)
        x = relu(x)
        x = self.dropout_1(x, training)

        x = self.dense_2(x)
        x = relu(x)
        x = self.dropout_2(x, training)

        logits = self.dense_3(x)
        return logits

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Performs the neural network's backward pass

        Args:
            grad (np.ndarray): calculated gradients

        Returns:
            (tuple[np.ndarray, np.ndarray]): updated weights and biases
        """

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Apply forward pass on function call"""
        return self.forward(x)


class Dense:
    """Fully connected layer

    Attributes:
        input_size (int): dimension of input features
        output_size (int): dimension of output features
    """

    def __init__(self, input_size: int, output_size: int):
        """Class constructor, initializes weights and biases"""
        self.weights = np.random.normal(size=(input_size, output_size)) * np.sqrt(
            2.0 / input_size
        )
        self.biases = np.zeros(output_size)
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the layer's forward pass

        Args:
            x (np.ndarray): input features

        Returns:
            (np.ndarray): output features
        """
        self.input = x @ self.weights + self.biases
        return self.input

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Performs the layer's backward pass

        Args:
            grad (np.ndarray): calculated gradients

        Returns:
            (tuple[np.ndarray, np.ndarray]): updated weights and biases
        """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply forward pass on function call"""
        return self.forward(x)


class Dropout:
    """Dropout layer

    Attributes:
        rate (float): percentage of neurons to be dropped
    """

    def __init__(self, rate: float):
        """Class constructor, initializes weights and biases"""
        if not 0 <= rate < 1:
            raise ValueError("Droupout rate must be in the range [0.0, 1.0)")

        self.keep_rate = 1 - rate

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Performs the layer's forward pass

        Args:
            x (np.ndarray): input features
            training (bool): tells whether the model is in training or inference mode

        Returns:
            probas (np.ndarray): output features
        """
        if not training or self.keep_rate == 0.0:
            return x

        self.mask = np.random.rand(*x.shape) < self.keep_rate
        return x * self.mask / self.keep_rate

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Performs the layer's backward pass

        Args:
            grad (np.ndarray): calculated gradients

        Returns:
            (tuple[np.ndarray, np.ndarray]): updated weights and biases
        """

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Apply forward pass on function call"""
        return self.forward(x, training)


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
