"""Model architecture"""

import os
import numpy as np
from src.model.helpers import relu, relu_derivative

NUM_INPUTS = 61  # handedness + 21 × (x, y, z) - wrist
NUM_CLASSES = 24  # letters A-I and K-Y
LEARNING_RATE = 0.01


class MLP:
    """Multi-Layer Perceptron model architecture

    Attributes:
        input_size               (int): dimension of input layer
        hidden_sizes (tuple[int, int]): dimensions of hidden layers
        output_size              (int): dimension of output layer
        dropout_rate           (float): percentage of neurons to be dropped
        lr                     (float): learning rate
    """

    def __init__(
        self,
        input_size: int = NUM_INPUTS,
        hidden_sizes: tuple[int, int] = (128, 64),
        output_size: int = NUM_CLASSES,
        lr: float = LEARNING_RATE,
    ):
        """Class constructor, create layers"""
        self.dense_1 = Dense(input_size, hidden_sizes[0], lr)
        self.dropout_1 = Dropout(0.2)
        self.dense_2 = Dense(hidden_sizes[0], hidden_sizes[1], lr)
        self.dropout_2 = Dropout(0.2)
        self.dense_3 = Dense(hidden_sizes[1], output_size, lr)

        self.relu_1_input = None
        self.relu_2_input = None

    def forward(self, x: np.ndarray, training=False) -> np.ndarray:
        """Perform the neural network's forward pass

        Args:
            x  (np.ndarray): input landmarks
            training (bool): tells whether the model is in training or inference mode. Defaults to False

        Returns:
            logits (np.ndarray): prediction scores for each class
        """
        x = self.dense_1(x)
        x = self.relu_1_input = relu(x)
        x = self.dropout_1(x, training)

        x = self.dense_2(x)
        x = self.relu_2_input = relu(x)
        x = self.dropout_2(x, training)

        logits = self.dense_3(x)
        return logits

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Perform the neural network's backward pass

        Args:
            grad (np.ndarray): calculated gradients

        Returns:
            (np.ndarray): final gradient
        """
        grad = self.dense_3.backward(grad)

        grad = self.dropout_2.backward(grad)
        grad = relu_derivative(self.relu_2_input, grad)
        grad = self.dense_2.backward(grad)

        grad = self.dropout_1.backward(grad)
        grad = relu_derivative(self.relu_1_input, grad)
        grad = self.dense_1.backward(grad)

        return grad

    def to_dict(self) -> dict[str, list[float]]:
        """Return model weights and biases in a JSON-serializable format.

        Returns:
            (dict[str, list[float]]): model weights and biases as a dict
        """
        return {
            "dense_1_weights": self.dense_1.weights.tolist(),
            "dense_1_biases": self.dense_1.biases.tolist(),
            "dense_2_weights": self.dense_2.weights.tolist(),
            "dense_2_biases": self.dense_2.biases.tolist(),
            "dense_3_weights": self.dense_3.weights.tolist(),
            "dense_3_biases": self.dense_3.biases.tolist(),
        }

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Apply forward pass on function call"""
        return self.forward(x, training)


class Dense:
    """Fully connected layer

    Attributes:
        input_size  (int): dimension of input features
        output_size (int): dimension of output features
        lr        (float): learning rate
    """

    def __init__(self, input_size: int, output_size: int, lr: float):
        """Class constructor, initialize weights and biases"""
        self.weights = np.random.normal(size=(input_size, output_size)) * np.sqrt(
            2.0 / input_size
        )
        self.biases = np.zeros(output_size)
        self.lr = lr
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the layer's forward pass

        Args:
            x (np.ndarray): input features

        Returns:
            (np.ndarray): output features
        """
        self.input = x
        return x @ self.weights + self.biases

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Perform the layer's backward pass

        Args:
            grad (np.ndarray): calculated gradients

        Returns:
            (np.ndarray): gradient to pass to the previous layer
        """
        if self.input is None:
            raise RuntimeError("Call forward before backward")

        grad_weights = np.outer(self.input, grad)
        grad_biases = grad

        grad_input = grad @ self.weights.T
        self.weights -= self.lr * grad_weights
        self.biases -= self.lr * grad_biases

        return grad_input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply forward pass on function call"""
        return self.forward(x)


class Dropout:
    """Dropout layer

    Attributes:
        rate (float): percentage of neurons to be dropped
    """

    def __init__(self, rate: float):
        """Class constructor, initialize weights and biases"""
        if not 0 <= rate < 1:
            raise ValueError("Droupout rate must be in the range [0.0, 1.0)")

        self.keep_rate = 1 - rate
        self.mask = None
        self.training = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Perform the layer's forward pass

        Args:
            x  (np.ndarray): input features
            training (bool): tells whether the model is in training or inference mode. Defaults to False

        Returns:
            probas (np.ndarray): output features
        """
        self.training = training
        if not training or self.keep_rate == 1.0:
            self.mask = None
            return x

        self.mask = np.random.rand(*x.shape) < self.keep_rate
        return x * self.mask / self.keep_rate

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Perform the layer's backward pass

        Args:
            grad (np.ndarray): calculated gradients

        Returns:
            (np.ndarray): gradient to pass to the previous layer
        """
        if not self.training or self.keep_rate == 1.0:
            return grad

        return grad * self.mask / self.keep_rate

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Apply forward pass on function call"""
        return self.forward(x, training)


def save_model(model: MLP, output_path: str = "models/model.json") -> None:
    """Saves model paramaters (weights and biases) into JSON file

    Args:
        model       (MLP): model whose parameters are to be saved
        output_path (str): Path to which the parameters are to be saved. Must be JSON.
                           Defaults to "models/model.json"
    """
    if len(output_path) < 6 or output_path[-5:] != ".json":
        print("File must be JSON. Returning...")
        return
    if os.path.exists(output_path):
        confirm = input("File already exists. Overide? [Y/n] ")
        if confirm not in ("Y", "y"):
            print("Returning...")
            return

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(model.to_dict()))

    except:
        print(f"File '{output_path}' not found or could not be opened. Exiting...")


def load_model(model: MLP, path: str = "models/model.json") -> None:
    """Loads paramaters (weights and biases) into model from JSON file

    Args:
        model (MLP): model to which the parameters are to be loaded into
        path  (str): Path from which the parameters are to be loaded. Must be JSON.
                     Defaults to "models/model.json"
    """
    if not os.path.exists(path):
        confirm = input("File does not exist.")
        if confirm not in ("Y", "y"):
            print("Returning...")
            return
    if len(path) < 6 or path[-5:] != ".json":
        print("File must be JSON. Returning...")
        return

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(model.to_dict()))

    except:
        print(f"File '{path}' could not be opened. Returning...")
