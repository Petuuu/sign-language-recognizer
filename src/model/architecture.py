"""Model architecture"""

import os
import json
import sys
from pathlib import Path
import numpy as np
from src.model.helpers import relu, relu_derivative, adam
from src.helpers import check_file_exists

NUM_INPUTS = 61  # handedness + 21 × (x, y, z) - wrist
NUM_CLASSES = 24  # letters A-I and K-Y
DROPOUT_RATE = 0.2
OPTIMIZER_PARAMS = (0.001, 0.9, 0.999)


class MLP:
    """Multi-Layer Perceptron model architecture

    Attributes:
        input_size                (int): dimension of input layer
        hidden_sizes (tuple[int,  int]): dimensions of hidden layers
        output_size               (int): dimension of output layer
        dropout_rate            (float): percentage of neurons to be dropped
        optimizer_params        (tuple): parameters for Adam optimizer
            (float): alpha (learning rate)
            (float): beta_1 (1. moment (mean) hyperparameter)
            (float): beta_2 (2. moment (variance) hyperparameter)
    """

    def __init__(
        self,
        input_size: int = NUM_INPUTS,
        hidden_sizes: tuple[int, int] = (128, 64),
        output_size: int = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
        optimizer_params: tuple[float] = OPTIMIZER_PARAMS,
    ):
        """Class constructor, create layers"""
        self.dense_1 = Dense(input_size, hidden_sizes[0], optimizer_params)
        self.dropout_1 = Dropout(dropout_rate)
        self.dense_2 = Dense(hidden_sizes[0], hidden_sizes[1], optimizer_params)
        self.dropout_2 = Dropout(dropout_rate)
        self.dense_3 = Dense(hidden_sizes[1], output_size, optimizer_params)

        self.relu_1_input = None
        self.relu_2_input = None

    def forward(self, x: np.ndarray, training=False) -> np.ndarray:
        """Perform the neural network's forward pass

        Args:
            x  (np.ndarray): input landmarks
            training (bool): tells whether the model is in training or inference mode.
                             Defaults to False

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
        input_size         (int): dimension of input features
        output_size        (int): dimension of output features
        optimizer_params (tuple): parameters for Adam optimizer
            (float): alpha (learning rate)
            (float): beta_1 (1. moment (mean) hyperparameter)
            (float): beta_2 (2. moment (variance) hyperparameter)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        optimizer_params: tuple[float],
    ):
        """Class constructor, initialize weights and biases"""
        self.weights = np.random.normal(size=(input_size, output_size)) * np.sqrt(
            2.0 / input_size
        )
        self.biases = np.zeros(output_size)
        self.lr = optimizer_params[0]
        self.beta_1, self.beta_2 = optimizer_params[1:]
        self.weights_moment_1 = np.zeros_like(self.weights)
        self.weights_moment_2 = np.zeros_like(self.weights)
        self.biases_moment_1 = np.zeros_like(self.biases)
        self.biases_moment_2 = np.zeros_like(self.biases)
        self.timestep = 0
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
        self.timestep += 1
        weights_update, self.weights_moment_1, self.weights_moment_2 = adam(
            grad_weights,
            self.beta_1,
            self.beta_2,
            self.weights_moment_1,
            self.weights_moment_2,
            self.timestep,
        )
        biases_update, self.biases_moment_1, self.biases_moment_2 = adam(
            grad_biases,
            self.beta_1,
            self.beta_2,
            self.biases_moment_1,
            self.biases_moment_2,
            self.timestep,
        )
        self.weights -= self.lr * weights_update
        self.biases -= self.lr * biases_update

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
            training (bool): tells whether the model is in training or inference mode.
                            Defaults to False

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
    check_file_exists(output_path)
    filename = Path(output_path).name
    if len(filename) < 6 or filename[-5:] != ".json":
        print("File must be JSON. Exiting...")
        sys.exit()
    parent = Path(output_path).parent
    if not parent.exists():
        raise FileNotFoundError(f"Output directory '{parent}' does not exist.")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(model.to_dict()))

    except Exception:
        print(f"File '{output_path}' could not be opened. Exiting...")
        sys.exit()


def load_model(model: MLP, path: str = "models/model.json") -> None:
    """Loads paramaters (weights and biases) into model from JSON file

    Args:
        model (MLP): model to which the parameters are to be loaded into
        path  (str): Path from which the parameters are to be loaded. Must be JSON.
                     Defaults to "models/model.json"
    """
    if not os.path.exists(path):
        print("File does not exist. Exiting...")
        sys.exit()
    filename = Path(path).name
    if len(filename) < 6 or filename[-5:] != ".json":
        print("File must be JSON. Exiting...")
        sys.exit()

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                params = json.loads(line.strip())

        model.dense_1.weights = np.asarray(params["dense_1_weights"])
        model.dense_1.biases = np.asarray(params["dense_1_biases"])
        model.dense_2.weights = np.asarray(params["dense_2_weights"])
        model.dense_2.biases = np.asarray(params["dense_2_biases"])
        model.dense_3.weights = np.asarray(params["dense_3_weights"])
        model.dense_3.biases = np.asarray(params["dense_3_biases"])

    except Exception:
        print(f"File '{path}' could not be opened or incorrect content. Exiting...")
        sys.exit()
