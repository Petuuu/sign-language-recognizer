"""Tests for model"""

import unittest
import os
from time import time
import numpy as np
import torch
from torch import nn
from src.model.architecture import MLP, Dropout
from src.model.helpers import create_dataset
from src.model.train import save_model, load_model

SIZE = 100000
SIZE_SMALL = 10

X_train, y_train, X_val, y_val = create_dataset("dataset/sample.csv")


class TestModel(unittest.TestCase):
    """Tests model architecture"""

    def test_architecture(self):
        """Tests that data passes through the model and is backpropagated correctly"""
        # Init
        x_np = np.random.rand(61)
        x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        model = MLP()
        torch_model = nn.Sequential(
            nn.Linear(61, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
        )

        # Copy weights from own model to PyTorch's
        with torch.no_grad():
            torch_model[0].weight.copy_(
                torch.from_numpy(model.dense_1.weights.T).float()
            )
            torch_model[0].bias.copy_(torch.from_numpy(model.dense_1.biases).float())
            torch_model[2].weight.copy_(
                torch.from_numpy(model.dense_2.weights.T).float()
            )
            torch_model[2].bias.copy_(torch.from_numpy(model.dense_2.biases).float())
            torch_model[4].weight.copy_(
                torch.from_numpy(model.dense_3.weights.T).float()
            )
            torch_model[4].bias.copy_(torch.from_numpy(model.dense_3.biases).float())

        # Forward
        start = time()
        own = model(x_np)
        end = time()
        print(f"\nOwn time: {end - start} s")

        start = time()
        correct = torch_model(x_torch)
        end = time()
        print(f"PyTorch time: {end - start} s")
        np.testing.assert_allclose(own, correct.detach().numpy(), rtol=1e-5, atol=1e-6)

        # Backward
        upstream = own.copy()
        start = time()
        own = model.backward(own)
        end = time()
        print(f"\nOwn time: {end - start} s")

        start = time()
        correct.backward(torch.tensor(upstream))
        end = time()
        print(f"PyTorch time: {end - start} s")
        np.testing.assert_allclose(own, x_torch.grad.numpy(), rtol=1e-5, atol=1e-6)

    def test_dropout(self):
        """Tests that dropout layer works correctly"""
        x = np.random.rand(100)
        drop = Dropout(0.3)
        zero_counts = [sum(drop(x, training=True) == 0.0) for _ in range(500)]
        self.assertAlmostEqual(np.mean(zero_counts), 30, delta=1)

    def test_error_checking(self):
        """Tests that invalid inputs and propagation orders are accounted for"""
        self.assertEqual("NOT DONE", "WIP")

    def test_serialization(self):
        """Tests that model parameters can be written to a JSON file
        and they are successfully loaded from said file"""
        model = MLP()
        original = model.to_dict()
        path = "test.json"

        save_model(model, path)
        load_model(model, path)
        os.remove(path)

        np.testing.assert_equal(original, model.to_dict())
