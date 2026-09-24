"""Tests for model"""

import unittest
from time import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.model import MLP, Dropout, relu, softmax
from src.train import binary_cross_entropy, train

SIZE = 10000000
SIZE_SMALL = 10


class TestModel(unittest.TestCase):
    """Tests model architecture"""

    def test_architecture(self):
        """Tests that data passes through the model correctly"""
        x = torch.rand(63)
        model = MLP()
        torch_model = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 22),
        )

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

        start = time()
        own = model(x.numpy())
        end = time()
        print(f"\nOwn time: {end - start} s")

        start = time()
        with torch.no_grad():
            correct = torch_model(x)
        end = time()
        print(f"PyTorch time: {end - start} s")
        np.testing.assert_allclose(own, correct.numpy(), rtol=1e-5, atol=1e-6)

    def test_dropout(self):
        """Tests that dropout layer works correctly"""
        x = np.random.rand(100)
        drop = Dropout(0.3)
        zero_counts = [sum(drop(x, training=True) == 0.0) for _ in range(500)]
        self.assertAlmostEqual(np.mean(zero_counts), 30, delta=1)

    def test_propagation(self):
        """Tests that loss is propagated and the weight
        updates makes the model improve"""
        # model = MLP()
        # train_losses, _, train_accs, _, _ = train(model)
        # self.assertAlmostEqual(train_losses[-1], 0)
        # self.assertEqual(train_accs[-1], 100)

    def test_gradients(self):
        """Tests that gradients are non-zero and loss decreases"""

    def test_layers_change(self):
        """Tests that all model layers change after each optimizer step"""


class TestMethods(unittest.TestCase):
    """Tests mathematical functions"""

    def test_binary_cross_entropy_small(self):
        """Tests loss function with small input"""
        inputi, target = np.random.rand(SIZE_SMALL), np.random.rand(SIZE_SMALL)

        start = time()
        own = binary_cross_entropy(inputi, target)
        end = time()
        print(f"\nOwn time: {end - start} s")

        start = time()
        correct = F.binary_cross_entropy(torch.tensor(inputi), torch.tensor(target))
        end = time()
        print(f"PyTorch time: {end - start} s")
        self.assertAlmostEqual(own, correct.item())

    def test_binary_cross_entropy(self):
        """Tests loss function with large input"""
        inputi, target = np.random.rand(SIZE), np.random.rand(SIZE)

        start = time()
        own = binary_cross_entropy(inputi, target)
        end = time()
        print(f"\nOwn time: {end - start} s")

        start = time()
        correct = F.binary_cross_entropy(torch.tensor(inputi), torch.tensor(target))
        end = time()
        print(f"PyTorch time: {end - start} s")
        self.assertAlmostEqual(own, correct.item())

    def test_relu_small(self):
        """Tests ReLU activation function with small input"""
        x = np.random.rand(SIZE_SMALL) - 1 / 2

        start = time()
        own = relu(x)
        end = time()
        print(f"\nOwn time: {end - start} s")

        start = time()
        correct = nn.ReLU()(torch.tensor(x))
        end = time()
        print(f"PyTorch time: {end - start} s")
        np.testing.assert_allclose(own, correct.numpy())

    def test_relu(self):
        """Tests ReLU activatio function with large input"""
        x = np.random.rand(SIZE) - 1 / 2

        start = time()
        own = relu(x)
        end = time()
        print(f"\nOwn time: {end - start} s")

        start = time()
        correct = nn.ReLU()(torch.tensor(x))
        end = time()
        print(f"PyTorch time: {end - start} s")
        np.testing.assert_allclose(own, correct.numpy())

    def test_softmax_small(self):
        """Test Softmax activation fuction with small input"""
        x = np.random.rand(SIZE_SMALL)

        start = time()
        own = softmax(x)
        end = time()
        print(f"\nOwn time: {end - start} s")

        start = time()
        correct = nn.Softmax(dim=-1)(torch.tensor(x))
        end = time()
        print(f"PyTorch time: {end - start} s")
        np.testing.assert_allclose(own, correct.numpy())

    def test_softmax_small(self):
        """Test Softmax activation fuction with large input"""
        x = np.random.rand(SIZE)

        start = time()
        own = softmax(x)
        end = time()
        print(f"\nOwn time: {end - start} s")

        start = time()
        correct = nn.Softmax(dim=-1)(torch.tensor(x))
        end = time()
        print(f"PyTorch time: {end - start} s")
        np.testing.assert_allclose(own, correct.numpy())
