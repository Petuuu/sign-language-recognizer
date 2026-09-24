"""Tests for model"""

import unittest
from time import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.model import MLP, relu, softmax
from src.train import binary_cross_entropy, train

SIZE = 10000000
SIZE_SMALL = 10


class TestModel(unittest.TestCase):
    """Tests model architecture"""

    def test_propagation(self):
        """Tests that data is passed through the model, loss is propagated, and the weight
        updates makes the model improve"""
        model = MLP()
        train_losses, _, train_accs, _, _ = train(model)
        self.assertAlmostEqual(train_losses[-1], 0)
        self.assertEqual(train_accs[-1], 100)

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
