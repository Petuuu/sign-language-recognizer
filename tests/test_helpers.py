"""Tests for model helper functions"""

import unittest
from time import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.model.helpers import (
    relu,
    softmax,
    cross_entropy,
    create_dataset,
    classify,
    adam,
)

SIZE = 100000
SIZE_SMALL = 10


class TestMethods(unittest.TestCase):
    """Tests mathematical functions"""

    def test_cross_entropy_small(self):
        """Tests loss function with small input"""
        logits = np.random.rand(SIZE_SMALL)
        label = np.random.randint(0, SIZE_SMALL)

        start = time()
        own, _ = cross_entropy(logits, label)
        end = time()
        print(f"\nOwn time: {end - start} s")

        start = time()
        correct = F.cross_entropy(torch.tensor(logits), torch.tensor(label))
        end = time()
        print(f"PyTorch time: {end - start} s")
        self.assertAlmostEqual(own, correct.item())

    def test_cross_entropy(self):
        """Tests loss function with large input"""
        logits = np.random.rand(SIZE)
        label = np.random.randint(0, SIZE)

        start = time()
        own, _ = cross_entropy(logits, label)
        end = time()
        print(f"\nOwn time: {end - start} s")

        start = time()
        correct = F.cross_entropy(torch.tensor(logits), torch.tensor(label))
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
        """Tests ReLU activation function with large input"""
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
        """Tests softmax activation fuction with small input"""
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

    def test_softmax(self):
        """Tests softmax activation fuction with large input"""
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

    def test_adam_small(self):
        """Tests Adam optimizer with small input"""
        grads = (
            np.random.rand(SIZE_SMALL),
            np.random.rand(SIZE_SMALL),
            np.random.rand(SIZE_SMALL),
        )
        values = np.random.rand(SIZE_SMALL)
        torch_param = nn.Parameter(torch.from_numpy(values.copy()))
        torch_optimizer = torch.optim.Adam(
            [torch_param], lr=0.01, betas=(0.9, 0.999), eps=1e-8
        )
        moment_1 = np.zeros_like(values)
        moment_2 = np.zeros_like(values)

        own_time = pytorch_time = 0

        for timestep, grad in enumerate(grads, start=1):
            start = time()
            update, moment_1, moment_2 = adam(
                grad, 0.9, 0.999, moment_1, moment_2, timestep
            )
            end = time()
            own_time += end - start

            torch_param.grad = torch.from_numpy(grad.copy())
            before = torch.detach(torch_param).numpy().copy()
            start = time()
            torch_optimizer.step()
            end = time()
            pytorch_time += end - start
            torch_update = before - torch.detach(torch_param).numpy()

            np.testing.assert_allclose(update * 0.01, torch_update)
            state = torch_optimizer.state[torch_param]
            np.testing.assert_allclose(moment_1, state["exp_avg"].numpy())
            np.testing.assert_allclose(moment_2, state["exp_avg_sq"].numpy())

        print(f"\nOwn time: {own_time} s")
        print(f"PyTorch time: {pytorch_time} s")

    def test_adam(self):
        """Tests Adam optimizer with large input"""
        grads = (
            np.random.rand(SIZE),
            np.random.rand(SIZE),
            np.random.rand(SIZE),
        )
        values = np.random.rand(SIZE)
        torch_param = nn.Parameter(torch.from_numpy(values.copy()))
        torch_optimizer = torch.optim.Adam(
            [torch_param], lr=0.01, betas=(0.9, 0.999), eps=1e-8
        )
        moment_1 = np.zeros_like(values)
        moment_2 = np.zeros_like(values)

        own_time = pytorch_time = 0

        for timestep, grad in enumerate(grads, start=1):
            start = time()
            update, moment_1, moment_2 = adam(
                grad, 0.9, 0.999, moment_1, moment_2, timestep
            )
            end = time()
            own_time += end - start

            torch_param.grad = torch.from_numpy(grad.copy())
            before = torch.detach(torch_param).numpy().copy()
            start = time()
            torch_optimizer.step()
            end = time()
            pytorch_time += end - start
            torch_update = before - torch.detach(torch_param).numpy()

            np.testing.assert_allclose(update * 0.01, torch_update)
            state = torch_optimizer.state[torch_param]
            np.testing.assert_allclose(moment_1, state["exp_avg"].numpy())
            np.testing.assert_allclose(moment_2, state["exp_avg_sq"].numpy())

        print(f"\nOwn time: {own_time} s")
        print(f"PyTorch time: {pytorch_time} s")

    def test_create_dataset(self):
        """Tests that dataset is prepared correctly"""
        with self.assertRaises(SystemExit):
            create_dataset("incorrect/path")

        X_train, y_train, X_val, y_val = create_dataset("dataset/sample.csv")

        self.assertEqual(X_train.shape, (16, 61))
        self.assertEqual(y_train.shape, (16,))
        self.assertEqual(X_val.shape, (5, 61))
        self.assertEqual(y_val.shape, (5,))

    def test_classification(self):
        """Tests that logits are correctly classified into labels"""
        x = np.array([10, 0.1, 0.1, 0.1])
        y = np.array([1, 1, 1, 1])

        self.assertEqual(classify(x)[1], 1)
        self.assertEqual(classify(y)[1], 0)
