"""Tests for model training algorithm"""

import unittest
from time import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.model.architecture import MLP
from src.model.train import train
from src.model.helpers import create_dataset

SIZE = 100000
SIZE_SMALL = 10


class TestTraining(unittest.TestCase):
    """Tests that model is correctly trained"""

    def test_propagation(self):
        """Tests that loss is propagated and the weight
        updates make the model improve by checking that the network
        is able to overfit"""
        model = MLP()
        X_train, y_train, X_val, y_val = create_dataset("dataset/sample.csv")
        train_losses, _, train_accs, _ = train(
            model, (X_train, y_train), (X_val, y_val)
        )
        self.assertAlmostEqual(train_losses[-1], 0)
        self.assertAlmostEqual(train_accs[-1], 100)

    def test_gradients(self):
        """Tests that gradients are non-zero and loss decreases"""
        self.assertEqual("NOT DONE", "WIP")

    def test_layers_change(self):
        """Tests that all model layers change after each optimizer step"""
        self.assertEqual("NOT DONE", "WIP")
