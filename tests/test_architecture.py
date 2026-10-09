"""Tests for model"""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from time import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from src.model.architecture import MLP, Dropout, save_model, load_model
from src.model.helpers import create_dataset
from src.model.train import train

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
        np.testing.assert_allclose(
            own, torch.detach(correct).numpy(), rtol=1e-5, atol=1e-6
        )

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
        and they are successfully loaded from said file. Also tests error
        handling"""
        model = MLP()
        train(model, X_train, y_train, X_val, y_val, eval_freq=0, verbose=False)
        original = model.to_dict()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.json"
            save_model(model, str(path))
            load_model(model, str(path))

        np.testing.assert_equal(original, model.to_dict())

        # incorrect path
        incorrect_path = "incorrect/path.json"
        with redirect_stdout(StringIO()) as output:
            with self.assertRaisesRegex(
                FileNotFoundError,
                r"Output directory 'incorrect' does not exist\.",
            ):
                save_model(model, incorrect_path)
        self.assertEqual(output.getvalue(), "")

        with redirect_stdout(StringIO()) as output:
            with self.assertRaises(SystemExit):
                load_model(model, incorrect_path)
        self.assertEqual(output.getvalue(), "File does not exist. Exiting...\n")

        # file not JSON
        with tempfile.TemporaryDirectory() as temp_dir:
            non_json_path = Path(temp_dir) / "non_json_path"
            with redirect_stdout(StringIO()) as output:
                with self.assertRaises(SystemExit):
                    save_model(model, str(non_json_path))
            self.assertEqual(output.getvalue(), "File must be JSON. Exiting...\n")

            non_json_path.touch()
            with redirect_stdout(StringIO()) as output:
                with self.assertRaises(SystemExit):
                    load_model(model, str(non_json_path))
            self.assertEqual(output.getvalue(), "File must be JSON. Exiting...\n")

        # file name too short
        with tempfile.TemporaryDirectory() as temp_dir:
            short_path = Path(temp_dir) / ".json"
            with redirect_stdout(StringIO()) as output:
                with self.assertRaises(SystemExit):
                    save_model(model, str(short_path))
            self.assertEqual(output.getvalue(), "File must be JSON. Exiting...\n")

            short_path.touch()
            with redirect_stdout(StringIO()) as output:
                with self.assertRaises(SystemExit):
                    load_model(model, str(short_path))
            self.assertEqual(output.getvalue(), "File must be JSON. Exiting...\n")

        # invalid JSON contents
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / "invalid.json"
            invalid_path.write_text("not valid JSON", encoding="utf-8")
            with redirect_stdout(StringIO()) as output:
                with self.assertRaises(SystemExit):
                    load_model(model, str(invalid_path))
            self.assertEqual(
                output.getvalue(),
                f"File '{invalid_path}' could not be opened or incorrect content. Exiting...\n",
            )
