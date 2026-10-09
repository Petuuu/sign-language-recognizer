"""Tests for model training algorithm"""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from src.model.architecture import MLP
from src.model.helpers import create_dataset
from src.model.train import plot_training, train

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
            model, X_train, y_train, X_val, y_val, n_epochs=500, verbose=False
        )
        self.assertAlmostEqual(train_losses[-1], 0, places=3)
        self.assertAlmostEqual(train_accs[-1], 100)

    def test_gradients(self):
        """Tests that gradients are non-zero and loss decreases"""
        self.assertEqual("NOT DONE", "WIP")

    def test_layers_change(self):
        """Tests that all model layers change after each optimizer step"""
        self.assertEqual("NOT DONE", "WIP")

    def test_plotting(self):
        """Tests that training results are saved as a PNG or JPG
        plot and errors are handled"""
        training_output = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "training.png"
            plot_training(*training_output, str(output_path))

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

        # incorrect path
        incorrect_path = "incorrect/path.png"
        with self.assertRaisesRegex(
            FileNotFoundError,
            r"Output directory 'incorrect' does not exist\.",
        ):
            plot_training(*training_output, incorrect_path)

        # file not PNG or JPG
        with tempfile.TemporaryDirectory() as temp_dir:
            non_image_path = Path(temp_dir) / "non_image_path"
            with redirect_stdout(StringIO()) as output:
                plot_training(*training_output, str(non_image_path))
            self.assertEqual(output.getvalue(), "File must be PNG or JPG\n")
            self.assertFalse(non_image_path.exists())

        # file name too short
        with tempfile.TemporaryDirectory() as temp_dir:
            short_path = Path(temp_dir) / ".png"
            with redirect_stdout(StringIO()) as output:
                plot_training(*training_output, str(short_path))
            self.assertEqual(output.getvalue(), "File must be PNG or JPG\n")
            self.assertFalse(short_path.exists())

        # incorrect sized training metrics
        training_output = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        with self.assertRaisesRegex(
            ValueError,
            "All training and validation lists must have the same length.",
        ):
            plot_training(*training_output, "pretraining.png")
