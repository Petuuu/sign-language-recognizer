"""Tests for image pipeline"""

import unittest
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from src.image_pipeline import normalize_landmarks


class TestPipeline(unittest.TestCase):
    """Tests image pipeline"""

    def test_normalization(self):
        """Tests that landmarks are normalized correctly"""
        landmarks = [NormalizedLandmark(i, i, i) for i in range(21)]
        normalized = normalize_landmarks(landmarks)

        correct = np.array([[i, i, i] for i in range(1, 21)], dtype=np.float64)
        correct /= np.sqrt(3 * 9**2)
        correct = correct.flatten().tolist()
        np.testing.assert_equal(normalized, correct)
