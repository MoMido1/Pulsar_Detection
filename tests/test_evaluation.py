import tempfile
import unittest
from pathlib import Path

import numpy as np

from main import load_dataset, min_dcf, normalized_dcf


class EvaluationMetricTests(unittest.TestCase):
    def test_min_dcf_is_zero_for_separable_scores(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([-2.0, -1.0, 1.0, 2.0])
        self.assertAlmostEqual(min_dcf(labels, scores, prior=0.5), 0.0)

    def test_normalized_dcf_uses_supplied_threshold(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([-2.0, 2.0, -1.0, 3.0])
        # One false positive and one false negative: normalized DCF is 0.5 at prior 0.5.
        self.assertAlmostEqual(normalized_dcf(labels, scores, prior=0.5, threshold=0.0), 0.5)

    def test_loader_reads_features_and_last_column_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.csv"
            path.write_text("1.0,2.0,0\n3.0,4.0,1\n", encoding="utf-8")
            X, y = load_dataset(path)
        np.testing.assert_array_equal(X, np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(y, np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
