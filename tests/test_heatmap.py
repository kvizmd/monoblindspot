import unittest
import torch
import bse


class TestHeatmap(unittest.TestCase):
    def test_peak_extraction(self):
        indices = [
            [20, 50],
            [60, 99],
            [88, 2]
        ]
        values = [0.5, 0.2, 0.9]
        r = 10

        heatmap = torch.zeros((1, 1, 100, 100), dtype=torch.float32)
        for point, value in zip(indices, values):
            heatmap = bse.utils.create_heatmap(heatmap, point, r, value)

        result_values, result_indices = \
            bse.utils.extract_heatmap_peak(heatmap)

        self.assertEqual(len(result_values), 1)
        self.assertEqual(len(result_indices), 1)

        expected_values = torch.tensor(
            [0.5, 0.9], dtype=torch.float32)
        self.assertTrue(
            torch.all(result_values[0] == expected_values))

        expected_indices = torch.tensor(
            [[0, 20, 50], [0, 88, 2]], dtype=int)
        self.assertTrue(
            torch.all(result_indices[0] == expected_indices))
