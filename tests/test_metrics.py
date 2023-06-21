import unittest

import numpy as np
import torch

from bse.utils.metric import \
    count_blindspot_confusion, \
    count_blindspot_zsubset_confusion, \
    compute_binary_metrics, \
    ignore_on_mask, \
    ignore_negative


class TestBSConfusion(unittest.TestCase):
    def test_count_blindspot_confusion(self):
        pred_bss = np.array([
            [20, 0, 20],
            [40, 0, 20],
            [20, 0, 40],
            [40, 0, 40],
            [20, 0, 60],
            [40, 0, 60]], dtype=np.float32)

        gt_bss = np.array([
            [20.5, 0, 20.1],
            [42, 0, 25],
            [20.1, 0, 40.1]], dtype=np.float32)

        result = count_blindspot_confusion(pred_bss, gt_bss, threshold=1)

        self.assertEqual(result["TP"], 2)
        self.assertEqual(result["FP"], 4)
        self.assertEqual(result["FN"], 1)

    def test_count_blindspot_zsubset_confusion(self):
        pred_bss = np.array([
            [20, 0, 20],
            [40, 0, 20],
            [20, 0, 40],
            [40, 0, 40],
            [20, 0, 60],
            [40, 0, 60]], dtype=np.float32)

        gt_bss = np.array([
            [20.5, 0, 20.1],
            [42, 0, 24.5],
            [20.1, 0, 40.1]], dtype=np.float32)

        ranges = {
            "all": (None, None),
            "short": (0, 30),
            "middle": (30, 60),
            "long": (60, 80)
        }

        result = count_blindspot_zsubset_confusion(
            pred_bss, gt_bss, ranges=ranges, threshold=1)

        self.assertEqual(result["all/TP"], 2)
        self.assertEqual(result["all/FP"], 4)
        self.assertEqual(result["all/FN"], 1)

        self.assertEqual(result["short/TP"], 1)
        self.assertEqual(result["short/FP"], 1)
        self.assertEqual(result["short/FN"], 1)

        self.assertEqual(result["middle/TP"], 1)
        self.assertEqual(result["middle/FP"], 1)
        self.assertEqual(result["middle/FN"], 0)

        self.assertTrue("long/TP" not in result)
        self.assertEqual(result["long/FP"], 2)
        self.assertTrue("long/FN" not in result)


class TestBSBinaryMetrics(unittest.TestCase):
    def test_case1(self):
        metrics = compute_binary_metrics(10, 20, 30)

        recall = 10 / (10 + 30)
        precision = 10 / (10 + 20)
        f1 = 2 * recall * precision / (recall + precision)

        self.assertTrue(np.isclose(metrics['recall'], recall))
        self.assertTrue(np.isclose(metrics['precision'], precision))
        self.assertTrue(np.isclose(metrics['f1'], f1))


class TestPointFiltering(unittest.TestCase):
    def test_ignore_nagative(self):
        x = torch.tensor([
            [0.5, 0.1],
            [0.2, 0.3],
            [-1, -1],
            [-1, -1]], dtype=torch.float32)

        y = ignore_negative(x)

        self.assertEqual(y.shape, (2, 2))
        self.assertEqual(y[0][0], 0.5)
        self.assertEqual(y[0][1], 0.1)
        self.assertEqual(y[1][0], 0.2)
        self.assertEqual(y[1][1], 0.3)

    def test_ignore_on_mask(self):
        x = torch.tensor([
            [0.5, 0.2],
            [0.1, 0.2],
            [0.2, 0.3]], dtype=torch.float32)

        mask = torch.zeros((1, 10, 10), dtype=torch.bool)
        mask[..., :4, :] = True

        y = ignore_on_mask(x, mask)
        self.assertEqual(y.shape, (1, 2))
        self.assertEqual(y[0][0], 0.5)
        self.assertEqual(y[0][1], 0.2)
