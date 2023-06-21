import unittest

import torch

from bse.utils.depth import disp_to_depth, depth_to_disp


class TestDepthConversion(unittest.TestCase):
    def test_consistency(self):
        d = torch.full((100, 10), 0.2, dtype=torch.float32)
        max_depth = 100
        min_depth = 0.1
        _, D = disp_to_depth(d, min_depth, max_depth)
        d_re = depth_to_disp(D, min_depth, max_depth)

        self.assertTrue(torch.allclose(d, d_re))

        D = torch.full((100, 10), 49, dtype=torch.float32)
        d = depth_to_disp(D, min_depth, max_depth)
        _, D_re = disp_to_depth(d, min_depth, max_depth)

        self.assertTrue(torch.allclose(D, D_re))
