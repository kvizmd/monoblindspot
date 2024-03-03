import unittest

import torch

import bse
from bse.models.depth import DepthNet
from bse.models.pose import PoseNet
from bse.ogm_models import CascadeOGMIntegrator


class TestCascadeOGMIntegrator(unittest.TestCase):
    def test_constract(self):
        depth_net = DepthNet(18)
        pose_net = PoseNet(18)
        try:
            CascadeOGMIntegrator(
                1, 192, 640, [0, -2, -1, 1, 2, 3, 4],
                depth_net, pose_net)
        except Exception:
            self.fail()
