import os

from PIL import Image
import numpy as np

from .kitti import KITTIDataset


class KITTIImprovedDataset(KITTIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_depth(
            self,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: bool) -> np.ndarray:
        depth_path = os.path.join(
            self.data_path,
            folder,
            'proj_depth/groundtruth',
            'image_0{}'.format(self.side_map[side]),
            '{:010d}.png'.format(frame_index))

        if not os.path.exists(depth_path):
            return None

        depth_gt = Image.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, Image.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt).copy()

        return depth_gt
