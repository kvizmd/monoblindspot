import json
import numpy as np
from collections import defaultdict


def load_annotated_blindspot(json_obj: dict) -> dict:
    """
    Load annotations of blind spots based on labelme formats.
    """
    height = json_obj['imageHeight']
    width = json_obj['imageWidth']
    shapes = json_obj['shapes']

    buf = defaultdict(list)
    for shape in shapes:
        label = shape['label']
        shape_t = shape['shape_type']

        points = np.array(shape['points'], dtype=np.float32)
        points[:, 0] /= width - 1
        points[:, 1] /= height - 1
        points = np.stack((points[:, 1], points[:, 0]), axis=-1)

        if shape_t == 'point' and label == 'blindspot':
            buf[label].append(points[0])

        elif shape_t == 'rectangle' and label == 'ambiguous':
            points = points[0:2]
            left = points[:, 1].min()
            right = points[:, 1].max()
            top = points[:, 0].min()
            bottom = points[:, 0].max()

            points = np.stack((left, top, right, bottom))
            buf[label].append({'type': 'box', 'points': points})

        elif shape_t == 'polygon' and label == 'ambiguous':
            buf[label].append({'type': 'polygon', 'points': points})

    return buf


def load_generated_blindspot(json_obj: dict) -> dict:
    """
    Load exported labels of blind spots
    """

    data = defaultdict(list)

    T_ogm2cam = np.array(json_obj['ogm2cam'], dtype=np.float32).reshape(4, 4)
    data['ogm2cam'].append(T_ogm2cam)

    shapes = json_obj['shapes']
    for shape in shapes:
        label = shape['label']
        if shape['shape_type'] != 'point' or label != 'blindspot':
            continue

        shape_data = []

        # [0, 1]
        point = np.array(shape['points'], dtype=np.float32)
        point = point.flatten()
        shape_data.append(point)

        # [2]
        score = np.array(shape['scores'], dtype=np.float32)
        score = score.flatten()
        shape_data.append(score)

        # [3, 4, 5]
        cell = np.array(shape['cells'], dtype=np.float32)
        cell = cell.flatten()
        shape_data.append(cell)

        data[label].append(np.concatenate(shape_data))

    return data


class BlindSpotLabelLoader:
    def __init__(self):
        super().__init__()
        self.caches = {}
        self.bs_nums = []
        self.bs_dims = []

    def clear_caches(self):
        self.caches.clear()
        self.bs_nums.clear()
        self.bs_dims.clear()

    def __getitem__(self, index):
        return self.caches[index]

    def load_labels(self, json_filename, cache_key):
        with open(json_filename, 'r') as f:
            obj = json.load(f)

        if obj['flags'].get('gtgen', False):
            points = load_generated_blindspot(obj)
        else:
            points = load_annotated_blindspot(obj)

        if 'blindspot' in points:
            bs_pt = np.stack(points['blindspot'], axis=0)
            self.bs_nums.append(len(bs_pt))
            self.bs_dims.append(bs_pt.shape[-1])
            points['blindspot'] = bs_pt

        self.caches[cache_key] = points

    def concat_labels(self, num_max_instances: int = None):
        if num_max_instances is None:
            num_max_instances = max(self.bs_nums)

        # Expand points to build mini-batch
        for key, val in self.caches.items():
            expand = np.full(
                (num_max_instances, max(self.bs_dims)), -1, dtype=np.float32)
            if 'blindspot' in val:
                pts = val['blindspot']
                expand[:pts.shape[0]] = pts
            self.caches[key]['blindspot'] = expand
