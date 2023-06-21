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

    points = defaultdict(list)

    shapes = json_obj['shapes']
    for shape in shapes:
        label = shape['label']
        if shape['shape_type'] != 'point' or label != 'blindspot':
            continue

        point = np.array(shape['points'], dtype=np.float32)
        score = np.array(shape['scores'], dtype=np.float32)

        point = point.flatten()
        score = score.flatten()

        augment_point = np.concatenate((point, score))
        points[label].append(augment_point)

    return points
