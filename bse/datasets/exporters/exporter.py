import torch


class Exporter:
    """
    Export the generated labels of blind spots as a json format.
    """

    def create_label_object(
            self,
            img_filename: str,
            bs_filename: str,
            bs_points: torch.Tensor,
            bs_confidence: torch.Tensor):
        # labelme style
        obj = {
            'version': '5.1.1',
            'shapes': [],
            'flags': {'gtgen': True},
            'imagePath': img_filename,
            'imageHeight': None,
            'imageWidth': None,
            'imageData': None
        }

        if len(bs_points) > 0:
            points = bs_points.tolist()
            scores = bs_confidence.tolist()

            for p, s in zip(points, scores):
                obj['shapes'].append({
                    'label': 'blindspot',
                    'points': [p],
                    'group_id': None,
                    'shape_type': 'point',
                    'flags': {},
                    'scores': [s]})

        return obj

    def get_image_path(
            self,
            folder: str,
            frame_index: str,
            side: str) -> str:
        raise NotImplementedError()

    def get_label_path(
            self,
            folder: str,
            frame_index: str,
            side: str) -> str:
        raise NotImplementedError()

    def __call__(
            self,
            folder: str,
            frame_index: str,
            side: str,
            bs_points: torch.Tensor,
            bs_confidence: torch.Tensor) -> dict:

        img_filename = self.get_image_path(folder, frame_index, side)
        bs_filename = self.get_label_path(folder, frame_index, side)

        obj = self.create_label_object(
            img_filename, bs_filename, bs_points, bs_confidence)

        return obj, bs_filename
