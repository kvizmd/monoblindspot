import torch


class Exporter:
    """
    Export the generated labels of blind spots as a json format.
    """

    def create_label_object(
            self,
            img_filename: str,
            bs_filename: str,
            shape_data: dict,
            property_data: dict = {}):
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

        for key, val in property_data.items():
            if isinstance(val, torch.Tensor):
                val = val.tolist()
            obj[key] = val

        num = len(shape_data['points'])
        for i in range(num):
            shape_item = {
                'label': 'blindspot',
                'group_id': None,
                'shape_type': 'point',
                'flags': {},
            }
            for key, val in shape_data.items():
                shape_item[key] = [val[i].tolist()]

            obj['shapes'].append(shape_item)

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
            shape_data: dict,
            property_data: dict = {}) -> dict:
        img_filename = self.get_image_path(folder, frame_index, side)
        bs_filename = self.get_label_path(folder, frame_index, side)

        obj = self.create_label_object(
            img_filename, bs_filename, shape_data, property_data)
        return obj, bs_filename
