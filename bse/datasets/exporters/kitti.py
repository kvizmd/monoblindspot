import os

from .exporter import Exporter


class KITTIExporter(Exporter):
    def __init__(self, img_ext: str = '.jpg'):
        super().__init__()
        self.img_ext = img_ext
        self.side_map = {'2': 2, '3': 3, 'l': 2, 'r': 3}

    def get_image_path(
            self,
            folder: str,
            frame_index: str,
            side: str) -> str:
        return os.path.join(
            folder,
            'image_0{}'.format(self.side_map[side]),
            'data',
            '{:010d}{}'.format(int(frame_index), self.img_ext))

    def get_label_path(
            self,
            folder: str,
            frame_index: str,
            side: str) -> str:
        return os.path.join(
            folder,
            'image_0{}'.format(self.side_map[side]),
            'bs',
            '{:010d}.json'.format(int(frame_index)))
