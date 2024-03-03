import os

from .exporter import Exporter


class CityscapesExporter(Exporter):
    def __init__(self, img_ext: str = '.jpg'):
        super().__init__()
        self.img_ext = img_ext

    def get_image_path(
            self,
            folder: str,
            frame_index: str,
            side: str) -> str:
        city_snippets = folder.split('_')
        city = city_snippets[0]
        snippets_idx = int(city_snippets[1])

        filename = os.path.join(
            city, '{}_{:06}_{:06}_leftImg8bit{}'.format(
                city, snippets_idx, frame_index, self.img_ext))
        return filename

    def get_label_path(
            self,
            folder: str,
            frame_index: str,
            side: str) -> str:
        city_snippets = folder.split('_')
        city = city_snippets[0]
        snippets_idx = int(city_snippets[1])

        filename = os.path.join(
            city, '{}_{:06}_{:06}_bs.json'.format(
                city, snippets_idx, frame_index))

        return filename
