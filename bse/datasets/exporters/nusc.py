import os

from .exporter import Exporter


class nuScenesExporter(Exporter):
    def get_image_path(
            self,
            folder: str,
            frame_index: str,
            side: str) -> str:
        return '{}/{}'.format(folder, frame_index)

    def get_label_path(
            self,
            folder: str,
            frame_index: str,
            side: str) -> str:
        return os.path.join(
            folder, '{:010d}.json'.format(int(frame_index)))
