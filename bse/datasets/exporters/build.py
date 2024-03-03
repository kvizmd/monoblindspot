from .kitti import KITTIExporter
from .nusc import nuScenesExporter
from .cityscapes import CityscapesExporter
from .ddad import DDADExporter
from .ithaca import Ithaca365Exporter


def build_exporter(cfg) -> dict:
    return {
        'kitti': KITTIExporter,
        'nusc': nuScenesExporter,
        'cityscapes': CityscapesExporter,
        'ddad': DDADExporter,
        'ithaca': Ithaca365Exporter
    }[str(cfg.DATA.NAME).lower()]()
