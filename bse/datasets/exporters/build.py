from .kitti import KITTIExporter
from .nusc import nuScenesExporter


def build_exporter(cfg) -> dict:
    return {
        'kitti': KITTIExporter,
        'nusc': nuScenesExporter,
    }[str(cfg.DATA.NAME).lower()]()
