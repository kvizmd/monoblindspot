from .depth import DepthCriterion
from .bs_bipart import BipartCriterion
from .bs_heatmap import HeatmapCriterion


def build_criterion(cfg) -> dict:
    crits = {}

    if cfg.LOSS.DEPTH.FACTOR > 0:
        crits['depth'] = DepthCriterion(
            cfg.LOSS.DEPTH.FRAME_IDXS,
            cfg.LOSS.DEPTH.SCALES,
            factor=cfg.LOSS.DEPTH.FACTOR,
            smooth_scaling=cfg.LOSS.DEPTH.SMOOTH_SCALING,
            smooth_factor=cfg.LOSS.DEPTH.SMOOTH_FACTOR)

    if cfg.LOSS.BS.FACTOR > 0:
        bs_crit = {
            'bipart': BipartCriterion,
            'heatmap': HeatmapCriterion,
        }[cfg.LOSS.BS.NAME.lower()]

        crits['bs'] = bs_crit(
            factor=cfg.LOSS.BS.FACTOR,
            cls_factor=cfg.LOSS.BS.CLS_FACTOR,
            pos_2d_factor=cfg.LOSS.BS.POS2D_FACTOR,
            offset_factor=cfg.LOSS.BS.OFFSET_FACTOR,
            score_match=cfg.LOSS.BS.SCORE_MATCH,
            pos_match=cfg.LOSS.BS.POS_MATCH,
            heatmap_radius=cfg.LOSS.BS.HEATMAP_RADIUS,
            occ_weighting=cfg.LOSS.BS.OCC_WEIGHTING)

    return crits
