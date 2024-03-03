from .ogm import OGMIntegrator
from .ambiguous import AmbiguousOGMIntegrator
from .cascade import CascadeOGMIntegrator
from .oracle_scale import OracleScaleOGMIntegrator
from .oracle_scale_oxts import OracleScaleOxtsOGMIntegrator
from .sequential import SequentialOGMIntegrator


def build_integrator(cfg, models) -> OGMIntegrator:
    integrator_class = {
        'cascade': CascadeOGMIntegrator,
        'sequential': SequentialOGMIntegrator,
        'oracle_scale': OracleScaleOGMIntegrator,
        'oracle_scale_oxts': OracleScaleOxtsOGMIntegrator,
        'ambiguous': AmbiguousOGMIntegrator
    }[str(cfg.OGM.NAME).lower()]

    return integrator_class(
        cfg.DATA.BATCH_SIZE,
        cfg.DATA.IMG_HEIGHT,
        cfg.DATA.IMG_WIDTH,
        cfg.DATA.FRAME_IDXS,
        models['depth'],
        models['pose'],
        min_depth=cfg.MODEL.DEPTH.MIN_DEPTH,
        max_depth=cfg.MODEL.DEPTH.MAX_DEPTH,
        ogm_size=cfg.OGM.SIZE,
        ogm_num_subgrids=cfg.OGM.SUBGRIDS,
        ogm_median_scale=cfg.OGM.MEDIAN_SCALE,
        ogm_ransac_iterations=cfg.OGM.RANSAC_ITERATIONS,
        ogm_invsnr_grad_thr=cfg.OGM.DEPTH_GRAD_THRESHOLD,
        ogm_invsnr_count_thr=cfg.OGM.COUNT_THRESHOLD,
        ogm_invsnr_min_prob=cfg.OGM.MIN_PRIOR_PROB,
        ogm_invsnr_max_prob=cfg.OGM.MAX_PRIOR_PROB,
        ogm_invsnr_height_quantile=cfg.OGM.HEIGHT_QUANTILE,
        ogm_update_prior_prob=cfg.OGM.UNKNOWN_PRIOR_PROB,
        ogm_project_prob_thr=cfg.OGM.THRESHOLD,
        ogm_project_pool_kernel=cfg.OGM.POOLING_KERNEL_SIZE,
        ogm_sampling_mask_acceptable_offset=cfg.OGM.SAMPLING_ACCEPT_OFFSET)
