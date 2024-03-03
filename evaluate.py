from argparse import ArgumentParser

import bse


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--exp_tag', type=str,
        default='debug',
        help='Tag string for experiment')
    parser.add_argument(
        '--run_name', type=str,
        default=None,
        help='Run name string for experiment')
    parser.add_argument(
        '--config', type=str,
        default='configs/debug.yaml',
        help='Config file path')
    parser.add_argument(
        '--opts',
        type=str,
        nargs='*',
        default=[],
        help='Override yaml configs with the same way as detectron2')
    args = parser.parse_args()

    # Overrides settings used only during training.
    # It can be further overridden by using args.opts.
    override_opts = [
        'DATA.BATCH_SIZE', '1',
        'DATA.SCALES', '[0]',
        'TRAIN.BS.BS_LABEL', '',
        'TRAIN.BS.BS_LABELS', '[]',
    ]
    override_opts += args.opts

    cfg = bse.load_config(
        args.config, override_opts=override_opts,
        check_requirements=False)

    bse.utils.fix_random_state(cfg.RANDOM_SEED, cfg.BENCHMARK)

    mode = cfg.TARGET_MODE.lower()
    if mode == 'depth':
        assert cfg.MODEL.DEPTH.WEIGHT
        assert cfg.MODEL.POSE.WEIGHT
        dataset = bse.build_depth_dataset(cfg, keys=[cfg.EVAL.SET_NAME])
        model = bse.build_model(cfg)['depth']
        evaluator = bse.DepthEvaluator(cfg, dataset, model)

    elif mode == 'bs':
        assert cfg.MODEL.BS.WEIGHT
        dataset = bse.build_bs_dataset(cfg, keys=[cfg.EVAL.SET_NAME])
        model = bse.build_model(cfg)['bs']
        evaluator = bse.BlindSpotEvaluator(cfg, dataset, model)

    elif mode == 'bs_gen':
        assert cfg.MODEL.DEPTH.WEIGHT
        assert cfg.MODEL.POSE.WEIGHT
        dataset = bse.build_bsgen_dataset(cfg, keys=[cfg.EVAL.SET_NAME])
        models = bse.build_model(cfg)
        integrator = bse.build_integrator(cfg, models)
        evaluator = bse.BlindSpotEvaluator(cfg, dataset, integrator)

    evaluator.run(experiment_tag=args.exp_tag, run_name=args.run_name)


if __name__ == '__main__':
    main()
