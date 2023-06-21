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
        '--config',
        type=str,
        required=True,
        help='Config file path')
    parser.add_argument(
        '--opts',
        type=str,
        nargs='*',
        help='Override yaml configs with the same way as detectron2')
    args = parser.parse_args()

    cfg = bse.load_config(args.config,  override_opts=args.opts)

    bse.fix_random_state(cfg.RANDOM_SEED, cfg.BENCHMARK)

    mode = cfg.TARGET_MODE.lower()
    if mode == 'depth':
        datasets = bse.build_depth_dataset(cfg)
        models = bse.build_model(cfg)
        crits = bse.build_criterion(cfg)
        trainer = bse.DepthTrainer(cfg, datasets, models, crits)

    elif mode == 'bs':
        datasets = bse.build_bs_dataset(cfg)
        models = bse.build_model(cfg)
        crits = bse.build_criterion(cfg)
        trainer = bse.BlindSpot2DTrainer(cfg, datasets, models, crits)

    elif mode == 'bs_gen':
        dataset = bse.build_bsgen_dataset(cfg, keys=['train'])
        models = bse.build_model(cfg)
        integrator = bse.build_integrator(cfg, models)
        exporter = bse.build_exporter(cfg)
        trainer = bse.BlindSpotGenerator(cfg, dataset, integrator, exporter)

    trainer.run(experiment_tag=args.exp_tag, run_name=args.run_name)


if __name__ == '__main__':
    main()
