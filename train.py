from argparse import ArgumentParser

import torch

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
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run in the DistributedDataParallel')
    args = parser.parse_args()

    cfg = bse.load_config(args.config,  override_opts=args.opts)

    bse.fix_random_state(cfg.RANDOM_SEED, cfg.BENCHMARK)

    if args.parallel:
        exp_id, run_id = bse.create_parallel_run(args.exp_tag, args.run_name)
        torch.multiprocessing.spawn(
            run_parallel, args=(cfg, exp_id, run_id),
            nprocs=torch.cuda.device_count(), join=True)
    else:
        trainer = setup(cfg)
        trainer.run(experiment_tag=args.exp_tag, run_name=args.run_name)


def setup(cfg, rank=None):
    mode = cfg.TARGET_MODE.lower()
    if mode == 'depth':
        datasets = bse.build_depth_dataset(cfg)
        models = bse.build_model(cfg)
        crits = bse.build_criterion(cfg)
        trainer = bse.DepthTrainer(cfg, datasets, models, crits, rank=rank)

    elif mode == 'bs':
        datasets = bse.build_bs_dataset(cfg)
        models = bse.build_model(cfg)
        crits = bse.build_criterion(cfg)
        trainer = bse.BlindSpot2DTrainer(
            cfg, datasets, models, crits, rank=rank)

    elif mode == 'bs_gen':
        dataset = bse.build_bsgen_dataset(cfg, keys=['train'])
        models = bse.build_model(cfg)
        integrator = bse.build_integrator(cfg, models)
        exporter = bse.build_exporter(cfg)

        trainer = bse.BlindSpotGenerator(
            cfg, dataset, integrator, exporter, rank=rank)

    return trainer


def run_parallel(rank, cfg, exp_id, run_id):
    trainer = setup(cfg, rank=rank)
    trainer.run_parallel(exp_id, run_id)


if __name__ == '__main__':
    main()
