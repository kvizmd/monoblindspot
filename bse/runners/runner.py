import os

import torch
from torch import nn
from torch.utils import data
import mlflow


class Runner:
    def __init__(self, cfg, rank=None):
        self.cfg = cfg

        self.rank = rank
        self.is_ddp = self.rank is not None
        self.n_gpu = torch.cuda.device_count()
        self.is_cuda_available = \
            self.n_gpu > 0 and 'cuda' in cfg.DEVICE.lower()
        self.is_parallel = \
            (self.is_ddp or cfg.MULTI_GPU) and self.is_cuda_available

        if self.is_ddp:
            self.rank = int(self.rank)
            torch.distributed.init_process_group(
                backend='nccl', rank=self.rank, world_size=self.n_gpu)
            self.device = self.rank
        else:
            self.device = cfg.DEVICE

    def build_parallel_sampler(self, dataset: data.Dataset, **kwargs):
        if not self.is_parallel:
            return None

        if not self.is_ddp:
            return None

        return data.distributed.DistributedSampler(
            dataset, rank=self.rank,
            seed=self.cfg.RANDOM_SEED,
            **kwargs)

    def convert_parallel_model(self, model: nn.Module) -> nn.Module:
        if not self.is_parallel:
            return model

        if not self.is_ddp:
            return nn.DataParallel(model)

        return nn.parallel.DistributedDataParallel(
            model, device_ids=[self.rank])

    def log_params(self, cfg, parent_key: str = ''):
        for key, val in cfg.items():
            key = parent_key + key
            if isinstance(val, dict):
                self.log_params(val, key + '/')
            else:
                mlflow.log_param(key, val)

    def transfer(self, inputs: dict):
        for key in inputs.keys():
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device, non_blocking=True)

    def run(
            self,
            experiment_tag: str = None,
            run_name: str = None):
        if experiment_tag is not None:
            mlflow.set_experiment(experiment_name=experiment_tag)

        with mlflow.start_run(run_name=run_name) as _:
            self.log_params(self.cfg)
            self.entry()

    def run_parallel(self, experiment_id: str, run_id: str):
        mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_run(
                run_id=run_id,
                experiment_id=experiment_id) as _:
            if self.rank == 0:
                self.log_params(self.cfg)
            self.entry()

    def entry(self):
        raise NotImplementedError()


def create_parallel_run(experiment_tag: str, run_name: str) -> tuple:

    exp = mlflow.set_experiment(experiment_name=experiment_tag)

    run_id = None
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11111'

    return exp.experiment_id, run_id
