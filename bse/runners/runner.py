import os
from datetime import timedelta

import torch
from torch import nn
from torch import distributed as dist
from torch.utils import data
import mlflow


class Runner:
    def __init__(self, cfg, rank=None):
        self.cfg = cfg

        self.is_ddp = rank is not None
        self.rank = rank if self.is_ddp else 0
        self.is_main_process = self.rank == 0
        self.n_gpu = torch.cuda.device_count()
        self.is_cuda_available = \
            self.n_gpu > 0 and 'cuda' in cfg.DEVICE.lower()
        self.is_parallel = \
            (self.is_ddp or cfg.MULTI_GPU) and self.is_cuda_available

        if self.is_ddp:
            self.rank = int(self.rank)
            dist.init_process_group(
                backend='nccl', rank=self.rank, world_size=self.n_gpu,
                timeout=timedelta(seconds=10000))
            self.device = self.rank
        else:
            self.device = cfg.DEVICE

    def __del__(self):
        if self.is_ddp:
            dist.destroy_process_group()

    def build_parallel_sampler(self, dataset: data.Dataset, **kwargs):
        if not self.is_parallel:
            return None

        if not self.is_ddp:
            return None

        return data.distributed.DistributedSampler(
            dataset, num_replicas=self.n_gpu, rank=self.rank,
            seed=self.cfg.RANDOM_SEED,
            **kwargs)

    def convert_parallel_model(self, model: nn.Module) -> nn.Module:
        if not self.is_parallel:
            return model

        if not self.is_ddp:
            return nn.DataParallel(model)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        return nn.parallel.DistributedDataParallel(
            model, device_ids=[self.rank])

    def log_params(self, cfg, parent_key: str = ''):
        for key, val in cfg.items():
            key = parent_key + key
            if isinstance(val, dict):
                self.log_params(val, key + '/')
            else:
                mlflow.log_param(key, val)

    def log_metric(self, *args, **kwargs):
        if self.is_main_process:
            mlflow.log_metric(*args, **kwargs)

    def log_state_dict(self, *args, **kwargs):
        if self.is_main_process:
            mlflow.pytorch.log_state_dict(*args, **kwargs)

    def _print(self, *args):
        if self.is_main_process:
            print(*args)

    def sync_process(self):
        if self.is_ddp:
            dist.barrier()

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

        with mlflow.start_run(run_name=run_name) as run:
            self.ml_run = run

            self.log_params(self.cfg)
            self.entry()

            self.ml_run = None

    def run_parallel(self, experiment_id: str, run_id: str):
        mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_run(
                run_id=run_id,
                experiment_id=experiment_id) as _:
            if self.is_main_process:
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
    os.environ['MASTER_PORT'] = '12323'

    return exp.experiment_id, run_id
