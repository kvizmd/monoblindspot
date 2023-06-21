import torch
import mlflow


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DEVICE

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
            mlflow.set_experiment(experiment_tag)

        with mlflow.start_run(run_name=run_name) as _:
            self.log_params(self.cfg)
            self.entry()

    def entry(self):
        raise NotImplementedError()
