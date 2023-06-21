import time
from collections import defaultdict

from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import mlflow

from bse.utils import seed_worker
from .runner import Runner


class Evaluator(Runner):
    def __init__(
            self,
            cfg,
            dataset: Dataset,
            model: nn.Module):
        super().__init__(cfg)

        self.batch_size = cfg.DATA.BATCH_SIZE

        assert len(dataset) % self.batch_size == 0
        self.dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            pin_memory=True, drop_last=False, worker_init_fn=seed_worker)

        self.model = model
        self.model.eval()
        self.model.to(self.device)

        self.runtime_warmup = 20

    def predict(self, inputs: dict) -> dict:
        raise NotImplementedError()

    def evaluate(
            self,
            inputs: dict,
            outputs: dict,
            iter_idx: dict) -> dict:
        raise NotImplementedError()

    def integrate(self, all_sample_metrics: dict) -> dict:
        raise NotImplementedError()

    @torch.inference_mode()
    def entry(self) -> dict:
        all_metrics = defaultdict(list)
        infer_sec = []
        with tqdm(
                self.dataloader,
                desc='Testing',
                leave=False,
                dynamic_ncols=True) as pbar:
            for i, inputs in enumerate(pbar):
                self.transfer(inputs)
                if i > self.runtime_warmup:
                    torch.cuda.synchronize()
                    start = time.time()

                with torch.inference_mode():
                    outputs = self.predict(inputs)

                for key in list(outputs.keys()):
                    new_key = [key[0]] + [0] + list(key[1:])
                    outputs[tuple(new_key)] = outputs.pop(key)

                if i > self.runtime_warmup:
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - start
                    infer_sec.append(elapsed_time)

                results = self.evaluate(inputs, outputs, i)
                for key, val in results.items():
                    if isinstance(val, torch.Tensor):
                        val = float(val.detach().cpu().items())
                    else:
                        val = float(val)

                    all_metrics[key].append(val)
                    mlflow.log_metric('frame/' + key, val, i)

                self.save_figure(inputs, outputs, '{:05}'.format(i))

        for key, vals in all_metrics.items():
            all_metrics[key] = np.array(vals)

        infer_sec = np.median(np.array(infer_sec))
        infer_fps = 1 / infer_sec
        infer_msec = infer_sec * 1e3
        print('Time: {:.2f} fps, {:.4f} ms'.format(infer_fps, infer_msec))
        mlflow.log_metric('runtime/fps', infer_fps)
        mlflow.log_metric('runtime/ms', infer_msec)

        all_metrics = self.integrate(all_metrics)
        for key, val in all_metrics.items():
            mlflow.log_metric(key, val)
        return all_metrics

    def save_figure(self, inputs: dict, outputs: dict, basename: str):
        raise NotImplementedError()
