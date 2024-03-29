import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from bse.utils import \
    GradualWarmupScheduler, \
    seed_worker

from .runner import Runner


class Trainer(Runner):
    def __init__(
            self,
            cfg,
            datasets: dict,
            models: dict,
            criterions: dict,
            **kwargs):
        super().__init__(cfg, **kwargs)

        self.max_epochs = cfg.TRAIN.MAX_EPOCHS
        self.batch_size = cfg.DATA.BATCH_SIZE
        self.height = cfg.DATA.IMG_HEIGHT
        self.width = cfg.DATA.IMG_WIDTH
        assert self.height % 32 == 0
        assert self.width % 32 == 0

        self.epoch_fmt = '{:04}-{:05}'
        self.factory_args = {'dtype': torch.float32, 'device': self.device}

        self.grad_clip = cfg.TRAIN.GRAD_CLIP
        self.logging_iter = cfg.TRAIN.LOGGING_ITER
        self.figsave_iter = cfg.TRAIN.FIGSAVE_ITER
        self.limit_batches = cfg.TRAIN.LIMIT_BATCHES

        self.ref_metric_key = cfg.TRAIN.REFERENCE_METRIC
        self.best_epoch_threshold = self.max_epochs // 2

        self.dataloaders, self.samplers = self.build_dataloader(cfg, datasets)
        self.val_iter = iter(self.dataloaders['val'])
        self.iter_per_epoch = min(
            len(self.dataloaders['train']),
            self.limit_batches)

        self.eval_batches = min(
            len(self.dataloaders['val']),
            cfg.TRAIN.EVAL_BATCHES,
            self.limit_batches)

        self.models = models
        for key in self.models.keys():
            self.models[key].train()
            self.models[key].to(self.device)
            self.models[key] = self.convert_parallel_model(self.models[key])

        self.crits = criterions
        for key in self.crits.keys():
            self.crits[key].to(self.device)

        self.optim = self.build_optimizer(cfg, self.models)
        self.scheduler = self.build_scheduler(cfg, self.optim)

        self.is_evaluable = None
        self.is_train = False

    def build_optimizer(self, cfg, models: dict) -> torch.optim.Optimizer:
        raise NotImplementedError()

    def build_scheduler(self, cfg, optim):
        lr_milestones = []
        for epoch in cfg.TRAIN.LR_MILESTONES:  # Convert to iteration
            lr_milestones.append(epoch * self.iter_per_epoch)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=lr_milestones, gamma=0.1)

        self.warmup_iters = cfg.TRAIN.WARMUP_ITERS
        if self.warmup_iters > 0:
            scheduler = GradualWarmupScheduler(
                optim, multiplier=1, total_epoch=self.warmup_iters,
                after_scheduler=scheduler)

        return scheduler

    def build_dataloader(self, cfg, datasets: dict) -> dict:
        samplers = {
            'train': self.build_parallel_sampler(
                datasets['train'], drop_last=True,
                shuffle=not cfg.DATA.NO_SHUFFLE),
            'val': None,
        }

        loaders = {}
        loaders['train'] = DataLoader(
            datasets['train'], batch_size=cfg.DATA.BATCH_SIZE,
            shuffle=(not cfg.DATA.NO_SHUFFLE) and (not self.is_ddp),
            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker, sampler=samplers['train'])

        loaders['val'] = DataLoader(
            datasets['val'], batch_size=cfg.DATA.BATCH_SIZE,
            shuffle=not cfg.DATA.NO_SHUFFLE,
            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker, sampler=samplers['val'])

        return loaders, samplers

    def get_model_module(self, key: str) -> nn.Module:
        if self.is_parallel:
            return self.models[key].module
        return self.models[key]

    def get_float_epoch(self, epoch: int, iteration: int) -> float:
        if iteration == 0:
            return float(epoch)
        return epoch + float(iteration) / self.iter_per_epoch

    def entry(self):
        self._print('Model Profiles:')
        for key in self.models.keys():
            params = 0
            for p in self.models[key].parameters():
                if p.requires_grad:
                    params += p.numel()
            self._print('  {}: {:.2f} M'.format(key, params / 1e6))
            self.log_metric('params/' + key, params)

        self._print('Target Metric: ' + self.ref_metric_key)

        best_metric = 0.0
        best_epoch = 0
        for epoch in range(self.max_epochs):
            self.sync_process()
            self.epoch = epoch

            if self.is_ddp:
                for sampler in self.samplers.values():
                    if sampler is not None:
                        sampler.set_epoch(epoch)

            self.is_train = True
            self.run_train()

            # The evaluation is only performed in the main process
            # to get these full results.
            # if not self.is_main_process:
            #     continue

            self.is_train = False
            metrics = self.run_val()

            checkpoints = self.export_checkpoints()
            for name, checkpoint in checkpoints.items():
                checkpoint['epoch'] = epoch + 1

                self.log_state_dict(
                    checkpoint, os.path.join('final_checkpoint', name))

            ref_metric = metrics.get(self.ref_metric_key, 0)
            if ref_metric >= best_metric \
                    and epoch > self.best_epoch_threshold:
                best_metric = ref_metric
                best_epoch = epoch
                for name, checkpoint in checkpoints.items():
                    self.log_state_dict(
                        checkpoint, os.path.join('best_checkpoint', name))

        if best_metric > 0.0 and self.is_main_process:
            self._print('Saved best model (Epoch: {}, {}: {:.2f})'.format(
                best_epoch, self.ref_metric_key, best_metric))

    def process_batch(self, inputs: dict) -> dict:
        raise NotImplementedError()

    def is_available_gt(self, inputs) -> bool:
        raise NotImplementedError()

    def _is_available_gt(self, inputs) -> bool:
        if self.is_evaluable is None:
            self.is_evaluable = self.is_available_gt(inputs)
        return self.is_evaluable

    def is_calculable_loss(self, inputs) -> bool:
        return True

    def evaluate(self, inputs: dict, outputs: dict) -> dict:
        raise NotImplementedError()

    def integrate_metrics(self, metrics: dict):
        raise NotImplementedError()

    def load_label(self, inputs: dict, outputs: dict):
        pass

    @torch.no_grad()
    def run_val(self) -> dict:
        """
        Validate the model on a single minibatch
        """
        all_losses = defaultdict(list)
        all_metrics = defaultdict(list)
        with tqdm(
                range(self.eval_batches),
                desc='({}/{}) Val'.format(self.epoch + 1, self.max_epochs),
                leave=False, dynamic_ncols=True) as pbar:
            for i in pbar:
                try:
                    inputs = next(self.val_iter)
                except StopIteration:
                    self.val_iter = iter(self.dataloaders['val'])
                    inputs = next(self.val_iter)

                self.transfer(inputs)

                outputs = self.process_batch(inputs)

                if self.is_calculable_loss(inputs):
                    loss, losses = self.compute_loss(inputs, outputs)
                    for key, val in losses.items():
                        all_losses[key].append(val)

                if self._is_available_gt(inputs):
                    metrics = self.evaluate(inputs, outputs)
                    for key, val in metrics.items():
                        if isinstance(val, torch.Tensor):
                            val = float(val.detach().cpu())
                        all_metrics[key].append(val)

        iteration = self.iter_per_epoch * (self.epoch + 1)
        for key, vals in all_losses.items():
            val = np.array(vals).mean()
            self.log_metric('val/' + key, val, iteration)

        self.save_figure(
            inputs, outputs, os.path.join('val', '{:010}'.format(iteration)))

        out_metrics = {}
        if self._is_available_gt(inputs) and all_metrics:
            for key, val in all_metrics.items():
                all_metrics[key] = np.array(val)

            out_metrics = self.integrate_metrics(all_metrics)

            for key, val in out_metrics.items():
                self.log_metric(key, val, iteration)

        return out_metrics

    def run_train(self):
        with tqdm(
                self.dataloaders['train'],
                desc='({}/{}) Train'.format(self.epoch + 1, self.max_epochs),
                leave=False,
                dynamic_ncols=True) as pbar:
            for i, inputs in enumerate(pbar):
                iteration = self.iter_per_epoch * self.epoch + i + 1
                self.transfer(inputs)
                outputs = self.process_batch(inputs)

                loss, losses = self.compute_loss(inputs, outputs)
                self.add_dummy_graph_of_unused_params(loss)

                self.optim.zero_grad()
                loss.backward()

                if (self.epoch == 0 and i == 0) \
                        or (i + 1) % self.logging_iter == 0:
                    # Plot the gradient L2-norm of the model before clip.
                    for key, model in self.models.items():
                        total_norm = self.get_model_gradient_norm(model)
                        self.log_metric(
                            key + '_grad_L2', total_norm, iteration)

                self.clip_grad()
                self.optim.step()
                self.scheduler.step()

                scalar_losses = {}
                for key, val in losses.items():
                    scalar_losses[key] = val
                pbar.set_postfix(scalar_losses)

                if (self.epoch == 0 and i == 0) \
                        or (i + 1) % self.logging_iter == 0:

                    for key, val in scalar_losses.items():
                        self.log_metric('train/' + key, val, iteration)

                    for group_id, group in enumerate(self.optim.param_groups):
                        self.log_metric(
                            'lr{}'.format(group_id), group['lr'], iteration)

                if (i + 1) % self.figsave_iter == 0:
                    self.save_figure(
                        inputs, outputs,
                        os.path.join(
                            'train',
                            self.epoch_fmt.format(self.epoch + 1, i + 1)))

                if (i + 1) > self.limit_batches:
                    break

        iteration = self.iter_per_epoch * (self.epoch + 1)
        for key, val in scalar_losses.items():
            self.log_metric('train/' + key, val, iteration)

    def _log_losses_as_float(self, losses: dict) -> dict:
        out = {}
        for lkey, lval in losses.items():
            if isinstance(lval, torch.Tensor):
                out[lkey] = float(lval.mean().detach().item())
        return out

    def compute_loss(
            self,
            inputs: dict,
            outputs: dict) -> tuple:
        self.load_label(inputs, outputs)

        losses = {}
        total_loss = torch.tensor(0, **self.factory_args)
        for crit_name, crit in self.crits.items():
            _losses = crit(inputs, outputs)

            total_loss += _losses.get('loss', 0)

            logged_loss = self._log_losses_as_float(_losses)
            for key, val in logged_loss.items():
                losses[crit_name + '/' + key] = val

        return total_loss, losses

    def save_figure(self, inputs: dict, outputs: dict, basename: str):
        raise NotImplementedError()

    def clip_grad(self):
        if self.grad_clip > 0:
            for key in self.models.keys():
                torch.nn.utils.clip_grad_norm_(
                    self.models[key].parameters(), self.grad_clip)

    def export_checkpoints(self) -> dict:
        checkpoints = {}
        for name in self.models.keys():
            checkpoint = {
                'config': self.cfg,
                'optimizer': self.optim.state_dict(),
                'model': self.get_model_module(name).export_weight()
            }
            checkpoints['{}'.format(name)] = checkpoint

        return checkpoints

    @torch.no_grad()
    def get_model_gradient_norm(self, model: torch.nn.Module) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is None or not p.requires_grad:
                continue
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def add_dummy_graph_of_unused_params(self, loss):
        # Multiply the sum of all params with zero.and add it to the loss
        # issue: https://github.com/pytorch/pytorch/issues/43259
        dummy = 0
        if self.is_ddp:
            for model in self.models.values():
                for p in model.parameters():
                    if p.requires_grad:
                        dummy += (0 * p).sum()
        loss += dummy  # inplace
