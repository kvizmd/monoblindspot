import os
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import mlflow

from bse.utils import seed_worker
from bse.ogm_models import OGMIntegrator
from bse.datasets import Exporter

from .runner import Runner
from .figure import make_bsgen_figure


class BlindSpotGenerator(Runner):
    def __init__(
            self,
            cfg,
            dataset: Dataset,
            integrator: OGMIntegrator,
            exporter: Exporter,
            **kwargs):
        super().__init__(cfg, **kwargs)

        self.max_epochs = cfg.TRAIN.MAX_EPOCHS
        self.batch_size = cfg.DATA.BATCH_SIZE
        self.figsave_iter = cfg.TRAIN.FIGSAVE_ITER

        self.exporter = exporter

        self.integrator = integrator
        self.integrator.to(self.device)
        self.integrator = self.convert_parallel_model(self.integrator)

        if len(dataset) % self.batch_size != 0:
            raise RuntimeError(
                'The batch size ({}) must be divisible by the number of '
                'data ({}).'.format(self.batch_size, len(dataset)))

        sampler = self.build_parallel_sampler(
            dataset, shuffle=False, drop_last=False)
        self.dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            pin_memory=True, drop_last=False, worker_init_fn=seed_worker,
            sampler=sampler)

    def entry(self):
        with tqdm(
                self.dataloader, desc='Exporting',
                leave=False, dynamic_ncols=True) as pbar:
            for i, inputs in enumerate(pbar):
                self.transfer(inputs)

                outputs = self.integrator(inputs)
                for key in list(outputs.keys()):
                    new_key = [key[0]] + [0] + list(key[1:])
                    outputs[tuple(new_key)] = outputs.pop(key)

                for b in range(self.batch_size):
                    folder, frame_idx, side = \
                        self.dataloader.dataset.parse_filename(
                            inputs['filename'][b])

                    property_data = {
                        'ogm2cam': outputs['T_norm2cam', 0, 0][b],
                    }
                    shape_data = {
                        'points': outputs['bs_point', 0, 0][b],
                        'scores': outputs['bs_confidence', 0, 0][b],
                        'cells': outputs['bs_peak_cell', 0, 0][b],
                    }
                    obj, out_filename = self.exporter(
                        folder, int(frame_idx), side,
                        shape_data, property_data)
                    mlflow.log_dict(obj, os.path.join('json', out_filename))

                if (i + 1) % self.figsave_iter == 0:
                    self.save_figure(
                        inputs, outputs,
                        os.path.join('figures', '{:05}'.format(i + 1)))

    def save_figure(self, inputs: dict, outputs: dict, basename: str):
        with make_bsgen_figure(inputs, outputs) as fig:
            mlflow.log_figure(fig, basename + '.jpg')
