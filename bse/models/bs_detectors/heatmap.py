import torch
from torch import nn

from bse.utils import extract_heatmap_peak

from bse.models.layers import ReLUBlock, XavierConv, FocalConv


class HeatmapDetector(nn.Module):
    """
    Heatmap Detector
    """

    def __init__(
            self,
            num_ch_enc: list,
            head_channels=256,
            inst_num=32,
            prior_prob=0.01,
            score_threshold=0.01,
            **kwargs):
        super().__init__()

        in_channels = num_ch_enc[-1]

        self.hm_head = nn.Sequential(
            ReLUBlock(in_channels, head_channels),
            FocalConv(head_channels, 1, prior_prob=prior_prob),
            nn.Sigmoid())

        self.offset_head = nn.Sequential(
            ReLUBlock(in_channels, head_channels),
            XavierConv(head_channels, 2),
            nn.Sigmoid())

        self.inst_num = inst_num
        self.score_threshold = score_threshold

    def forward(self, features, postprocess=True):
        outputs = {}
        if isinstance(features, (list, tuple)):
            ft = features[-1]
        else:
            ft = features

        outputs['pred_hm', 0] = self.hm_head(ft)
        outputs['pred_offset', 0] = self.offset_head(ft)

        if postprocess:
            self.postprocess_nms(outputs)

        return outputs

    def postprocess_nms(self, outputs):
        pred_hm = outputs['pred_hm', 0]
        factory_args = {'dtype': pred_hm.dtype, 'device': pred_hm.device}

        scores, indices = extract_heatmap_peak(
            pred_hm, threshold=self.score_threshold)

        peak_scores = []
        peak_indices = []
        for b, (score, indice) in enumerate(zip(scores, indices)):
            peak_scr = torch.zeros((self.inst_num,), **factory_args)
            peak_idx = torch.zeros((self.inst_num, 2), **factory_args)

            num = min(self.inst_num, len(score))
            if num > 0:
                top_scr, top_idx = torch.topk(score, num)
                peak_idx[:num] = indice[top_idx, -2:]
                peak_scr[:num] = top_scr

            peak_scores.append(peak_scr)
            peak_indices.append(peak_idx)

        outputs['bs_confidence', 0] = torch.stack(peak_scores)

        peak_indices = torch.stack(peak_indices)

        H, W = pred_hm.shape[-2:]
        sampling_indices = peak_indices[..., 0] * W + peak_indices[..., 1]
        sampling_indices = sampling_indices.view(-1, 1, self.inst_num)
        sampling_indices = sampling_indices.long()

        peak_offsets = torch.gather(
            outputs['pred_offset', 0].view(-1, 2, H * W),
            2, sampling_indices.expand(-1, 2, -1))
        peak_offsets = peak_offsets.permute(0, 2, 1)

        peak_point = peak_indices + peak_offsets
        peak_point[..., 0] /= H - 1
        peak_point[..., 1] /= W - 1
        outputs['bs_point', 0] = peak_point
