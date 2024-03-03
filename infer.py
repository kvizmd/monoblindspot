import os

from argparse import ArgumentParser
from PIL import Image
from glob import glob

import numpy as np
import tqdm

import torch
from torchvision import transforms
from torch.nn import functional as F

import bse
from bse.utils.figure import \
    put_colorized_points, \
    alpha_blend, \
    to_numpy


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input', type=str,
        default='debug',
        help='Tag string for experiment')
    parser.add_argument(
        '--out_dir', type=str,
        default=None,
        help='Run name string for experiment')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Config file path')
    parser.add_argument(
        '--score_thr',
        type=float,
        default=0.25,
        help='Threshold of confidence score')
    parser.add_argument(
        '--opts',
        type=str,
        nargs='*',
        default=[],
        help='Override yaml configs with the same way as detectron2')
    args = parser.parse_args()

    override_opts = ['EVAL.BS.SCORE_THRESHOLD', str(args.score_thr)]

    cfg = bse.load_config(
        args.config,
        override_opts=override_opts + args.opts,
        check_requirements=False)
    inference(
        cfg,
        args.input,
        args.out_dir,
        score_thr=args.score_thr)


def inference(
        cfg,
        input_path: str,
        out_dir: str,
        score_thr: float = 0.3):
    if os.path.isfile(input_path):
        filenames = [input_path]
    elif os.path.isdir(input_path):
        filenames = [
            x for x in sorted(glob(os.path.join(input_path, '*.*')))
            if os.path.splitext(x)[1][1:].lower() in ['jpeg', 'jpg', 'png']
        ]
    else:
        raise RuntimeError('Invalid Path: {}'.format(input_path))

    to_tensor = transforms.ToTensor()
    device = cfg.DEVICE
    width = cfg.DATA.IMG_WIDTH
    height = cfg.DATA.IMG_HEIGHT

    model = bse.build_model(cfg)['bs']
    model.eval()
    model.to(device)

    os.makedirs(out_dir, exist_ok=True)
    for i, filename in enumerate(tqdm.tqdm(filenames, leave=False)):
        # Preprocess
        img = Image.open(filename)

        # Cropping to fit the aspect ratio of the traning image.
        crop_h = abs(img.height - int(height / width * img.width))
        crop_w = img.width % 32
        img = img.crop((
            crop_w // 2, crop_h // 2,
            img.width - crop_w // 2, img.height - crop_h // 2))

        original_img = np.array(img)

        img = to_tensor(img).unsqueeze(0)
        img = img.to(device)

        original_height, original_width = img.shape[-2:]
        img = F.interpolate(
            img, size=(int(width / original_width * original_height), width),
            align_corners=True, mode='bilinear')

        # Prediction
        with torch.inference_mode():
            predictions = model(img)
        scores = predictions['bs_confidence', 0][0]
        points = predictions['bs_point', 0][0]

        # Thresholding
        mask = scores.view(-1) >= score_thr
        scores = scores[mask]
        points = points[mask]

        # Visualization
        scores = to_numpy(scores)
        points = to_numpy(points)

        pred_overlay = put_colorized_points(
            original_height, original_width, points, scores)
        visualization = alpha_blend(original_img, pred_overlay, None)

        out_filename = os.path.join(
            out_dir, os.path.basename(filename))
        visualization = Image.fromarray(visualization)
        visualization.save(out_filename)


if __name__ == '__main__':
    main()
