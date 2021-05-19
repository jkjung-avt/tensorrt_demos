"""export.py

This script is an adatped copy of:
https://github.com/ZHKKKe/MODNet/blob/master/onnx/export_onnx.py

This script is for converting a PyTorch MODNet model to ONNX.  The
output ONNX model will have fixed batch size (1) and input image
width/height.  The input image width and height could be specified
by command-line options (default to 512x288).

Example usage: (Recommended to run this inside a virtual environment)
$ python export.py --width 512 --height 288 \
                   modnet_photographic_portrait_matting.ckpt \
                   modnet.onnx
"""


import os
import argparse

import torch
from torch.autograd import Variable

from .modnet import MODNet


BATCH_SIZE = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--width', type=int, default=512,
        help='image width of the converted ONNX model [512]')
    parser.add_argument(
        '--height', type=int, default=288,
        help='image width of the converted ONNX model [288]')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose logging [False]')
    parser.add_argument(
        'input_ckpt', type=str, help='the input PyTorch checkpoint file path')
    parser.add_argument(
        'output_onnx', type=str, help='the output ONNX file path')
    args = parser.parse_args()

    if not os.path.isfile(args.input_ckpt):
        raise SystemExit('ERROR: file (%s) not found!' % args.input_ckpt)

    # define model & load checkpoint
    modnet = torch.nn.DataParallel(MODNet()).cuda()
    modnet.load_state_dict(torch.load(args.input_ckpt))
    modnet.eval()

    # prepare dummy input
    dummy_img = torch.rand(BATCH_SIZE, 3, args.height, args.width) * 2. - 1.
    dummy_img = dummy_img.cuda()

    # export to onnx model
    torch.onnx.export(
        modnet.module, dummy_img, args.output_onnx,
        opset_version=11, export_params=True, verbose=args.verbose,
        input_names=['input'], output_names=['output'])
