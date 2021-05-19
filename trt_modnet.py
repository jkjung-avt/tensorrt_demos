"""trt_modnet.py

This script demonstrates how to do real-time "image matting" with
TensorRT optimized MODNet engine.
"""


import os
import argparse

import numpy as np
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.camera import add_camera_args, Camera
from utils.writer import get_video_writer
from utils.background import Background
from utils.display import open_window, show_fps
from utils.display import FpsCalculator, ScreenToggler
from utils.modnet import TrtMODNet


WINDOW_NAME = 'TrtMODNetDemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time image matting with TensorRT optimized MODNet')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '--background', type=str, default='',
        help='background image or video file name [None]')
    parser.add_argument(
        '--create_video', type=str, default='',
        help='create output video (either .ts or .mp4) [None]')
    args = parser.parse_args()
    return args


def blend_img(img, bg, matte):
    """Blend foreground and background using the 'matte'.

    # Arguments
        img: uint8 np.array of shape (H, W, 3), the foreground image
        bg:  uint8 np.array of shape (H, W, 3), the background image
        matte: float32 np.array of shape (H, W), values between 0.0 and 1.0
    """
    return (img * matte[..., np.newaxis] +
            bg * (1 - matte[..., np.newaxis])).astype(np.uint8)


class TrtMODNetRunner():
    """TrtMODNetRunner

    TODO: Add a demo mode...

    # Arguments
        modnet: TrtMODNet instance
        cam: Camera object (for reading input images)
        bggen: background generator (for reading background images)
        writer: VideoWriter object (for saving output video)
    """

    def __init__(self, modnet, cam, bggen, writer=None):
        self.modnet = modnet
        self.cam = cam
        self.bggen = bggen
        self.writer = writer
        open_window(
            WINDOW_NAME, 'TensorRT MODNet Demo', cam.img_width, cam.img_height)

    def run(self):
        """Get img and bg, infer matte, blend and show img, then repeat."""
        fps_calc = FpsCalculator()
        scrn_tog = ScreenToggler()
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:  break
            img, bg = self.cam.read(), self.bggen.read()
            if img is None:  break
            matte = self.modnet.infer(img)
            matted_img = blend_img(img, bg, matte)
            fps = fps_calc.update()
            matted_img = show_fps(matted_img, fps)
            if self.writer:  self.writer.write(matted_img)
            cv2.imshow(WINDOW_NAME, matted_img)
            key = cv2.waitKey(1)
            if key == ord('F') or key == ord('f'):  # Toggle fullscreen
                scrn_tog.toggle()
            elif key == 27:                         # ESC key: quit
                break

    def __del__(self):
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    writer = None
    if args.create_video:
        writer = get_video_writer(
            args.create_video, cam.img_width, cam.img_height)

    modnet = TrtMODNet()
    bggen = Background(args.background, cam.img_width, cam.img_height)

    runner = TrtMODNetRunner(modnet, cam, bggen, writer)
    runner.run()

    if writer:
        writer.release()
    cam.release()


if __name__ == '__main__':
    main()
