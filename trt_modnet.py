"""trt_modnet.py

This script demonstrates how to do real-time "image matting" with
TensorRT optimized MODNet engine.
"""


import os
import time
import argparse
import subprocess

import numpy as np
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.camera import add_camera_args, Camera
from utils.background import Background
from utils.display import open_window, set_display, show_fps
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


def get_video_writer(name, width, height, fps=30):
    """Get a VideoWriter object for saving output video.

    This function tries to use Jetson's hardware H.264 encoder (omxh264enc)
    if available, in which case the output video would be a MPEG-2 TS file.
    Otherwise, it uses cv2's built-in encoding mechanism and saves a MP4
    file.
    """
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'omxh264dec' in gst_elements:
        filename = name + '.ts'  # Transport Stream
        gst_str = ('appsrc ! videoconvert ! omxh264enc ! mpegtsmux ! '
                   'filesink location=%s') % filename
        return cv2.VideoWriter(
            gst_str, cv2.CAP_GSTREAMER, 0, fps, (width, height))
    else:
        filename = name + '.mp4'  # MP4
        return cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


def blend_img(img, bg, matte):
    """Blend foreground and background using the 'matte'.

    # Arguments
        img: uint8 np.array of shape (H, W, 3), the foreground image
        bg:  uint8 np.array of shape (H, W, 3), the background image
        matte: float32 np.array of shape (H, W), values between 0.0 and 1.0
    """
    return (img * matte[..., np.newaxis] +
            bg * (1 - matte[..., np.newaxis])).astype(np.uint8)


class FpsCalculator():
    """Helper class for calculating frames-per-second (FPS)."""

    def __init__(self, decay_factor=0.95):
        self.fps = 0.0
        self.tic = time.time()
        self.decay_factor = decay_factor

    def update(self):
        toc = time.time()
        curr_fps = 1.0 / (toc - self.tic)
        self.fps = curr_fps if self.fps == 0.0 else self.fps
        self.fps = self.fps * self.decay_factor + \
                   curr_fps * (1 - self.decay_factor)
        self.tic = toc
        return self.fps

    def reset(self):
        self.fps = 0.0


class ScreenToggler():
    """Helper class for toggling between non-fullscreen and fullscreen."""

    def __init__(self):
        self.full_scrn = False

    def toggle(self):
        self.full_scrn = not self.full_scrn
        set_display(WINDOW_NAME, self.full_scrn)


class TrtMODNetRunner():
    """TrtMODNetRunner

    # Arguments
        modnet: TrtMODNet instance
        cam: Camera object (for reading input images)
        bggen: background generator (for reading background images)
        writer: VideoWriter object (for saving output video)
    """

    def __init__(self, modnet, cam, bggen, writer=None):
        self.modnet = modnet
        sefl.cam = cam
        self.bggen = bggen
        self.writer = writer
        open_window(
            WINDOW_NAME, 'TensorRT MODNet Demo', cam.img_width, cam.img_height)

    def run(self):
        fps_calc = FpsCalculator()
        scrn_tog = ScreenToggler()
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0: break
            img, bg = self.cam.read(), self.bggen.read()
            if img is None: break
            matte = self.modnet.infer(img)
            matted_img = blend_img(img, bg, matte)
            matted_img = show_fps(matted_img, fps)
            cv2.imshow(WINDOW_NAME, matted_img)
            fps = fps_calc.update()
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

    cam.release()


if __name__ == '__main__':
    main()
