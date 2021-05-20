"""trt_modnet.py

This script demonstrates how to do real-time "image matting" with
TensorRT optimized MODNet engine.
"""


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
    parser.add_argument(
        '--demo_mode', action='store_true',
        help='run the program in a special "demo mode" [False]')
    args = parser.parse_args()
    return args


class BackgroundBlender():
    """BackgroundBlender

    # Arguments
        demo_mode: if True, do foreground/background blending in a
                   special "demo mode" which alternates among the
                   original, replaced and black backgrounds.
    """

    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        self.count = 0

    def blend(self, img, bg, matte):
        """Blend foreground and background using the 'matte'.

        # Arguments
            img: uint8 np.array of shape (H, W, 3), the foreground image
            bg:  uint8 np.array of shape (H, W, 3), the background image
            matte: float32 np.array of shape (H, W), values between 0.0 and 1.0
        """
        if self.demo_mode:
            img, bg, matte = self._mod_for_demo(img, bg, matte)
        return (img * matte[..., np.newaxis] +
                bg * (1 - matte[..., np.newaxis])).astype(np.uint8)

    def _mod_for_demo(self, img, bg, matte):
        """Modify img, bg and matte for "demo mode"

        # Demo script (based on "count")
              0~ 59: black background left to right
             60~119: black background only
            120~179: replaced background left to right
            180~239: replaced background
            240~299: original background left to right
            300~359: original background
        """
        img_h, img_w, _ = img.shape
        if self.count < 120:
            bg = np.zeros(bg.shape, dtype=np.uint8)
            if self.count < 60:
                offset = int(img_w * self.count / 59)
                matte[:, offset:img_w] = 1.0
        elif self.count < 240:
            if self.count < 180:
                offset = int(img_w * (self.count - 120) / 59)
                bg[:, offset:img_w, :] = 0
        else:
            if self.count < 300:
                offset = int(img_w * (self.count - 240) / 59)
                matte[:, 0:offset] = 1.0
            else:
                matte[:, :] = 1.0
        self.count = (self.count + 1) % 360
        return img, bg, matte


class TrtMODNetRunner():
    """TrtMODNetRunner

    # Arguments
        modnet: TrtMODNet instance
        cam: Camera object (for reading foreground images)
        bggen: background generator (for reading background images)
        blender: BackgroundBlender object
        writer: VideoWriter object (for saving output video)
    """

    def __init__(self, modnet, cam, bggen, blender, writer=None):
        self.modnet = modnet
        self.cam = cam
        self.bggen = bggen
        self.blender = blender
        self.writer = writer
        open_window(
            WINDOW_NAME, 'TensorRT MODNet Demo', cam.img_width, cam.img_height)

    def run(self):
        """Get img and bg, infer matte, blend and show img, then repeat."""
        scrn_tog = ScreenToggler()
        fps_calc = FpsCalculator()
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:  break
            img, bg = self.cam.read(), self.bggen.read()
            if img is None:  break
            matte = self.modnet.infer(img)
            matted_img = self.blender.blend(img, bg, matte)
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
    blender = BackgroundBlender(args.demo_mode)

    runner = TrtMODNetRunner(modnet, cam, bggen, blender, writer)
    runner.run()

    if writer:
        writer.release()
    cam.release()


if __name__ == '__main__':
    main()
