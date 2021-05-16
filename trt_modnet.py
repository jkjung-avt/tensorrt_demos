"""trt_modnet.py

This script demonstrates how to do real-time "image matting" with
TensorRT optimized MODNet engine.
"""


import os
import time
import argparse

import numpy as np
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.modnet import TrtMODNet


WINDOW_NAME = 'TrtMODNetDemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time image matting with TensorRT optimized MODNet')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    modnet = TrtMODNet()
    open_window(
        WINDOW_NAME, 'Camera TensorRT MODNet Demo',
        cam.img_width, cam.img_height)

    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        matte = modnet.infer(img)
        matted_img = (img * matte[..., np.newaxis]).astype(np.uint8)
        matted_img = show_fps(matted_img, fps)
        cv2.imshow(WINDOW_NAME, matted_img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
