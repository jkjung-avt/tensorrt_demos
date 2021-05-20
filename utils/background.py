"""background.py

This code implements the Background class for the TensorRT MODNet
demo.  The Background class could generate background images from
either a still image, a video file or nothing (pure black bg).
"""


import numpy as np
import cv2


class Background():
    """Backgrounf class which supports one of the following sources:

    1. Image (jpg, png, etc.) file, repeating indefinitely
    2. Video file, looping forever
    3. None -> black background

    # Arguments
        src: if not spcified, use black background; else, src should be
             a filename of an image (jpg/png) or video (mp4/ts)
        width & height: width & height of the output background image
    """

    def __init__(self, src, width, height, demo_mode=False):
        self.src = src
        self.width = width
        self.height = height
        self.demo_mode = demo_mode
        if not src:  # empty source: black background
            self.is_video = False
            self.bg_frame = np.zeros((height, width, 3), dtype=np.uint8)
        elif not isinstance(src, str):
            raise ValueError('bad src')
        elif src.endswith('.jpg') or src.endswith('.png'):
            self.is_video = False
            self.bg_frame = cv2.resize(cv2.imread(src), (width, height))
            assert self.bg_frame is not None and self.bg_frame.ndim == 3
        elif src.endswith('.mp4') or src.endswith('.ts'):
            self.is_video = True
            self.cap = cv2.VideoCapture(src)
            assert self.cap.isOpened()
        else:
            raise ValueError('unknown src')

    def read(self):
        """Read a frame from the Background object."""
        if self.is_video:
            _, frame = self.cap.read()
            if frame is None:
                # assume end of video file has been reached, so loop around
                self.cap.release()
                self.cap = cv2.VideoCapture(self.src)
                _, frame = self.cap.read()
            return cv2.resize(frame, (self.width, self.height))
        else:
            return self.bg_frame.copy()

    def __del__(self):
        if self.is_video:
            try:
                self.cap.release()
            except:
                pass
