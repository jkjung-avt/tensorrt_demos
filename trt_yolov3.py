"""trt_yolov3.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLOv3 engine.
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolov3_classes import get_cls_dict
from utils.yolov3 import TrtYOLOv3
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization


WINDOW_NAME = 'TrtYOLOv3Demo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLOv3 model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, default='yolov3-416',
                        help='yolov3[-spp|-tiny]-[288|416|608]')
    parser.add_argument('--category_num', type=int, default=80,
                        help='number of object categories [80]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolov3, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolov3: the TRT YOLOv3 object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            boxes, confs, clss = trt_yolov3.detect(img, conf_th)
            img = vis.draw_bboxes(img, boxes, confs, clss)
            img = show_fps(img, fps)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolov3_onnx/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolov3_onnx/%s.trt) not found!' % args.model)

    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

    trt_yolov3 = TrtYOLOv3(args.model, (h, w), args.category_num)

    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height,
                'Camera TensorRT YOLOv3 Demo')
    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam, trt_yolov3, conf_th=0.3, vis=vis)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
