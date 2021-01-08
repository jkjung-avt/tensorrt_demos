"""trt_yolo_cv.py

This script helps making inference object detection video with
TensorRT optimized YOLO engine.
"cv"means "create video"
made by BigJoon
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

from utils.yolo_with_plugins import TrtYOLO


#WINDOW_NAME = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    desc = ('capture Video input file and save BBoxed Video'
            'object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-v', '--video_name',type=str, required=True,
        help='You need to put input video')
    parser.add_argument(
        '-o', '--result_video',type=str, required=True,
        help='You need to put output video name')
    parser.add_argument(
        '-c', '--category_num', type=int, default=15,
        help='number of object categories [15]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    args = parser.parse_args()
    return args


def loop_and_detect(cap, trt_yolo,result_video, conf_th,vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cap: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    #full_scrn = False
    fps = 0.0
    tic = time.time()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(result_video,cv2.VideoWriter_fourcc(*'mp4v'),30,(frame_width,frame_height))
    while True:
        ret,frame = cap.read()
        
        boxes, confs, clss = trt_yolo.detect(frame, conf_th)
        frame = vis.draw_bboxes(frame, boxes, confs, clss)
        frame = show_fps(frame, fps)

        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        out.write(frame)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc

    cap.release()
    out.release()


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cap = cv2.VideoCapture(args.video_name)
    if(cap.isOpened() == False):
        print("unable to read read source video feed")

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

    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)
    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cap, trt_yolo, args.result_video , conf_th=0.3, vis=vis)


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
