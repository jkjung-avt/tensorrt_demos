"""trt_yolo_cv.py

This script could be used to make object detection video with
TensorRT optimized YOLO engine.

"cv" means "create video"
made by BigJoon (ref. jkjung-avt)
"""


import os
import argparse
import time
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization


## do all changes 


WINDOW_NAME = 'TrtYOLODemo' # leave as it is/can change

model_name = 'yolov4-mish_25_weapons_DA_BG_Extra_best' # put model name only (not absolute path)
category_num = 25 # 
letterbox = False # dont change
video_path = "/home/ubuntu/tensorrt_demos/doc/kgf2.mp4" # put absolute video path
vid_name = (video_path.split("/")[-1]).split(".")[0]
threshold = 0.3
output_path= "/home/ubuntu/tensorrt_demos/doc/output/demoop"



labels_weapons  = ['pistol','revolver','rifle','time bomb','tank','sniper','rocket launcher','dagger','sword','axe','artillery'
    ,'torpedo','missile','nanchucks','cigarette','blood','cigar','hookah','bong','shotgun','alcohol','machine gun'
    ,'human bleeding','pipe','grenade']
    
    
def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 80:       ##  for coco 
        #return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
        return {i: n for i, n in enumerate(labels_weapons)}
    else:
        #return {i: 'CLS%d' % i for i in range(category_num)}
        return {i: n for i, n in enumerate(labels_weapons)}       ##



def loop_and_detect(cap, trt_yolo, conf_th, vis, writer):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cap: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      writer: the VideoWriter object for the output video.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()

    while True:
        ret, frame = cap.read()
        if frame is None:  break
        boxes, confs, clss = trt_yolo.detect(frame, conf_th)
        print("classes",clss)
        print("confi",confs)
        frame = vis.draw_bboxes(frame, boxes, confs, clss)
        
        ####
        frame = show_fps(frame, fps)
       # cv2.imshow(WINDOW_NAME, frame)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        print("fps",fps)
        
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        ###
        
        writer.write(frame)
       # print('.', end='', flush=True)

    print('\nDone.')


def main():
    #if args.category_num <= 0:
    #    raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % model_name):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % model_name)
    desc = ('Run the TensorRT optimized object detecion model on an input '
            'video and save BBoxed overlaid output as another video.')
    #parser = argparse.ArgumentParser(description=desc)

    #args = parser.parse_args()
    #args.video=video_path
    
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit('ERROR: failed to open the input video file!')
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(
        output_path +'/'+ "output_of{}.mp4".format(vid_name),
        cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    cls_dict = get_cls_dict(category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(model_name, category_num, letterbox)

    loop_and_detect(cap, trt_yolo, conf_th=threshold, vis=vis, writer=writer)
    ###0.5
    writer.release()
    cap.release()


if __name__ == '__main__':
    main()
