
import os
import time
import argparse
import sys
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

#from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

WINDOW_NAME = 'TrtYOLODemo' # leave as it is/can change
model_name = 'yolov4-mish_25_weapons_DA_BG_Extra_best' # put model name only (not absolute path)
category_num = 25 # dont change
letterbox = False # dont change
image_path = "/home/ubuntu/tensorrt_demos/doc/0classes_499.jpg" # put absolute image path
threshold = 0.3
output_path= "/home/ubuntu/tensorrt_demos/doc/output"

img_name = (image_path.split("/")[-1]).split(".")[0]
labels_office = ['typewriter','file storage','photocopy machine/printer','paper weight','conference table','printer/photocopy machine',
    'projector','book/diary','blackboard','whiteboard','pen/pencil','pencil/pen','board marker',
    'duster','diary/book','scissors','pen stand','file','Desk','Collection of Files']



labels_weapons  = ['pistol','revolver','rifle','time bomb','tank','sniper','rocket launcher','dagger','sword','axe','artillery'
    ,'torpedo','missile','nanchucks','cigarette','blood','cigar','hookah','bong','shotgun','alcohol','machine gun'
    ,'human bleeding','pipe','grenade']
    
    
def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 25:
        #return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
        return {i: n for i, n in enumerate(labels_weapons)}
    else:
        #return {i: 'CLS%d' % i for i in range(category_num)}
        return {i: n for i, n in enumerate(labels_weapons)}       ##
        



#################################
def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.
    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    xx=0
    if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
        sys.exit("Window name error")
    img = cam.read()
    if img is None:
        sys.exit("Image empty error")
    boxes, confs, clss = trt_yolo.detect(img, conf_th)
    print("class: ",clss)
    print("boxes :",boxes)
    
    img = vis.draw_bboxes(img, boxes, confs, clss)
    img = show_fps(img, fps)
    cv2.imshow(WINDOW_NAME, img)
    cv2.imwrite(output_path +'/'+ "output_of{}.jpg".format(img_name),img)
   # "output_of_"+  +'.jpg',img)
    #"template {0}.jpg".format(xx),img
    toc = time.time()
    curr_fps = 1.0 / (toc - tic)
    # calculate an exponentially decaying average of fps number
    fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
    tic = toc
    key = cv2.waitKey(10)                        ## put zero for more time ^ press ESC key to quite
    if key == 27:  # ESC key: quit program
        cam.release()
        cv2.destroyAllWindows()
    elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
        full_scrn = not full_scrn
        set_display(WINDOW_NAME, full_scrn)

def main():
    if not os.path.isfile('yolo/%s.trt' % model_name):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % model_name)

    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    args = parser.parse_args()
    args.image = image_path
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(model_name, category_num, letterbox)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, conf_th=threshold, vis=vis)
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
