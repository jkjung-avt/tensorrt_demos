# Added by Adriano Santos

import cv2
import time
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils.yolo_with_plugins import TrtYOLO
from utils.yolo_classes import get_class_name


# Parameters to test
conf_th = 0.5
nms_threshold = 0.5
opt_device = '0,1'
color = (255, 0, 0) 

# Create a model instance
trt_yolo = TrtYOLO("yolov3-spp3-416", (416, 416), 1)

def main(video):
    # Start the fake camera
    camera = cv2.VideoCapture(video)

    while camera.isOpened():
        # Get a frame
        (status, frame) = camera.read()
        
        # Stop conditions
        key = cv2.waitKey(1) & 0xFF
        if (status == False or key == ord("q")):
            break

        # Get object detection information
        boxes, confs, clss = trt_yolo.detect(frame, conf_th, nms_threshold)

        # For all detection do:
        for i, box in enumerate(boxes):

            # x1, y1, x2, y2
            x1, y1, x2, y2 =int(box[0]) ,int(box[1]) ,int(box[2]) ,int(box[3])
            
            # Centro point 
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids = (cX, cY)
            
            # Drawing the box
            cv2.rectangle(frame,(x1, y1), (x2, y2), color ,1)
            
            # Get class and conf values
            obj_class = get_class_name(int(clss[i]))
            conf = "{:.2f}".format(float(confs[i]))

            # Format a label
            label = "{}-{}".format(obj_class, conf)

            # Print a label into the box
            cv2.putText(frame,label,(inputCentroids),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,cv2.LINE_AA)
        
        cv2.imshow("Video", frame)
        # Slowing video
        time.sleep(0.02)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Video file path
    video = ' ... set file path'
    main(video)