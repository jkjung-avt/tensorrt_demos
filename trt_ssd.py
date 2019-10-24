"""trt_ssd.py

This script demonstrates how to do real-time object detection with
TensorRT optimized Single-Shot Multibox Detector (SSD) engine.
"""


import sys
import time
import ctypes
import argparse

import numpy as np
import cv2
import pycuda.driver as cuda
import tensorrt as trt

from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display


WINDOW_NAME = 'TrtSsdDemo'


ctypes.CDLL("lib/libflattenconcat.so")
COCO_LABELS = coco.COCO_CLASSES_LIST


# initialize
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)

# create engine
with open(model.TRTbin, 'rb') as f:
    buf = f.read()
    engine = runtime.deserialize_cuda_engine(buf)

# create buffer
host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []
stream = cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    host_mem = cuda.pagelocked_empty(size, np.float32)
    cuda_mem = cuda.mem_alloc(host_mem.nbytes)

    bindings.append(int(cuda_mem))
    if engine.binding_is_input(binding):
        host_inputs.append(host_mem)
        cuda_inputs.append(cuda_mem)
    else:
        host_outputs.append(host_mem)
        cuda_outputs.append(cuda_mem)
context = engine.create_execution_context()


# inference
#TODO enable video pipeline
#TODO using pyCUDA for preprocess
ori = cv2.imread(sys.argv[1])
image = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (model.dims[2],model.dims[1]))
image = (2.0/255.0) * image - 1.0
image = image.transpose((2, 0, 1))
np.copyto(host_inputs[0], image.ravel())

start_time = time.time()
cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
context.execute_async(bindings=bindings, stream_handle=stream.handle)
cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
stream.synchronize()
print("execute times "+str(time.time()-start_time))

output = host_outputs[0]
height, width, channels = ori.shape
for i in range(int(len(output)/model.layout)):
    prefix = i*model.layout
    index = int(output[prefix+0])
    label = int(output[prefix+1])
    conf  = output[prefix+2]
    xmin  = int(output[prefix+3]*width)
    ymin  = int(output[prefix+4]*height)
    xmax  = int(output[prefix+5]*width)
    ymax  = int(output[prefix+6]*height)

    if conf > 0.7:
        print("Detected {} with confidence {}".format(COCO_LABELS[label], "{0:.0%}".format(conf)))
        cv2.rectangle(ori, (xmin,ymin), (xmax, ymax), (0,0,255),3)
        cv2.putText(ori, COCO_LABELS[label],(xmin+10,ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

cv2.imwrite("result.jpg", ori)
cv2.imshow("result", ori)
cv2.waitKey(0)


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'SSD model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, default='egohands',
                        choices=['coco', 'egohands'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height,
                'Camera TensorRT SSD Demo for Jetson Nano')
    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam, model, vis)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
