"""run_onnx.py

A simple script for verifying the modnet.onnx model.

I used the following image for testing:
$ gdown --id 1fkyh03NEuSwvjFttYVwV7TjnJML04Xn6 -O image.jpg
"""


import numpy as np
import cv2
import onnx
import onnxruntime

img = cv2.imread('image.jpg')
img = cv2.resize(img, (512, 288), cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1)).astype(np.float32)
img = (img - 127.5) / 127.5
img = np.expand_dims(img, axis=0)

session = onnxruntime.InferenceSession('modnet.onnx', None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: img})
matte = np.squeeze(result[0])
cv2.imshow('Matte', (matte * 255.).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
