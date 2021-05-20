import numpy as np
import cv2

import pycuda.autoinit
from utils.modnet import TrtMODNet

img = cv2.imread('modnet/image.jpg')
modnet = TrtMODNet()
matte = modnet.infer(img)
cv2.imshow('Matte', matte)
cv2.waitKey(0)
cv2.destroyAllWindows()
