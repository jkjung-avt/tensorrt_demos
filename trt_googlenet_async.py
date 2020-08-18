"""trt_googlenet.py

This is the 'async' version of trt_googlenet.py implementation.

Refer to trt_ssd_async.py for description about the design and
synchronization between the main and child threads.
"""


import sys
import time
import argparse
import threading

import numpy as np
import cv2
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from pytrt import PyTrtGooglenet


PIXEL_MEANS = np.array([[[104., 117., 123.]]], dtype=np.float32)
DEPLOY_ENGINE = 'googlenet/deploy.engine'
ENGINE_SHAPE0 = (3, 224, 224)
ENGINE_SHAPE1 = (1000, 1, 1)
RESIZED_SHAPE = (224, 224)

WINDOW_NAME = 'TrtGooglenetDemo'
MAIN_THREAD_TIMEOUT = 10.0  # 10 seconds

# 'shared' global variables
s_img, s_probs, s_labels = None, None, None


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time image classification with TrtGooglenet '
            'on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--crop', dest='crop_center',
                        help='crop center square of image for '
                             'inferencing [False]',
                        action='store_true')
    args = parser.parse_args()
    return args


def classify(img, net, labels, do_cropping):
    """Classify 1 image (crop)."""
    crop = img
    if do_cropping:
        h, w, _ = img.shape
        if h < w:
            crop = img[:, ((w-h)//2):((w+h)//2), :]
        else:
            crop = img[((h-w)//2):((h+w)//2), :, :]

    # preprocess the image crop
    crop = cv2.resize(crop, RESIZED_SHAPE)
    crop = crop.astype(np.float32) - PIXEL_MEANS
    crop = crop.transpose((2, 0, 1))  # HWC -> CHW

    # inference the (cropped) image
    out = net.forward(crop[None])  # add 1 dimension to 'crop' as batch

    # output top 3 predicted scores and class labels
    out_prob = np.squeeze(out['prob'][0])
    top_inds = out_prob.argsort()[::-1][:3]
    return (out_prob[top_inds], labels[top_inds])


class TrtGooglenetThread(threading.Thread):
    def __init__(self, condition, cam, labels, do_cropping):
        """__init__

        # Arguments
            condition: the condition variable used to notify main
                       thread about new frame and detection result
            cam: the camera object for reading input image frames
            labels: a numpy array of class labels
            do_cropping: whether to do center-cropping of input image
        """
        threading.Thread.__init__(self)
        self.condition = condition
        self.cam = cam
        self.labels = labels
        self.do_cropping = do_cropping
        self.running = False

    def run(self):
        """Run until 'running' flag is set to False by main thread."""
        global s_img, s_probs, s_labels

        print('TrtGooglenetThread: loading the TRT Googlenet engine...')
        self.net = PyTrtGooglenet(DEPLOY_ENGINE, ENGINE_SHAPE0, ENGINE_SHAPE1)
        print('TrtGooglenetThread: start running...')
        self.running = True
        while self.running:
            img = self.cam.read()
            if img is None:
                break
            top_probs, top_labels = classify(
                img, self.net, self.labels, self.do_cropping)
            with self.condition:
                s_img, s_probs, s_labels = img, top_probs, top_labels
                self.condition.notify()
        del self.net
        print('TrtGooglenetThread: stopped...')

    def stop(self):
        self.running = False
        self.join()


def show_top_preds(img, top_probs, top_labels):
    """Show top predicted classes and softmax scores."""
    x = 10
    y = 40
    for prob, label in zip(top_probs, top_labels):
        pred = '{:.4f} {:20s}'.format(prob, label)
        #cv2.putText(img, pred, (x+1, y), cv2.FONT_HERSHEY_PLAIN, 1.0,
        #            (32, 32, 32), 4, cv2.LINE_AA)
        cv2.putText(img, pred, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (0, 0, 240), 1, cv2.LINE_AA)
        y += 20


def loop_and_display(condition):
    """Continuously capture images from camera and do classification."""
    global s_img, s_probs, s_labels

    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        with condition:
            if condition.wait(timeout=MAIN_THREAD_TIMEOUT):
                img, top_probs, top_labels = s_img, s_probs, s_labels
            else:
                raise SystemExit('ERROR: timeout waiting for img from child')
        show_top_preds(img, top_probs, top_labels)
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
        elif key == ord('H') or key == ord('h'):  # Toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    labels = np.loadtxt('googlenet/synset_words.txt', str, delimiter='\t')
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    open_window(
        WINDOW_NAME, 'Camera TensorRT GoogLeNet Demo',
        cam.img_width, cam.img_height)
    condition = threading.Condition()
    trt_thread = TrtGooglenetThread(condition, cam, labels, args.crop_center)
    trt_thread.start()  # start the child thread
    loop_and_display(condition)
    trt_thread.stop()   # stop the child thread

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
