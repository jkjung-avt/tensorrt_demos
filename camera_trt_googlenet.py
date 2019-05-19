"""camera_trt_googlenet.py

This script demonstrates how to do real-time image classification
(inferencing) with Cython wrapped TensorRT optimized googlenet engine.

Note a large portion of the code was copied from my previous example:
'tegra-cam-caffe-threaded.py'
https://gist.github.com/jkjung-avt/d408aaabebb5b0041c318f4518bd918f
"""


import sys
import timeit
import argparse
import threading

import numpy as np
import cv2
from pytrt import PyTrtGooglenet


PIXEL_MEANS = np.array([[[104., 117., 123.]]], dtype=np.float32)
DEPLOY_ENGINE = 'googlenet/deploy.engine'
ENGINE_SHAPE0 = (3, 224, 224)
ENGINE_SHAPE1 = (1000, 1, 1)
RESIZED_SHAPE = (224, 224)

WINDOW_NAME = 'TrtGooglenetDemo'

# The following 2 global variables are shared between threads
THREAD_RUNNING = False
IMG_HANDLE = None


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time image classification with TrtGooglenet '
            'on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [0]',
                        default=0, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [640]',
                        default=640, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [480]',
                        default=480, type=int)
    parser.add_argument('--crop', dest='crop_center',
                        help='crop center square of image for Caffe '
                             'inferencing [False]',
                        action='store_true')
    args = parser.parse_args()
    return args


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
    gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)2592, height=(int)1458, '
               'format=(string)I420, framerate=(fraction)30/1 ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(width, height):
    """Open the display window."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME,
                       'Camera TensorRT GoogLeNet Classification Demo '
                       'for Jetson Nano')


def grab_img(cap):
    """
    This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab new image and put it
    into the global IMG_HANDLE, until THREAD_RUNNING is set to False.
    """
    global THREAD_RUNNING
    global IMG_HANDLE
    while THREAD_RUNNING:
        _, IMG_HANDLE = cap.read()


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


def show_help_text(img, help_text):
    """Draw help text on image."""
    cv2.putText(img, help_text, (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(img, help_text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (240, 240, 240), 1, cv2.LINE_AA)


def set_display(full_scrn):
    """Set disply window to either full screen or normal."""
    if full_scrn:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)


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
    tic = timeit.default_timer()
    out = net.forward(crop[None])  # add 1 dimension to 'crop' as batch
    toc = timeit.default_timer()
    print('{:.3f}s'.format(toc-tic))

    # output top 3 predicted scores and class labels
    out_prob = np.squeeze(out['prob'][0])
    top_inds = out_prob.argsort()[::-1][:3]
    return (out_prob[top_inds], labels[top_inds])


def loop_and_classify(net, labels, do_cropping):
    """Continuously capture images from camera and do classification."""
    global IMG_HANDLE
    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = IMG_HANDLE
        if img is not None:
            top_probs, top_labels = classify(img, net, labels, do_cropping)
            show_top_preds(img, top_probs, top_labels)
            if show_help:
                show_help_text(img, help_text)
            cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'):  # Toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(full_scrn)


def main():
    global THREAD_RUNNING
    args = parse_args()
    print('Called with args:')
    print(args)

    labels = np.loadtxt('googlenet/synset_words.txt', str, delimiter='\t')

    # Open camera
    if args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri,
                            args.image_width,
                            args.image_height,
                            args.rtsp_latency)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev,
                           args.image_width,
                           args.image_height)
    else:  # By default, use the Jetson onboard camera
        cap = open_cam_onboard(args.image_width,
                               args.image_height)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    # initialize the tensorrt googlenet engine
    net = PyTrtGooglenet(DEPLOY_ENGINE, ENGINE_SHAPE0, ENGINE_SHAPE1)

    # Start the sub-thread, which is responsible for grabbing images
    THREAD_RUNNING = True
    th = threading.Thread(target=grab_img, args=(cap,))
    th.start()

    open_window(args.image_width, args.image_height)
    loop_and_classify(net, labels, args.crop_center)

    # Terminate the sub-thread
    THREAD_RUNNING = False
    th.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
