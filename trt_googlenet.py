"""trt_googlenet.py

This script demonstrates how to do real-time image classification
(inferencing) with Cython wrapped TensorRT optimized googlenet engine.
"""


import timeit
import argparse

import numpy as np
import cv2
from utils.camera import add_camera_args, Camera
from utils.display import open_window, show_help_text, set_display
from pytrt import PyTrtGooglenet


PIXEL_MEANS = np.array([[[104., 117., 123.]]], dtype=np.float32)
DEPLOY_ENGINE = 'googlenet/deploy.engine'
ENGINE_SHAPE0 = (3, 224, 224)
ENGINE_SHAPE1 = (1000, 1, 1)
RESIZED_SHAPE = (224, 224)

WINDOW_NAME = 'TrtGooglenetDemo'


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


def loop_and_classify(cam, net, labels, do_cropping):
    """Continuously capture images from camera and do classification."""
    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
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
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    labels = np.loadtxt('googlenet/synset_words.txt', str, delimiter='\t')
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # initialize the tensorrt googlenet engine
    net = PyTrtGooglenet(DEPLOY_ENGINE, ENGINE_SHAPE0, ENGINE_SHAPE1)

    open_window(
        WINDOW_NAME, 'Camera TensorRT GoogLeNet Demo',
        cam.img_width, cam.img_height)
    loop_and_classify(cam, net, labels, args.crop_center)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
