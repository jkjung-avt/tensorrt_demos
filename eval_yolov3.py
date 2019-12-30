"""eval_yolov3.py

This script is for evaluating mAP (accuracy) of YOLOv3 models.
"""


import os
import sys
import json
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from progressbar import progressbar

from utils.yolov3 import TrtYOLOv3
from utils.yolov3_classes import yolov3_cls_to_ssd


HOME = os.environ['HOME']
VAL_IMGS_DIR = HOME + '/data/coco/images/val2014'
VAL_ANNOTATIONS = HOME + '/data/coco/annotations/instances_val2014.json'


def parse_args():
    """Parse input arguments."""
    desc = 'Evaluate mAP of SSD model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--imgs_dir', type=str, default=VAL_IMGS_DIR,
                        help='directory of validation images [%s]' % VAL_IMGS_DIR)
    parser.add_argument('--annotations', type=str, default=VAL_ANNOTATIONS,
                        help='groundtruth annotations [%s]' % VAL_ANNOTATIONS)
    parser.add_argument('--model', type=str, default='yolov3-416',
                        choices=['yolov3-288', 'yolov3-416', 'yolov3-608'])
    args = parser.parse_args()
    return args


def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.imgs_dir):
        sys.exit('%s is not a valid directory' % args.imgs_dir)
    if not os.path.isfile(args.annotations):
        sys.exit('%s is not a valid file' % args.annotations)


def generate_results(yolov3, imgs_dir, jpgs, results_file):
    """Run detection on each jpg and write results to file."""
    results = []
    for jpg in progressbar(jpgs):
        img = cv2.imread(os.path.join(imgs_dir, jpg))
        image_id = int(jpg.split('.')[0].split('_')[-1])
        boxes, confs, clss = yolov3.detect(img, conf_th=1e-2)
        for box, conf, cls in zip(boxes, confs, clss):
            x = float(box[0])
            y = float(box[1])
            w = float(box[2] - box[0] + 1)
            h = float(box[3] - box[1] + 1)
            cls = yolov3_cls_to_ssd[cls]
            results.append({'image_id': image_id,
                            'category_id': int(cls),
                            'bbox': [x, y, w, h],
                            'score': float(conf)})
    with open(results_file, 'w') as f:
        f.write(json.dumps(results, indent=4))


def main():
    args = parse_args()
    check_args(args)

    results_file = 'yolov3_onnx/results_%s.json' % args.model
    yolo_dim = int(args.model.split('-')[-1])  # 416 or 608
    trt_yolov3 = TrtYOLOv3(args.model, (yolo_dim, yolo_dim))

    jpgs = [j for j in os.listdir(args.imgs_dir) if j.endswith('.jpg')]
    generate_results(trt_yolov3, args.imgs_dir, jpgs, results_file)

    # Run COCO mAP evaluation
    # Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(args.annotations)
    cocoDt = cocoGt.loadRes(results_file)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    print(cocoEval.summarize())


if __name__ == '__main__':
    main()
