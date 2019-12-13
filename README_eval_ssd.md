# Instructions for evaluating accuracy (mAP) of SSD models

Preparation
-----------

1. Prepare image data and label ('bbox') file for the evaluation.  I used COCO [2014 Val images (41K/6GB)](http://images.cocodataset.org/zips/val2014.zip) and [2014 Train/Val annotations (241MB)](http://images.cocodataset.org/annotations/annotations_trainval2014.zip).  You could try to use your own dataset for evaluation, but you'd need to convert the labels into [COCO Object Detection ('bbox') format](http://cocodataset.org/#format-data) if you want to use code in this repository without modifications.

   More specifically, I downloaded the images and labels, and unzipped files into `${HOME}/data/coco/`.

   ```shell
   $ wget http://images.cocodataset.org/zips/val2014.zip \
          -O ${HOME}/Downloads/val2014.zip
   $ wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip \
          -O ${HOME}/Downloads/annotations_trainval2014.zip
   $ mkdir -p ${HOME}/data/coco/images
   $ cd ${HOME}/data/coco/images
   $ unzip ${HOME}/Downloads/val2014.zip
   $ cd ${HOME}/data/coco
   $ unzip ${HOME}/Downloads/annotations_trainval2014.zip
   ```

   Later on I would be using the following (unzipped) image and annotation files for the evaluation.

   ```
   ${HOME}/data/coco/images/val2014/COCO_val2014_*.jpg
   ${HOME}/data/coco/annotations/instances_val2014.json
   ```

2. Install 'pycocotools'.  The easiest way is to use `pip3 install`.

   ```shell
   $ sudo pip3 install pycocotools
   ```

   Alternatively, you could build and install it from [source](https://github.com/cocodataset/cocoapi).

Evaluation
----------

I've created the [eval_ssd.py](eval_ssd.py) script to do the [mAP evaluation](http://cocodataset.org/#detection-eval).

```
usage: eval_ssd.py [-h] [--mode {tf,trt}] [--imgs_dir IMGS_DIR]
                   [--annotations ANNOTATIONS]
                   {ssd_mobilenet_v1_coco,ssd_mobilenet_v2_coco}
```

The script takes 1 mandatory argument: either 'ssd_mobilenet_v1_coco' or 'ssd_mobilenet_v2_coco'.  In addition, it accepts the following options:

* `--mode {tf,trt}`: to evaluate either the unoptimized TensorFlow frozen inference graph (tf) or the optimized TensorRT engine (trt).
* `--imgs_dir IMGS_DIR`: to specify an alternative directory for reading image files.
* `--annotations ANNOTATIONS`: to specify an alternative annotation/label file.

For example, I evaluated both 'ssd_mobilenet_v1_coco' and 'ssd_mobilenet_v2_coco' TensorRT engines on my x86_64 PC and got these results.  The overall mAP values are `0.230` and `0.246`, respectively.

```shell
$ python3 eval_ssd.py --mode trt ssd_mobilenet_v2_coco
loading annotations into memory...
Done (t=3.34s)
creating index...
index created!
Loading and preparing results...
DONE (t=1.04s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=74.94s).
Accumulating evaluation results...
DONE (t=10.07s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.230
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.352
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.252
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.163
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.208
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.265
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.024
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.604
$
$ python3 eval_ssd.py --mode trt ssd_mobilenet_v2_coco
......
loading annotations into memory...
Done (t=3.17s)
creating index...
index created!
Loading and preparing results...
DONE (t=1.03s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=77.38s).
Accumulating evaluation results...
DONE (t=10.15s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.246
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.271
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.021
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.170
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.220
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.279
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.280
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.643
```
