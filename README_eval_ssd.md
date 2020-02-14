# Instructions for evaluating accuracy (mAP) of SSD models

Preparation
-----------

1. Prepare image data and label ('bbox') file for the evaluation.  I used COCO [2017 Val images (5K/1GB)](http://images.cocodataset.org/zips/val2017.zip) and [2017 Train/Val annotations (241MB)](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).  You could try to use your own dataset for evaluation, but you'd need to convert the labels into [COCO Object Detection ('bbox') format](http://cocodataset.org/#format-data) if you want to use code in this repository without modifications.

   More specifically, I downloaded the images and labels, and unzipped files into `${HOME}/data/coco/`.

   ```shell
   $ wget http://images.cocodataset.org/zips/val2017.zip \
          -O ${HOME}/Downloads/val2017.zip
   $ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
          -O ${HOME}/Downloads/annotations_trainval2017.zip
   $ mkdir -p ${HOME}/data/coco/images
   $ cd ${HOME}/data/coco/images
   $ unzip ${HOME}/Downloads/val2017.zip
   $ cd ${HOME}/data/coco
   $ unzip ${HOME}/Downloads/annotations_trainval2017.zip
   ```

   Later on I would be using the following (unzipped) image and annotation files for the evaluation.

   ```
   ${HOME}/data/coco/images/val2017/*.jpg
   ${HOME}/data/coco/annotations/instances_val2017.json
   ```

2. Install 'pycocotools'.  The easiest way is to use `pip3 install`.

   ```shell
   $ sudo pip3 install pycocotools
   ```

   Alternatively, you could build and install it from [source](https://github.com/cocodataset/cocoapi).

3. Install additional requirements.

   ```shell
   $ sudo pip3 install progressbar2
   ```

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
$ python3 eval_ssd.py --mode trt ssd_mobilenet_v1_coco
......
100% (5000 of 5000) |####################| Elapsed Time: 0:00:26 Time:  0:00:26
loading annotations into memory...
Done (t=0.36s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.11s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.89s).
Accumulating evaluation results...
DONE (t=1.37s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.351
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.209
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.022
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.606
None
$
$ python3 eval_ssd.py --mode trt ssd_mobilenet_v2_coco
......
100% (5000 of 5000) |####################| Elapsed Time: 0:00:29 Time:  0:00:29
loading annotations into memory...
Done (t=0.37s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.12s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.47s).
Accumulating evaluation results...
DONE (t=1.42s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.248
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.021
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.221
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.279
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.027
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.202
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.643
None
```
