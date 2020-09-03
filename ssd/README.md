Reference:

1. [AastaNV/TRT_object_detection](https://github.com/AastaNV/TRT_object_detection)
2. ['sampleUffSSD' in TensorRT samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#uffssd_sample)

Sources of the trained models:

* 'ssd_mobilenet_v1_coco.pb' and 'ssd_mobilnet_v2_coco.pb': This is just the 'frozen_inference_graph.pb' file in [ssd_mobilenet_v1_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) and [ssd_mobilenet_v2_coco_2018_03_29.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz), i.e. 2 of the trained models in [TensorFlow 1 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md).

* 'ssd_mobilenet_v1_egohands.pb' and 'ssd_mobilenet_v2_egohands.pb': These models are trained using my [Hand Detection Tutorial](https://github.com/jkjung-avt/hand-detection-tutorial) code.  After training, just run the [export.sh](https://github.com/jkjung-avt/hand-detection-tutorial/blob/master/export.sh) script to generated the frozen graph (pb) files.

* I've also added support for [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) in the code.  You could download the .pb by following the link.
