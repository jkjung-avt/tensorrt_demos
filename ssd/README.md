Reference:

1. [AastaNV/TRT_object_detection](https://github.com/AastaNV/TRT_object_detection)
2. ['sampleUffSSD' in TensorRT samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#uffssd_sample)

Sources of the trained models:

* 'ssd_mobilenet_v1_coco.pb': This is just the 'frozen_inference_graph.pb' file in [ssd_mobilenet_v1_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz), i.e. one of the trained models in [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

* 'ssd_mobilenet_v1_egohands.pb': This model is trained using my [Hand Detection Tutorial](https://github.com/jkjung-avt/hand-detection-tutorial) code.  After training, just run the [export.sh](https://github.com/jkjung-avt/hand-detection-tutorial/blob/master/export.sh) script to generated the frozen graph (pb) file.
