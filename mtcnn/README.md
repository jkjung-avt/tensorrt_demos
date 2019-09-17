The MTCNN caffe model files are taken from [https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT](https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT).  These model files contains a workaround which replaces 'PReLU' with 'ReLU', 'Scale' and 'Elementwise Addition' layers.  I use them to get around the issue of TensorRT 3.x/4.x not supporting PReLU layers.  Please refer to the original GitHub page (linked above) for more details.

* det1_relu.prototxt
* det1_relu.caffemodel
* det2_relu.prototxt
* det2_relu.caffemodel
* det3_relu.prototxt
* det3_relu.caffemodel
