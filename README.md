# tensorrt_demos

Examples demonstrating how to optimize caffe models with TensorRT and run inferencing on Jetson Nano/TX2.

Table of contents
-----------------

* [Blog posts](#blog)
* [Prerequisite](#prerequisite)
* [Demo #1: googlenet](#googlenet)
* [Demo #2: mtcnn](#mtcnn)

<a name="blog"></a>
Blog posts related to this repository
-------------------------------------

* [Running TensorRT Optimized GoogLeNet on Jetson Nano](https://jkjung-avt.github.io/tensorrt-googlenet/)
* [TensorRT MTCNN Face Detector](https://jkjung-avt.github.io/tensorrt-mtcnn/)
* [Optimizing TensorRT MTCNN](https://jkjung-avt.github.io/optimize-mtcnn/)

<a name="prerequisite"></a>
Prerequisite
------------

The code in this repository was tested on both Jetson Nano DevKit and Jetson TX2.  In order to run the demo programs below, first make sure you have the target Jetson Nano system with the proper version of image installed.  Reference: [Setting up Jetson Nano: The Basics](https://jkjung-avt.github.io/setting-up-nano/).

More specifically, the target Jetson Nano/TX2 system should have TensorRT libraries installed.  For example, TensorRT v5.0.6 was present on the tested Jetson Nano system.

```shell
$ ls /usr/lib/aarch64-linux-gnu/libnvinfer.so*
/usr/lib/aarch64-linux-gnu/libnvinfer.so
/usr/lib/aarch64-linux-gnu/libnvinfer.so.5
/usr/lib/aarch64-linux-gnu/libnvinfer.so.5.0.6
```

Furthermore, the demo programs require the 'cv2' (OpenCV) module in python3.  You could refer to [Installing OpenCV 3.4.6 on Jetson Nano](https://jkjung-avt.github.io/opencv-on-nano/) about how to install opencv-3.4.6 on the Jetson system.

<a name="googlenet"></a>
Demo #1: googlenet
------------------

Step-by-step:

1. Clone this repository.

   ```shell
   $ cd ${HOME}/project
   $ git clone https://github.com/jkjung-avt/tensorrt_demos
   $ cd tensorrt_demos
   ```

2. Build the TensorRT engine from the trained googlenet (ILSVRC2012) model.  Note that I downloaded the trained model files from [BVLC caffe](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) and have put a copy of all necessary files in this repository.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/googlenet
   $ make
   $ ./create_engine
   ```

3. Build the Cython code.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos
   $ make
   ```

4. Run the `trt_googlenet.py` demo program.  For example, run the demo with a USB webcam as the input.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos
   $ python3 trt_googlenet.py --usb --vid 0 --width 1280 --height 720
   ```

   Here's a screenshot of the demo.

   ![A Picture of a Golden Retriever](https://raw.githubusercontent.com/jkjung-avt/tensorrt_demos/master/doc/golden_retriever.png)

5. The demo program supports a number of different image inputs.  You could do `python3 trt_googlenet.py --help` to read the help messages.  Or more specifically, the following inputs could be specified:

   * `--file --filename test_video.mp4`: a video file, e.g. mp4 or ts.
   * `--image --filename test_image.jpg`: an image file, e.g. jpg or png.
   * `--usb --vid 0`: USB webcam (/dev/video0).
   * `--rtsp --uri rtsp://admin:123456@192.168.1.1/live.sdp`: RTSP source, e.g. an IP cam.

<a name="mtcnn"></a>
Demo #2: mtcnn
--------------

Assuming this repository has been cloned at `${HOME}/project/tensorrt_demos`, follow these steps:

1. Build the TensorRT engines from the trained MTCNN model.  (Refer to [mtcnn/README.md](https://github.com/jkjung-avt/tensorrt_demos/blob/master/mtcnn/README.md) for more information about the prototxt and caffemodel files.)

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/mtcnn
   $ make
   $ ./create_engines
   ```

2. Build the Cython code if it has not been done yet.  Refer to step 3 in Demo #1.

3. Run the `trt_mtcnn.py` demo program.  For example, I just grabbed from the internet a poster of The Avengers for testing.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos
   $ python3 trt_mtcnn.py --image --filename ${HOME}/Pictures/avengers.jpg
   ```

   Here's the result.

   ![Avengers faces detected](https://raw.githubusercontent.com/jkjung-avt/tensorrt_demos/master/doc/avengers.png)

4. The `trt_mtcnn.py` demo program could also take various image inputs.  Refer to step 5 in Demo #1 again.
