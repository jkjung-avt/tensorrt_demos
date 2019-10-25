# tensorrt_demos

Examples demonstrating how to optimize caffe/tensorflow models with TensorRT and run inferencing on Jetson Nano/TX2.

Table of contents
-----------------

* [Blog posts](#blog)
* [Prerequisite](#prerequisite)
* [Demo #1: googlenet](#googlenet)
* [Demo #2: mtcnn](#mtcnn)
* [Demo #3: ssd](#ssd)

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

More specifically, the target Jetson Nano/TX2 system should have TensorRT libraries installed.  For example, TensorRT v5.1.6 (from JetPack-4.2.2) was present on the tested Jetson Nano system.

```shell
$ ls /usr/lib/aarch64-linux-gnu/libnvinfer.so*
/usr/lib/aarch64-linux-gnu/libnvinfer.so
/usr/lib/aarch64-linux-gnu/libnvinfer.so.5
/usr/lib/aarch64-linux-gnu/libnvinfer.so.5.1.6
```

Furthermore, the demo programs require the 'cv2' (OpenCV) module in python3.  You could refer to [Installing OpenCV 3.4.6 on Jetson Nano](https://jkjung-avt.github.io/opencv-on-nano/) for how to install opencv-3.4.6 on the Jetson system.

Lastly, if you plan to run demo #3 (ssd), you'd also need to have 'tensorflowi-1.x' installed.  You could refer to [Building TensorFlow 1.12.2 on Jetson Nano](https://jkjung-avt.github.io/build-tensorflow-1.12.2/) for how to install tensorflow-1.12.2 on the Jetson Nano/TX2.

<a name="googlenet"></a>
Demo #1: googlenet
------------------

This demo illustrates how to convert a prototxt file and a caffemodel file into a tensorrt engine file, and to classify images with the optimized tensorrt engine.

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

This demo builds upon the previous example.  It converts 3 sets of prototxt and caffemodel files into 3 tensorrt engines, namely the PNet, RNet and ONet.  Then it combines the 3 engine files to implement MTCNN, a very good face detector.

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

<a name="ssd"></a>
Demo #3: ssd
------------

This demo shows how to convert trained tensorflow Single-Shot Multibox Detector (SSD) models through UFF to TensorRT engines, and to do real-time object detection with the optimized engines.

NOTE: This particular demo requires TensorRT 'Python API'.  So, unlike the previous 2 demos, this one only works for TensorRT 5.x on Jetson Nano/TX2.  In other words, it only works on Jetson systems properly set up with JetPack-4.x, but **not** Jetson-3.x or earlier versions.

Assuming this repository has been cloned at `${HOME}/project/tensorrt_demos`, follow these steps:

1. Install requirements (pycuda, etc.) and build TensorRT engines.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/ssd
   $ ./install.sh
   $ python3 ./build_engines.py
   ```

2. Run the `trt_ssd.py` demo program.  The demo supports 2 models: either 'coco' or 'egohands'.  For example, I tested the 'coco' model with the previous 'huskies' picture.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos
   $ python3 trt_ssd.py --model coco --image \
                        --filename ${HOME}/project/tf_trt_models/examples/detection/data/huskies.jpg
   ```

   Here's the result.  (Frame rate was around 22.8 fps on Jetson Nano, which is pretty good.)

   ![Huskies detected](https://raw.githubusercontent.com/jkjung-avt/tensorrt_demos/master/doc/huskies.png)

   I also tested the 'egohands' (hand detector) model with a video clip from YouTube, and got the following result.  Again, frame rate (27~28 fps) was good.  But the detection didn't seem very accurate though :-(

   ```shell
   $ python3 trt_ssd.py --model egohands --file \
                        --filename ${HOME}/Videos/Secret_Handshake_with_Justin_Bieber.mp4
   ```

   [![Handshake detected](https://raw.githubusercontent.com/jkjung-avt/tensorrt_demos/master/doc/handshake.png)](https://youtu.be/Fv6OSf0QNmU)

3. The `trt_ssd.py` demo program could also take various image inputs.  Refer to step 5 in Demo #1 again.
