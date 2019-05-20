# tensorrt_demos

Examples demonstrating how to optimize caffe models with TensorRT and run inferencing on Jetson Nano.

Related blog posts:

* To be updated...

Table of contents
-----------------

* [Prerequisite](#prerequisite)
* [Demo #1: googlenet](#googlenet)
* [Demo #2: mtcnn](#mtcnn)

<a name="prerequisite"></a>
Prerequisite
------------

The code in this repository was tested on a Jetson Nano DevKit running the "jetson-nano-sd-r32.1-2019-03-18.zip” image (JetPack-4.2).  In order to run the demo programs below, first make sure you have the target Jetson Nano system with the proper version of image installed.  Reference: [Setting up Jetson Nano: The Basics](https://jkjung-avt.github.io/setting-up-nano/).

More specifically, the target Jetson Nano system should have TensorRT libraries installed.  For example, TensorRT v5.0.6 was present on the tested system.

```shell
$ ls /usr/lib/aarch64-linux-gnu/libnvinfer.so*
/usr/lib/aarch64-linux-gnu/libnvinfer.so
/usr/lib/aarch64-linux-gnu/libnvinfer.so.5
/usr/lib/aarch64-linux-gnu/libnvinfer.so.5.0.6
```

Furthermore, the demo programs require the 'cv2' (OpenCV) module in python3.  You could refer to [Installing OpenCV 3.4.6 on Jetson Nano](https://jkjung-avt.github.io/opencv-on-nano/) about how opencv-3.4.6 was installed on the tested Jetson Nano DevKit.

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

4. Run the `camera_trt_googlenet.py` demo program.  You could use `--help` option to display help messages.  In short, use `--usb` for USB webcam or `rtsp` for RTSP video input.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos
   $ python3 camera_trt_googlenet.py --usb --vid 0 --width 1280 --height 720
   ```

   Here's a screenshot of the demo.

   ![A Picture of a Golden Retriever](https://raw.githubusercontent.com/jkjung-avt/tensorrt_demos/master/doc/golden_retriever.png)

<a name="mtcnn"></a>
Demo #2: mtcnn
--------------

To be updated...
