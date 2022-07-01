# tensorrt_demos
-----------------
### installation for yolov3 & yolov4 


<a name="prerequisite"></a>
Prerequisite
------------

The code in this repository was tested on Jetson Nano, TX2, and Xavier NX DevKits.  In order to run the demos below, first make sure you have the proper version of image (JetPack) installed on the target Jetson system.  For example, [Setting up Jetson Nano: The Basics](https://jkjung-avt.github.io/setting-up-nano/) and [Setting up Jetson Xavier NX](https://jkjung-avt.github.io/setting-up-xavier-nx/).

More specifically, the target Jetson system must have TensorRT libraries installed.
* For yolov3 & yolov4 it requires TensorRT 6.x+.
* Demo #1 part 1: INT8 requires TensorRT 6.x+ and only works on GPUs with CUDA compute 6.1+.
* Demo #2 part 2: DLA core requires TensorRT 7.x+ (is only tested on Jetson Xavier NX).


You could check which version of TensorRT has been installed on your Jetson system by looking at file names of the libraries.  For example, TensorRT v5.1.6 (JetPack-4.2.2) was present on one of my Jetson Nano DevKits.

```shell
$ ls /usr/lib/aarch64-linux-gnu/libnvinfer.so*
/usr/lib/aarch64-linux-gnu/libnvinfer.so
/usr/lib/aarch64-linux-gnu/libnvinfer.so.5
/usr/lib/aarch64-linux-gnu/libnvinfer.so.5.1.6
```

Furthermore, all demo programs in this repository require "cv2" (OpenCV) module for python3.  You could use the "cv2" module which came in the JetPack.  Or, if you'd prefer building your own, refer to [Installing OpenCV 3.4.6 on Jetson Nano](https://jkjung-avt.github.io/opencv-on-nano/) for how to build from source and install opencv-3.4.6 on your Jetson system.


<a name="yolov3"></a>
Demo #: YOLOv3
---------------

(Merged with Demo #: YOLOv4...)

<a name="yolov4"></a>
Demo #2: YOLOv4
---------------



Assuming this repository has been cloned at "${HOME}/project/tensorrt_demos", follow these steps:

1. Install "pycuda" in case you haven't done so in Demo #3.  Note that the installation script resides in the "ssd" folder.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/ssd
   $ ./install_pycuda.sh
   ```

2. Install **version "1.4.1" (not the latest version)** of python3 **"onnx"** module.  Note that the "onnx" module would depend on "protobuf" as stated in the [Prerequisite](#prerequisite) section.  Reference: [information provided by NVIDIA](https://devtalk.nvidia.com/default/topic/1052153/jetson-nano/tensorrt-backend-for-onnx-on-jetson-nano/post/5347666/#5347666).

   ```shell
   $ sudo pip3 install onnx==1.4.1
   ```

3. Go to the "plugins/" subdirectory and build the "yolo_layer" plugin.  When done, a "libyolo_layer.so" would be generated.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/plugins
   $ make
   ```

4. Download the pre-trained yolov3/yolov4 COCO models and convert the targeted model to ONNX and then to TensorRT engine.  I use "yolov4-416" as example below.  (Supported models: "yolov3-tiny-288", "yolov3-tiny-416", "yolov3-288", "yolov3-416", "yolov3-608", "yolov3-spp-288", "yolov3-spp-416", "yolov3-spp-608", "yolov4-tiny-288", "yolov4-tiny-416", "yolov4-288", "yolov4-416", "yolov4-608", "yolov4-csp-256", "yolov4-csp-512", "yolov4x-mish-320", "yolov4x-mish-640", and [custom models](https://jkjung-avt.github.io/trt-yolo-custom-updated/) such as "yolov4-416x256".)

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/yolo
   $ ./download_yolo.sh
   $ python3 yolo_to_onnx.py -m yolov4-416
   $ python3 onnx_to_tensorrt.py -m yolov4-416
   ```

   The last step ("onnx_to_tensorrt.py") takes a little bit more than half an hour to complete on my Jetson Nano DevKit.  When that is done, the optimized TensorRT engine would be saved as "yolov4-416.trt".

   In case "onnx_to_tensorrt.py" fails (process "Killed" by Linux kernel), it could likely be that the Jetson platform runs out of memory during conversion of the TensorRT engine.  This problem might be solved by adding a larger swap file to the system.  Reference: [Process killed in onnx_to_tensorrt.py Demo#5](https://github.com/jkjung-avt/tensorrt_demos/issues/344).

## inference
#### add path of required arguments inside .py file
### using simple python.py file    
   ```shell
   $ cd ${HOME}/project/tensorrt_demos
   $ python3 trt_ad_yolo_image.py
   $ python3 trt_ad_yolo_video.py 
           
   ```


### using argument 
5. Test the TensorRT "yolov4-416" engine with the "dog.jpg" image.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos
   $ wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg -O ${HOME}/Pictures/dog.jpg
   $ python3 trt_yolo.py --image ${HOME}/Pictures/dog.jpg \
                         -m yolov4-416
   ```
   
   
   This is a screenshot of the demo against JetPack-4.4, i.e. TensorRT 7.

   !["yolov4-416" detection result on dog.jpg](doc/dog_trt_yolov4_416.jpg)

6. The "trt_yolo.py" demo program could also take various image inputs.  Refer to step 5 in Demo #1 again.

   For example, I tested my own custom trained ["yolov4-crowdhuman-416x416"](https://github.com/jkjung-avt/yolov4_crowdhuman) TensorRT engine with the "Avengers: Infinity War" movie trailer:

   [![Testing with the Avengers: Infinity War trailer](https://raw.githubusercontent.com/jkjung-avt/yolov4_crowdhuman/master/doc/infinity_war.jpg)](https://youtu.be/7Qr_Fq18FgM)




7. (Optional) Test other models than "yolov4-416".

8. (Optional) If you would like to stream TensorRT YOLO detection output over the network and view the results on a remote host, check out my [trt_yolo_mjpeg.py example](https://github.com/jkjung-avt/tensorrt_demos/issues/226).

9. Similar to step 5 of Demo #3, I created an "eval_yolo.py" for evaluating mAP of the TensorRT yolov3/yolov4 engines.  Refer to [README_mAP.md](README_mAP.md) for details.

   ```shell
   $ python3 eval_yolo.py -m yolov3-tiny-288
   $ python3 eval_yolo.py -m yolov4-tiny-416
   ......
   $ python3 eval_yolo.py -m yolov4-608
   $ python3 eval_yolo.py -l -m yolov4-csp-256
   ......
   $ python3 eval_yolo.py -l -m yolov4x-mish-640
   ```

   I evaluated all these TensorRT yolov3/yolov4 engines with COCO "val2017" data and got the following results.  I also checked the FPS (frames per second) numbers on my Jetson Nano DevKit with JetPack-4.4 (TensorRT 7).

   | TensorRT engine         | mAP @<br>IoU=0.5:0.95 |  mAP @<br>IoU=0.5  | FPS on Nano |
   |:------------------------|:---------------------:|:------------------:|:-----------:|
   | yolov3-tiny-288 (FP16)  |         0.077         |        0.158       |     35.8    |
   | yolov3-tiny-416 (FP16)  |         0.096         |        0.202       |     25.5    |
   | yolov3-288 (FP16)       |         0.331         |        0.601       |     8.16    |
   | yolov3-416 (FP16)       |         0.373         |        0.664       |     4.93    |
   | yolov3-608 (FP16)       |         0.376         |        0.665       |     2.53    |
   | yolov3-spp-288 (FP16)   |         0.339         |        0.594       |     8.16    |
   | yolov3-spp-416 (FP16)   |         0.391         |        0.664       |     4.82    |
   | yolov3-spp-608 (FP16)   |         0.410         |        0.685       |     2.49    |
   | yolov4-tiny-288 (FP16)  |         0.179         |        0.344       |     36.6    |
   | yolov4-tiny-416 (FP16)  |         0.196         |        0.387       |     25.5    |
   | yolov4-288 (FP16)       |         0.376         |        0.591       |     7.93    |
   | yolov4-416 (FP16)       |         0.459         |        0.700       |     4.62    |
   | yolov4-608 (FP16)       |         0.488         |        0.736       |     2.35    |
   | yolov4-csp-256 (FP16)   |         0.336         |        0.502       |     12.8    |
   | yolov4-csp-512 (FP16)   |         0.436         |        0.630       |     4.26    |
   | yolov4x-mish-320 (FP16) |         0.400         |        0.581       |     4.79    |
   | yolov4x-mish-640 (FP16) |         0.470         |        0.668       |     1.46    |

10. Check out my blog posts for implementation details:

   * [TensorRT ONNX YOLOv3](https://jkjung-avt.github.io/tensorrt-yolov3/)
   * [TensorRT YOLOv4](https://jkjung-avt.github.io/tensorrt-yolov4/)
   * [Verifying mAP of TensorRT Optimized SSD and YOLOv3 Models](https://jkjung-avt.github.io/trt-detection-map/)
   * For training your own custom yolov4 model: [Custom YOLOv4 Model on Google Colab](https://jkjung-avt.github.io/colab-yolov4/)
   * For adapting the code to your own custom trained yolov3/yolov4 models: [TensorRT YOLO For Custom Trained Models (Updated)](https://jkjung-avt.github.io/trt-yolo-custom-updated/)

<a name="int8_and_dla"></a>
Demo #6: Using INT8 and DLA core
--------------------------------

NVIDIA introduced [INT8 TensorRT inferencing](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf) since CUDA compute 6.1+.  For the embedded Jetson product line, INT8 is available on Jetson AGX Xavier and Xavier NX.  In addition, NVIDIA further introduced [Deep Learning Accelerator (NVDLA)](http://nvdla.org/) on Jetson Xavier NX.  I tested both features on my Jetson Xavier NX DevKit, and shared the source code in this repo.

Please make sure you have gone through the steps of [Demo #5](#yolov4) and are able to run TensorRT yolov3/yolov4 engines successfully, before following along:

1. In order to use INT8 TensorRT, you'll first have to prepare some images for "calibration".  These images for calibration should cover all distributions of possible image inputs at inference time.  According to [official documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimizing_int8_c), 500 of such images are suggested by NVIDIA.  As an example, I used 1,000 images from the COCO "val2017" dataset for that purpose.  Note that I've previously downloaded the "val2017" images for [mAP evaluation](README_mAP.md).

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/yolo
   $ mkdir calib_images
   ### randomly pick and copy over 1,000 images from "val207"
   $ for jpg in $(ls -1 ${HOME}/data/coco/images/val2017/*.jpg | sort -R | head -1000); do \
       cp ${HOME}/data/coco/images/val2017/${jpg} calib_images/; \
     done
   ```

   When this is done, the 1,000 images for calibration should be present in the "${HOME}/project/tensorrt_demos/yolo/calib_images/" directory.

2. Build the INT8 TensorRT engine.  I use the "yolov3-608" model in the example commands below.  (I've also created a "build_int8_engines.sh" script to facilitate building multiple INT8 engines at once.)  Note that building the INT8 TensorRT engine on Jetson Xavier NX takes quite long.  By enabling verbose logging ("-v"), you would be able to monitor the progress more closely.

   ```
   $ ln -s yolov3-608.cfg yolov3-int8-608.cfg
   $ ln -s yolov3-608.onnx yolov3-int8-608.onnx
   $ python3 onnx_to_tensorrt.py -v --int8 -m yolov3-int8-608
   ```

3. (Optional) Build the TensorRT engines for the DLA cores.  I use the "yolov3-608" model as example again.  (I've also created a "build_dla_engines.sh" script for building multiple DLA engines at once.)

   ```
   $ ln -s yolov3-608.cfg yolov3-dla0-608.cfg
   $ ln -s yolov3-608.onnx yolov3-dla0-608.onnx
   $ python3 onnx_to_tensorrt.py -v --int8 --dla_core 0 -m yolov3-dla0-608
   $ ln -s yolov3-608.cfg yolov3-dla1-608.cfg
   $ ln -s yolov3-608.onnx yolov3-dla1-608.onnx
   $ python3 onnx_to_tensorrt.py -v --int8 --dla_core 1 -m yolov3-int8-608
   ```

4. Test the INT8 TensorRT engine with the "dog.jpg" image.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos
   $ python3 trt_yolo.py --image ${HOME}/Pictures/dog.jpg \
                         -m yolov3-int8-608
   ```

   (Optional) Also test the DLA0 and DLA1 TensorRT engines.

   ```shell
   $ python3 trt_yolo.py --image ${HOME}/Pictures/dog.jpg \
                         -m yolov3-dla0-608
   $ python3 trt_yolo.py --image ${HOME}/Pictures/dog.jpg \
                         -m yolov3-dla1-608
   ```

5. Evaluate mAP of the INT8 and DLA TensorRT engines.

   ```shell
   $ python3 eval_yolo.py -m yolov3-int8-608
   $ python3 eval_yolo.py -m yolov3-dla0-608
   $ python3 eval_yolo.py -m yolov3-dla1-608
   ```

6. I tested the 5 original yolov3/yolov4 models on my Jetson Xavier NX DevKit with JetPack-4.4 (TensorRT 7.1.3.4).  Here are the results.

   The following **FPS numbers** were measured under "15W 6CORE" mode, with CPU/GPU clocks set to maximum value (`sudo jetson_clocks`).

   | TensorRT engine  |   FP16   |   INT8   |   DLA0   |   DLA1   |
   |:-----------------|:--------:|:--------:|:--------:|:--------:|
   | yolov3-tiny-416  |    58    |    65    |    42    |    42    |
   | yolov3-608       |   15.2   |   23.1   |   14.9   |   14.9   |
   | yolov3-spp-608   |   15.0   |   22.7   |   14.7   |   14.7   |
   | yolov4-tiny-416  |    57    |    60    |     X    |     X    |
   | yolov4-608       |   13.8   |   20.5   |   8.97   |   8.97   |
   | yolov4-csp-512   |   19.8   |   27.8   |    --    |    --    |
   | yolov4x-mish-640 |   9.01   |   14.1   |    --    |    --    |

   And the following are **"mAP@IoU=0.5:0.95" / "mAP@IoU=0.5"** of those TensorRT engines.

   | TensorRT engine  |       FP16      |       INT8      |       DLA0      |       DLA1      |
   |:-----------------|:---------------:|:---------------:|:---------------:|:---------------:|
   | yolov3-tiny-416  |  0.096 / 0.202  |  0.094 / 0.198  |  0.096 / 0.199  |  0.096 / 0.199  |
   | yolov3-608       |  0.376 / 0.665  |  0.378 / 0.670  |  0.378 / 0.670  |  0.378 / 0.670  |
   | yolov3-spp-608   |  0.410 / 0.685  |  0.407 / 0.681  |  0.404 / 0.676  |  0.404 / 0.676  |
   | yolov4-tiny-416  |  0.196 / 0.387  |  0.190 / 0.376  |        X        |        X        |
   | yolov4-608       |  0.488 / 0.736  | *0.317 / 0.507* |  0.474 / 0.727  |  0.473 / 0.726  |
   | yolov4-csp-512   |  0.436 / 0.630  |  0.391 / 0.577  |       --        |       --        |
   | yolov4x-mish-640 |  0.470 / 0.668  |  0.434 / 0.631  |       --        |       --        |

7. Issues:

   * For some reason, I'm not able to build DLA TensorRT engines for the "yolov4-tiny-416" model.  I have [reported the issue](https://forums.developer.nvidia.com/t/problem-building-tensorrt-engines-for-dla-core/155749) to NVIDIA.
   * There is no method in TensorRT 7.1 Python API to specifically set DLA core at inference time.  I also [reported this issue](https://forums.developer.nvidia.com/t/no-method-in-tensorrt-python-api-for-setting-dla-core-for-inference/155874) to NVIDIA.  When testing, I simply deserialize the TensorRT engines onto Jetson Xavier NX.  I'm not 100% sure whether the engine is really executed on DLA core 0 or DLA core 1.
   * mAP of the INT8 TensorRT engine of the "yolov4-608" model is not good.  Originally, I thought it was [an issue of TensorRT library's handling of "Concat" nodes](https://forums.developer.nvidia.com/t/concat-in-caffe-parser-is-wrong-when-working-with-int8-calibration/142639/3?u=jkjung13).  But after some more investigation, I saw that was not the case.  Currently, I'm still not sure what the problem is...


Licenses
--------

1. I referenced source code of [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) samples to develop most of the demos in this repository.  Those NVIDIA samples are under [Apache License 2.0](https://github.com/NVIDIA/TensorRT/blob/master/LICENSE).
2. [GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet): "This model is released for unrestricted use."
3. [MTCNN](https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT): license not specified.  Note [the original MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment) is under [MIT License](https://github.com/kpzhang93/MTCNN_face_detection_alignment/blob/master/LICENSE).
4. [TensorFlow Object Detection Models](https://github.com/tensorflow/models/tree/master/research/object_detection): [Apache License 2.0](https://github.com/tensorflow/models/blob/master/LICENSE).
5. YOLOv3/YOLOv4 models ([DarkNet](https://github.com/AlexeyAB/darknet)): [YOLO LICENSE](https://github.com/AlexeyAB/darknet/blob/master/LICENSE).
6. [MODNet](https://github.com/ZHKKKe/MODNet): [Creative Commons Attribution NonCommercial ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.
7. For the rest of the code (developed by jkjung-avt and other contributors): [MIT License](https://github.com/jkjung-avt/tensorrt_demos/blob/master/LICENSE).
