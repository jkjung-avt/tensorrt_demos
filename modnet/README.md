# How to convert the original PyTorch MODNet model to ONNX

The original pre-trained PyTorch MODNet model comes from [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet).  Note that this pre-trained model is under [Creative Commons Attribution NonCommercial ShareAlike 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

You could use the script in this repository to convert the original PyTorch model to ONNX.  I recommend to do such conversion within a python3 virtual environment, since you'd need to use some specific versions of pip3 packages.  Below is a step-by-step guide about how to build the python3 virtual environment and then convert the PyTorch MODNet model to ONNX.

1. Make sure python3 "venv" module is installed.

   ```shell
   $ sudo apt install python3-venv
   ```

2. Create a virtual environment named "venv-onnx" and activate it.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/modnet
   $ python3 -m venv venv-onnx
   $ source venv-onnx/bin/activate
   ```

   At this point, you should have entered the virtual environment and would see shell prompt proceeded with "(venv-onnx) ".  You could do `deactivate` to quit the virtual environment when you are done using it.

   Download "torch-1.7.0-cp36-cp36m-linux_aarch64.whl" from here: [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048).  Then install all required packages into the virtual environment.  (Note the following should be done inside the "venv-onnx" virtual environment.)

   ```shell
   ### update pip to the latest version in the virtual env
   $ curl https://bootstrap.pypa.io/get-pip.py | python
   ### udpate these essential packages
   $ python -m pip install -U setuptools Cython
   ### I recommend numpy 1.16.x on Jetson
   $ python -m pip install "numpy<1.17.0"
   ### install cv2 into the virtual env
   $ cp -r /usr/lib/python3.6/dist-packages/cv2 venv-onnx/lib/python3.6/site-packages/
   ### install PyImage, onnx and onnxruntime
   $ python -m pip install PyImage onnx==1.8.1 onnxruntime==1.6.0
   ### install PyTorch v1.7.0
   $ sudo apt install libopenblas-base libopenmpi-dev
   $ python -m pip install ${HOME}/Downloads/torch-1.7.0-cp36-cp36m-linux_aarch64.whl
   ```

   In addition, you might also install [onnx-graphsurgeon](https://pypi.org/project/onnx-graphsurgeon/) and [polygraphy](https://pypi.org/project/polygraphy/) for debugging.  Otherwise, you could do some simple testing to make sure "onnx" and "torch" are working OK in the virtual env.

3. Download the pre-trained MODNet model (PyTorch checkpoint file) from the link on this page: [/ZHKKKe/MODNet/pretrained](https://github.com/ZHKKKe/MODNet/tree/master/pretrained).  I recommend using "modnet_webcam_portrait_matting.ckpt".  Just put the file in the current directory.

4. Do the conversion using the following command.  The ouput "modnet.onnx" would be generated.

   ```shell
   $ python -m torch2onnx.export modnet_webcam_portrait_matting.ckpt modnet.onnx
   ```

   By default, the "torch2onnx.export" script sets input image width and height to 512x288.  They could be modified by the "--width" and "--height" command-line options.  In addition, the "-v" command-line option could be used to enable verbose logs of `torch.onnx.export()`.
