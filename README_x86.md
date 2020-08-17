# Instructions for x86_64 platforms

All demos in this repository, with minor tweaks, should also work on x86_64 platforms with NVIDIA GPU(s).  Here is a list of required modifications if you'd like to run the demos on an x86_64 PC/server.


Make sure you have TensorRT installed properly on your x86_64 system.  You could follow NVIDIA's official [Installation Guide :: NVIDIA Deep Learning TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) documentation.

Demo #1 (GoogLeNet) and #2 (MTCNN)
----------------------------------

1. Set `TENSORRT_INCS` and `TENSORRT_LIBS` in "common/Makefile.config" correctly for your x86_64 system.  More specifically, you should find the following lines in "common/Mafefile.config" and modify them if needed.

   ```
   # These are the directories where I installed TensorRT on my x86_64 PC.
   TENSORRT_INCS=-I"/usr/local/TensorRT-7.1.3.4/include"
   TENSORRT_LIBS=-L"/usr/local/TensorRT-7.1.3.4/lib"
   ```

2. Set `library_dirs` and `include_dirs` in "setup.py".  More specifically, you should check and make sure the 2 TensorRT path lines are correct.

   ```python
   library_dirs = [
       '/usr/local/cuda/lib64',
       '/usr/local/TensorRT-7.1.3.4/lib',  # for my x86_64 PC
       '/usr/local/lib',
   ]
   ......
   include_dirs = [
       # in case the following numpy include path does not work, you
       # could replace it manually with, say,
       # '-I/usr/local/lib/python3.6/dist-packages/numpy/core/include',
       '-I' + numpy.__path__[0] + '/core/include',
       '-I/usr/local/cuda/include',
       '-I/usr/local/TensorRT-7.1.3.4/include',  # for my x86_64 PC
       '-I/usr/local/include',
   ]
   ```

3. Follow the steps in the original [README.md](https://github.com/jkjung-avt/tensorrt_demos/blob/master/README.md), and the demos should work on x86_64 as well.

Demo #3 (SSD)
-------------

1. Make sure to follow NVIDIA's official [Installation Guide :: NVIDIA Deep Learning TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) documentation and pip3 install "tensorrt", "uff", and "graphsurgeon" packages.

2. Patch `/usr/local/lib/python3.?/dist-packages/graphsurgeon/node_manipulation.py` by adding the following line (around line #42):

   ```python
    def shape(node):
        ......
        node.name = name or node.name
        node.op = op or node.op or node.name
   +    node.attr["dtype"].type = 1
        for key, val in kwargs.items():
        ......
   ```
3. (I think this step is only required for TensorRT 6 or earlier versions.)  Re-build `libflattenconcat.so` from TensorRT's 'python/uff_ssd' sample source code.  For example,

   ```shell
   $ mkdir -p ${HOME}/src/TensorRT-5.1.5.0
   $ cp -r /usr/local/TensorRT-5.1.5.0/samples ${HOME}/src/TensorRT-5.1.5.0
   $ cd ${HOME}/src/TensorRT-5.1.5.0/samples/python/uff_ssd
   $ mkdir build
   $ cd build
   $ cmake -D NVINFER_LIB=/usr/local/TensorRT-5.1.5.0/lib/libnvinfer.so \
           -D TRT_INCLUDE=/usr/local/TensorRT-5.1.5.0/include ..
   $ make
   $ cp libflattenconcat.so ${HOME}/project/tensorrt_demos/ssd/
   ```

4. Install "pycuda".

   ```shell
   $ sudo apt-get install -y build-essential python-dev
   $ sudo apt-get install -y libboost-python-dev libboost-thread-dev
   $ sudo pip3 install setuptools
   $ export boost_pylib=$(basename /usr/lib/x86_64-linux-gnu/libboost_python3-py3?.so)
   $ export boost_pylibname=${boost_pylib%.so}
   $ export boost_pyname=${boost_pylibname/lib/}
   $ cd ${HOME}/src
   $ wget https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
   $ tar xzvf pycuda-2019.1.2.tar.gz
   $ cd pycuda-2019.1.2
   $ ./configure.py --python-exe=/usr/bin/python3 \
                    --cuda-root=/usr/local/cuda \
                    --cudadrv-lib-dir=/usr/lib/x86_64-linux-gnu \
                    --boost-inc-dir=/usr/include \
                    --boost-lib-dir=/usr/lib/x86_64-linux-gnu \
                    --boost-python-libname=${boost_pyname} \
                    --boost-thread-libname=boost_thread \
                    --no-use-shipped-boost
   $ make -j4
   $ python3 setup.py build
   $ sudo python3 setup.py install
   $ python3 -c "import pycuda; print('pycuda version:', pycuda.VERSION)"
   ```

5. Follow the steps in the original [README.md](https://github.com/jkjung-avt/tensorrt_demos/blob/master/README.md) but skip `install.sh`.  You should be able to build the SSD TensorRT engines and run them on on x86_64 as well.

Demo #4 (YOLOv3) & Demo #5 (YOLOv4)
-----------------------------------

Checkout "plugins/Makefile".  You'll need to make sure in "plugins/Makefile":

* CUDA `compute` is set correctly for your GPU (reference: [CUDA GPUs | NVIDIA Developer]());
* `TENSORRT_INCS` and `TENSORRT_LIBS` point to the right paths.

```
......
else ifeq ($(cpu_arch), x86_64)  # x86_64 PC
  $(warning "compute=75" is for GeForce RTX-2080 Ti.  Please make sure CUDA compute is set correctly for your system in the Makefile.)
  compute=75
......
NVCCFLAGS=-m64 -gencode arch=compute_$(compute),code=sm_$(compute) \
               -gencode arch=compute_$(compute),code=compute_$(compute)
......
# These are the directories where I installed TensorRT on my x86_64 PC.
TENSORRT_INCS=-I"/usr/local/TensorRT-7.1.3.4/include"
TENSORRT_LIBS=-L"/usr/local/TensorRT-7.1.3.4/lib"
......
```

Otherwise, you should be able to follow the steps in the original [README.md](https://github.com/jkjung-avt/tensorrt_demos/blob/master/README.md) to get these 2 demos working.
