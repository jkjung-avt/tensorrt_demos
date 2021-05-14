* Convert PyTorch model to ONNX

1. Create venv-onnx
2. `python -m torch2onnx.export modnet_photographic_portrait_matting.ckpt modnet.onnx`

* Build TensorRT engine for 7.1

1. `git submodule update --init --recursive`
2. Patch CMakeLists.txt
3. Build onnx-tensorrt
   ```
   $ mkdir -p onnx-tensorrt/build
   $ cd onnx-tensorrt/build
   $ cmake -DCMAKE_CXX_FLAGS=-I/usr/local/cuda/targets/aarch64-linux/include \
           -DONNX_NAMESPACE=onnx2trt_onnx ..
   $ make
   ```
4. Build engine
   ```
   $ LD_LIBRARY_PATH=$(pwd)/onnx-tensorrt/build \
     onnx-tensorrt/build/onnx2trt modnet.onnx -o modnet.engine -v
   ```
