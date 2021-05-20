"""modnet.py

Implementation of TrtMODNet class.
"""


import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda


# Code in this module is only for TensorRT 7+
if trt.__version__[0] < '7':
    raise SystemExit('TensorRT version < 7')


def _preprocess_modnet(img, input_shape):
    """Preprocess an image before TRT MODNet inferencing.

    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    img = cv2.resize(img, (input_shape[1], input_shape[0]), cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = (img - 127.5) / 127.5
    return img


def _postprocess_modnet(output, output_shape):
    """Postprocess TRT MODNet output.

    # Args
        output: inferenced output by the TensorRT engine
        output_shape: (H, W), e.g. (480, 640)
    """
    matte = cv2.resize(
        output, (output_shape[1], output_shape[0]),
        interpolation=cv2.INTER_AREA)
    return matte


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return 'Host:\n' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, context):
    """Allocates all host/device in/out buffers required for an engine."""
    assert len(engine) == 2 and engine[0] == 'input' and engine[1] == 'output'
    dtype = trt.nptype(engine.get_binding_dtype('input'))
    assert trt.nptype(engine.get_binding_dtype('output')) == dtype
    bindings = []

    dims_in = context.get_binding_shape(0)
    assert len(dims_in) == 4 and dims_in[0] == 1 and dims_in[1] == 3
    hmem_in = cuda.pagelocked_empty(trt.volume(dims_in), dtype)
    dmem_in = cuda.mem_alloc(hmem_in.nbytes)
    bindings.append(int(dmem_in))
    inputs = [HostDeviceMem(hmem_in, dmem_in)]

    dims_out = context.get_binding_shape(1)
    assert len(dims_out) == 4 and dims_out[0] == 1 and dims_out[1] == 1
    assert dims_out[2] == dims_in[2] and dims_out[3] == dims_in[3]
    hmem_out = cuda.pagelocked_empty(trt.volume(dims_out), dtype)
    dmem_out = cuda.mem_alloc(hmem_out.nbytes)
    bindings.append(int(dmem_out))
    outputs = [HostDeviceMem(hmem_out, dmem_out)]

    return bindings, inputs, outputs


def do_inference_v2(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.  Inputs and outputs are expected to be lists
    of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class TrtMODNet(object):
    """TrtMODNet class encapsulates things needed to run TRT MODNet."""

    def __init__(self, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt.

        # Arguments
            cuda_ctx: PyCUDA context for inferencing (usually only needed
                      in multi-threaded cases
        """
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()
        assert self.engine.get_binding_dtype('input') == trt.tensorrt.DataType.FLOAT

        try:
            self.context = self.engine.create_execution_context()
            self.output_shape = self.context.get_binding_shape(1)  # (1, 1, 480, 640)
            self.stream = cuda.Stream()
            self.bindings, self.inputs, self.outputs = allocate_buffers(
                self.engine, self.context)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()
        dims = self.context.get_binding_shape(0)  # 'input'
        self.input_shape = (dims[2], dims[3])

    def _load_engine(self):
        if not trt.init_libnvinfer_plugins(self.trt_logger, ''):
            raise RuntimeError('fail to init built-in plugins')
        engine_path = 'modnet/modnet.engine'
        with open(engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def infer(self, img):
        """Infer an image.

        The output is a matte (matting mask), which is a grayscale image
        with either 0 or 255 pixels.
        """
        img_resized = _preprocess_modnet(img, self.input_shape)

        self.inputs[0].host = np.ascontiguousarray(img_resized)
        if self.cuda_ctx:
            self.cuda_ctx.push()
        trt_outputs = do_inference_v2(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        output = trt_outputs[0].reshape(self.output_shape[-2:])
        return _postprocess_modnet(output, img.shape[:2])
