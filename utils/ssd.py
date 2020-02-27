"""ssd.py

This module implements the TrtSSD class.
"""


import ctypes

import numpy as np
import cv2
import tensorflow as tf
import tensorrt as trt
import pycuda.driver as cuda


def _preprocess_trt(img, shape=(300, 300)):
    """Preprocess an image before TRT SSD inferencing."""
    img = cv2.resize(img, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img *= (2.0/255.0)
    img -= 1.0
    return img


def _postprocess_trt(img, output, conf_th, output_layout):
    """Postprocess TRT SSD output."""
    img_h, img_w, _ = img.shape
    boxes, confs, clss = [], [], []
    for prefix in range(0, len(output), output_layout):
        #index = int(output[prefix+0])
        conf = float(output[prefix+2])
        if conf < conf_th:
            continue
        x1 = int(output[prefix+3] * img_w)
        y1 = int(output[prefix+4] * img_h)
        x2 = int(output[prefix+5] * img_w)
        y2 = int(output[prefix+6] * img_h)
        cls = int(output[prefix+1])
        boxes.append((x1, y1, x2, y2))
        confs.append(conf)
        clss.append(cls)
    return boxes, confs, clss


class TrtSSD(object):
    """TrtSSD class encapsulates things needed to run TRT SSD."""

    def _load_plugins(self):
        if trt.__version__[0] < '7':
            ctypes.CDLL("ssd/libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        TRTbin = 'ssd/TRT_%s.bin' % self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()

    def __init__(self, model, input_shape, output_layout=7):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.input_shape = input_shape
        self.output_layout = output_layout
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()

    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs

    def detect(self, img, conf_th=0.3):
        """Detect objects in the input image."""
        img_resized = _preprocess_trt(img, self.input_shape)
        np.copyto(self.host_inputs[0], img_resized.ravel())

        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        output = self.host_outputs[0]
        return _postprocess_trt(img, output, conf_th, self.output_layout)


def _preprocess_tf(img, shape=(300, 300)):
    """Preprocess an image before TensorFlow SSD inferencing."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    return img


def _postprocess_tf(img, boxes, scores, classes, conf_th):
    """Postprocess TensorFlow SSD output."""
    h, w, _ = img.shape
    out_boxes = boxes[0] * np.array([h, w, h, w])
    out_boxes = out_boxes.astype(np.int32)
    out_boxes = out_boxes[:, [1, 0, 3, 2]]  # swap x's and y's
    out_confs = scores[0]
    out_clss = classes[0].astype(np.int32)

    # only return bboxes with confidence score above threshold
    mask = np.where(out_confs >= conf_th)
    return out_boxes[mask], out_confs[mask], out_clss[mask]


class TfSSD(object):
    """TfSSD class encapsulates things needed to run TensorFlow SSD."""

    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape

        # load detection graph
        ssd_graph = tf.Graph()
        with ssd_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile('ssd/%s.pb' % model, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

        # define input/output tensors
        self.image_tensor = ssd_graph.get_tensor_by_name('image_tensor:0')
        self.det_boxes = ssd_graph.get_tensor_by_name('detection_boxes:0')
        self.det_scores = ssd_graph.get_tensor_by_name('detection_scores:0')
        self.det_classes = ssd_graph.get_tensor_by_name('detection_classes:0')

        # create the session for inferencing
        self.sess = tf.Session(graph=ssd_graph)

    def __del__(self):
        self.sess.close()

    def detect(self, img, conf_th):
        img_resized = _preprocess_tf(img, self.input_shape)
        boxes, scores, classes = self.sess.run(
            [self.det_boxes, self.det_scores, self.det_classes],
            feed_dict={self.image_tensor: np.expand_dims(img_resized, 0)})
        return _postprocess_tf(img, boxes, scores, classes, conf_th)
