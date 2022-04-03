"""plugins.py

I referenced the code from https://github.com/dongfangduoshou123/YoloV3-TensorRT/blob/master/seralizeEngineFromPythonAPI.py
"""


import ctypes

import numpy as np
import tensorrt as trt

from yolo_to_onnx import (is_pan_arch, DarkNetParser, get_category_num,
                          get_h_and_w, get_output_convs, get_anchors)


try:
    ctypes.cdll.LoadLibrary('../plugins/libyolo_layer.so')
except OSError as e:
    raise SystemExit('ERROR: failed to load ../plugins/libyolo_layer.so.  '
                     'Did you forget to do a "make" in the "../plugins/" '
                     'subdirectory?') from e


def get_scales(cfg_file_path):
    """Get scale_x_y's of all yolo layers from the cfg file."""
    with open(cfg_file_path, 'r') as f:
        cfg_lines = f.readlines()
    yolo_lines = [l.strip() for l in cfg_lines if l.startswith('[yolo]')]
    scale_lines = [l.strip() for l in cfg_lines if l.startswith('scale_x_y')]
    if len(scale_lines) == 0:
        return [1.0] * len(yolo_lines)
    else:
        assert len(scale_lines) == len(yolo_lines)
        return [float(l.split('=')[-1]) for l in scale_lines]


def get_new_coords(cfg_file_path):
    """Get new_coords flag of yolo layers from the cfg file."""
    with open(cfg_file_path, 'r') as f:
        cfg_lines = f.readlines()
    yolo_lines = [l.strip() for l in cfg_lines if l.startswith('[yolo]')]
    newc_lines = [l.strip() for l in cfg_lines if l.startswith('new_coords')]
    if len(newc_lines) == 0:
        return 0
    else:
        assert len(newc_lines) == len(yolo_lines)
        return int(newc_lines[-1].split('=')[-1])


def get_plugin_creator(plugin_name, logger):
    """Get the TensorRT plugin creator."""
    trt.init_libnvinfer_plugins(logger, '')
    plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
    for c in plugin_creator_list:
        if c.name == plugin_name:
            return c
    return None


def add_yolo_plugins(network, model_name, logger):
    """Add yolo plugins into a TensorRT network."""
    cfg_file_path = model_name + '.cfg'
    parser = DarkNetParser()
    layer_configs = parser.parse_cfg_file(cfg_file_path)
    num_classes = get_category_num(cfg_file_path)
    output_tensor_names = get_output_convs(layer_configs)
    h, w = get_h_and_w(layer_configs)
    if len(output_tensor_names) == 2:
        yolo_whs = [
            [w // 32, h // 32], [w // 16, h // 16]]
    elif len(output_tensor_names) == 3:
        yolo_whs = [
            [w // 32, h // 32], [w // 16, h // 16],
            [w //  8, h //  8]]
    elif len(output_tensor_names) == 4:
        yolo_whs = [
            [w // 64, h // 64], [w // 32, h // 32],
            [w // 16, h // 16], [w //  8, h //  8]]
    else:
        raise TypeError('bad number of outputs: %d' % len(output_tensor_names))
    if is_pan_arch(cfg_file_path):
        yolo_whs.reverse()
    anchors = get_anchors(cfg_file_path)
    if len(anchors) != len(yolo_whs):
        raise ValueError('bad number of yolo layers: %d vs. %d' %
                         (len(anchors), len(yolo_whs)))
    if network.num_outputs != len(anchors):
        raise ValueError('bad number of network outputs: %d vs. %d' %
                         (network.num_outputs, len(anchors)))
    scales = get_scales(cfg_file_path)
    if any([s < 1.0 for s in scales]):
        raise ValueError('bad scale_x_y: %s' % str(scales))
    if len(scales) != len(anchors):
        raise ValueError('bad number of scales: %d vs. %d' %
                         (len(scales), len(anchors)))
    new_coords = get_new_coords(cfg_file_path)

    plugin_creator = get_plugin_creator('YoloLayer_TRT', logger)
    if not plugin_creator:
        raise RuntimeError('cannot get YoloLayer_TRT plugin creator')
    old_tensors = [network.get_output(i) for i in range(network.num_outputs)]
    new_tensors = [None] * network.num_outputs
    for i, old_tensor in enumerate(old_tensors):
        input_multiplier = w // yolo_whs[i][0]
        new_tensors[i] = network.add_plugin_v2(
            [old_tensor],
            plugin_creator.create_plugin('YoloLayer_TRT', trt.PluginFieldCollection([
                trt.PluginField("yoloWidth", np.array(yolo_whs[i][0], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("yoloHeight", np.array(yolo_whs[i][1], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("inputMultiplier", np.array(input_multiplier, dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("newCoords", np.array(new_coords, dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("numClasses", np.array(num_classes, dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("numAnchors", np.array(len(anchors[i]) // 2, dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("anchors", np.ascontiguousarray(anchors[i], dtype=np.float32), trt.PluginFieldType.FLOAT32),
                trt.PluginField("scaleXY", np.array(scales[i], dtype=np.float32), trt.PluginFieldType.FLOAT32),
            ]))
        ).get_output(0)

    for new_tensor in new_tensors:
        network.mark_output(new_tensor)
    for old_tensor in old_tensors:
        network.unmark_output(old_tensor)

    return network


def add_concat(network, model_name, logger):
    """Add a final concatenation output into a TensorRT network."""
    if network.num_outputs < 2 or network.num_outputs > 4:
        raise TypeError('bad number of yolo layers: %d' % network.num_outputs)
    yolo_tensors = [network.get_output(i) for i in range(network.num_outputs)]
    concat_tensor = network.add_concatenation(yolo_tensors).get_output(0)
    for yolo_tensor in yolo_tensors:
        network.unmark_output(yolo_tensor)
    concat_tensor.name = 'detections'
    network.mark_output(concat_tensor)
    return network
