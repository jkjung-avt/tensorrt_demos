"""build_engine.py

This script converts a SSD model (pb) to UFF and subsequently builds
the TensorRT engine.

Input : ssd_mobilenet_v[1|2]_[coco|egohands].pb
Output: TRT_ssd_mobilenet_v[1|2]_[coco|egohands].bin
"""


import os
import ctypes
import argparse

import uff
import tensorrt as trt
import graphsurgeon as gs


DIR_NAME = os.path.dirname(__file__)
LIB_FILE = os.path.abspath(os.path.join(DIR_NAME, 'libflattenconcat.so'))
MODEL_SPECS = {
    'ssd_mobilenet_v1_coco': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v1_coco.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'tmp_v1_coco.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssd_mobilenet_v1_coco.bin')),
        'num_classes': 91,
        'min_size': 0.2,
        'max_size': 0.95,
        'input_order': [0, 2, 1],  # order of loc_data, conf_data, priorbox_data
    },
    'ssd_mobilenet_v1_egohands': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v1_egohands.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'tmp_v1_egohands.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssd_mobilenet_v1_egohands.bin')),
        'num_classes': 2,
        'min_size': 0.05,
        'max_size': 0.95,
        'input_order': [0, 2, 1],  # order of loc_data, conf_data, priorbox_data
    },
    'ssd_mobilenet_v2_coco': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v2_coco.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'tmp_v2_coco.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssd_mobilenet_v2_coco.bin')),
        'num_classes': 91,
        'min_size': 0.2,
        'max_size': 0.95,
        'input_order': [1, 0, 2],  # order of loc_data, conf_data, priorbox_data
    },
    'ssd_mobilenet_v2_egohands': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v2_egohands.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'tmp_v2_egohands.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssd_mobilenet_v2_egohands.bin')),
        'num_classes': 2,
        'min_size': 0.05,
        'max_size': 0.95,
        'input_order': [0, 2, 1],  # order of loc_data, conf_data, priorbox_data
    },
}
INPUT_DIMS = (3, 300, 300)
DEBUG_UFF = False


def add_plugin(graph, model, spec):
    """add_plugin

    Reference:
    1. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v1_coco_2018_01_28.py
    2. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v2_coco_2018_03_29.py
    3. https://devtalk.nvidia.com/default/topic/1050465/jetson-nano/how-to-write-config-py-for-converting-ssd-mobilenetv2-to-uff-format/post/5333033/#5333033
    """
    numClasses = spec['num_classes']
    minSize = spec['min_size']
    maxSize = spec['max_size']
    inputOrder = spec['input_order']

    all_assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    all_identity_nodes = graph.find_nodes_by_op("Identity")
    graph.forward_inputs(all_identity_nodes)

    Input = gs.create_plugin_node(
        name="Input",
        op="Placeholder",
        shape=(1,) + INPUT_DIMS
    )

    PriorBox = gs.create_plugin_node(
        name="MultipleGridAnchorGenerator",
        op="GridAnchor_TRT",
        minSize=minSize,  # was 0.2
        maxSize=maxSize,  # was 0.95
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1, 0.1, 0.2, 0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6
    )

    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=0.3,  # was 1e-8
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=numClasses,  # was 91
        inputOrder=inputOrder,
        confSigmoid=1,
        isNormalized=1
    )

    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        axis=2
    )

    if trt.__version__[0] >= '7':
        concat_box_loc = gs.create_plugin_node(
            "concat_box_loc",
            op="FlattenConcat_TRT",
            axis=1,
            ignoreBatch=0
        )
        concat_box_conf = gs.create_plugin_node(
            "concat_box_conf",
            op="FlattenConcat_TRT",
            axis=1,
            ignoreBatch=0
        )
    else:
        concat_box_loc = gs.create_plugin_node(
            "concat_box_loc",
            op="FlattenConcat_TRT"
        )
        concat_box_conf = gs.create_plugin_node(
            "concat_box_conf",
            op="FlattenConcat_TRT"
        )

    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "ToFloat": Input,
        "image_tensor": Input,
        "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,  # for 'ssd_mobilenet_v1_coco'
        "Concatenate": concat_priorbox,  # for other models
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }

    graph.collapse_namespaces(namespace_plugin_map)
    graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
    graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")
    if model == 'ssd_mobilenet_v1_coco':
        graph.find_nodes_by_name("Input")[0].input.remove("image_tensor:0")

    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=list(MODEL_SPECS.keys()))
    args = parser.parse_args()

    # initialize
    if trt.__version__[0] < '7':
        ctypes.CDLL(LIB_FILE)
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    # compile the model into TensorRT engine
    model = args.model
    spec = MODEL_SPECS[model]
    dynamic_graph = add_plugin(
        gs.DynamicGraph(spec['input_pb']),
        model,
        spec)
    _ = uff.from_tensorflow(
        dynamic_graph.as_graph_def(),
        output_nodes=['NMS'],
        output_filename=spec['tmp_uff'],
        text=True,
        debug_mode=DEBUG_UFF)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        parser.register_input('Input', INPUT_DIMS)
        parser.register_output('MarkOutput_0')
        parser.parse(spec['tmp_uff'], network)
        engine = builder.build_cuda_engine(network)

        buf = engine.serialize()
        with open(spec['output_bin'], 'wb') as f:
            f.write(buf)


if __name__ == '__main__':
    main()
