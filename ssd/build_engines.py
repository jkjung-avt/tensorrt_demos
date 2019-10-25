"""build_engines.py

This script converts the SSD models (pb) to UFF and subsequently builds
the TensorRT engines.

Inputs : ssd_mobilenet_v1_coco.pb
         ssd_mobilenet_v1_egohands.pb
Outputs: TRT_ssd_mobilenet_v1_coco.bin
         TRT_ssd_mobilenet_v1_egohands.bin
"""


import os
import ctypes

import uff
import tensorrt as trt
import graphsurgeon as gs


DIR_NAME = os.path.dirname(__file__)
LIB_FILE = os.path.abspath(os.path.join(DIR_NAME, 'libflattenconcat.so'))
MODEL_SPECS = {
    'coco': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v1_coco.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'tmp_coco.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssd_mobilenet_v1_coco.bin')),
        'num_classes': 91,
    },
    'egohands': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v1_egohands.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'tmp_egohands.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssd_mobilenet_v1_egohands.bin')),
        'num_classes': 2,
    },
}
INPUT_DIMS = (3, 300, 300)
LAYOUT = 7


def add_plugin(graph, model):
    """add_plugin

    Code taken from: https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v1_coco_2018_01_28.py
    """
    if model not in ('coco', 'egohands'):
        raise ValueError('Bad model name')
    minSize = 0.1 if model == 'egohands' else 0.2
    maxSize = 0.9 if model == 'egohands' else 0.95
    numClasses = 2 if model == 'egohands' else 91

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

    Postprocessor = gs.create_plugin_node(
        name="Postprocessor",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=0.3,  # was 1e-8
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=numClasses,  # was 91
        inputOrder=[0, 2, 1],
        confSigmoid=1,
        isNormalized=1
    )

    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        axis=2
    )

    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
    )

    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
    )

    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": Postprocessor,
        "Preprocessor": Input,
        "ToFloat": Input,
        "image_tensor": Input,
        "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,  # for 'coco'
        "Concatenate": concat_priorbox,  # for 'egohands'
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }

    graph.collapse_namespaces(namespace_plugin_map)
    graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
    graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")
    if model != 'egohands':
        graph.find_nodes_by_name("Input")[0].input.remove("image_tensor:0")

    return graph


def main():
    # initialize
    ctypes.CDLL(LIB_FILE)
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    #runtime = trt.Runtime(TRT_LOGGER)

    # compile models into TensorRT
    for model, spec in MODEL_SPECS.items():
        dynamic_graph = add_plugin(
            gs.DynamicGraph(spec['input_pb']),
            model)
        _ = uff.from_tensorflow(
            dynamic_graph.as_graph_def(),
            ['Postprocessor'],
            output_filename=spec['tmp_uff'])
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
