"""onnx_to_tensorrt.py

For converting a MODNet ONNX model to a TensorRT engine.
"""


import os
import argparse

import tensorrt as trt

if trt.__version__[0] < '7':
    raise SystemExit('TensorRT version < 7')


BATCH_SIZE = 1


def parse_args():
    """Parse command-line options and arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose output (for debugging) [False]')
    parser.add_argument(
        '--int8', action='store_true',
        help='build INT8 TensorRT engine [False]')
    parser.add_argument(
        '--dla_core', type=int, default=-1,
        help='id of DLA core for inference, ranging from 0 to N-1 [-1]')
    parser.add_argument(
        '--width', type=int, default=640,
        help='input image width of the model [640]')
    parser.add_argument(
        '--height', type=int, default=480,
        help='input image height of the model [480]')
    parser.add_argument(
        'input_onnx', type=str, help='the input onnx file')
    parser.add_argument(
        'output_engine', type=str, help='the output TensorRT engine file')
    args = parser.parse_args()
    return args


def load_onnx(onnx_file_path):
    """Read the ONNX file."""
    with open(onnx_file_path, 'rb') as f:
        return f.read()


def set_net_batch(network, batch_size):
    """Set network input batch size."""
    shape = list(network.get_input(0).shape)
    shape[0] = batch_size
    network.get_input(0).shape = shape
    return network


def build_engine(onnx_file_path, width, height,
                 do_int8=False, dla_core=False, verbose=False):
    """Build a TensorRT engine from ONNX using the older API."""
    onnx_data = load_onnx(onnx_file_path)

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    EXPLICIT_BATCH = [1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if do_int8 and not builder.platform_has_fast_int8:
            raise RuntimeError('INT8 not supported on this platform')
        if not parser.parse(onnx_data):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
        network = set_net_batch(network, BATCH_SIZE)

        builder.max_batch_size = BATCH_SIZE
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            'Input',                         # input tensor name
            (BATCH_SIZE, 3, height, width),  # min shape
            (BATCH_SIZE, 3, height, width),  # opt shape
            (BATCH_SIZE, 3, height, width))  # max shape
        config.add_optimization_profile(profile)
        if do_int8:
            raise RuntimeError('INT8 not implemented yet')
        if dla_core >= 0:
            raise RuntimeError('DLA_core not implemented yet')
        engine = builder.build_engine(network, config)

        return engine


def main():
    args = parse_args()
    if not os.path.isfile(args.input_onnx):
        raise FileNotFoundError(args.input_onnx)

    print('Building an engine.  This would take a while...')
    print('(Use "-v" or "--verbose" to enable verbose logging.)')
    engine = build_engine(
        args.input_onnx, args.width, args.height,
        args.int8, args.dla_core, args.verbose)
    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')
    print('Completed creating engine.')

    with open(args.output_engine, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % args.output_engine)


if __name__ == '__main__':
    main()
