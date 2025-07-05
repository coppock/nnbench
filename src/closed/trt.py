import logging
import time

import tensorrt as trt

# <https://github.com/NVIDIA/TensorRT/blob/main/samples/python/common_runtime.py>
from .common_runtime import allocate_buffers, do_inference


class Logger(trt.ILogger):
    def __init__(self):
        super().__init__()

    def log(self, severity, msg):
        logging.log(logging.CRITICAL - 10 * severity.value, msg)


def generator(model, batch_size, _args):
    with open(model, 'rb') as f, trt.Runtime(Logger()) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    logging.debug(engine.num_optimization_profiles)
    inputs, outputs, bindings, stream = allocate_buffers(engine, 0)
    context = engine.create_execution_context()
    for i in range(engine.num_io_tensors):
        logging.debug(f'tensor {engine[i]} shape {context.get_tensor_shape(engine[i])}')
        binding = engine[i]
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            context.set_input_shape(
                binding,
                [x if x >= 0 else batch_size
                    for x in engine.get_tensor_shape(binding)],
            )
        logging.debug(f'tensor {engine[i]} shape {context.get_tensor_shape(engine[i])}')
    while True:
        t = time.perf_counter()
        do_inference(context, engine, bindings, inputs, outputs, stream)
        yield time.perf_counter() - t
