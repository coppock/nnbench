import argparse
import time

import kserve
import numpy as np


class _InputType:
    def __init__(self, s):
        self.name, shape, self.dtype = s.split(',')
        self.shape = [int(x) for x in shape.split(':')]


def _random(shape, datatype):
    rng = np.random.default_rng()
    match datatype:
        case 'FP32':
            return rng.random(shape, np.float32)
        case 'INT64':
            dtype = np.int64
            return rng.integers(0, np.iinfo(dtype).max, shape, np.int64, True)


async def Client(model, args):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_type', nargs='*', type=_InputType)
    args = parser.parse_args(args)

    client = kserve.InferenceGRPCClient('localhost:8001')

    async def request():
        inputs = kserve.InferRequest([kserve.InferInput(
            input_type.name,
            input_type.shape,
            input_type.dtype,
            _random(input_type.shape, input_type.dtype),
        ) for input_type in args.input_type])
        t = time.perf_counter()
        await client.infer(model, inputs)
        return time.perf_counter() - t

    return request
