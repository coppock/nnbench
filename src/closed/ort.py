import time

import onnx
import onnxruntime as ort

_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
_DTYPES = {'tensor(float)': onnx.TensorProto.FLOAT}


def generator(model, _batch_size, _args):
    session = ort.InferenceSession(model, providers=_PROVIDERS)
    while True:
        # For some reason, latencies are bimodal when plain numpy arrays are
        # used: use IO bindings instead.
        io_binding = session.io_binding()
        for input in session.get_inputs():
            value = ort.OrtValue.ortvalue_from_shape_and_type(
                input.shape,
                _DTYPES[input.type],
                'cpu',
            )
            io_binding.bind_ortvalue_input(input.name, value)
        for output in session.get_outputs():
            io_binding.bind_output(output.name, 'cpu')
        t = time.perf_counter()
        session.run_with_iobinding(io_binding)
        yield time.perf_counter() - t
