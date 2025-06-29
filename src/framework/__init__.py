import argparse
import os
import signal
import sys

from . import ort
from . import trt
from . import vllm_

_DONE = False
_CHOICES = {
    'ort': ort,
    'trt': trt,
    'vllm': vllm_,
}


def _handler(*_):
    global _DONE
    _DONE = True


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('framework', type=_CHOICES.get,
                        choices=_CHOICES.values())
    parser.add_argument('model')
    parser.add_argument('-n', '--batch-size', type=int, default=1)
    args = parser.parse_args(args)

    for signalnum in signal.SIGTERM, signal.SIGINT:
        signal.signal(signalnum, _handler)

    iterator = args.framework.generator(args.model, args.batch_size, args)
    while not _DONE:
        try:
            print(next(iterator), flush=True)
        except BrokenPipeError:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            break


if __name__ == '__main__':
    main()
