import argparse
import asyncio
import importlib
import os
import signal
import sys
import itertools
import random

_APIS = {
    'kserve': 'kserve_',
    'openai': 'openai_',
}


async def loop(request, num_requests, rate):
    event_loop = asyncio.get_running_loop()
    for sig in signal.SIGTERM, signal.SIGINT:
        event_loop.add_signal_handler(sig, event_loop.stop)

    async def time_request():
        try:
            print(await request(), flush=True)
        except BrokenPipeError:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            event_loop.stop()

    async with asyncio.TaskGroup() as tg:
        for _ in range(num_requests) if num_requests else itertools.count():
            tg.create_task(time_request())
            if rate:
                await asyncio.sleep(random.expovariate(rate))


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-requests', type=int)
    parser.add_argument('-r', '--rate', type=float)
    parser.add_argument('api', choices=_APIS)
    parser.add_argument('model')
    namespace, args = parser.parse_known_args(args)

    api = importlib.import_module(f'{__name__}.{_APIS[namespace.api]}', __package__)
    request = api.Client(namespace.model, args)
    try:
        asyncio.run(loop(request, namespace.num_requests, namespace.rate))
    except RuntimeError as e:
        if str(e) != 'Event loop stopped before Future completed.':
            raise


if __name__ == '__main__':
    main()
