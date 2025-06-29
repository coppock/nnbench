import argparse
import asyncio
import os
import signal
import sys
import itertools
import random

from . import kserve_
from . import openai_

_CHOICES = {
    'kserve': kserve_,
    'openai': openai_,
}


async def load(client, num_requests, rate):
    event_loop = asyncio.get_running_loop()
    for sig in signal.SIGTERM, signal.SIGINT:
        event_loop.add_signal_handler(sig, event_loop.stop)

    async def time_request():
        try:
            print(await client(), flush=True)
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
    parser.add_argument('api', type=_CHOICES.get, choices=_CHOICES.values())
    parser.add_argument('model')
    args = parser.parse_args(args)
    namespace, args = parser.parse_known_args(args)

    client = namespace.api.Client(namespace.model, args)
    asyncio.run(load(client, namespace.num_requests, namespace.rate))


if __name__ == '__main__':
    main()
