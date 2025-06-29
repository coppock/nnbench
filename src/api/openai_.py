import argparse
import random
import time

from openai import AsyncOpenAI

def Client(model, args):
    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--trace', type=float)
    parser.add_argument('vocab_size', type=int)
    parser.add_argument('input_len', type=int)
    parser.add_argument('output_len', type=int)
    args = parser.parse_args(args)

    client = AsyncOpenAI(
        api_key='empty',
        base_url='http://localhost:8000/v1',
    )

    async def request():
        prompt = [random.randint(0, args.vocab_size - 1)
                  for _ in range(args.input_len)]
        t = time.perf_counter()
        await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=args.output_len,
            stop=None,
        )
        return time.perf_counter() - t

    return request
