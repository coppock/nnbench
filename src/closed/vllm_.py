import argparse
import csv
import itertools
import os
import random
import sys
import time

os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
from vllm import LLM
from vllm.inputs import TokensPrompt


def generator(model, batch_size, args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--gpu-memory-utilization', type=float)
    args = parser.parse_args(args)

    kwargs = {'model': model, 'enforce_eager': True}
    if args.gpu_memory_utilization:
        kwargs['gpu_memory_utilization'] = args.gpu_memory_utilization
    llm = LLM(**kwargs)
    vocab_size = llm.get_tokenizer().vocab_size
    reader = csv.reader(sys.stdin)
    while True:
        # I don't have a great way both to batch and to limit the number of
        # output tokens.
        prompts = [TokensPrompt(prompt_token_ids=[random.randint(0, vocab_size - 1)
                                                  for _ in range(int(row['ContextTokens']))])
                   for row in itertools.islice(reader, batch_size)]
        t = time.perf_counter()
        llm.generate(prompts)
        yield time.perf_counter() - t
