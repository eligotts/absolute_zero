import verifiers as vf
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
# Load your environment
env = vf.load_environment("absolute_zero")

# Test with a model
aclient = AsyncOpenAI()
sclient = OpenAI()

import asyncio

async def main():
    await env.seed_buffers(aclient, "gpt-4.1", target_triplets=2, target_induction=2)
    results = env.evaluate(
        sclient, "gpt-4.1",
        num_examples=6,
        rollouts_per_example=1,
        # max_concurrent=32,
    )
    print(results)

asyncio.run(main())