import asyncio
import json
import logging
import os
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

from absolute_zero import (
    AZRXMLParser,
    AZRExecutor,
    AZRBufferManager,
    AZREnv,
    load_environment,
)


def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def configure_logging():
    log_path = os.getenv("AZR_LOG_FILE", "azr_runs.log")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.getLogger("AZREnv").setLevel(logging.INFO)
    logging.getLogger("verifiers.envs.AZREnv").setLevel(logging.INFO)
    logging.getLogger("AZRRunLog").setLevel(logging.INFO)


async def test_parser():
    header("Phase 1: AZRXMLParser sanity")
    parser = AZRXMLParser()
    ok_text = (
        "<think>t</think><answer>\n"
        "```python\n"
        "def f(x):\n    return x\n"
        "```\n"
        "```input\n"
        "1\n"
        "```\n"
        "</answer>"
    )
    # Parse just the <answer> text
    answer_txt = parser.parse_answer(ok_text) or ""
    print(f"answer_len: {len(answer_txt)} contains_python: {'def f(' in answer_txt}")
    # Parse fenced blocks within <answer>
    blocks = parser.parse_answer(ok_text, fences=["python", "input"]) or {}
    py_blocks = blocks.get("python", []) if isinstance(blocks, dict) else []
    in_blocks = blocks.get("input", []) if isinstance(blocks, dict) else []
    print(f"python_blocks: {len(py_blocks)}, input_blocks: {len(in_blocks)}")
    assert len(py_blocks) == 1 and "def f(" in py_blocks[0]
    assert len(in_blocks) == 1 and in_blocks[0].strip() == "1"

    bad_texts = [
        "no tags",
        "<answer>{}</answer>",
        "<think>only</think>",
        "<think></think><answer>no fences here</answer>",
    ]
    for i, bt in enumerate(bad_texts):
        ans = parser.parse_answer(bt)
        blks = parser.parse_answer(bt, fences=["python", "input"]) or {}
        pyc = len(blks.get("python", [])) if isinstance(blks, dict) else 0
        inc = len(blks.get("input", [])) if isinstance(blks, dict) else 0
        print(f"bad_case_{i}: answer_present={ans is not None and len(ans) > 0}, py_blocks={pyc}, in_blocks={inc}")


def test_executor():
    header("Phase 2: AZRExecutor sanity")
    ex = AZRExecutor()
    code = "def f(a, b):\n    return a + b"
    f, err = ex.compile_program(code)
    print(f"compile_ok: {f is not None}, err: {err}")
    if f:
        ok, out, e2 = ex.run_deterministic(f, (2, 3), runs=3)
        print(f"deterministic_ok: {ok}, out: {out}, err: {e2}")
    bad_code = "import os\n\ndef f(x):\n    return x"
    f2, err2 = ex.compile_program(bad_code)
    print(f"compile_disallowed_ok: {f2 is None}, err_contains: {str(err2)[:60] if err2 else None}")
    # Verify single-arg wrapped input convention
    code_list = "def f(x):\n    return len(x)"
    f3, err3 = ex.compile_program(code_list)
    assert f3 is not None, f"unexpected compile error: {err3}"
    ok1, out1, e3 = ex.run_deterministic(f3, [1,2,3], runs=2)
    ok2, out2, e4 = ex.run_deterministic(f3, [[1,2,3]], runs=2)
    print(f"single_arg_list_direct: ok={ok1}, out={out1}, err={e3}")
    print(f"single_arg_list_wrapped: ok={ok2}, out={out2}, err={e4}")
    assert ok1 and out1 == 3
    assert ok2 and out2 == 3


def test_buffers():
    header("Phase 3: AZRBufferManager basics")
    bm = AZRBufferManager(init_zero_triplet=True)
    print(
        f"initial_triplets: {len(bm.triplet_set)}, "
        f"deduction: {len(bm.deduction)}, abduction: {len(bm.abduction)}, induction: {len(bm.induction)}"
    )
    bm.add_triplet("def f(x):\n    return x[::-1]", "abcd", "dcba")
    print(
        f"after_add_triplet -> triplets: {len(bm.triplet_set)}, "
        f"last_triplet_program_prefix: {bm.triplet_set[-1].program[:10]}"
    )
    bm.add_induction(
        program="def f(x):\n    return x",
        message="Return input",
        io_pairs=[("A", "A"), ("B", "B")],
        visible=[("A", "A")],
        hidden=[("B", "B")],
    )
    print(f"after_add_induction -> induction: {len(bm.induction)}")


async def test_env_propose_and_seed():
    header("Phase 4: AZREnv propose rollouts with fake model")
    env = load_environment()
    print(
        f"env_init -> triplets: {len(env.buffers.triplet_set)}, "
        f"induction: {len(env.buffers.induction)}"
    )
    load_dotenv()
    model_name = os.getenv("AZR_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    # Async client for rollout/seed, sync client for evaluate convenience
    aclient = AsyncOpenAI()
    sclient = OpenAI()

    async def run_propose(task: str):
        before = (
            len(env.buffers.triplet_set),
            len(env.buffers.induction),
        )
        completion, state = await env.rollout(
            client=aclient,
            model=model_name,
            prompt=[],
            task=task,
            info={"K": env.K, "mc_samples": env.mc_samples, "determinism_runs": env.j},
            sampling_args={"temperature": 0.2, "max_tokens": 2048},
        )
        after = (
            len(env.buffers.triplet_set),
            len(env.buffers.induction),
        )
        print(
            f"propose:{task} -> valid:{state.get('valid')}, "
            f"triplets:{before[0]}→{after[0]}, induction:{before[1]}→{after[1]}, "
            f"format_ok:{state.get('format_ok')}, error={(state.get('error') or '')[:60]}"
        )
        return state

    # Run each proposer once with the real model
    await run_propose("deduction.propose")
    await run_propose("abduction.propose")
    await run_propose("induction.propose")

    # Also demonstrate evaluate-style run against the env dataset (real calls)
    header("Phase 4b: env.evaluate one example")
    eval_results = env.evaluate(
        client=sclient,
        model=model_name,
        num_examples=1,
        rollouts_per_example=1,
        score_rollouts=True,
    )
    print(f"evaluate -> rewards={eval_results.reward}, metrics={eval_results.metrics.keys()}")
    if eval_results.state:
        st = eval_results.state[0]
        print(
            f"evaluate state -> task={st.get('task')}, format_ok={st.get('format_ok')}, json_ok={st.get('json_ok')}, error={(st.get('error') or '')[:60]}"
        )

    # Now test seeding with targets (real calls)
    header("Phase 5: AZREnv seed_buffers with real model")
    await env.seed_buffers(
        client=aclient,
        model=model_name,
        target_triplets=4,
        target_induction=3,
        sampling_args={"temperature": 0.2, "max_tokens": 2048},
    )
    print(
        f"after_seed -> triplets:{len(env.buffers.triplet_set)}, induction:{len(env.buffers.induction)}"
    )
    # Preview last triplet and induction
    if env.buffers.triplet_set:
        t = env.buffers.triplet_set[-1]
        print(f"last_triplet: input={t.input!r}, output={t.output!r}, program_prefix={t.program[:20]!r}")
    if env.buffers.induction:
        it = env.buffers.induction[-1]
        print(
            f"last_induction: message={it.message!r}, "
            f"visible={it.visible_pairs}, hidden={it.hidden_pairs}"
        )


async def main():
    configure_logging()
    await test_parser()
    test_executor()
    test_buffers()
    await test_env_propose_and_seed()


if __name__ == "__main__":
    asyncio.run(main())



