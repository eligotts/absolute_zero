# Absolute Zero Reasoner (AZR) Environment — Detailed Technical Design

Author: Cline  
Target repo for implementation: /Users/eligottlieb/Documents/environments/absolute_zero  
Primary integration: /Users/eligottlieb/Documents/verifiers (verifiers library)

This document is a complete implementation plan to recreate the Absolute Zero Reasoner (AZR) paper as a verifiers-compatible environment. It specifies data structures, control flow, prompts, parsing, safety validation, buffer management, propose/solve reward computation, and training loop integration. A software engineer with this design doc and access to the referenced repos can implement the environment end-to-end.

- Upstream AZR reference code: /Users/eligottlieb/Documents/Absolute-Zero-Reasoner (not accessible from verifiers repo; line refs are from your summary)
- Verifiers framework: /Users/eligottlieb/Documents/verifiers
  - Key files we integrate with:
    - verifiers/envs/environment.py
    - verifiers/envs/multiturn_env.py
    - verifiers/envs/singleturn_env.py
    - verifiers/trainers/grpo_trainer.py
    - verifiers/trainers/async_batch_generator.py
    - verifiers/rubrics/rubric.py
    - verifiers/parsers/think_parser.py
    - verifiers/types.py
  - Example envs to mirror structure:
    - environments/doublecheck/doublecheck.py
    - environments/self_reward/self_reward.py

--------------------------------------------------------------------------------
## 1. Goals and Scope

Recreate AZR as a verifiers Environment supporting three task types and two roles:

- Task types (α): deduction, abduction, induction
- Roles: proposer, solver
- Six total “question” classes per batch step:
  - deduction.propose (dp), abduction.propose (ap), induction.propose (ip)
  - deduction.solve (ds), abduction.solve (as), induction.solve (is)

Environment responsibilities:

- Maintain validated buffers for each task and a unified triplet set.
- Build prompts dynamically (with K reference examples for proposer, with sampled problems for solver).
- Parse, validate, and safely execute proposed code and solver predictions.
- For proposer tasks, compute Monte Carlo (MC) solver accuracy using N rollouts, then compute r_propose per AZR.
- For solver tasks, compute binary correctness with format-aware penalties.
- Integrate with verifiers GRPOTrainer (dataset providing six tasks; environment constructs actual content).
- Return completion and a State carrying all metadata required by the Rubric to compute final rewards, without requiring Rubric to perform model calls.

Non-goals:

- Full veRL/TRR++ optimization. We use verifiers’ GRPO trainer.
- Exact prompt parity with AZR. We adhere to DeepSeek R1-style <think> and <answer> with JSON payloads inside <answer> for structured parsing and penalties.

--------------------------------------------------------------------------------
## 2. Core Concepts and Mapping to verifiers

AZR summary (from your provided notes):

- Proposer samples tasks; environment validates to obtain (x, y*), then assigns proposer reward r_propose = {0 if mean solve acc in {0,1}; 1 - mean solve acc otherwise}, with MC estimate using N=8 solver rollouts.
- Solver receives x and produces y; environment computes r_solve = I(y == y*).
- Format-aware penalties:
  - -1.0 if formatting errors
  - -0.5 if formatted but wrong
  - r_role if formatted and passable (valid)
- Safety: block disallowed modules; run program determinism checks (j=2 identical runs).

verifiers integration:

- Environment implements async rollout(client, model, prompt, answer, task, info, sampling_args)
- GRPOTrainer provides:
  - AsyncBatchGenerator that calls env.a_generate(...), which:
    - Submits many prompts/tasks in a batch
    - Runs env.run_rollouts (parallelizable) — which calls env.rollout per sample
    - Scores via rubric.score_rollouts(prompts, completions, answers, states, tasks, infos)
- Reward functions should be pure on prompt/completion/answer/state. We will compute heavy work (validation, code exec, MC rollouts) inside env.rollout and store results in state for rubric to read.

--------------------------------------------------------------------------------
## 3. Data Model and Buffers

AZR works over code triplets: (p, i, o)
- p: Python function program (e.g., def f(x): ...)
- i: input(s)
- o: output(s)
- For induction: we use a set of IO examples {(in_n, out_n)} and a message m (“hint”)

We need four collections:

- triplet_set: List[Triplet] — unified validated triplets, with metadata.
- D_deduction: List[DeductionItem] — items primarily served as (p, i) with stored o for solver check.
- D_abduction: List[AbductionItem] — items primarily served as (p, o) with stored i for solver check.
- D_induction: List[InductionItem] — {program p, examples [(in, out)], message m, split {visible IO half, hidden IO half}}.

Types (Python pseudo):

```python
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Dict
import time

@dataclass
class Triplet:
    program: str         # code snippet defining a pure function
    input: Any           # json-serializable input
    output: Any          # json-serializable output
    step_id: int         # recency
    created_at: float    # timestamp

@dataclass
class DeductionItem:
    program: str
    input: Any
    output: Any
    step_id: int
    created_at: float

@dataclass
class AbductionItem:
    program: str
    input: Any
    output: Any
    step_id: int
    created_at: float

@dataclass
class InductionItem:
    program: str
    message: str
    io_pairs: List[Tuple[Any, Any]]     # full set validated
    visible_pairs: List[Tuple[Any, Any]]
    hidden_pairs: List[Tuple[Any, Any]]
    step_id: int
    created_at: float
````

BufferManager:

- Methods:

  - add_triplet(t: Triplet)
  - add_deduction(item), add_abduction(item), add_induction(item)
  - sample_deduction(recency_bias=True) -> DeductionItem | None
  - sample_abduction(recency_bias=True) -> AbductionItem | None
  - sample_induction(recency_bias=True) -> InductionItem | None

- Recency bias: prefer items with highest step_id; fallback to random sample from historical when recent < capacity (mirroring AZR’s “max_new” then fallback).

- Thread-safety: Since AsyncBatchGenerator processes batches sequentially in its worker thread, environment mutations during generation are serialized. We still protect with a simple threading.Lock to be safe.

Seeding:

- zero triplet: def f(x): return x with input like "Hello World" output identical.

- Optionally, during early steps if buffers are empty:

  - deduction/abduction propose will try to generate new triplets (preferred)
  - solve tasks will fallback to sampling from triplet_set if their specific buffer is empty (or skip/return a neutral prompt with zero reward?). To avoid training stalls, we always ensure at least one valid seed exists in each buffer via converting the zero triplet into D_deduction and D_abduction entries during environment init.

---

## 4. Prompting and Output Format

We adopt DeepSeek R1-style with JSON inside . This enables structured parsing plus format gating:

General outer format (both roles and all tasks):

```javascript
<think>
... chain-of-thought ...
</think>
<answer>
{ JSON payload specific to task }
</answer>
```

JSON payload specifications:

- deduction.propose (dp): propose a new program and input
  - {"program": "<python code for def f(...): ...>", "input": }
- abduction.propose (ap): propose a new program and input (we keep symmetry with deduction.propose)
  - {"program": "<python code for def f(...): ...>", "input": }
- induction.propose (ip): given a referenced program p (sampled from union of buffers, provided in prompt), propose message and N inputs
  - {"message": "", "inputs": [, ...]}
- deduction.solve (ds): predict output given p and i
  - {"output": }
- abduction.solve (as): predict input given p and o
  - {"input": }
- induction.solve (is): synthesize program given visible IO examples and message
  - {"program": "<python code for def f(...): ...>"}

Reference examples (K=6 for proposer):

- Provided as context in user prompt:

  - For dp/ap: list of K recent validated triplets (program, input, output).
  - For ip: a single program p sampled (from union), plus K reference triplets for diversity (optional).

Deterministic validation description and forbidden modules list are included in the prompt.

---

## 5. Safety and Execution

We must safely execute python code:

- Execution sandbox:

  - No filesystem/network/system access.
  - Disallow dangerous builtins: __import__, open, eval, exec, compile, globals, locals, vars, setattr, delattr, getattr for module-level import.
  - Provide restricted __builtins__ with: abs, all, any, bool, dict, enumerate, filter, float, int, len, list, map, max, min, pow, range, set, sorted, str, sum, tuple, zip — plus math module (pre-imported, but no os/sys/etc).

- Disallowed modules (hard block on source text and on attempted import):
  - os, sys, shutil, subprocess, pathlib, socket, requests, urllib, http, jsonpickle, pickle, marshal, ctypes, builtins (to rebind), importlib, threading, multiprocessing, time (sleep), random (stochastic), numpy.random random state unless seeded – simplest: disallow numpy entirely to start.

- Determinism check:

  - j=2 runs with same input; requires outputs be strictly equal (==) and no timing side-effects.
  - For induction solve validation on hidden pairs: all must satisfy p_pred(i) == o.

Executor pseudo:

```python
def safe_exec_program(program: str) -> Callable:
    # static scan for forbidden substrings (import os, __import__, etc.)
    # create restricted globals dict with limited builtins and allowed modules
    # exec(program, restricted_globals, restricted_locals)
    # find a top-level function def (prefer named f), return callable
```

```python
def run_twice_and_equal(f: Callable, inp: Any) -> Tuple[bool, Any]:
    out1 = f(copy.deepcopy(inp))
    out2 = f(copy.deepcopy(inp))
    return (out1 == out2), out1
```

Error handling:

- Any parsing/safety/execution error yields valid=False; environment will:

  - Not add to buffers
  - For propose: set mc_acc=None and let rubric assign penalties
  - For solve: mark correct=False; rubric applies -0.5 or -1 based on formatting

---

## 6. Environment Class Design

File: /Users/eligottlieb/Documents/environments/absolute_zero/absolute_zero.py

Exports:

- load_environment(...) -> vf.Environment

Main classes:

1. AZRParser (R1JSONParser)

- Extracts DeepSeek format and JSON payload. Provides two metrics:

  - format_ok: whether ... and ... exist and in correct order with non-empty answer
  - json_ok: whether content is valid JSON mapping

- parse_answer(text) -> returns raw content (string) or None

- parse_json(completion) -> dict | None

2. AZRExecutor

- Implements safe execution and determinism checks (see Section 5)

3. AZRBufferManager

- Holds the three buffers and triplet_set with recency bias sampling utilities
- step_id increments per proposal added

4. AZREnv(Environment)

- message_type="chat"

- Overrides rollout(...) to implement the full propose/solve logic per task:

  - Compose dynamic prompt messages (base prompt from dataset + dynamic references/problems)
  - Call get_model_response
  - Parse and validate
  - For propose: MC solver rollouts (N) using the same get_model_response with solver prompts
  - For solve: evaluate straightforwardly
  - Construct completion (assistant message) and state dict

State fields (must be JSON-serializable; used by Rubric):

```python
state = {
  "prompt": prompt,           # verifiers
  "completion": completion,   # verifiers
  "answer": answer,           # verifiers
  "task": task,               # verifiers
  "info": info,               # verifiers + {params}
  "responses": [response],    # verifiers raw response object
  "turn": 1,                  # verifiers
  # AZR-specific:
  "format_ok": bool,
  "json_ok": bool,
  "valid": bool,              # proposal validity or solve evaluation passable
  "error": str | None,        # validation error description
  "propose": {
    "mc_samples": int,
    "mc_accuracy": float | None,
  } | None,
  "solve": {
    "correct": bool | None,
  } | None,
  # payloads:
  "payload": {
    "program": str | None,
    "input": Any | None,
    "output": Any | None,
    "message": str | None,
    "io_pairs": list[tuple] | None,
    "visible_pairs": list[tuple] | None,
    "hidden_pairs": list[tuple] | None,
  },
  "sampled_problem_id": str | int | None,
}
```

5. AZRRubric(Rubric)

- Single composite reward function `azr_reward(...)` that inspects task and state:

  - If not state["format_ok"] -> -1.0

  - Else if task endswith ".propose":

    - If not state["valid"] or state["propose"]["mc_accuracy"] is None -> -0.5
    - Else if mc in {0.0, 1.0} -> 0.0
    - Else -> 1.0 - mc

  - Else (solve task):

    - If state["solve"]["correct"] is True -> 1.0
    - Else -> -0.5

- Optionally include separate metric keys for logging (e.g., "format", "mc_acc", "solve_correct") via the underlying RolloutScore metrics.

load_environment config:

```python
def load_environment(
  max_turns: int = 1,
  mc_samples: int = 8,
  proposer_K: int = 6,
  determinism_runs: int = 2,
  seed: int = 1337420,
  system_prompt: str = BASE_SYSTEM_PROMPT,
  # initial seed zero triplet:
  init_zero_triplet: bool = True,
  # dataset scheduling:
  dataset_repeats: int = 1
) -> vf.Environment
```

- Returns AZREnv(dataset=..., rubric=AZRRubric(...), parser=AZRParser(), ...)
- Dataset will include 6 “task rows” (dp, ap, ip, ds, as, is), optionally repeated; The dynamic content (references/problems) is injected by env.rollout just-in-time.

---

## 7. Control Flow by Task

We denote client/model/sampling_args passed through rollout.

### 7.1 deduction.propose (dp)

- Build user message:

  - “You are the proposer for deduction tasks. Using prior references, propose JSON with {"program": "...", "input": ...}. Program must define a pure function (def f(...): ...). Avoid forbidden modules. Your response must follow: ... {"program": "...", "input": ...}
  - K=6 references: display as a compact list of {program, input, output} with triple backticks for code sections.

- Call model, get assistant content.

- Parse R1 + JSON; if format fails: state.format_ok=False, valid=False; return completion+state.

- Validate:

  - Static safety scan of program string
  - Compile+exec in sandbox to get callable f
  - Determinism: run f(input) twice equal? -> o

- If any fails: valid=False; return

- Construct triplet (p, i, o). Add to triplet_set, D_deduction, D_abduction (since both share same triplet content).

- MC solver accuracy estimation:

  - Build solver prompt for deduction.solve with the just-created (p, i) and ground-truth o internal (not shown to model).

  - Repeat N times:

    - Call model once per trial; parse ds JSON, safe-parse the “output” value (pure JSON)
    - Check correctness: (pred_output == o)

  - mc_accuracy = mean(correct)

- Update state with propose.mc_accuracy.

### 7.2 abduction.propose (ap)

- Same as dp for propose stage, except the role description names abduction; Still propose (program, input) and derive o by execution; Reason: AZR seeded both from generated (p, i) pairs for both deduction and abduction buffers.
- Add triplet; compute MC accuracy using abduction.solve prompt:
  - Provide (p, o); Model predicts input; Check f(pred_input) == o
- State updated accordingly.

### 7.3 induction.propose (ip)

- Sample a program p from union of D_deduction ∪ D_abduction (favor recency)

- Build user message:

  - “You are the proposer for induction tasks. Given the following function (p), propose a message ‘m’ describing the function at a high level, and generate N new inputs (list). You must obey R1+JSON schema: {"message": "...", "inputs": [ ... ]}”
  - Include p code; include K references optionally for context/style.

- Call model; parse JSON; validate all inputs via executor against program p to produce outputs; if any errors -> valid=False

- Form io_pairs [(in, out)] for all; split randomly into visible and hidden halves (strict half for N even; for N odd, floor or customized)

- Build InductionItem and add to D_induction

- MC solver accuracy estimation:

  - Build induction.solve prompt with visible pairs and the message
  - Repeat N times: model outputs {"program": "..."}; parse; safe exec; validate across all hidden pairs: all f_pred(in) == out ? 1 else 0
  - mc_accuracy = mean

### 7.4 deduction.solve (ds)

- Sample DeductionItem from D_deduction (recency-biased; fallback historical)
- Build user message:
  - “You are the solver for deduction: given program p and input i, output JSON {"output": ...}. Format R1+JSON.”
- Call model; parse; if format_ok=False -> penalize; else compute correct = (pred == o). No buffer updates.
- State updated with solve.correct.

### 7.5 abduction.solve (as)

- Sample AbductionItem (recency-bias)
- Build user message: “Given program p and output o, predict input i -> JSON {"input": ...}”
- Call model; parse; execute f(pred_i) twice; check (f(pred_i) == o) and determinism; correct boolean.
- Edge cases: if f fails -> incorrect.

### 7.6 induction.solve (is)

- Sample InductionItem
- Build user message:
  - Provide message m + visible pairs only; ask to synthesize program code JSON {"program": "..."}; enforce R1+JSON.
- Call model; parse program; sandbox exec; validate on hidden pairs: all pass -> correct=True else False.

---

## 8. Rubric: Reward Functions and Penalties

Implement AZRRubric with one composite reward function `azr_reward(...)`. It reads:

- state["format_ok"] (from parser) for formatting penalty

- For propose tasks:

  - state["valid"] and state["propose"]["mc_accuracy"] (float)
  - r_propose = 0.0 if mc in {0,1}, else 1 - mc

- For solve tasks:
  - state["solve"]["correct"] boolean

- Composite logic:

  - if not format_ok: return -1.0

  - else if task.endswith(".propose"):

    - if not valid or mc is None: return -0.5
    - else return r_propose

  - else (solve):
    - return 1.0 if correct else -0.5

Expose metrics in RolloutScore.metrics for logging:

- "format_ok": 0/1
- "json_ok": 0/1
- "valid": 0/1
- "mc_acc": float or 0.0
- "solve_correct": 0/1

This matches AZR’s composite reward shape in spirit while using verifiers’ rubric pattern (verifiers/rubrics/rubric.py).

---

## 9. Dataset Construction and Scheduling

Dataset shape:

- 6 rows; each is one “task type”:
  - {prompt: base messages for that role, task: "deduction.propose"} and so on for all six
- answer: "" (unused; AZR is self-validated)
- info: include defaults and tunables:
  - {"K": 6, "mc_samples": 8, "determinism_runs": 2, "oai_tools": optional}

In load_environment():

- Construct a small HF Dataset via Dataset.from_list([...]) with 6 entries; optionally apply `repeat(dataset_repeats)` to adjust size.
- The actual problem/instructions (references/problems) are added during rollout, not baked into dataset.

GRPOTrainer / AsyncBatchGenerator integration:

- RepeatSampler ensures each unique prompt is repeated num_generations times and distributed across processes consistently.
- Our env.rollout is agnostic to distribution; it samples from buffers per call and handles propose/solve appropriately.
- This aligns with your note “num_generations maps to B in the paper,” as the trainer groups generations for advantage normalization (see verifiers/trainers/grpo_trainer.py _compute_advantages).

---

## 10. Prompt Templates

Base system prompt (configurable):

```javascript
You are an expert reasoning model using the DeepSeek R1 format.
Always respond strictly with:
<think>
... step-by-step internal reasoning ...
</think>
<answer>
... final JSON payload only ...
</answer>

JSON-only inside <answer>. Do not include explanations there.
```

Per-role base user preface (added to base prompt):

- Proposer (dp/ap):
  - “Role: Proposer for {deduction|abduction}. Using the K references below, propose a new function and input adhering to safety rules. Output JSON: {"program": "...", "input": ...} Rules: define a pure function def f(...): return ..., no imports of disallowed modules, deterministic behavior. References:
    1. Program:
       ```python
       ...
       ```
       Input: ...

```javascript
```

Command Output

````shell
...
    ...
  - End with a reminder about DeepSeek R1 format.

- Proposer (ip):
  - Provide sampled program p (from union of buffers), show code, and ask for:
    {"message": "...", "inputs": [ ... ]}
  - Same safety + determinism description.

- Solver (ds):
  - Provide p code + input i; ask for {"output": ...}.

- Solver (as):
  - Provide p code + output o; ask for {"input": ...}.

- Solver (is):
  - Provide message m + visible IO pairs; ask for {"program": "..."}.

These templates will be implemented as string builder functions in AZREnv.

--------------------------------------------------------------------------------
## 11. Rollout Implementation (Pseudo)

```python
class AZREnv(vf.Environment):
    def __init__(..., parser=None, rubric=None, ...):
        super().__init__(..., parser=parser or AZRParser(), rubric=rubric or AZRRubric())
        self.buffers = AZRBufferManager(seed=seed, init_zero_triplet=init_zero_triplet)
        self.mc_samples = mc_samples
        self.K = proposer_K
        self.j = determinism_runs
        self.step_counter = 0

    async def rollout(self, client, model, prompt, answer="", task="default", info=None, sampling_args=None, **kwargs):
        info = info or {}
        K = info.get("K", self.K)
        mc = info.get("mc_samples", self.mc_samples)
        j = info.get("determinism_runs", self.j)

        # 1) Build dynamic prompt based on task
        if task == "deduction.propose":
            user_msg = self._build_dp_prompt(K)
        elif task == "abduction.propose":
            user_msg = self._build_ap_prompt(K)
        elif task == "induction.propose":
            user_msg, program_context = self._build_ip_prompt(K)
        elif task == "deduction.solve":
            user_msg, item = self._build_ds_prompt()
        elif task == "abduction.solve":
            user_msg, item = self._build_as_prompt()
        elif task == "induction.solve":
            user_msg, item = self._build_is_prompt()
        else:
            # default pass-through
            user_msg = {"role":"user","content":"Continue"}
            item = None

        # Merge with base prompt
        rollout = list(prompt) if isinstance(prompt, list) else [{"role":"user","content":prompt}]
        rollout.append(user_msg)

        # 2) Call model once for the main role output
        response = await self.get_model_response(client, model, rollout, sampling_args=sampling_args, message_type=self.message_type)
        assistant_text = response.choices[0].message.content or ""  # Chat

        completion = [{"role": "assistant", "content": assistant_text}]
        state = {
            "prompt": rollout,
            "completion": completion,
            "answer": answer,
            "task": task,
            "info": info,
            "responses": [response],
            "turn": 1,
        }

        # 3) Parse
        format_ok, json_obj, error = self.parser.try_parse_r1_json(assistant_text)
        state["format_ok"] = format_ok
        state["json_ok"] = json_obj is not None
        state["error"] = error

        # 4) Branch by task
        if task.endswith(".propose"):
            valid, propose_state, payload = await self._handle_propose(task, json_obj, j, mc, client, model, sampling_args)
            state["valid"] = valid
            state["propose"] = propose_state
            state["solve"] = None
            state["payload"] = payload
        else:
            correct, payload = await self._handle_solve(task, json_obj, j)
            state["valid"] = True if format_ok else False
            state["propose"] = None
            state["solve"] = {"correct": bool(correct)}
            state["payload"] = payload

        return completion, state
````

Helper methods:

- _handle_propose:

  - For dp/ap: validate p,i -> o; add to buffers; compute mc via N solver calls with ds/as prompts.
  - For ip: given sampled p, validate inputs and compute outputs; split visible/hidden; add to induction buffer; compute mc via N solver calls with is prompt.

- _handle_solve:

  - For ds/as/is: sample from corresponding buffer, build prompt (already done), parse predicted JSON, validate with executor, compute correctness.

---

## 12. Proposer MC Evaluation Details

Important alignment with AZR:

- AZR used the same rollout actor to evaluate MC samples (reward_managers.py lines ~642-681 and ~720-799 in your summary).
- Our environment uses `get_model_response` with the same client/model and sampling args to generate MC samples for the solver tasks. This mirrors AZR’s approach.

Parameters:

- mc_samples default 8
- For reproducibility, seed is set during env init for any stochastic control (but model sampling uses temperature and top-p per GRPO trainer config).

Performance:

- Each proposer rollout invokes N additional model calls. GRPO’s AsyncBatchGenerator processes batches in a background thread; ensure `generation_timeout` is large enough.

---

## 13. Recency-Biased Sampling and Fallback

Strategy:

- When sampling for solve:

  - Prefer items with the highest step_id that were added recently.
  - If insufficient items exist (e.g., cold start), fallback sample uniformly from triplet_set / relevant buffer.

- Proposers don’t fallback-pad invalid proposals (AZR behavior). If an invalid proposal occurs, it’s discarded (remain in PPO batch with zero or penalty).

Implementation:

- Keep a deque of recent indices; on add, push; sample from deque first; if empty, sample from full list via random.choice.

---

## 14. Parser and Format Rewards

Parser (AZRParser/R1JSONParser):

- try_parse_r1_json(text) -> (format_ok, json_obj | None, error_str | None)

- format_ok criteria (DeepSeek R1-like):

  - text contains single and , and a subsequent ... block
  - Non-empty answer block

- json_obj parsing: json.loads(answer_block)

- The rubric will use state["format_ok"] for gating.

We intentionally do not depend on ThinkParser semantics (which blank out content if missing). Our custom parser strictly enforces both tags and JSON in .

---

## 15. Reward Computation Mapping

From AZR summary:

- Proposer reward:

  - r_propose = 0 if r̄solve ∈ {0,1}
  - else 1 - r̄solve

- Solver reward:
  - r_solve = I(y == y*)

- Composite gating:

  - r_role if formatted and passable
  - -0.5 if formatted but wrong
  - -1.0 if formatting error

Our rubric exact mapping:

- If not format_ok: -1.0

- If propose:

  - If not valid or mc_accuracy is None: -0.5
  - Else if mc ∈ {0.0, 1.0}: 0.0
  - Else: 1.0 - mc

- If solve:
  - 1.0 if correct else -0.5

---

## 16. Training Configuration and Interaction

Use verifiers/trainers/grpo_trainer.py:

- num_generations = G (maps to AZR’s B notion in your notes for normalization)
- The RepeatSampler ensures group-wise advantage normalization across generations of the same prompt (compute mean and std per group).
- Our dataset of 6 tasks will be repeated and batched by GRPO; environment.rollout dynamically injects buffer content; newly proposed tasks are only available for solver sampling in subsequent steps (matches AZR finding that solve batches draw from buffers existing before the current step’s proposals).

Sampling args:

- Provided by GRPO config via AsyncBatchGenerator; environment passes them into get_model_response, including temperature/top_p/logprobs.

vLLM integration:

- GRPOTrainer handles weight sync to vLLM server periodically; environment is agnostic.

---

## 17. Edge Cases and Error Handling

- Cold start: ensure at least the zero-triplet is converted into both D_deduction and D_abduction. For induction buffer seeding, can bootstrap with the zero program and a small set of trivial inputs (e.g., ["A", "B"]) to create initial induction items.
- Non-JSON or missing tags: format_ok=False -> -1 reward.
- Syntactically valid JSON but missing required keys: valid=False -> -0.5 for solve; for propose, also avoid MC.
- Program without function def: invalid.
- Non-deterministic code: invalid.
- Abduction solve: If candidate input evaluation throws, incorrect.
- Induction solve: Program compile failure or any hidden pair mismatch -> incorrect.

---

## 18. Testing Plan

- Unit test executor with trivial functions and forbidden patterns.

- Unit test parser with correct and malformed R1 texts.

- Unit test buffer sampling recency bias.

- Dry-run env.rollout for each task with hand-authored responses (mocking client via verifiers/tests/mock_openai_client.py pattern) to verify:

  - State flags (format_ok, valid, correct) and reward mapping
  - Proposer adds to buffers and computes MC accuracy
  - Solve samples from buffers and scores

- Small GRPO training run over 60 steps with mc_samples=2 to validate pipelines.

---

## 19. Implementation Steps (Checklist)

1. Scaffolding

- Create absolute_zero.py exporting load_environment()
- Implement AZRParser, AZRExecutor, AZRBufferManager, AZRRubric, AZREnv

2. Parser

- Implement strict R1+JSON parsing and helpers

3. Executor

- Implement sandboxed exec, function discovery (prefer def f), determinism checks

4. Buffers

- Implement add/sample with recency bias, conversion between triplet and per-task items

5. Prompt templates

- Implement builder methods for dp/ap/ip/ds/as/is

6. Environment.rollout

- Implement control flow for each task, JSON parsing, validation
- Implement proposer MC evaluation logic

7. Rubric

- Implement azr_reward() composite mapping; log key metrics

8. Dataset

- Build 6-entry dataset with base system prompt and minimal per-task base user message; env.rollout appends dynamic content

9. Seeding

- Add zero triplet; (optional) bootstrap induction buffer with trivial inputs

10. Tests / validation

- Add small smoke tests and a demo train config

---

## 20. References to AZR Code (from your summary)

- absolute_zero_reasoner/trainer/ppo/azr_ray_trainer.py:

  - fit loop orchestrating propose then solve
  - fallback sampling (lines ~740-747)
  - valid-only buffer updates (lines ~918-936)

- absolute_zero_reasoner/rewards/reward_managers.py:

  - proposer MC reward computation (lines ~531-876)
  - format gating and correctness integration (lines ~447-529)
  - intrinsic rewards (we omit initially; plan allows adding later)

- absolute_zero_reasoner/rewards/code_reward.py:
  - code/input/output parsing utilities

Behavior we replicate:

- Propose and solve decoupled in same step; new proposals appear for solve in subsequent steps due to buffer lifecycle.
- Invalid proposals kept in PPO batch but not added to buffers; proposer reward 0-ish (we gate -0.5 for formatted invalid, -1 for format error).
- MC evaluation uses identical generation path as solve (we reuse get_model_response).

---

## 21. Future Extensions (Optional)

- Add intrinsic rewards for proposer when format_ok and mc_acc > 0 (complexity/diversity/ast edit distance), controlled by flags.
- More sophisticated safety (AST-level checks), and richer deterministic constraints.
- Program schemas beyond single-function f, including multi-argument inputs.
- Persist buffers across runs via disk-backed store.

---

## 22. File Layout (target)

- /Users/eligottlieb/Documents/environments/absolute_zero/absolute_zero.py

  - class AZRParser
  - class AZRExecutor
  - class AZRBufferManager
  - class AZRRubric
  - class AZREnv
  - def load_environment(...)

End of design.
