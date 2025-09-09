In-depth code review guide

1. Architecture and entry points

- File: environments/absolute_zero/absolute_zero.py

- Entrypoint: load_environment(...) returns a configured AZREnv with:

  - dataset: a 6-row Dataset (optionally repeated), one per task type
  - parser: AZRParser
  - rubric: AZRRubric
  - message_type: "chat"

- Key classes:

  - AZRParser: Extracts DeepSeek R1 blocks and parses the JSON payload; provides format gating.
  - AZRExecutor: Sandboxed Python executor with static safety scan and determinism checks.
  - AZRBufferManager: Maintains validated triplets and per-task buffers with recency-biased sampling.
  - AZRRubric: Composite reward mapping for propose/solve with formatting penalties.
  - AZREnv: Implements prompts, rollout control flow, propose validation, MC evaluation, and solver scoring.

2. Parser (AZRParser)

- Expected format: The assistant must return two blocks:

  - <think>...</think> (ignored for scoring but required)
  - <answer>{ JSON payload }</answer> (strictly JSON)

- try_parse_r1_json(text) returns (format_ok, json_obj, error):

  - format_ok ensures both blocks exist in order and the answer block is non-empty JSON.
  - Code fences are stripped if present; language hints are removed before json.loads.
  - json_obj must be a JSON object (dict). Otherwise format_ok=False.

- Downstream effect:

  - If format_ok=False, AZRRubric returns -1.0 regardless of role/task.
  - If format_ok=True but JSON missing required keys, the environment marks valid=False or correct=False and rubric applies role-appropriate penalties.

3. Executor (AZRExecutor)

- Static safety scan (regex-based) blocks:

  - Any import usage, dynamic code exec functions, filesystem/network/system modules, multiprocessing/threads, stochastic modules, and other unsafe builtins.
  - The list includes: os, sys, shutil, subprocess, pathlib, socket, requests, urllib, http, jsonpickle, pickle, marshal, ctypes, importlib, threading, multiprocessing, random, numpy, pandas, time.sleep, builtins rebinding, reflective builtins.

- Allowed builtins: Minimal functional subset (abs, all, any, bool, dict, enumerate, filter, float, int, len, list, map, max, min, pow, range, set, sorted, str, sum, tuple, zip) and math module.

- Callable discovery: Prefers a function named f; otherwise the first user-defined callable.

- Input calling conventions:

  - dict -> **kwargs
  - list/tuple -> *args
  - else -> single positional argument

- Determinism check:
  - run_deterministic runs f multiple times with deepcopied input; outputs must be identical.

- Evaluators:

  - eval_abduction_input: Checks that f(agent_input) == gold_output.
  - eval_program_on_pairs: Validates predicted program against a list of (input, output) pairs.
  - Note: eval_output_prediction exists but is not used in this environment.

4. Buffers and data model (AZRBufferManager)

- Data classes:

  - Triplet: (program, input, output, step_id, created_at)
  - DeductionItem: (program, input, output, step_id, created_at)
  - AbductionItem: (program, input, output, step_id, created_at)
  - InductionItem: (program, message, io_pairs, visible_pairs, hidden_pairs, step_id, created_at)

- Initialization:

  - Optional zero triplet seeding with identity function f(x)=x on "Hello World".
  - Adds to triplet_set, deduction, and abduction buffers.

- add_triplet/add_induction: Inserts validated items with monotonically increasing step_id protected by a thread lock.

- Sampling:

  - _recency_sample: 70% picks most recent; otherwise picks uniformly from the rest.
  - sample_program_from_union: Returns the most recent program across deduction/abduction, falling back to triplet_set.
  - get_references(K): Returns K most recent triplets for proposer context.

5. Prompts (builder methods)

- All prompts remind the model of DeepSeek R1 format and specify a strict JSON schema in the answer block.

- Proposer:

  - _build_dp_prompt(K): Ask for {"program": "...", "input": ...}, include safety and determinism, with K references.
  - _build_ap_prompt(K): Same schema as deduction proposer to unify triplet generation.
  - _build_ip_prompt(program, K): Given a sampled program, ask for {"message": "...", "inputs": [...]}, include K references for style guidance.

- Solver:

  - _build_ds_prompt(item): Given (program, input), ask for {"output": ...}.
  - _build_as_prompt(item): Given (program, output), ask for {"input": ...}.
  - _build_is_prompt(item): Given message + visible_pairs, ask for {"program": "..."}; validates later on hidden_pairs.

6. Rollout control flow (AZREnv.rollout)

- Prepares rollout messages:

  - For proposer tasks, appends proposer user message to the dataset’s system message.
  - For solver tasks, constructs a fresh messages list including system + user with sampled items.

- Model call via get_model_response; assistant content stored in completion.

- State initialization:

  - Stores prompt, completion, answer (unused), task, info, responses, turn.
  - Adds flags: format_ok, json_ok, error. Prepares payload structure and sampled_problem_id.

- Early exit on formatting failure:

  - Propose: valid=False, propose.mc_accuracy=None.
  - Solve: correct=False.

- Task branches:

  - deduction.propose/_handle_dp:

    - Compile and run deterministically on provided input, derive output.
    - Add triplet to buffers (triplet_set, deduction, abduction).
    - MC evaluate deduction.solve N times; mc_accuracy = mean(correct).

  - abduction.propose/_handle_ap:
    - Same validation; MC evaluate abduction.solve N times; correctness via eval_abduction_input.

  - induction.propose/_handle_ip:

    - Given program_context, validate proposed inputs, compute io_pairs.
    - Split into visible/hidden halves, ensure both non-empty.
    - Add induction item; MC evaluate induction.solve N times by synthesizing a program and scoring on hidden pairs.

  - deduction.solve/_handle_ds:
    - Compare predicted output to gold output directly.

  - abduction.solve/_handle_as:
    - Execute program on predicted input and compare to gold output.

  - induction.solve/_handle_is:
    - Compile predicted program and validate on hidden pairs.

- State payload:

  - Proposer: includes program/input and derived outputs (or message/io_pairs for induction).
  - Solver: includes sampled program/problem context and predicted fields.

7. Monte Carlo evaluation (proposer reward estimation)

- _mc_deduction_solve: Repeats deduction.solve N times; counts exact matches on output.
- _mc_abduction_solve: Repeats abduction.solve N times; counts cases where f(pred_input) == gold_output.
- _mc_induction_solve: Repeats induction.solve N times; compiles predicted program and validates on hidden_pairs.
- Uses the same client/model/sampling_args path as main rollout to mirror AZR.

8. Reward mapping (AZRRubric)

- Format gating:
  - If not state["format_ok"] -> -1.0.

- Propose tasks:

  - If not state["valid"] or mc_accuracy is None -> -0.5.
  - Else if mc in {0.0, 1.0} -> 0.0.
  - Else -> 1.0 - mc.

- Solve tasks:
  - 1.0 if correct else -0.5.

- Notes:

  - This adheres to the design’s “format-aware penalties” and “proposer reward from MC accuracy” mapping.
  - Metrics such as format_ok/json_ok/valid/mc_acc/solve_correct are captured in state for downstream logging/analysis.

9. Dataset construction and scheduling

- _make_azr_dataset(system_prompt, repeats, K, mc_samples, determinism_runs) builds six rows (or repeats thereof), each with:

  - prompt: system message only
  - answer: empty
  - task: one of dp/ap/ip/ds/as/is
  - info: JSON string carrying K/mc_samples/determinism_runs

- The environment reads these and injects dynamic content at rollout time.

- This is compatible with verifiers’ GRPO flow where AsyncBatchGenerator calls env.rollout for each dataset row across generations.

10. Safety and determinism policy

- Safety via static scan:

  - Blocks dangerous modules/functions by regex prior to exec.
  - Rejects any code using import or disallowed constructs (including inside strings).

- Execution:

  - Restricted builtins and math only.
  - Any exception during exec or calls marks proposal invalid or solve incorrect as applicable.

- Determinism:
  - At least two runs per check; all outputs must match exactly.

11. Edge cases and fallbacks

- Cold start:

  - Zero-triplet ensures both deduction and abduction buffers are non-empty from the start.
  - For induction.solve, if buffer empty, it bootstraps a trivial identity program with simple pairs.

- Formatting errors:
  - The parser sets format_ok=False, which the rubric maps to -1.0.

- Missing keys even with valid JSON object:

  - Propose: valid=False → -0.5.
  - Solve: correct=False → -0.5.

12. Extensibility points

- Parser:
  - Can add more structural constraints (e.g., enforce presence of specific keys per task).

- Executor:

  - Consider an AST-based analyzer for stricter safety and better error messages.
  - Expand allowed builtins or permitted modules if tasks require (e.g., fractions).

- Buffers:
  - Persist buffers to disk; add capacity management or diversity constraints.

- Rubric:
  - Add intrinsic rewards or logging metrics via Rubric APIs if supported.

- Prompts:
  - Make K, mc_samples, and determinism_runs task-conditional via info.

13. Testing checklist

- Parser:

  - Valid R1 format yields (True, dict, None).
  - Missing tags, wrong order, empty answer, or malformed JSON yields (False, None, error).

- Executor:

  - Safe code executes deterministically with repeated runs equal.
  - Disallowed constructs are rejected pre-exec.

- Propose handlers:

  - Valid dp/ap produce new triplets and non-None mc_accuracy in [0,1].
  - Valid ip produces io_pairs with non-empty visible and hidden.

- Solver handlers:
  - ds exact equality scoring, as functional inverse check, is hidden-pair validation.

- MC evaluation:
  - Returns 0 ≤ mc_accuracy ≤ 1; respects mc_samples parameter.

- End-to-end:
  - Small mc_samples=2 run to confirm stability and state fields.

14. Known limitations and notes

- Regex-based safety can over-block benign strings and under-block exotic evasion; acceptable for initial sandboxing.
- Allowed builtins are intentionally minimal; expand only if necessary.
- Equality checks are strict; special types (e.g., floating-point tolerance) are not handled.
- eval_output_prediction is unused; retained for potential future evaluation variants.
