import atexit
import builtins as _py_builtins
import inspect
import math
import multiprocessing as mp
import os
import pickle
import queue
import re
import signal
import threading
import time
import traceback
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

# =========================
# Safe Python executor
# =========================

class _ExecutionTimeout(Exception):
    """Raised when user code exceeds the configured runtime budget."""


class AZRExecutor:
    """
    Sandboxed Python executor for single-function programs.
    - Disallows import and dangerous builtins/modules.
    - Provides a whitelisted set of builtins and math module.
    - Supports calling with single arg, *args (if input is list/tuple), or **kwargs (if input is dict).
    - Determinism check by repeated runs.
    """

    DISALLOWED_PATTERNS = [
        r"\bimport\b",
        r"__import__",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bopen\s*\(",
        r"\bcompile\s*\(",
        r"\bos\b",
        r"\bsys\b",
        r"\bshutil\b",
        r"\bsubprocess\b",
        r"\bpathlib\b",
        r"\bsocket\b",
        r"\brequests\b",
        r"\burllib\b",
        r"\bhttp\b",
        r"\bjsonpickle\b",
        r"\bpickle\b",
        r"\bmarshal\b",
        r"\bctypes\b",
        r"\bimportlib\b",
        r"\bthreading\b",
        r"\bmultiprocessing\b",
        r"\brandom\b",  # disallow stochastic behavior
        r"\bnumpy\b",
        r"\bpandas\b",
        r"\btime\.sleep\s*\(",
        r"\bbuiltins\b",
        r"\bglobals\s*\(",
        r"\blocals\s*\(",
        r"\bvars\s*\(",
        r"\bsetattr\s*\(",
        r"\bdelattr\s*\(",
        r"\bgetattr\s*\(",
    ]

    ALLOWED_BUILTINS = {
        "__build_class__": _py_builtins.__build_class__,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "chr": chr,
        "object": object,
        "type": type,
        "super": super,
        "isinstance": isinstance,
        "staticmethod": staticmethod,
        "classmethod": classmethod,
        "property": property,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "ord": ord,
        "pow": pow,
        "range": range,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "reversed": reversed,
    }

    def __init__(
        self,
        max_exec_seconds: float = 5.0,
        start_method: Optional[str] = None,
        max_workers: Optional[int] = None,
    ):
        """Initialize the executor and worker pool.

        We retain the ``start_method`` parameter for compatibility, but
        execution is now delegated to a persistent process pool instead of
        spawning a fresh interpreter per snippet. This mirrors the reference
        Absolute-Zero-Reasoner setup and avoids repeated model re-initialisation.
        """

        self.max_exec_seconds = float(max_exec_seconds)
        self.start_method = None
        self.max_workers = max_workers or max(1, (os.cpu_count() or 2) // 2)

        requested_method = start_method or os.getenv("AZR_EXECUTOR_START_METHOD")
        available_methods = mp.get_all_start_methods()
        candidates: List[str] = []
        if requested_method and requested_method in available_methods:
            candidates.append(requested_method)
        if not candidates:
            if "spawn" in available_methods:
                candidates.append("spawn")
            if "forkserver" in available_methods:
                candidates.append("forkserver")
            if "fork" in available_methods:
                candidates.append("fork")

        ctx: Optional[mp.context.BaseContext] = None
        for cand in candidates:
            try:
                ctx = mp.get_context(cand)
                self.start_method = cand
                break
            except ValueError:
                continue

        if ctx is None:
            ctx = mp.get_context()
            self.start_method = ctx.get_start_method()

        self._mp_context = ctx
        self._task_queue = None
        self._result_queue = None
        self._workers: List[mp.Process] = []
        self._job_counter = 0
        self._job_lock = threading.Lock()
        self._result_cache: Dict[int, Tuple[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._pool_lock = threading.Lock()
        atexit.register(self.shutdown)

    @classmethod
    def _static_scan(cls, program: str) -> Optional[str]:
        for pat in cls.DISALLOWED_PATTERNS:
            if re.search(pat, program):
                return f"Program contains disallowed pattern: {pat}"
        return None

    @staticmethod
    def _discover_callable(env: Dict[str, Any]) -> Optional[Callable]:
        # Prefer 'f', else first callable defined by user (skip dunder)
        if "f" in env and callable(env["f"]):
            return env["f"]
        for k, v in env.items():
            if callable(v) and not k.startswith("__"):
                return v
        return None

    @staticmethod
    def _call_function(f: Callable, inp: Any) -> Any:
        sig = inspect.signature(f)
        try:
            params = list(sig.parameters.values())
            param_count = len(params)
            if param_count == 0:
                return f()
            # If function expects exactly one parameter, prefer passing dict as a single positional
            # unless the parameter is **kwargs (VAR_KEYWORD), in which case unpack the dict.
            if param_count == 1:
                single_param = params[0]
                if isinstance(inp, dict) and single_param.kind == inspect.Parameter.VAR_KEYWORD:
                    return f(**inp)
                # Accept "wrapped single-arg" convention: [[...]] or [{...}] or [(...)]
                if isinstance(inp, list) and len(inp) == 1 and isinstance(inp[0], (list, dict, tuple)):
                    return f(inp[0])
                return f(inp)
            # For multi-arg functions, unpack mappings and sequences appropriately
            if isinstance(inp, dict):
                return f(**inp)
            if isinstance(inp, (list, tuple)):
                return f(*inp)
            # Fallback: pass as single positional
            return f(inp)
        except Exception as e:
            raise e

    def compile_program(self, program: str) -> Tuple[Optional[Callable], Optional[str]]:
        # Static filter before attempting to exec user code in restricted globals
        err = self._static_scan(program)
        if err:
            return None, err
        # Prepare restricted globals
        safe_globals: Dict[str, Any] = {
            "__builtins__": self.ALLOWED_BUILTINS,
            "__build_class__": _py_builtins.__build_class__,
            "__name__": "__main__",
            "math": math,
        }
        try:
            # Execute with the same globals/locals so top-level defs are visible to functions' globals
            exec(program, safe_globals, safe_globals)
        except Exception as e:
            return None, f"Program exec failed: {e}"
        # discover callable
        f = self._discover_callable(safe_globals)
        if f is None:
            return None, "No callable function found (expected def f(...): ...)"
        try:
            setattr(f, "__azr_source__", program)
        except Exception:
            pass
        return f, None

    def run_deterministic(self, f: Callable, inp: Any, runs: int = 2) -> Tuple[bool, Optional[Any], Optional[str]]:
        # Execute multiple times in a sandboxed subprocess with timeout to ensure determinism
        runs = max(2, runs)
        program_src = getattr(f, "__azr_source__", None)
        if not isinstance(program_src, str):
            return False, None, "Missing program source for execution"
        ok, outputs, error = self._execute_with_timeout(program_src, inp, runs)
        if not ok:
            return False, None, error
        if not outputs:
            return False, None, "No output produced"
        first = outputs[0]
        for out in outputs[1:]:
            if out != first:
                return False, None, "Non-deterministic output across runs"
        return True, first, None

    # Convenience helpers for common checks

    def eval_output_prediction(
        self,
        code: str,
        program_input: Any,
        gold_output: Any,
        agent_output: Any,
        runs: int = 2,
    ) -> bool:
        f, err = self.compile_program(code)
        if f is None:
            return False
        ok, out, _ = self.run_deterministic(f, program_input, runs=runs)
        if not ok:
            return False
        return out == agent_output or agent_output == gold_output

    def eval_abduction_input(
        self, code: str, gold_output: Any, agent_input: Any, runs: int = 2
    ) -> bool:
        f, err = self.compile_program(code)
        if f is None:
            return False
        ok, out, _ = self.run_deterministic(f, agent_input, runs=runs)
        if not ok:
            return False
        return out == gold_output

    def eval_program_on_pairs(
        self, code: str, io_pairs: List[Tuple[Any, Any]], runs: int = 2
    ) -> bool:
        f, err = self.compile_program(code)
        if f is None:
            return False
        for inp, out_exp in io_pairs:
            ok, out, _ = self.run_deterministic(f, inp, runs=runs)
            if not ok or out != out_exp:
                return False
        return True

    def _execute_with_timeout(
        self, program: str, inp: Any, runs: int
    ) -> Tuple[bool, Optional[List[Any]], Optional[str]]:
        job_id = self._submit_job(program, inp, runs)
        wait_timeout = max(self.max_exec_seconds, 0.1) + 0.5
        try:
            status, payload = self._get_result(job_id, wait_timeout)
        except mp.TimeoutError:
            self._restart_pool()
            return False, None, "Execution timed out"
        except RuntimeError as exc:
            self._restart_pool()
            return False, None, str(exc)
        except Exception as exc:
            self._restart_pool()
            return False, None, str(exc)

        if status == "ok":
            return True, payload, None
        if status == "timeout":
            return False, None, "Execution timed out"
        return False, None, str(payload)

    def _submit_job(self, program: str, inp: Any, runs: int) -> int:
        self._ensure_pool()
        assert self._task_queue is not None
        with self._job_lock:
            job_id = self._job_counter
            self._job_counter += 1
        try:
            self._task_queue.put((job_id, program, inp, runs, self.max_exec_seconds))
        except Exception as exc:
            raise RuntimeError(f"Failed to enqueue job: {exc}") from exc
        return job_id

    def _get_result(self, job_id: int, timeout: float) -> Tuple[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            with self._cache_lock:
                cached = self._result_cache.pop(job_id, None)
            if cached is not None:
                return cached

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise mp.TimeoutError()

            if self._result_queue is None:
                raise RuntimeError("Worker result queue unavailable")
            try:
                job_result = self._result_queue.get(timeout=remaining)
            except queue.Empty:
                raise mp.TimeoutError()
            except (EOFError, OSError) as exc:
                raise RuntimeError(f"Worker pool communication error: {exc}") from exc

            if not isinstance(job_result, tuple) or len(job_result) != 3:
                continue
            result_job_id, status, payload = job_result
            if result_job_id == job_id:
                return status, payload
            with self._cache_lock:
                self._result_cache[result_job_id] = (status, payload)

    def _ensure_pool(self) -> None:
        with self._pool_lock:
            if not self._workers:
                self._start_workers()

    def _start_workers(self) -> None:
        self._task_queue = self._mp_context.Queue(maxsize=self.max_workers * 2)
        self._result_queue = self._mp_context.Queue()
        with self._cache_lock:
            self._result_cache.clear()
        self._workers = []
        for _ in range(self.max_workers):
            proc = self._mp_context.Process(
                target=_worker_loop,
                args=(self._task_queue, self._result_queue),
                daemon=True,
            )
            proc.start()
            self._workers.append(proc)

    def _restart_pool(self) -> None:
        with self._pool_lock:
            self._stop_workers_locked()
            self._start_workers()

    def _stop_workers_locked(self) -> None:
        if self._task_queue is not None:
            for _ in self._workers:
                try:
                    self._task_queue.put_nowait(None)
                except Exception:
                    pass
        for proc in self._workers:
            try:
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
            except Exception:
                pass
        self._workers = []
        if self._task_queue is not None:
            try:
                self._task_queue.close()
            except Exception:
                pass
            self._task_queue = None
        if self._result_queue is not None:
            try:
                self._result_queue.close()
            except Exception:
                pass
            self._result_queue = None
        with self._cache_lock:
            self._result_cache.clear()

    def shutdown(self) -> None:
        with self._pool_lock:
            self._stop_workers_locked()

    def __del__(self):  # pragma: no cover - defensive cleanup
        try:
            self.shutdown()
        except Exception:
            pass


def _worker_loop(task_queue, result_queue) -> None:
    while True:
        try:
            task = task_queue.get()
        except (EOFError, OSError):
            break
        if task is None:
            break
        if not isinstance(task, tuple) or len(task) != 5:
            continue
        job_id, program, inp, runs, timeout_seconds = task
        try:
            status, payload = _run_program_with_timeout(program, inp, runs, timeout_seconds)
        except Exception as exc:  # pragma: no cover - defensive
            status, payload = "err", f"Worker execution failed: {exc}"
        try:
            result_queue.put((job_id, status, payload))
        except Exception as exc:  # payload not serializable
            safe_payload = f"Failed to serialize result: {exc}"
            try:
                result_queue.put((job_id, "err", safe_payload))
            except Exception:
                pass


def _run_program_with_timeout(
    program: str,
    inp: Any,
    runs: int,
    timeout_seconds: Optional[float],
) -> Tuple[str, Any]:
    try:
        err = AZRExecutor._static_scan(program)
        if err:
            return "err", err

        safe_globals: Dict[str, Any] = {
            "__builtins__": AZRExecutor.ALLOWED_BUILTINS,
            "__build_class__": _py_builtins.__build_class__,
            "__name__": "__main__",
            "math": math,
        }
        exec(program, safe_globals, safe_globals)
        f = AZRExecutor._discover_callable(safe_globals)
        if f is None:
            return "err", "No callable function found (expected def f(...): ...)"

        max_runs = max(2, runs)
        timer_supported = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
        per_run_timeout = None
        use_timer = False
        previous_handler = None
        if timeout_seconds and timeout_seconds > 0 and timer_supported:
            per_run_timeout = max(timeout_seconds / max_runs, 0.01)
            use_timer = True

            def _raise_timeout(signum, frame):  # type: ignore[unused-argument]
                raise _ExecutionTimeout(f"Execution exceeded {per_run_timeout:.2f}s budget")

            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _raise_timeout)

        outputs: List[Any] = []
        try:
            for _ in range(max_runs):
                if use_timer and per_run_timeout is not None:
                    signal.setitimer(signal.ITIMER_REAL, per_run_timeout)
                try:
                    outputs.append(AZRExecutor._call_function(f, deepcopy(inp)))
                except _ExecutionTimeout as exc:
                    return "timeout", str(exc)
                finally:
                    if use_timer:
                        signal.setitimer(signal.ITIMER_REAL, 0.0)
        finally:
            if use_timer:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if previous_handler is not None:
                    signal.signal(signal.SIGALRM, previous_handler)

        try:
            pickle.dumps(outputs)
        except Exception as exc:
            return "err", f"Serialization error: {exc}"
        return "ok", outputs
    except Exception:
        return "err", traceback.format_exc()


# Re-export helper for process pool pickling
__all__ = ["AZRExecutor", "_run_program_with_timeout"]
