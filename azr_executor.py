import builtins as _py_builtins
import inspect
import math
import pickle
import re
import signal
import subprocess
import sys
import traceback
from pathlib import Path
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
    ):
        """Initialize the executor.

        The previous multiprocessing based implementation selected a context
        start method here. We keep the signature for backwards compatibility,
        but execution now happens via a dedicated helper subprocess so no
        multiprocessing context is required.
        """

        _ = start_method  # preserved for compatibility; intentionally unused
        self.max_exec_seconds = float(max_exec_seconds)
        self.start_method = "subprocess"
        self._worker_script = Path(__file__).with_name("azr_executor_worker.py")

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
        job = {
            "program": program,
            "input": inp,
            "runs": runs,
            "timeout": self.max_exec_seconds,
        }

        if not self._worker_script.exists():  # pragma: no cover - defensive
            return False, None, f"Worker script not found at {self._worker_script}"

        try:
            completed = subprocess.run(
                [sys.executable, str(self._worker_script)],
                input=pickle.dumps(job),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=max(self.max_exec_seconds, 0.1) + 1.0,
            )
        except subprocess.TimeoutExpired:
            return False, None, "Execution timed out"
        except Exception as exc:
            return False, None, f"Failed to launch sandbox process: {exc}"

        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", "ignore").strip()
            stdout = completed.stdout.decode("utf-8", "ignore").strip()
            message = stderr or stdout or f"Worker exited with code {completed.returncode}"
            return False, None, message

        try:
            status, payload = pickle.loads(completed.stdout)
        except Exception as exc:
            return False, None, f"Failed to decode worker response: {exc}"

        if status == "ok":
            return True, payload, None
        if status == "timeout":
            return False, None, "Execution timed out"
        return False, None, str(payload)


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

        return "ok", outputs
    except Exception:
        return "err", traceback.format_exc()


# Exported for the worker module
__all__ = ["AZRExecutor", "_run_program_with_timeout"]
