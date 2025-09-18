import builtins as _py_builtins
import os
import inspect
import math
import multiprocessing
import re
import traceback
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

# =========================
# Safe Python executor
# =========================

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
        self.max_exec_seconds = float(max_exec_seconds)

        requested_method = start_method or os.getenv("AZR_EXECUTOR_START_METHOD")
        available_methods = multiprocessing.get_all_start_methods()

        candidates: list[str] = []
        if requested_method:
            candidates.append(requested_method)
        else:
            if "fork" in available_methods:
                candidates.append("fork")
            if "forkserver" in available_methods and "forkserver" not in candidates:
                candidates.append("forkserver")
            candidates.append("spawn")

        ctx: Optional[multiprocessing.context.BaseContext] = None
        self.start_method = None
        for cand in candidates:
            try:
                ctx = multiprocessing.get_context(cand)
                self.start_method = cand
                break
            except Exception:
                continue

        if ctx is None:
            ctx = multiprocessing.get_context()
            self.start_method = ctx.get_start_method()

        self._mp_context = ctx

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
        parent_conn, child_conn = self._mp_context.Pipe(duplex=False)
        process = self._mp_context.Process(
            target=_azr_executor_worker,
            args=(child_conn, program, inp, runs),
            daemon=True,
        )
        try:
            process.start()
        except Exception as exc:
            child_conn.close()
            parent_conn.close()
            return False, None, f"Failed to start sandbox process: {exc}"
        child_conn.close()
        status = "timeout"
        payload: Any = "Execution timed out"
        try:
            if parent_conn.poll(self.max_exec_seconds):
                try:
                    status, payload = parent_conn.recv()
                except EOFError:
                    status, payload = "err", "Sandbox exited without returning a result"
            elif not process.is_alive():
                status, payload = "err", "Sandbox exited without returning a result"
        finally:
            parent_conn.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
        if status == "ok":
            return True, payload, None
        if status == "timeout":
            return False, None, "Execution timed out"
        return False, None, str(payload)


def _azr_executor_worker(conn, program: str, inp: Any, runs: int) -> None:
    try:
        err = AZRExecutor._static_scan(program)
        if err:
            conn.send(("err", err))
            return
        safe_globals: Dict[str, Any] = {
            "__builtins__": AZRExecutor.ALLOWED_BUILTINS,
            "__build_class__": _py_builtins.__build_class__,
            "__name__": "__main__",
            "math": math,
        }
        exec(program, safe_globals, safe_globals)
        f = AZRExecutor._discover_callable(safe_globals)
        if f is None:
            conn.send(("err", "No callable function found (expected def f(...): ...)"))
            return
        outputs: List[Any] = []
        for _ in range(max(2, runs)):
            outputs.append(AZRExecutor._call_function(f, deepcopy(inp)))
        conn.send(("ok", outputs))
    except Exception:
        conn.send(("err", traceback.format_exc()))
    finally:
        try:
            conn.close()
        except Exception:
            pass
