import builtins as _py_builtins
import json as _json
import math
import sys
from typing import Any, Dict, List
from copy import deepcopy


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


def _discover_callable(env: Dict[str, Any]):
    if "f" in env and callable(env["f"]):
        return env["f"]
    for k, v in env.items():
        if callable(v) and not k.startswith("__"):
            return v
    return None


def _call_function(f, inp: Any):
    import inspect

    sig = inspect.signature(f)
    params = list(sig.parameters.values())
    param_count = len(params)
    if param_count == 0:
        return f()
    if param_count == 1:
        single_param = params[0]
        if isinstance(inp, dict) and single_param.kind == inspect.Parameter.VAR_KEYWORD:
            return f(**inp)
        if isinstance(inp, list) and len(inp) == 1 and isinstance(inp[0], (list, dict, tuple)):
            return f(inp[0])
        return f(inp)
    if isinstance(inp, dict):
        return f(**inp)
    if isinstance(inp, (list, tuple)):
        return f(*inp)
    return f(inp)


def main() -> int:
    stdin = sys.stdin
    stdout = sys.stdout
    while True:
        line = stdin.readline()
        if not line:
            break
        try:
            req = _json.loads(line)
            program: str = req.get("program", "")
            runs: int = int(req.get("runs", 2))
            # input arrives as repr; eval safely using Python literals only
            inp_literal: str = req.get("input_literal", "None")
            inp = None
            try:
                import ast

                inp = ast.literal_eval(inp_literal)
            except Exception:
                inp = inp_literal

            # static scan (disallow imports and dangerous builtins)
            import re

            disallowed = [
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
                r"\brandom\b",
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
            for pat in disallowed:
                if re.search(pat, program):
                    raise ValueError(f"Program contains disallowed pattern: {pat}")

            safe_globals = {
                "__builtins__": ALLOWED_BUILTINS,
                "__build_class__": _py_builtins.__build_class__,
                "__name__": "__main__",
                "math": math,
            }
            exec(program, safe_globals, safe_globals)
            f = _discover_callable(safe_globals)
            if f is None:
                raise ValueError("No callable function found (expected def f(...): ...)")

            outputs: List[Any] = []
            for _ in range(max(2, runs)):
                outputs.append(_call_function(f, deepcopy(inp)))
            resp = {"status": "ok", "payload": outputs}
        except Exception as exc:
            import traceback

            resp = {"status": "err", "error": traceback.format_exc()}
        stdout.write(_json.dumps(resp, ensure_ascii=False) + "\n")
        stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

