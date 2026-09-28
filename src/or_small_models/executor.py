"""Bounded subprocess execution for generated Gurobi programs."""

from __future__ import annotations

import ast
import math
import os
import re
import site
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .records import ExecutionResult

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_OBJECTIVE_PATTERN = re.compile(rf"^FINAL_OBJECTIVE=\s*({_NUMBER})\s*$", re.MULTILINE)
_ALLOWED_IMPORTS = {
    "collections",
    "functools",
    "gurobipy",
    "itertools",
    "math",
    "statistics",
    "sys",
    "typing",
}
_BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
_ALLOWED_DUNDER_NAMES = {"__", "__name__"}
_ENV_ALLOWLIST = {
    "GRB_LICENSE_FILE",
    "GUROBI_HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "LICENSEID",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "PYTHONPATH",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
    "WLSACCESSID",
    "WLSSECRET",
}


class CodeSafetyError(ValueError):
    """Raised when generated code attempts an operation blocked by the runtime."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_generated_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        raise
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    raise CodeSafetyError(
                        "blocked_import", f"Import '{alias.name}' is not allowed."
                    )
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module not in _ALLOWED_IMPORTS:
                raise CodeSafetyError(
                    "blocked_import", f"Import from '{node.module}' is not allowed."
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
                raise CodeSafetyError(
                    "blocked_call", f"Call '{node.func.id}' is not allowed."
                )
        elif (
            isinstance(node, ast.Name)
            and node.id.startswith("__")
            and node.id not in _ALLOWED_DUNDER_NAMES
        ):
            raise CodeSafetyError(
                "blocked_dunder_access", "Dunder names are not allowed."
            )
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CodeSafetyError(
                "blocked_dunder_access", "Dunder attribute access is not allowed."
            )
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        ):
            raise CodeSafetyError(
                "blocked_module_access", "Access to sys attributes is not allowed."
            )
class GurobiExecutor:
    """Execute one generated program once; this is not a security sandbox."""

    def __init__(self, timeout_seconds: float = 120) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str) -> ExecutionResult:
        started = time.perf_counter()
        try:
            validate_generated_code(code)
            compile(code, "generated_model.py", "exec")
        except SyntaxError as error:
            return ExecutionResult(
                outcome="compilation_error",
                success=False,
                stderr=str(error),
                latency_seconds=time.perf_counter() - started,
            )
        except CodeSafetyError as error:
            return ExecutionResult(
                outcome=error.code,
                success=False,
                stderr=str(error),
                latency_seconds=time.perf_counter() - started,
            )

        runtime = Path(__file__).with_name("gurobi_runtime.py")
        with tempfile.TemporaryDirectory(prefix="or-small-model-") as directory:
            script = Path(directory) / "generated_model.py"
            script.write_text(code, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, str(runtime), str(script)],
                    cwd=directory,
                    env=_minimal_environment(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                return ExecutionResult(
                    outcome="timeout",
                    success=False,
                    stdout=_text(error.stdout),
                    stderr=f"Code execution exceeded {self.timeout_seconds} seconds.",
                    latency_seconds=time.perf_counter() - started,
                )

        stdout = completed.stdout[-20000:]
        stderr = completed.stderr[-20000:]
        matches = _OBJECTIVE_PATTERN.findall(stdout)
        objective = float(matches[0]) if len(matches) == 1 else None
        if objective is not None and not math.isfinite(objective):
            objective = None
        if completed.returncode != 0:
            outcome = (
                "compilation_error"
                if "SyntaxError" in stderr
                else "runtime_error"
            )
            return ExecutionResult(
                outcome=outcome,
                success=False,
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                latency_seconds=time.perf_counter() - started,
            )
        if objective is None:
            return ExecutionResult(
                outcome="output_contract_error",
                success=False,
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                latency_seconds=time.perf_counter() - started,
            )
        return ExecutionResult(
            outcome="executed",
            success=True,
            objective_value=objective,
            return_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            latency_seconds=time.perf_counter() - started,
        )


def _minimal_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in _ENV_ALLOWLIST
    }
    # On Windows, Python's user site directory depends on APPDATA, which is
    # intentionally excluded from the allowlist. Preserve package discovery for
    # installations such as a user-level gurobipy package by adding that path
    # explicitly to the child interpreter's PYTHONPATH.
    user_site = site.getusersitepackages()
    python_path = environment.get("PYTHONPATH", "")
    paths = [path for path in python_path.split(os.pathsep) if path]
    if user_site and user_site not in paths:
        paths.append(user_site)
    if paths:
        environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-20000:]
    return (value or "")[-20000:]
