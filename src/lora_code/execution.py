"""Runs generated programs in a subprocess with rlimits and a timeout.

This catches infinite loops, memory blowups and stray file writes, but it is
not a real sandbox, so run evals somewhere disposable.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

PASSED, FAILED, TIMEOUT = "passed", "failed", "timeout"


@dataclass
class ExecResult:
    status: str
    stderr_tail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASSED


def _limits(cpu_seconds: int, memory_mb: int):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        mem = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        os.setsid()
    return apply


def run_program(program: str, timeout: float = 10.0, memory_mb: int = 2048) -> ExecResult:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "HOME": tmp, "TMPDIR": tmp}
        preexec = _limits(int(timeout) + 1, memory_mb) if sys.platform.startswith("linux") else None
        proc = subprocess.Popen(
            [sys.executable, "-I", path], cwd=tmp, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            preexec_fn=preexec,
        )
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.communicate()
            return ExecResult(TIMEOUT)
        tail = err.decode(errors="replace")[-500:]
        if proc.returncode == 0:
            return ExecResult(PASSED, tail)
        if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
            return ExecResult(TIMEOUT, tail)
        return ExecResult(FAILED, tail)


def run_many(programs: list[str], timeout: float = 10.0, workers: int = 8) -> list[ExecResult]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: run_program(p, timeout), programs))
