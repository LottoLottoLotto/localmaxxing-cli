#!/usr/bin/env python3
"""Execution-based verifier harness for code eval shards (HumanEval / MBPP).

Reads one JSON task per line from stdin and writes one JSON result per line to
stdout. Each task: {"question_id": str, "program": str}. The program is the full
self-contained Python source (model solution + hidden tests + an invocation that
must raise on failure). A task passes iff the program exits 0 within limits.

This file is the in-container entrypoint, but it has no third-party dependencies
so it can also be invoked directly (e.g. `python3 run_sandbox.py`) for
environments without Docker. Network isolation and the read-only root filesystem
are provided by the container flags; this harness adds per-task CPU/memory/time
limits and process isolation as defense in depth.
"""

import json
import os
import resource
import subprocess
import sys
import tempfile
import time

CPU_SECONDS = int(os.environ.get("SANDBOX_CPU_SECONDS", "10"))
WALL_SECONDS = float(os.environ.get("SANDBOX_WALL_SECONDS", "15"))
MEM_BYTES = int(os.environ.get("SANDBOX_MEM_MB", "512")) * 1024 * 1024
STDERR_LIMIT = int(os.environ.get("SANDBOX_STDERR_LIMIT", "4000"))


def _limit() -> None:
    # Applied in the child before exec. Caps CPU time, address space, file size,
    # and process count so a hostile or runaway solution cannot exhaust the box.
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS + 1))
    resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ValueError, OSError):
        pass
    os.setsid()


def run_one(program: str) -> dict:
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "solution.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(program)
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                cwd=work,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=_limit,
                timeout=WALL_SECONDS,
            )
            returncode = proc.returncode
            stderr = proc.stderr.decode("utf-8", "replace")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = -1
            stderr = (exc.stderr.decode("utf-8", "replace") if exc.stderr else "") + "\n[timeout]"
        duration_ms = int((time.monotonic() - started) * 1000)
        if len(stderr) > STDERR_LIMIT:
            stderr = stderr[:STDERR_LIMIT] + "\n[...truncated]"
        return {
            "passed": (not timed_out) and returncode == 0,
            "returncode": returncode,
            "timed_out": timed_out,
            "stderr": stderr.strip(),
            "duration_ms": duration_ms,
        }


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
            qid = task.get("question_id", "")
            program = task.get("program", "")
        except (ValueError, AttributeError) as exc:
            sys.stdout.write(json.dumps({"question_id": "", "passed": False, "error": f"bad task: {exc}"}) + "\n")
            sys.stdout.flush()
            continue
        if not program:
            sys.stdout.write(json.dumps({"question_id": qid, "passed": False, "error": "empty program"}) + "\n")
            sys.stdout.flush()
            continue
        result = run_one(program)
        result["question_id"] = qid
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
