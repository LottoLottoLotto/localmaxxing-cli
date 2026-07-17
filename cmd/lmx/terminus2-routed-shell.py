#!/usr/bin/env python3
"""Run Harbor's actual Terminus-2 agent against a localmaxxing task container.

This adapter is intentionally thin: localmaxxing-cli owns task/container/verifier
lifecycle, while Harbor Terminus-2 owns the terminal-agent loop. Commands execute
inside the already-running task container through Docker, matching Harbor's tmux
interaction model instead of the localmaxxing built-in shell protocol.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _prefer_harbor_tool_python() -> None:
    """Re-exec under uv's Harbor tool Python when this script was launched with system Python."""
    if os.environ.get("LMX_TERMINUS2_REEXEC") == "1":
        return
    home = Path.home()
    for candidate in (
        home / ".local/share/uv/tools/harbor/bin/python3",
        home / ".local/share/uv/tools/harbor/bin/python",
    ):
        if candidate.exists() and candidate.is_file() and Path(sys.executable).resolve() != candidate.resolve():
            env = os.environ.copy()
            env["LMX_TERMINUS2_REEXEC"] = "1"
            os.execve(str(candidate), [str(candidate), *sys.argv], env)


_prefer_harbor_tool_python()

try:
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.environments.base import ExecResult
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by integration runs.
    raise SystemExit(
        "Could not import Harbor Terminus-2. Install Harbor or run this adapter with "
        "the uv tool environment that provides the `harbor` Python package."
    ) from exc


@dataclass
class RoutedEnvironment:
    """Small duck-typed Harbor environment backed by `docker exec`.

    Terminus-2 and TmuxSession only need a subset of BaseEnvironment for terminal
    tasks. Implement that subset directly instead of letting Harbor create its own
    container; localmaxxing-cli has already started the canonical task image.
    """

    container: str
    trace_dir: Path
    workdir: str
    default_user: str | None
    command_timeout_sec: float

    def __post_init__(self) -> None:
        self.session_id = os.environ.get("LMX_TERMINAL_TASK_ID", "lmx-terminus2")
        self.environment_name = self.session_id
        self.trial_paths = SimpleNamespace(agent_dir=self.trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.exec_log = self.trace_dir / "environment-exec.jsonl"

    def _docker_args(self, user: str | int | None = None, cwd: str | None = None) -> list[str]:
        args = ["docker", "exec", "-i"]
        resolved_user = user if user is not None else self.default_user
        if resolved_user not in (None, ""):
            args += ["--user", str(resolved_user)]
        args += ["-w", cwd or self.workdir, self.container, "bash", "-lc"]
        return args

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
        timeout_sec: float | None = None,
        **_: Any,
    ) -> ExecResult:
        effective_timeout = self.command_timeout_sec
        if timeout_sec is not None and timeout_sec > 0:
            effective_timeout = min(effective_timeout, timeout_sec)
        full_command = command
        if env:
            exports = " ".join(f"{shlex.quote(k)}={shlex.quote(v)}" for k, v in env.items())
            full_command = f"export {exports}; {command}"
        token = os.urandom(12).hex()
        pidfile = f"/tmp/lmx-terminal-command-{token}.pid"
        wrapped_command = (
            f"pidfile={shlex.quote(pidfile)}; rm -f \"$pidfile\"; "
            f"setsid bash -lc {shlex.quote(full_command)} & child=$!; "
            "printf '%s\\n' \"$child\" > \"$pidfile\"; "
            "wait \"$child\"; rc=$?; rm -f \"$pidfile\"; exit \"$rc\""
        )
        proc = await asyncio.create_subprocess_exec(
            *self._docker_args(user=user, cwd=cwd),
            wrapped_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            terminated, termination_detail = await self._terminate_container_group(pidfile)
            proc.kill()
            stdout_b, stderr_b = await proc.communicate()
            stderr = stderr_b.decode(errors="replace") + f"\n[command_timeout after {effective_timeout}s]"
            if termination_detail:
                stderr += "\n" + termination_detail
            self._log_exec(command, cwd, user, 124, stdout_b.decode(errors="replace"), stderr)
            if not terminated:
                raise RuntimeError("timed-out command process group could not be terminated inside the task container")
            return ExecResult(stdout=stdout_b.decode(errors="replace"), stderr=stderr, return_code=124)

        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        return_code = int(proc.returncode or 0)
        self._log_exec(command, cwd, user, return_code, stdout, stderr)
        return ExecResult(stdout=stdout, stderr=stderr, return_code=return_code)

    async def _terminate_container_group(self, pidfile: str) -> tuple[bool, str]:
        cleanup = (
            f"pidfile={shlex.quote(pidfile)}; "
            "test -s \"$pidfile\" || exit 2; pid=$(cat \"$pidfile\"); "
            "case \"$pid\" in ''|*[!0-9]*) exit 3;; esac; "
            "kill -TERM -- -\"$pid\" 2>/dev/null || true; "
            "i=0; while kill -0 -- -\"$pid\" 2>/dev/null && [ \"$i\" -lt 50 ]; do sleep 0.1; i=$((i+1)); done; "
            "if kill -0 -- -\"$pid\" 2>/dev/null; then kill -KILL -- -\"$pid\" 2>/dev/null || true; fi; "
            "i=0; while kill -0 -- -\"$pid\" 2>/dev/null && [ \"$i\" -lt 20 ]; do sleep 0.1; i=$((i+1)); done; "
            "alive=0; kill -0 -- -\"$pid\" 2>/dev/null && alive=1; rm -f \"$pidfile\"; exit \"$alive\""
        )
        killer = await asyncio.create_subprocess_exec(
            "docker", "exec", self.container, "bash", "-lc", cleanup,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(killer.communicate(), timeout=10)
        except asyncio.TimeoutError:
            killer.kill()
            await killer.communicate()
            return False, "[command_timeout cleanup itself timed out]"
        detail = (stdout_b + stderr_b).decode(errors="replace").strip()
        return killer.returncode == 0, detail

    def _log_exec(self, command: str, cwd: str | None, user: str | int | None, return_code: int, stdout: str, stderr: str) -> None:
        event = {
            "command": command,
            "cwd": cwd or self.workdir,
            "user": str(user if user is not None else self.default_user or ""),
            "return_code": return_code,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }
        with self.exec_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    async def is_dir(self, path: str | os.PathLike[str]) -> bool:
        result = await self.exec(f"test -d {shlex.quote(str(path))}")
        return result.return_code == 0

    async def upload_file(self, source_path: str | os.PathLike[str], target_path: str | os.PathLike[str], **_: Any) -> None:
        target = str(target_path)
        subprocess.run(["docker", "exec", self.container, "mkdir", "-p", str(Path(target).parent)], check=False)
        with open(source_path, "rb") as src:
            proc = subprocess.run(["docker", "exec", "-i", self.container, "tee", target], stdin=src, stdout=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise RuntimeError(f"docker upload failed for {target}")

    async def download_file(self, source_path: str | os.PathLike[str], target_path: str | os.PathLike[str], **_: Any) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as dst:
            proc = subprocess.run(["docker", "exec", self.container, "cat", str(source_path)], stdout=dst)
        if proc.returncode != 0:
            raise RuntimeError(f"docker download failed for {source_path}")


def _openai_base_url(raw: str) -> str:
    raw = raw.rstrip("/")
    return raw if raw.endswith("/v1") else raw + "/v1"


def _model_info() -> dict[str, Any]:
    max_input = int(os.environ.get("LMX_TERMINUS2_MAX_INPUT_TOKENS", "262144"))
    max_output = int(os.environ.get("LMX_TERMINUS2_MAX_OUTPUT_TOKENS", "16384"))
    return {
        "litellm_provider": "openai",
        "mode": "chat",
        "max_input_tokens": max_input,
        "max_output_tokens": max_output,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }


def _write_usage_jsonl(trace_dir: Path, context: AgentContext) -> None:
    usage = {
        "input_tokens": int(context.n_input_tokens or 0),
        "output_tokens": int(context.n_output_tokens or 0),
        "cache_read_tokens": int(context.n_cache_tokens or 0),
        "total_tokens": int((context.n_input_tokens or 0) + (context.n_output_tokens or 0) + (context.n_cache_tokens or 0)),
    }
    n_calls = max(1, int((context.metadata or {}).get("n_episodes") or 1))
    keys = ("input_tokens", "output_tokens", "cache_read_tokens", "total_tokens")
    with (trace_dir / "terminus-usage.jsonl").open("w", encoding="utf-8") as fh:
        for index in range(n_calls):
            event_usage = {
                key: (value // n_calls) + (1 if index < (value % n_calls) else 0)
                for key, value in usage.items()
                if key in keys
            }
            event = {"type": "message_end", "message": {"role": "assistant", "usage": event_usage}}
            fh.write(json.dumps(event) + "\n")
    with (trace_dir / "usage-summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"usage": usage, "metadata": context.metadata or {}}, fh, indent=2)
        fh.write("\n")


async def main() -> int:
    required = [
        "LMX_TERMINAL_CONTAINER",
        "LMX_TERMINAL_INSTRUCTION_FILE",
        "LMX_TERMINAL_TRACE_DIR",
        "LMX_TERMINAL_MODEL",
        "LMX_TERMINAL_BASE_URL",
    ]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        print(f"missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2

    trace_dir = Path(os.environ["LMX_TERMINAL_TRACE_DIR"])
    trace_dir.mkdir(parents=True, exist_ok=True)
    instruction = Path(os.environ["LMX_TERMINAL_INSTRUCTION_FILE"]).read_text(encoding="utf-8")
    model = os.environ["LMX_TERMINAL_MODEL"]
    base_url = _openai_base_url(os.environ["LMX_TERMINAL_BASE_URL"])
    max_turns = int(os.environ.get("LMX_TERMINAL_MAX_TURNS") or os.environ.get("LMX_TERMINUS2_MAX_TURNS") or "100")
    temperature_raw = os.environ.get("LMX_TERMINUS2_TEMPERATURE")
    temperature = float(temperature_raw) if temperature_raw not in (None, "") else None
    default_user = os.environ.get("LMX_TERMINAL_AGENT_USER") or None
    command_timeout_sec = float(os.environ.get("LMX_TERMINAL_COMMAND_TIMEOUT_SECONDS") or "14400")

    env = RoutedEnvironment(
        container=os.environ["LMX_TERMINAL_CONTAINER"],
        trace_dir=trace_dir,
        workdir=os.environ.get("LMX_TERMINAL_WORKDIR", "/app"),
        default_user=default_user,
        command_timeout_sec=command_timeout_sec,
    )
    context = AgentContext()
    agent = Terminus2(
        logs_dir=trace_dir,
        model_name=model,
        max_turns=max_turns,
        parser_name="json",
        api_base=base_url,
        temperature=temperature,
        model_info=_model_info(),
        record_terminal_session=False,
        suppress_max_turns_warning=True,
        enable_summarize=os.environ.get("LMX_TERMINUS2_ENABLE_SUMMARIZE", "1") != "0",
        proactive_summarization_threshold=int(os.environ.get("LMX_TERMINUS2_SUMMARIZE_THRESHOLD", "8000")),
        llm_call_kwargs={"api_key": os.environ.get("LMX_TERMINAL_MODEL_API_KEY") or "localmaxxing"},
    )

    await agent.setup(env)
    try:
        await agent.run(instruction=instruction, environment=env, context=context)
    finally:
        if getattr(agent, "_session", None) is not None:
            await agent._session.stop()
        _write_usage_jsonl(trace_dir, context)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
