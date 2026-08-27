from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from bv.core.redaction import redact_text


class CommandResult(BaseModel):
    """The safe, diagnostic result of a child-process invocation.

    ``argv`` contains only the redacted display form of the command.  The raw
    command is intentionally not retained on the result object.
    """

    model_config = ConfigDict(extra="forbid")

    argv: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def displayed_argv(self) -> list[str]:
        return self.argv


def _normalise_argv(argv: Sequence[str]) -> list[str]:
    if isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of strings, not a string")
    if not isinstance(argv, Sequence):
        raise TypeError("argv must be a sequence of strings")

    values = list(argv)
    if not values:
        raise ValueError("argv must not be empty")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("argv must contain only strings")
    return values


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


_TASKKILL_TIMEOUT_SECONDS = 5.0
_POST_KILL_COMMUNICATE_TIMEOUT_SECONDS = 1.0
_POST_KILL_WAIT_TIMEOUT_SECONDS = 1.0


def _close_pipe(pipe: object | None) -> None:
    close = getattr(pipe, "close", None)
    if close is None:
        return
    try:
        close()
    except OSError:
        pass


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for name in ("stdin", "stdout", "stderr"):
        _close_pipe(getattr(process, name, None))


def _force_kill(process: subprocess.Popen[str]) -> None:
    try:
        if process.poll() is None:
            process.kill()
    except (OSError, ProcessLookupError):
        pass


def _wait_for_exit(process: subprocess.Popen[str]) -> None:
    try:
        process.wait(timeout=_POST_KILL_WAIT_TIMEOUT_SECONDS)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        pass


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate the process and its practical child tree where supported."""

    if process.poll() is not None:
        return

    if os.name == "nt":
        taskkill_succeeded = False
        try:
            taskkill_result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_TASKKILL_TIMEOUT_SECONDS,
            )
            taskkill_succeeded = taskkill_result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            taskkill_succeeded = False
        if not taskkill_succeeded or process.poll() is None:
            _force_kill(process)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            _force_kill(process)
        else:
            if process.poll() is None:
                _force_kill(process)
    _wait_for_exit(process)


def _partial_timeout_output(
    primary: subprocess.TimeoutExpired,
    fallback: subprocess.TimeoutExpired,
    attribute: str,
) -> str:
    value = getattr(primary, attribute, None)
    if value is None:
        value = getattr(fallback, attribute, None)
    return _text(value)


def run_command(
    argv: Sequence[str],
    cwd: Path | str | None = None,
    stdin_text: str | None = None,
    timeout: float = 60.0,
    secrets: Iterable[str] = (),
) -> CommandResult:
    """Run a command without shell interpretation and redact its diagnostics."""

    raw_argv = _normalise_argv(argv)
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    secret_values = tuple(secret for secret in secrets if secret)
    displayed_argv = tuple(redact_text(value, secret_values) for value in raw_argv)
    popen_kwargs: dict[str, object] = {
        "shell": False,
        "cwd": cwd,
        "stdin": subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if creationflags:
            popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(raw_argv, **popen_kwargs)
    except OSError as exc:
        message = exc.strerror or "unable to start command"
        return CommandResult(
            argv=list(displayed_argv),
            returncode=-1,
            stderr=redact_text(f"command failed to start: {message}", secret_values),
        )

    timed_out = False
    post_kill_communication_timed_out = False
    stdout: str | bytes | None = None
    stderr: str | bytes | None = None
    try:
        stdout, stderr = process.communicate(input=stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired as first_timeout:
        timed_out = True
        _terminate_process(process)
        try:
            stdout, stderr = process.communicate(
                timeout=_POST_KILL_COMMUNICATE_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as second_timeout:
            post_kill_communication_timed_out = True
            stdout = _partial_timeout_output(second_timeout, first_timeout, "output")
            stderr = _partial_timeout_output(second_timeout, first_timeout, "stderr")
            _close_process_pipes(process)
            _force_kill(process)
            _wait_for_exit(process)
        except OSError:
            stdout = _text(getattr(first_timeout, "output", None))
            stderr = _text(getattr(first_timeout, "stderr", None))
            _close_process_pipes(process)
            _force_kill(process)
            _wait_for_exit(process)

    stdout_text = redact_text(_text(stdout), secret_values)
    stderr_text = redact_text(_text(stderr), secret_values)
    if timed_out:
        timeout_message = f"command timed out after {timeout:g} seconds"
        stderr_text = f"{stderr_text}\n{timeout_message}".strip()
        if post_kill_communication_timed_out:
            stderr_text = (
                f"{stderr_text}\npost-kill communication timed out; "
                "diagnostics may be partial"
            )

    returncode = -9 if timed_out else process.returncode
    if returncode is None:
        returncode = -1

    return CommandResult(
        argv=list(displayed_argv),
        returncode=returncode,
        stdout=stdout_text,
        stderr=stderr_text,
    )
