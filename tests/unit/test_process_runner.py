from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import bv.core.process as process_module
from bv.core.process import CommandResult, run_command


def test_runner_passes_arguments_without_shell_expansion(tmp_path: Path) -> None:
    result = run_command(
        ["python", "-c", "import sys; print(sys.argv[1])", "x & whoami"],
        cwd=tmp_path,
        timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "x & whoami"


def test_runner_redacts_secrets(tmp_path: Path) -> None:
    result = run_command(
        ["python", "-c", "print('secret-value')"],
        cwd=tmp_path,
        timeout=10,
        secrets={"secret-value"},
    )
    assert "secret-value" not in result.stdout


def test_runner_rejects_string_argv(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        run_command("python -c print('unsafe')", cwd=tmp_path, timeout=10)  # type: ignore[arg-type]


def test_runner_redacts_stderr_and_displayed_argv(tmp_path: Path) -> None:
    result = run_command(
        [
            sys.executable,
            "-c",
            "import sys; print('secret-value', file=sys.stderr)",
            "secret-value",
        ],
        cwd=tmp_path,
        timeout=10,
        secrets={"secret-value"},
    )
    assert "secret-value" not in result.stderr
    assert all("secret-value" not in value for value in result.displayed_argv)


def test_runner_marks_timeout_and_returns(tmp_path: Path) -> None:
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=tmp_path,
        timeout=0.1,
    )
    assert result.returncode == -9
    assert "timed out" in result.stderr


def test_command_result_has_fixed_pydantic_contract() -> None:
    result = CommandResult(
        argv=["tool", "<REDACTED>"],
        returncode=0,
        stdout="out",
        stderr="err",
    )

    assert isinstance(result, process_module.BaseModel)
    assert set(CommandResult.model_fields) == {
        "argv",
        "returncode",
        "stdout",
        "stderr",
    }
    assert isinstance(result.argv, list)
    assert result.displayed_argv == result.argv
    assert not hasattr(result, "timed_out")


def test_runner_startup_failure_uses_stable_integer_sentinel(tmp_path: Path) -> None:
    result = run_command(
        [str(tmp_path / "missing-command.exe")],
        cwd=tmp_path,
        timeout=10,
    )

    assert result.returncode == -1
    assert isinstance(result.returncode, int)


def test_timeout_bounds_taskkill_and_second_communicate(monkeypatch) -> None:
    class Pipe:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class Process:
        pid = 1234

        def __init__(self) -> None:
            self.returncode = None
            self.communicate_calls = 0
            self.kill_calls = 0
            self.stdin = Pipe()
            self.stdout = Pipe()
            self.stderr = Pipe()

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.kill_calls += 1
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

        def communicate(self, input=None, timeout=None):
            del input
            self.communicate_calls += 1
            raise subprocess.TimeoutExpired(
                ["fake-command"],
                timeout,
                output=f"partial-{self.communicate_calls}",
                stderr=f"stderr-{self.communicate_calls}",
            )

    process = Process()
    taskkill_calls = []

    def fake_taskkill(argv, **kwargs):
        taskkill_calls.append((argv, kwargs))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(process_module.os, "name", "nt")
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(process_module.subprocess, "run", fake_taskkill)

    result = run_command(["fake-command"], timeout=0.01)

    assert result.returncode == -9
    assert result.stdout == "partial-2"
    assert "stderr-2" in result.stderr
    assert "timed out" in result.stderr
    assert process.communicate_calls == 2
    assert process.kill_calls >= 1
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert taskkill_calls[0][1]["timeout"] > 0
