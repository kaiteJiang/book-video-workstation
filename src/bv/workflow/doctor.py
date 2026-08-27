from __future__ import annotations

import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from bv.config import AppConfig, VolcAsrConfig
from bv.core.process import CommandResult, run_command


CheckStatus = Literal["READY", "MISSING", "WARNING", "UNVERIFIED"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str = ""
    blocking: bool = True
    displayed_argv: tuple[str, ...] = ()


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[CheckResult, ...]
    live: bool = False

    @property
    def has_blocking_missing(self) -> bool:
        return any(
            check.blocking and check.status == "MISSING" for check in self.checks
        )

    @property
    def ready(self) -> bool:
        return not self.has_blocking_missing and all(
            check.status == "READY" or not check.blocking for check in self.checks
        )

    def __iter__(self):
        return iter(self.checks)


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: list[str],
        cwd: Path | str | None = None,
        stdin_text: str | None = None,
        timeout: float = 60.0,
        secrets: set[str] | tuple[str, ...] = (),
    ) -> CommandResult: ...


Probe = Callable[[], bool]
ExecutableFinder = Callable[[str], str | None]
WriteTest = Callable[[Path], bool]


def classify_check(
    name: str,
    executable_exists: bool,
    functional_probe_passed: bool | None,
    detail: str = "",
    blocking: bool = True,
    displayed_argv: tuple[str, ...] = (),
) -> CheckResult:
    """Classify presence separately from demonstrated functionality."""

    if not executable_exists:
        status: CheckStatus = "MISSING"
    elif functional_probe_passed is True:
        status = "READY"
    else:
        status = "UNVERIFIED"
    return CheckResult(
        name=name,
        status=status,
        detail=detail,
        blocking=blocking,
        displayed_argv=displayed_argv,
    )


def _presence(
    name: str,
    exists: bool,
    *,
    detail: str,
    blocking: bool = True,
) -> CheckResult:
    return CheckResult(
        name=name,
        status="READY" if exists else "MISSING",
        detail=detail,
        blocking=blocking,
    )


def _command_exists(command: str | Path, finder: ExecutableFinder) -> bool:
    value = str(command)
    path = Path(value)
    if path.is_absolute() or path.parent != Path("."):
        return path.is_file()
    return finder(value) is not None


def _command_result_ok(result: CommandResult) -> bool:
    return result.returncode == 0


def _run_probe(
    runner: CommandRunner,
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 15.0,
) -> CommandResult:
    return runner(argv, cwd=cwd, timeout=timeout, secrets=())


def _command_check(
    *,
    name: str,
    command: str | Path,
    runner: CommandRunner,
    finder: ExecutableFinder,
    probe_argv: list[str] | None,
    cwd: Path | None = None,
    timeout: float = 15.0,
    blocking: bool = True,
    missing_detail: str = "executable not found",
) -> CheckResult:
    exists = _command_exists(command, finder)
    if not exists:
        return _presence(name, False, detail=missing_detail, blocking=blocking)
    if probe_argv is None:
        return _presence(name, True, detail="executable found", blocking=blocking)
    result = _run_probe(runner, probe_argv, cwd=cwd, timeout=timeout)
    return CheckResult(
        name=name,
        status="READY" if _command_result_ok(result) else "UNVERIFIED",
        detail="probe passed" if _command_result_ok(result) else "probe failed",
        blocking=blocking,
        displayed_argv=tuple(getattr(result, "displayed_argv", probe_argv)),
    )


def _default_write_test(workspace: Path) -> bool:
    if not workspace.is_dir():
        return False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=workspace, prefix=".bv-doctor-", delete=True
        ) as marker:
            marker.write("ok")
            marker.flush()
        return True
    except OSError:
        return False


def _python_check() -> CheckResult:
    version = platform.python_version()
    supported = sys.version_info[:2] == (3, 11)
    return CheckResult(
        name="Python version",
        status="READY" if supported else "WARNING",
        detail=f"{version} (requires Python 3.11)",
        blocking=True,
    )


def _feature_check(
    *,
    name: str,
    result: CommandResult,
    feature: Callable[[str], bool],
) -> CheckResult:
    output = _safe_probe_output(result)
    passed = feature(output)
    passed = _command_result_ok(result) and passed
    return CheckResult(
        name=name,
        status="READY" if passed else "UNVERIFIED",
        detail="feature reported by FFmpeg" if passed else "feature not reported by FFmpeg",
        displayed_argv=tuple(result.displayed_argv),
    )


def _safe_probe_output(result: CommandResult) -> str:
    return f"{result.stdout}\n{result.stderr}"


def run_doctor(
    config: AppConfig,
    *,
    live: bool = False,
    runner: CommandRunner | None = None,
    executable_finder: ExecutableFinder = shutil.which,
    environment: Mapping[str, str] | None = None,
    write_test: WriteTest | None = None,
    codex_probe: Probe | None = None,
    grok_probe: Probe | None = None,
    synthesis_probe: Probe | None = None,
    asr_probe: Callable[[VolcAsrConfig], bool] | None = None,
) -> DoctorReport:
    """Run local readiness checks, with quota-using probes gated by ``live``."""

    command_runner = runner or run_command
    env = environment if environment is not None else os.environ
    checks: list[CheckResult] = [_python_check()]

    uv_command = config.indextts2.uv_command
    uv_check = _command_check(
        name="uv version",
        command=uv_command,
        runner=command_runner,
        finder=executable_finder,
        probe_argv=[str(uv_command), "--version"],
    )
    checks.append(uv_check)

    codex_command = config.codex.command
    checks.append(
        _command_check(
            name="Codex executable",
            command=codex_command,
            runner=command_runner,
            finder=executable_finder,
            probe_argv=None,
        )
    )
    checks.append(
        _live_cli_probe(
            name="Codex real probe",
            command=codex_command,
            live=live,
            finder=executable_finder,
            probe=codex_probe,
            blocking=True,
        )
    )

    grok_blocking = config.grok.blocking
    grok_command = config.grok.command
    checks.append(
        _command_check(
            name="Grok executable",
            command=grok_command,
            runner=command_runner,
            finder=executable_finder,
            probe_argv=None,
            blocking=grok_blocking,
        )
    )
    checks.append(
        _live_cli_probe(
            name="Grok real probe",
            command=grok_command,
            live=live,
            finder=executable_finder,
            probe=grok_probe,
            blocking=grok_blocking,
        )
    )

    install_dir = config.indextts2.install_dir
    model_dir = config.indextts2.model_dir
    checks.append(
        _presence(
            "IndexTTS2 project",
            install_dir.is_dir(),
            detail=str(install_dir) if install_dir.is_dir() else "project directory not found",
        )
    )
    checks.append(
        _presence(
            "IndexTTS2 model directory",
            model_dir.is_dir(),
            detail=str(model_dir) if model_dir.is_dir() else "model directory not found",
        )
    )

    indextts_check_argv = [
        str(uv_command),
        "run",
        "--project",
        str(install_dir),
        "indextts2",
        "check",
        "--model-dir",
        str(model_dir),
        "--device",
        config.indextts2.device,
    ]
    check_ready = (
        install_dir.is_dir()
        and model_dir.is_dir()
        and _command_exists(uv_command, executable_finder)
    )
    if not check_ready:
        checks.append(
            CheckResult(
                name="IndexTTS2 check command",
                status="MISSING",
                detail="project, model directory, or uv is missing",
                displayed_argv=tuple(indextts_check_argv),
            )
        )
    else:
        result = _run_probe(
            command_runner,
            indextts_check_argv,
            cwd=install_dir,
            timeout=60.0,
        )
        checks.append(
            CheckResult(
                name="IndexTTS2 check command",
                status="READY" if _command_result_ok(result) else "UNVERIFIED",
                detail="check passed" if _command_result_ok(result) else "check failed",
                displayed_argv=tuple(
                    getattr(result, "displayed_argv", indextts_check_argv)
                ),
            )
        )

    checks.append(
        _live_optional_probe(
            name="IndexTTS2 synthesis probe",
            live=live,
            probe=synthesis_probe,
            detail_when_skipped="skipped; use --live",
        )
    )

    checks.append(_cuda_check(command_runner, executable_finder))

    ffmpeg_command = config.ffmpeg.command
    ffmpeg_exists = _command_exists(ffmpeg_command, executable_finder)
    ffmpeg_version: CommandResult | None = None
    if ffmpeg_exists:
        ffmpeg_version = _run_probe(
            command_runner, [str(ffmpeg_command), "-version"], timeout=15.0
        )
    checks.append(
        CheckResult(
            name="FFmpeg",
            status=(
                "READY"
                if ffmpeg_version is not None and _command_result_ok(ffmpeg_version)
                else "MISSING"
                if not ffmpeg_exists
                else "UNVERIFIED"
            ),
            detail="version probe passed"
            if ffmpeg_version is not None and _command_result_ok(ffmpeg_version)
            else "executable not found"
            if not ffmpeg_exists
            else "version probe failed",
            displayed_argv=(str(ffmpeg_command), "-version"),
        )
    )

    ffprobe_command = config.ffmpeg.ffprobe_command
    checks.append(
        _command_check(
            name="FFprobe",
            command=ffprobe_command,
            runner=command_runner,
            finder=executable_finder,
            probe_argv=[str(ffprobe_command), "-version"],
        )
    )

    filters_argv = [str(ffmpeg_command), "-hide_banner", "-filters"]
    encoders_argv = [str(ffmpeg_command), "-hide_banner", "-encoders"]
    if ffmpeg_version is None or not _command_result_ok(ffmpeg_version):
        checks.append(
            CheckResult(
                name="FFmpeg libass/subtitles",
                status="MISSING" if not ffmpeg_exists else "UNVERIFIED",
                detail="FFmpeg version probe did not pass",
                displayed_argv=tuple(filters_argv),
            )
        )
        checks.append(
            CheckResult(
                name="H.264 encoder",
                status="MISSING" if not ffmpeg_exists else "UNVERIFIED",
                detail="FFmpeg version probe did not pass",
                displayed_argv=tuple(encoders_argv),
            )
        )
    else:
        filters_result = _run_probe(command_runner, filters_argv, timeout=15.0)
        encoders_result = _run_probe(command_runner, encoders_argv, timeout=15.0)
        checks.append(
            _feature_check(
                name="FFmpeg libass/subtitles",
                result=filters_result,
                feature=lambda output: "subtitles" in output,
            )
        )
        checks.append(
            _feature_check(
                name="H.264 encoder",
                result=encoders_result,
                feature=lambda output: any(
                    encoder in output
                    for encoder in (
                        "libx264",
                        "h264_nvenc",
                        "h264_amf",
                        "h264_qsv",
                        "h264_mf",
                    )
                ),
            )
        )

    api_key_present = bool(env.get(config.volc_asr.api_key_env))
    legacy_pair_present = all(
        bool(env.get(name))
        for name in (
            config.volc_asr.app_key_env,
            config.volc_asr.access_key_env,
        )
    )
    credentials_present = api_key_present or legacy_pair_present
    checks.append(
        CheckResult(
            name="Volcengine credentials",
            status="READY" if credentials_present else "MISSING",
            detail="required credential values are present"
            if credentials_present
            else "one or more required credential values are absent",
        )
    )
    if not credentials_present:
        checks.append(
            CheckResult(
                name="Volcengine live ASR probe",
                status="MISSING",
                detail="credentials are missing",
            )
        )
    else:
        checks.append(
            _live_optional_probe(
                name="Volcengine live ASR probe",
                live=live,
                probe=(lambda: asr_probe(config.volc_asr)) if asr_probe else None,
                detail_when_skipped="skipped; use --live",
            )
        )

    checks.append(
        _presence(
            "Reference voice",
            config.voice_path.is_file(),
            detail=str(config.voice_path)
            if config.voice_path.is_file()
            else "reference voice file not found",
        )
    )
    checks.append(
        _presence(
            "Subtitle font",
            config.subtitle_font_path.is_file(),
            detail=str(config.subtitle_font_path)
            if config.subtitle_font_path.is_file()
            else "subtitle font file not found",
        )
    )

    writer = write_test or _default_write_test
    workspace_writable = writer(config.workspace_dir)
    checks.append(
        CheckResult(
            name="Workspace write test",
            status="READY" if workspace_writable else "MISSING",
            detail="temporary write succeeded"
            if workspace_writable
            else "workspace is missing or not writable",
        )
    )

    return DoctorReport(checks=tuple(checks), live=live)


def _live_cli_probe(
    *,
    name: str,
    command: str,
    live: bool,
    finder: ExecutableFinder,
    probe: Probe | None,
    blocking: bool,
) -> CheckResult:
    if not live:
        return CheckResult(
            name=name,
            status="UNVERIFIED",
            detail="skipped; use --live",
            blocking=blocking,
        )
    if not _command_exists(command, finder):
        return CheckResult(
            name=name,
            status="MISSING",
            detail="executable not found",
            blocking=blocking,
        )
    if probe is None:
        return CheckResult(
            name=name,
            status="UNVERIFIED",
            detail="no real probe configured",
            blocking=blocking,
        )
    try:
        passed = probe()
    except Exception:
        passed = False
    return CheckResult(
        name=name,
        status="READY" if passed else "UNVERIFIED",
        detail="live probe passed" if passed else "live probe failed",
        blocking=blocking,
    )


def _live_optional_probe(
    *,
    name: str,
    live: bool,
    probe: Probe | None,
    detail_when_skipped: str,
) -> CheckResult:
    if not live:
        return CheckResult(name=name, status="UNVERIFIED", detail=detail_when_skipped)
    if probe is None:
        return CheckResult(
            name=name,
            status="UNVERIFIED",
            detail="no live probe configured",
        )
    try:
        passed = probe()
    except Exception:
        passed = False
    return CheckResult(
        name=name,
        status="READY" if passed else "UNVERIFIED",
        detail="live probe passed" if passed else "live probe failed",
    )


def _cuda_check(
    runner: CommandRunner,
    finder: ExecutableFinder,
) -> CheckResult:
    if finder("nvidia-smi") is not None:
        argv = [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ]
        result = _run_probe(runner, argv, timeout=15.0)
        return CheckResult(
            name="CUDA device",
            status="READY" if _command_result_ok(result) else "UNVERIFIED",
            detail="GPU query passed" if _command_result_ok(result) else "GPU query failed",
            displayed_argv=tuple(getattr(result, "displayed_argv", argv)),
        )

    argv = [
        sys.executable,
        "-c",
        "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
    ]
    result = _run_probe(runner, argv, timeout=30.0)
    return CheckResult(
        name="CUDA device",
        status="READY" if _command_result_ok(result) else "MISSING",
        detail="PyTorch CUDA probe passed"
        if _command_result_ok(result)
        else "nvidia-smi is unavailable and CUDA is not ready",
        displayed_argv=tuple(getattr(result, "displayed_argv", argv)),
    )


def render_doctor_report(report: DoctorReport) -> None:
    for check in report.checks:
        suffix = f": {check.detail}" if check.detail else ""
        print(f"{check.status:<10} {check.name}{suffix}")
