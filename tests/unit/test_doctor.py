from pathlib import Path

from bv.config import AppConfig, FFmpegConfig, IndexTTS2Config
from bv.core.process import CommandResult
from bv.workflow.doctor import classify_check
from bv.workflow.doctor import _feature_check
from bv.workflow.doctor import run_doctor


def _minimal_doctor(
    *,
    environment: dict[str, str] | None = None,
    live: bool = False,
    codex_probe=None,
    grok_probe=None,
    synthesis_probe=None,
    asr_probe=None,
):
    calls: list[list[str]] = []

    def fake_runner(argv, **kwargs) -> CommandResult:
        del kwargs
        calls.append(list(argv))
        return CommandResult(argv=list(argv), returncode=0, stdout="")

    report = run_doctor(
        AppConfig(),
        live=live,
        runner=fake_runner,
        executable_finder=lambda command: command,
        environment=environment or {},
        write_test=lambda path: True,
        codex_probe=codex_probe,
        grok_probe=grok_probe,
        synthesis_probe=synthesis_probe,
        asr_probe=asr_probe,
    )
    return report, calls


def _status(report, name: str) -> str:
    return next(check for check in report.checks if check.name == name).status


def test_directory_does_not_count_as_function_ready() -> None:
    result = classify_check(
        name="IndexTTS2 synthesis",
        executable_exists=True,
        functional_probe_passed=False,
    )
    assert result.status == "UNVERIFIED"


def test_non_live_doctor_uses_injected_local_probes_only(tmp_path: Path) -> None:
    install_dir = tmp_path / "index-tts"
    model_dir = install_dir / "checkpoints"
    install_dir.mkdir()
    model_dir.mkdir()
    voice_path = tmp_path / "reference.wav"
    font_path = tmp_path / "subtitle.ttf"
    ffmpeg_path = tmp_path / "ffmpeg.exe"
    ffprobe_path = tmp_path / "ffprobe.exe"
    for path in (voice_path, font_path, ffmpeg_path, ffprobe_path):
        path.write_bytes(b"test")

    config = AppConfig(
        workspace_dir=tmp_path,
        voice_path=voice_path,
        subtitle_font_path=font_path,
        indextts2=IndexTTS2Config(
            install_dir=install_dir,
            model_dir=model_dir,
        ),
        ffmpeg=FFmpegConfig(
            command=ffmpeg_path,
            ffprobe_command=ffprobe_path,
        ),
    )
    calls: list[list[str]] = []
    live_probe_called = False

    def fake_runner(
        argv: list[str],
        *,
        cwd: Path | None = None,
        timeout: float = 60.0,
        secrets: tuple[str, ...] = (),
    ) -> CommandResult:
        del cwd, timeout, secrets
        calls.append(argv)
        output = ""
        if argv[-1] == "-filters":
            output = " ... subtitles ..."
        if argv[-1] == "-encoders":
            output = " ... libx264 ..."
        return CommandResult(argv=list(argv), returncode=0, stdout=output)

    def fake_finder(command: str) -> str | None:
        return command if command in {"uv", "codex", "grok"} else None

    def forbidden_live_probe() -> bool:
        nonlocal live_probe_called
        live_probe_called = True
        return True

    report = run_doctor(
        config,
        runner=fake_runner,
        executable_finder=fake_finder,
        environment={},
        write_test=lambda path: path == tmp_path,
        synthesis_probe=forbidden_live_probe,
    )

    assert live_probe_called is False
    assert [
        "uv",
        "run",
        "--project",
        str(install_dir),
        "indextts2",
        "check",
        "--model-dir",
        str(model_dir),
        "--device",
        "cuda",
    ] in calls
    assert next(
        check for check in report.checks if check.name == "IndexTTS2 check command"
    ).status == "READY"


def test_non_live_does_not_call_injected_real_probes() -> None:
    calls: list[str] = []
    report, runner_calls = _minimal_doctor(
        environment={"BV_VOLC_ASR_API_KEY": "set"},
        codex_probe=lambda: calls.append("codex") or True,
        grok_probe=lambda: calls.append("grok") or True,
        synthesis_probe=lambda: calls.append("synthesis") or True,
        asr_probe=lambda config: calls.append("asr") or True,
    )

    assert calls == []
    assert _status(report, "Codex real probe") == "UNVERIFIED"
    assert _status(report, "Grok real probe") == "UNVERIFIED"
    assert _status(report, "IndexTTS2 synthesis probe") == "UNVERIFIED"
    assert _status(report, "Volcengine live ASR probe") == "UNVERIFIED"
    assert not any(
        argv[:2] in (["codex", "--version"], ["grok", "--version"])
        for argv in runner_calls
    )


def test_live_without_real_probes_is_unverified_and_does_not_use_version() -> None:
    report, runner_calls = _minimal_doctor(live=True)

    assert _status(report, "Codex real probe") == "UNVERIFIED"
    assert _status(report, "Grok real probe") == "UNVERIFIED"
    assert not any(
        argv[:2] in (["codex", "--version"], ["grok", "--version"])
        for argv in runner_calls
    )


def test_live_uses_injected_codex_and_grok_verdicts() -> None:
    report, runner_calls = _minimal_doctor(
        live=True,
        codex_probe=lambda: True,
        grok_probe=lambda: False,
    )

    assert _status(report, "Codex real probe") == "READY"
    assert _status(report, "Grok real probe") == "UNVERIFIED"
    assert not any(
        argv[:2] in (["codex", "--version"], ["grok", "--version"])
        for argv in runner_calls
    )


def test_live_synthesis_and_asr_use_injected_verdicts() -> None:
    called: list[str] = []
    report, _ = _minimal_doctor(
        live=True,
        environment={"BV_VOLC_ASR_API_KEY": "set"},
        synthesis_probe=lambda: called.append("synthesis") or False,
        asr_probe=lambda config: called.append("asr") or True,
    )

    assert called == ["synthesis", "asr"]
    assert _status(report, "IndexTTS2 synthesis probe") == "UNVERIFIED"
    assert _status(report, "Volcengine live ASR probe") == "READY"


def test_api_only_volc_credentials_are_ready() -> None:
    report, _ = _minimal_doctor(environment={"BV_VOLC_ASR_API_KEY": "set"})

    assert _status(report, "Volcengine credentials") == "READY"


def test_legacy_pair_volc_credentials_are_ready() -> None:
    report, _ = _minimal_doctor(
        environment={
            "BV_VOLC_ASR_APP_KEY": "set",
            "BV_VOLC_ASR_ACCESS_KEY": "set",
        }
    )

    assert _status(report, "Volcengine credentials") == "READY"


def test_incomplete_volc_credentials_are_missing() -> None:
    report, _ = _minimal_doctor(
        environment={"BV_VOLC_ASR_APP_KEY": "set"}
    )

    assert _status(report, "Volcengine credentials") == "MISSING"


def test_ffmpeg_feature_requires_successful_command() -> None:
    result = CommandResult(
        argv=["ffmpeg", "-filters"],
        returncode=1,
        stdout="subtitles",
        stderr="failed",
    )

    check = _feature_check(
        name="FFmpeg libass/subtitles",
        result=result,
        feature=lambda output: "subtitles" in output,
    )

    assert check.status == "UNVERIFIED"
