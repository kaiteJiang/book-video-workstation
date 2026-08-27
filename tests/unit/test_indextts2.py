from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from bv.core.process import CommandResult
from bv.voice.indextts2 import (
    VoiceSynthesisError,
    build_synth_argv,
    synthesize_voice,
)


def _write_wav(path: Path, *, frames: int = 480) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x00\x00" * frames)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    episode = tmp_path / "episode root"
    install = episode / "Index TTS install"
    model = install / "checkpoints"
    private = episode / ".private" / "voice"
    install.mkdir(parents=True)
    model.mkdir()
    private.mkdir(parents=True)
    script = episode / "approved script.txt"
    script.write_text("这是已经批准的旁白。", encoding="utf-8")
    reference = episode / "authorized reference.wav"
    _write_wav(reference)
    raw = private / "raw-unique.wav"
    return episode, install, model, script, reference, raw


def test_build_synth_argv_uses_private_text_file_and_exact_runtime_flags(tmp_path: Path) -> None:
    episode, install, model, script, reference, raw = _paths(tmp_path)

    argv = build_synth_argv(
        install_dir=install,
        model_dir=model,
        script_path=script,
        reference_voice_path=reference,
        raw_output_path=raw,
        uv_command="C:/tools/uv.exe",
    )

    assert argv == [
        "C:/tools/uv.exe", "run", "--project", str(install), "indextts2", "synth",
        "--text-file", str(script), "--voice", str(reference), "--output", str(raw),
        "--model-dir", str(model), "--device", "cuda", "--fp16", "--no-deepspeed",
        "--no-cuda-kernel", "--no-accel", "--no-torch-compile",
    ]
    assert "这是已经批准的旁白。" not in argv
    assert "--force" not in argv
    assert episode.name in argv[3]


@pytest.mark.parametrize("bad_text", ["", "\x00unsafe", "[REVIEW REQUIRED]"])
def test_synthesize_voice_rejects_unapproved_or_unsafe_script_before_runner(
    tmp_path: Path, bad_text: str
) -> None:
    episode, install, model, script, reference, raw = _paths(tmp_path)
    script.write_text(bad_text, encoding="utf-8")
    called = False

    def runner(**kwargs: object) -> CommandResult:
        nonlocal called
        called = True
        return CommandResult(argv=[], returncode=0)

    with pytest.raises(VoiceSynthesisError, match="invalid_approved_script"):
        synthesize_voice(
            episode_root=episode, install_dir=install, model_dir=model,
            script_path=script, approved_script_sha256=_sha256(script),
            reference_voice_path=reference, raw_output_path=raw, runner=runner,
        )

    assert called is False
    assert not raw.exists()


@pytest.mark.parametrize(
    "marker",
    ["<!-- BV:VOICEOVER:START -->", "<!-- BV:VOICEOVER:END -->"],
)
def test_synthesize_voice_rejects_task14_voiceover_markers_before_runner(
    tmp_path: Path, marker: str
) -> None:
    episode, install, model, script, reference, raw = _paths(tmp_path)
    script.write_text(f"{marker}\n这是待复核的旁白。", encoding="utf-8")
    called = False

    def runner(**kwargs: object) -> CommandResult:
        nonlocal called
        called = True
        return CommandResult(argv=[], returncode=0)

    with pytest.raises(VoiceSynthesisError, match="invalid_approved_script"):
        synthesize_voice(
            episode_root=episode, install_dir=install, model_dir=model,
            script_path=script, approved_script_sha256=_sha256(script),
            reference_voice_path=reference, raw_output_path=raw, runner=runner,
        )

    assert called is False
    assert not raw.exists()


def test_synthesize_voice_rejects_hash_mismatch_existing_output_and_invalid_reference(
    tmp_path: Path,
) -> None:
    episode, install, model, script, reference, raw = _paths(tmp_path)
    raw.write_bytes(b"not task owned")

    with pytest.raises(VoiceSynthesisError, match="raw_output_exists"):
        synthesize_voice(
            episode_root=episode, install_dir=install, model_dir=model,
            script_path=script, approved_script_sha256=_sha256(script),
            reference_voice_path=reference, raw_output_path=raw,
        )

    raw.unlink()
    reference.write_bytes(b"not a wave")
    with pytest.raises(VoiceSynthesisError, match="invalid_reference_wav"):
        synthesize_voice(
            episode_root=episode, install_dir=install, model_dir=model,
            script_path=script, approved_script_sha256=_sha256(script),
            reference_voice_path=reference, raw_output_path=raw,
        )


def test_synthesize_voice_returns_verified_fake_runner_wav_without_retaining_private_output(
    tmp_path: Path,
) -> None:
    episode, install, model, script, reference, raw = _paths(tmp_path)
    seen: dict[str, object] = {}

    def runner(*, argv: list[str], cwd: Path, timeout: float, secrets: tuple[str, ...]) -> CommandResult:
        seen.update(argv=argv, cwd=cwd, timeout=timeout, secrets=secrets)
        _write_wav(raw)
        return CommandResult(argv=["<REDACTED>"], returncode=0)

    result = synthesize_voice(
        episode_root=episode, install_dir=install, model_dir=model,
        script_path=script, approved_script_sha256=_sha256(script),
        reference_voice_path=reference, raw_output_path=raw, runner=runner,
    )

    assert result.raw_sha256 == _sha256(raw)
    assert result.raw_path == raw
    assert seen["cwd"] == install
    assert "--text-file" in seen["argv"]
    assert script.read_text(encoding="utf-8") not in seen["argv"]
    assert all(script.read_text(encoding="utf-8") not in value for value in seen["secrets"])


def test_synthesize_voice_uses_native_batch_concat_for_multiple_paragraphs(
    tmp_path: Path,
) -> None:
    episode, install, model, script, reference, raw = _paths(tmp_path)
    direct_cli = install / ".venv" / "Scripts" / "indextts2.exe"
    direct_cli.parent.mkdir(parents=True)
    direct_cli.write_bytes(b"launcher")
    script.write_text("第一段。\n\n第二段。\n\n第三段。", encoding="utf-8")
    seen: dict[str, object] = {}

    def runner(*, argv: list[str], cwd: Path, timeout: float, secrets: tuple[str, ...]) -> CommandResult:
        seen.update(argv=argv, cwd=cwd, timeout=timeout, secrets=secrets)
        batch_path = Path(argv[argv.index("--batch-file") + 1])
        seen["tasks"] = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines()]
        _write_wav(raw)
        return CommandResult(argv=["<REDACTED>"], returncode=0)

    result = synthesize_voice(
        episode_root=episode,
        install_dir=install,
        model_dir=model,
        script_path=script,
        approved_script_sha256=_sha256(script),
        reference_voice_path=reference,
        raw_output_path=raw,
        runner=runner,
    )

    assert result.raw_path == raw
    assert seen["argv"][:2] == [str(direct_cli), "batch"]
    assert "--concat" in seen["argv"]
    assert seen["tasks"] == [
        {"text": "第一段。", "silence_after_ms": 280},
        {"text": "第二段。", "silence_after_ms": 280},
        {"text": "第三段。", "silence_after_ms": 0},
    ]
    assert script.read_text(encoding="utf-8") not in seen["argv"]


def test_synthesize_voice_accepts_workspace_voice_outside_episode_without_modifying_it(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    episode = workspace / "books" / "book-001" / "episodes" / "E001"
    install = tmp_path / "tools" / "index-tts"
    model = install / "checkpoints"
    script = episode / "script" / "approved.txt"
    reference = workspace / "voices" / "reference_voice.wav"
    raw = episode / ".private" / "voice" / "raw-unique.wav"
    for directory in (model, script.parent, reference.parent, raw.parent):
        directory.mkdir(parents=True, exist_ok=True)
    script.write_text("这是已经批准的旁白。", encoding="utf-8")
    _write_wav(reference)
    reference_hash = _sha256(reference)

    def runner(**kwargs: object) -> CommandResult:
        _write_wav(raw)
        return CommandResult(argv=["<REDACTED>"], returncode=0)

    result = synthesize_voice(
        episode_root=episode, install_dir=install, model_dir=model,
        script_path=script, approved_script_sha256=_sha256(script),
        reference_voice_path=reference, expected_reference_sha256=reference_hash,
        raw_output_path=raw, runner=runner,
    )

    assert result.raw_path == raw
    assert _sha256(reference) == reference_hash


def test_synthesize_voice_rejects_redirected_external_reference_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bv.voice.indextts2 as module

    episode, install, model, script, _, raw = _paths(tmp_path)
    reference_root = tmp_path / "workspace" / "voices"
    reference_root.mkdir(parents=True)
    reference = reference_root / "reference_voice.wav"
    _write_wav(reference)
    reference_hash = _sha256(reference)
    called = False
    monkeypatch.setattr(module, "_is_redirected", lambda path: Path(path) == reference_root)

    def runner(**kwargs: object) -> CommandResult:
        nonlocal called
        called = True
        return CommandResult(argv=[], returncode=0)

    with pytest.raises(VoiceSynthesisError, match="unsafe_voice_path"):
        synthesize_voice(
            episode_root=episode, install_dir=install, model_dir=model,
            script_path=script, approved_script_sha256=_sha256(script),
            reference_voice_path=reference, raw_output_path=raw, runner=runner,
        )

    assert called is False
    assert not raw.exists()
    assert _sha256(reference) == reference_hash


@pytest.mark.parametrize(
    ("returncode", "write_output", "expected_code"),
    [(-9, False, "synthesis_timeout"), (3, False, "synthesis_failed"), (0, False, "missing_raw_output")],
)
def test_synthesize_voice_fails_closed_and_removes_only_new_raw_output(
    tmp_path: Path, returncode: int, write_output: bool, expected_code: str
) -> None:
    episode, install, model, script, reference, raw = _paths(tmp_path)

    def runner(**kwargs: object) -> CommandResult:
        if write_output:
            _write_wav(raw)
        return CommandResult(argv=[], returncode=returncode, stderr="private failure")

    with pytest.raises(VoiceSynthesisError, match=expected_code) as raised:
        synthesize_voice(
            episode_root=episode, install_dir=install, model_dir=model,
            script_path=script, approved_script_sha256=_sha256(script),
            reference_voice_path=reference, raw_output_path=raw, runner=runner,
        )

    assert str(raised.value) == expected_code
    assert not raw.exists()


def test_synthesize_voice_detects_script_hash_race_and_deletes_generated_raw(tmp_path: Path) -> None:
    episode, install, model, script, reference, raw = _paths(tmp_path)

    def runner(**kwargs: object) -> CommandResult:
        _write_wav(raw)
        script.write_text("changed after approval", encoding="utf-8")
        return CommandResult(argv=[], returncode=0)

    with pytest.raises(VoiceSynthesisError, match="input_hash_changed"):
        synthesize_voice(
            episode_root=episode, install_dir=install, model_dir=model,
            script_path=script, approved_script_sha256=_sha256(script),
            reference_voice_path=reference, raw_output_path=raw, runner=runner,
        )

    assert not raw.exists()


def test_synthesize_voice_rejects_redirected_output_ancestor_before_external_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bv.voice.indextts2 as module

    episode, install, model, script, reference, raw = _paths(tmp_path)
    monkeypatch.setattr(module, "_is_redirected", lambda path: Path(path) == raw.parent)

    with pytest.raises(VoiceSynthesisError, match="unsafe_voice_path"):
        synthesize_voice(
            episode_root=episode, install_dir=install, model_dir=model,
            script_path=script, approved_script_sha256=_sha256(script),
            reference_voice_path=reference, raw_output_path=raw,
        )

    assert not raw.exists()


def test_synthesize_voice_rejects_output_with_input_file_as_ancestor_before_runner(tmp_path: Path) -> None:
    episode, install, model, script, reference, _ = _paths(tmp_path)
    raw = script / "raw.wav"
    called = False

    def runner(**kwargs: object) -> CommandResult:
        nonlocal called
        called = True
        return CommandResult(argv=[], returncode=0)

    with pytest.raises(VoiceSynthesisError, match="unsafe_voice_path"):
        synthesize_voice(
            episode_root=episode, install_dir=install, model_dir=model,
            script_path=script, approved_script_sha256=_sha256(script),
            reference_voice_path=reference, raw_output_path=raw, runner=runner,
        )

    assert called is False
