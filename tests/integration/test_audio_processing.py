from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from bv.core.process import CommandResult
from bv.voice.processing import (
    VoiceProcessingError,
    build_voice_master,
    classify_narration_duration,
    detect_qualifying_edge_silence,
    inspect_pcm_wav,
)


FFMPEG = Path(r"D:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe")
pytestmark = pytest.mark.skipif(not FFMPEG.is_file(), reason="FFmpeg executable is absent at configured path")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tone_wav(
    path: Path, *, sample_rate: int = 48_000, channels: int = 1,
    leading_s: float = 0.0, tone_s: float = 1.0, internal_s: float = 0.0,
    trailing_s: float = 0.0,
) -> None:
    samples: list[int] = []
    for seconds, amplitude in ((leading_s, 0), (tone_s / 2, 8_000), (internal_s, 0), (tone_s / 2, 8_000), (trailing_s, 0)):
        for index in range(round(seconds * sample_rate)):
            samples.append(round(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate)))
    frames = b"".join(struct.pack("<h", sample) * channels for sample in samples)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(frames)


def test_pcm_inspection_and_edge_silence_are_format_aware_and_never_trim_internal_pause(
    tmp_path: Path,
) -> None:
    non_master = tmp_path / "stereo-44k.wav"
    _write_tone_wav(non_master, sample_rate=44_100, channels=2, leading_s=0.2, tone_s=1, internal_s=0.5, trailing_s=0.4)

    inspected = inspect_pcm_wav(non_master)
    edge = detect_qualifying_edge_silence(non_master)

    assert (inspected.sample_rate, inspected.channels, inspected.sample_width) == (44_100, 2, 2)
    assert edge.start_trim_ms == 0
    assert 390 <= edge.end_trim_ms <= 410
    assert edge.all_silence is False


def test_all_silence_is_detected_without_claiming_it_is_safe_to_trim(tmp_path: Path) -> None:
    silent = tmp_path / "silent.wav"
    _write_tone_wav(silent, leading_s=1, tone_s=0)

    edge = detect_qualifying_edge_silence(silent)

    assert edge.all_silence is True
    assert (edge.start_trim_ms, edge.end_trim_ms) == (0, 0)


@pytest.mark.parametrize(
    ("seconds", "level", "band", "has_warning"),
    [(44.999, "fail", "too_short", False), (45.0, "pass", "edge_short", True),
     (48.0, "pass", "ideal", False), (56.0, "pass", "ideal", False),
     (56.001, "pass", "edge_long", True), (60.0, "pass", "edge_long", True),
     (60.001, "fail", "too_long", False)],
)
def test_duration_gate_uses_exact_boundaries(seconds: float, level: str, band: str, has_warning: bool) -> None:
    result = classify_narration_duration(seconds)

    assert (result.level, result.band) == (level, band)
    assert bool(result.warnings) is has_warning


def test_build_voice_master_uses_ffmpeg_two_pass_normalization_atomic_manifest_and_frame_duration(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    episode = workspace / "books" / "book-001" / "episodes" / "E001"
    raw = episode / "audio" / "raw.wav"
    reference = workspace / "voices" / "reference_voice.wav"
    script = episode / "approved.txt"
    master = episode / "audio" / "voice_master.wav"
    raw.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    _write_tone_wav(raw, leading_s=0.5, tone_s=1.5, internal_s=0.45, trailing_s=0.5)
    _write_tone_wav(reference, tone_s=0.1)
    script.write_text("private approved script", encoding="utf-8")
    raw_hash = _sha256(raw)

    manifest = build_voice_master(
        episode_root=episode, raw_path=raw, master_path=master, script_path=script,
        reference_voice_path=reference, approved_script_sha256=_sha256(script),
        voice_id="authorized-v1", ffmpeg_command=FFMPEG,
    )

    inspected = inspect_pcm_wav(master)
    assert (inspected.sample_rate, inspected.channels, inspected.sample_width) == (48_000, 1, 2)
    assert 1_900 <= manifest.master_duration_ms <= 2_000
    assert manifest.master_duration_ms == round(inspected.duration_seconds * 1000)
    assert _sha256(raw) == raw_hash
    assert manifest.raw_sha256 == raw_hash
    assert manifest.master_sha256 == _sha256(master)
    assert manifest.script_sha256 == _sha256(script)
    assert manifest.reference_sha256 == _sha256(reference)
    assert manifest.processing_sha256
    assert manifest.model_extra is None
    assert "private approved script" not in manifest.model_dump_json()
    assert master.with_suffix(".json").is_file()

    idempotent = build_voice_master(
        episode_root=episode, raw_path=raw, master_path=master, script_path=script,
        reference_voice_path=reference, approved_script_sha256=_sha256(script),
        voice_id="authorized-v1", ffmpeg_command=episode / "must-not-run-ffmpeg.exe",
    )
    assert idempotent == manifest


def test_build_voice_master_accepts_doubao_receipt_without_reference(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    raw = episode / "audio" / "raw.wav"
    script = episode / "script" / "approved.txt"
    receipt = episode / ".private" / "tts" / "doubao-formal-receipt.json"
    master = episode / "audio" / "voice_master.wav"
    raw.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    receipt.parent.mkdir(parents=True)
    _write_tone_wav(raw, tone_s=1.0)
    script.write_text("approved", encoding="utf-8")
    receipt.write_text(
        json.dumps(
            {
                "provider": "doubao",
                "request_sha256": "1" * 64,
                "task_id_sha256": "2" * 64,
                "response_sha256": "3" * 64,
                "status": "success",
                "error_code": None,
            }
        ),
        encoding="utf-8",
    )

    manifest = build_voice_master(
        episode_root=episode,
        raw_path=raw,
        master_path=master,
        script_path=script,
        approved_script_sha256=_sha256(script),
        provider="doubao",
        provider_voice_id="reader-a",
        provider_receipt_path=receipt,
        reference_voice_path=None,
        ffmpeg_command=FFMPEG,
    )

    assert manifest.provider == "doubao"
    assert manifest.provider_voice_id == "reader-a"
    assert manifest.voice_id == "reader-a"
    assert manifest.reference_sha256 is None
    assert manifest.provider_receipt_sha256 == _sha256(receipt)


def test_loudnorm_passes_use_identical_trim_sample_bounds(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    raw = episode / "audio" / "raw.wav"
    reference = episode / "reference.wav"
    script = episode / "approved.txt"
    master = episode / "audio" / "voice_master.wav"
    raw.parent.mkdir(parents=True)
    _write_tone_wav(raw, leading_s=0.5, tone_s=1.0, trailing_s=0.5)
    _write_tone_wav(reference, tone_s=0.1)
    script.write_text("approved", encoding="utf-8")
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: object) -> CommandResult:
        calls.append(argv)
        if argv[-2:] == ["null", "NUL"]:
            return CommandResult(
                argv=["ffmpeg"], returncode=0,
                stderr=(
                    '{"input_i":"-20.0","input_tp":"-8.0","input_lra":"0.0",'
                    '"input_thresh":"-30.0","target_offset":"0.0"}'
                ),
            )
        output = Path(argv[-1])
        source = raw if "-af" not in argv else next(
            Path(value) for index, value in enumerate(argv) if argv[index - 1] == "-i"
        )
        shutil.copyfile(source, output)
        return CommandResult(argv=["ffmpeg"], returncode=0)

    build_voice_master(
        episode_root=episode, raw_path=raw, master_path=master, script_path=script,
        reference_voice_path=reference, approved_script_sha256=_sha256(script),
        voice_id="authorized-v1", ffmpeg_command="ffmpeg", runner=runner,
    )

    filtered_calls = [argv for argv in calls if "-af" in argv]
    first_pass = filtered_calls[0][filtered_calls[0].index("-af") + 1]
    second_pass = filtered_calls[1][filtered_calls[1].index("-af") + 1]
    trim_prefix = "atrim=start_sample=24000:end_sample=72000,asetpts=N/SR/TB,"
    assert first_pass.startswith(trim_prefix)
    assert second_pass.startswith(trim_prefix)


def test_real_ffmpeg_long_edge_silence_is_trimmed_before_two_pass_loudnorm(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    episode = workspace / "books" / "book-001" / "episodes" / "E001"
    raw = episode / "audio" / "raw.wav"
    reference = workspace / "voices" / "reference_voice.wav"
    script = episode / "script" / "approved.txt"
    master = episode / "audio" / "voice_master.wav"
    raw.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    _write_tone_wav(raw, leading_s=3.0, tone_s=2.0, trailing_s=3.0)
    _write_tone_wav(reference, tone_s=0.1)
    script.write_text("approved", encoding="utf-8")
    reference_hash = _sha256(reference)

    manifest = build_voice_master(
        episode_root=episode, raw_path=raw, master_path=master, script_path=script,
        reference_voice_path=reference, expected_reference_sha256=reference_hash,
        approved_script_sha256=_sha256(script), voice_id="authorized-v1",
        ffmpeg_command=FFMPEG,
    )

    assert 2_950 <= manifest.qualifying_start_trim_ms <= 3_050
    assert 2_950 <= manifest.qualifying_end_trim_ms <= 3_050
    assert 1_950 <= manifest.master_duration_ms <= 2_050
    assert _sha256(reference) == reference_hash


def test_real_ffmpeg_applies_bounded_tempo_only_to_slightly_overlong_narration(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    raw = episode / "audio" / "raw.wav"
    reference = episode / "reference.wav"
    script = episode / "approved.txt"
    master = episode / "audio" / "voice_master.wav"
    raw.parent.mkdir(parents=True)
    _write_tone_wav(raw, tone_s=62.5)
    _write_tone_wav(reference, tone_s=0.1)
    script.write_text("approved", encoding="utf-8")

    manifest = build_voice_master(
        episode_root=episode,
        raw_path=raw,
        master_path=master,
        script_path=script,
        reference_voice_path=reference,
        approved_script_sha256=_sha256(script),
        voice_id="authorized-v1",
        ffmpeg_command=FFMPEG,
    )

    assert 1.04 < manifest.tempo_factor < 1.06
    assert 59_500 <= manifest.master_duration_ms <= 60_000
    assert manifest.duration.level == "pass"


def test_build_voice_master_rejects_redirected_external_reference_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bv.voice.processing as module

    workspace = tmp_path / "workspace"
    episode = workspace / "books" / "book-001" / "episodes" / "E001"
    raw = episode / "audio" / "raw.wav"
    script = episode / "script" / "approved.txt"
    reference_root = workspace / "voices"
    reference = reference_root / "reference_voice.wav"
    master = episode / "audio" / "voice_master.wav"
    for directory in (raw.parent, script.parent, reference_root):
        directory.mkdir(parents=True, exist_ok=True)
    _write_tone_wav(raw, tone_s=0.2)
    _write_tone_wav(reference, tone_s=0.1)
    script.write_text("approved", encoding="utf-8")
    reference_hash = _sha256(reference)
    monkeypatch.setattr(module, "_is_redirected", lambda path: Path(path) == reference_root)

    with pytest.raises(VoiceProcessingError, match="unsafe_voice_path"):
        build_voice_master(
            episode_root=episode, raw_path=raw, master_path=master, script_path=script,
            reference_voice_path=reference, approved_script_sha256=_sha256(script),
            voice_id="authorized-v1", ffmpeg_command=FFMPEG,
        )

    assert not master.exists()
    assert _sha256(reference) == reference_hash


def test_build_voice_master_does_not_publish_master_when_processing_fails(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    raw = episode / "audio" / "raw.wav"
    reference = episode / "reference.wav"
    script = episode / "approved.txt"
    master = episode / "audio" / "voice_master.wav"
    raw.parent.mkdir(parents=True)
    _write_tone_wav(raw, tone_s=0.4)
    _write_tone_wav(reference, tone_s=0.1)
    script.write_text("approved", encoding="utf-8")
    with pytest.raises(VoiceProcessingError, match="ffmpeg_decode_failed"):
        build_voice_master(
            episode_root=episode, raw_path=raw, master_path=master, script_path=script,
            reference_voice_path=reference, approved_script_sha256=_sha256(script),
            voice_id="authorized-v1", ffmpeg_command=episode / "missing-ffmpeg.exe",
        )

    assert not master.exists()


def test_existing_master_with_invalid_manifest_fails_before_processing_and_preserves_master(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    raw = episode / "audio" / "raw.wav"
    reference = episode / "reference.wav"
    script = episode / "approved.txt"
    master = episode / "audio" / "voice_master.wav"
    raw.parent.mkdir(parents=True)
    _write_tone_wav(raw, tone_s=0.4)
    _write_tone_wav(reference, tone_s=0.1)
    script.write_text("approved", encoding="utf-8")
    master.write_bytes(b"approved master to preserve")
    master.with_suffix(".json").mkdir()

    with pytest.raises(VoiceProcessingError, match="existing_master_integrity_invalid"):
        build_voice_master(
            episode_root=episode, raw_path=raw, master_path=master, script_path=script,
            reference_voice_path=reference, approved_script_sha256=_sha256(script),
            voice_id="authorized-v1", ffmpeg_command=FFMPEG,
        )

    assert master.read_bytes() == b"approved master to preserve"


def test_build_voice_master_rejects_real_windows_junction_before_external_target_write(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    external_target = tmp_path / "external-target"
    episode.mkdir()
    external_target.mkdir()
    junction = episode / "audio-junction"
    created = subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", str(junction), str(external_target)],
        capture_output=True, text=True, check=False,
    )
    if created.returncode != 0:
        pytest.skip("Windows Junction creation is unavailable in this environment")
    try:
        raw = episode / "raw.wav"
        reference = episode / "reference.wav"
        script = episode / "approved.txt"
        _write_tone_wav(raw, tone_s=0.2)
        _write_tone_wav(reference, tone_s=0.1)
        script.write_text("approved", encoding="utf-8")

        with pytest.raises(VoiceProcessingError, match="unsafe_voice_path"):
            build_voice_master(
                episode_root=episode, raw_path=raw, master_path=junction / "voice_master.wav",
                script_path=script, reference_voice_path=reference,
                approved_script_sha256=_sha256(script), voice_id="authorized-v1",
                ffmpeg_command=FFMPEG,
            )

        assert list(external_target.iterdir()) == []
    finally:
        junction.rmdir()
