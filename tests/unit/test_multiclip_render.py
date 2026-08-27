from __future__ import annotations

import hashlib
import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from bv.video.render import (
    CoverOverlay,
    RenderError,
    RenderInputs,
    RenderManifest,
    NormalizedSegment,
    SegmentRenderInput,
    build_concat_manifest,
    build_render_argv,
    build_segment_normalize_argv,
    preflight_render,
    render_final,
)
from bv.video.qc import QCError, inspect_final
from bv.core.process import CommandResult
from bv.subtitles.generate import SubtitleCue, SubtitleManifest
from bv.video.cover import CoverManifest
from bv.voice.processing import NarrationDuration, VoiceManifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(root: Path, name: str, content: bytes = b"fixture") -> Path:
    path = root / name
    path.write_bytes(content)
    return path


def _inputs(tmp_path: Path, *, segments: int = 4) -> RenderInputs:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shared = "a" * 64
    source_paths = tuple(_file(tmp_path, f"S{index:02d}.mp4", f"{index}".encode()) for index in range(1, segments + 1))
    master = tmp_path / "voice_master.wav"
    with wave.open(str(master), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x00\x00" * 48_000 * 48)
    subtitles = _file(tmp_path, "final.ass", b"[Script Info]\n")
    cover = _file(tmp_path, "cover.png")
    master_sha = _sha(master)
    cues = (SubtitleCue(index=0, start_ms=0, end_ms=48_000, text="fixture"),)
    cue_hash = hashlib.sha256(json.dumps([{"index": 0, "start_ms": 0, "end_ms": 48_000, "text": "fixture"}], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return RenderInputs(
        book_id="book-01", episode_id="E001",
        segments=tuple(SegmentRenderInput(
            segment_id=f"S{index:02d}", media_path=path, media_sha256=_sha(path),
            prompt_sha256=(str(index) * 64)[:64], storyboard_sha256=shared,
            semantic_lock_sha256=shared, script_sha256=shared, audio_sha256=master_sha,
            subtitle_sha256=cue_hash, expected_duration_ms=12_000,
            source_duration_ms=12_000, warning_codes=(),
        ) for index, path in enumerate(source_paths, start=1)),
        voice_master_path=master, voice_master_sha256=master_sha,
        voice_manifest=VoiceManifest(voice_id="fixture", script_sha256=shared, reference_sha256="b" * 64, raw_sha256="c" * 64, master_sha256=master_sha, processing_sha256="d" * 64, sample_rate=48_000, channels=1, sample_width=2, raw_duration_ms=48_000, master_duration_ms=48_000, qualifying_start_trim_ms=0, qualifying_end_trim_ms=0, duration=NarrationDuration(seconds=48.0, level="pass", band="ideal"), created_at=datetime.now(UTC)),
        master_duration_ms=48_000, ass_path=subtitles, ass_sha256=_sha(subtitles), asr_sha256="e" * 64,
        subtitle_manifest=SubtitleManifest(approved_sha256=shared, audio_sha256=master_sha, asr_sha256="e" * 64, cue_content_sha256=cue_hash, style_template_sha256="f" * 64, font_family="fixture", font_sha256="0" * 64, width=1080, height=1920, cue_count=1, duration_ms=48_000), subtitle_cues=cues,
        cover=CoverOverlay(
            cover_path=cover, cover_sha256=_sha(cover), manifest=CoverManifest(book_id="book-01", episode_id="E001", source_sha256=_sha(cover), copied_sha256=_sha(cover), source_type="user_provided", matches_product_version=True, width=40, height=60, image_format="PNG"), start_ms=46_000, end_ms=48_000,
            overlay_zone_text="top right", overlay_zone_sha256=hashlib.sha256(b"top right").hexdigest(),
            x=760, y=80, width=240, height=360,
        ),
        render_root=tmp_path / "render", final_path=tmp_path / "output" / "final.mp4",
    )


@pytest.mark.parametrize("cover_value", [None, "missing"])
def test_legacy_multiclip_render_still_requires_product_cover(
    tmp_path: Path,
    cover_value: object,
) -> None:
    payload = _inputs(tmp_path).model_dump()
    if cover_value == "missing":
        payload.pop("cover")
    else:
        payload["cover"] = cover_value

    with pytest.raises(ValidationError):
        RenderInputs.model_validate(payload)


def test_preflight_rejects_reordered_or_stale_inputs_before_ffmpeg(tmp_path: Path) -> None:
    """A changed media hash or S02-before-S01 must prevent any render invocation."""
    inputs = _inputs(tmp_path).model_copy(update={"segments": tuple(reversed(_inputs(tmp_path).segments))})

    with pytest.raises(RenderError, match="invalid_segment_set"):
        preflight_render(inputs)

    stale = _inputs(tmp_path)
    stale.segments[0].media_path.write_bytes(b"changed")
    with pytest.raises(RenderError, match="input_hash_mismatch"):
        preflight_render(stale)


def test_normalization_argv_discards_h3_audio_and_only_pads_recorded_shortfall(tmp_path: Path) -> None:
    """Removing -an or padding an exact-length source would reintroduce prohibited H3 audio/frame invention."""
    inputs = _inputs(tmp_path)
    exact = build_segment_normalize_argv(inputs.segments[0], tmp_path / "normalized.mp4", ffmpeg_command="ffmpeg")
    assert "-an" in exact
    assert "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,setsar=1" in " ".join(exact)
    assert "tpad=" not in " ".join(exact)

    short = inputs.segments[0].model_copy(update={"source_duration_ms": 11_800, "warning_codes": ("duration_shortfall_repair_required",)})
    padded = build_segment_normalize_argv(short, tmp_path / "padded.mp4", ffmpeg_command="ffmpeg")
    assert "tpad=stop_mode=clone:stop_duration=0.200" in " ".join(padded)
    assert "trim=duration=12.000" in " ".join(padded)


def test_concat_manifest_is_ordered_utf8_and_refuses_control_or_redirect_paths(tmp_path: Path) -> None:
    """Changing the segment order or accepting a newline path must change/falsify the concat input."""
    first = _file(tmp_path, "第一段.mp4")
    second = _file(tmp_path, "S02.mp4")
    manifest = build_concat_manifest((first, second), tmp_path / "concat.txt")
    assert manifest.path.read_text(encoding="utf-8") == "file '" + str(first).replace("\\", "/") + "'\nfile '" + str(second).replace("\\", "/") + "'\n"
    assert manifest.sha256 == _sha(manifest.path)
    with pytest.raises(RenderError, match="unsafe_render_path"):
        build_concat_manifest((tmp_path / "bad\nname.mp4",), tmp_path / "bad.txt")


def test_final_argv_maps_master_burns_ass_and_limits_cover_to_approved_interval(tmp_path: Path) -> None:
    """Dropping the master map, ASS burn-in, cover enable window, or loudnorm would violate delivery policy."""
    inputs = _inputs(tmp_path)
    argv = build_render_argv(inputs, tmp_path / "concat.mp4", tmp_path / "final.tmp.mp4", ffmpeg_command="ffmpeg")
    text = " ".join(argv)
    assert "-map [v] -map 1:a:0" in text and "-an" not in text
    assert "subtitles=filename=" in text and "loudnorm=I=-16:TP=-1.5" in text
    assert "enable='between(t,46.000,48.000)'" in text
    assert "-c:v libx264 -pix_fmt yuv420p -r 30 -c:a aac -ac 2 -ar 48000 -movflags +faststart" in text


def test_preflight_refuses_existing_final_and_cover_interval_outside_final_beat(tmp_path: Path) -> None:
    """An overwrite or a cover shown before the final segment must fail before a process can start."""
    inputs = _inputs(tmp_path)
    inputs.final_path.parent.mkdir()
    inputs.final_path.write_bytes(b"accepted prior final")
    with pytest.raises(RenderError, match="output_already_exists"):
        preflight_render(inputs)

    other = tmp_path / "outside"
    other.mkdir()
    baseline = _inputs(other)
    invalid = baseline.model_copy(update={
        "cover": baseline.cover.model_copy(update={"start_ms": 35_000, "end_ms": 36_000}),
    })
    with pytest.raises(RenderError, match="invalid_cover_overlay"):
        preflight_render(invalid)


def test_qc_reports_technical_facts_and_explicitly_declines_aesthetic_claims(tmp_path: Path) -> None:
    """A wrong stream shape/duration must fail, while a valid technical probe cannot claim visual understanding."""
    inputs = _inputs(tmp_path)
    final = _file(tmp_path, "delivery.mp4", b"delivered bytes")
    payload = {
        "format": {"duration": "48.000"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "width": 1080, "height": 1920, "r_frame_rate": "30/1", "tags": {"rotate": "0"}},
            {"codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "48000"},
        ],
    }
    report = inspect_final(final, inputs=inputs, runner=lambda *args, **kwargs: CommandResult(argv=["ffprobe"], returncode=0, stdout=__import__("json").dumps(payload)))
    assert report.facts.duration_ms == 48_000
    assert report.loudness_status == "NOT_MEASURED"
    assert report.face_character_consistency == report.hand_artifact_check == report.aesthetic_check == "NOT_PERFORMED"

    payload["streams"].append({"codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "48000"})
    with pytest.raises(QCError, match="unexpected_stream_layout"):
        inspect_final(final, inputs=inputs, runner=lambda *args, **kwargs: CommandResult(argv=["ffprobe"], returncode=0, stdout=__import__("json").dumps(payload)))


def test_persisted_render_manifest_never_serializes_a_private_normalized_path(tmp_path: Path) -> None:
    """Replacing a render-work path must not leak it through the persisted delivery manifest."""
    private_output = tmp_path / "private" / "normalized" / "S01.mp4"
    manifest = RenderManifest(
        segment_ids=("S01",), normalized=(NormalizedSegment(segment_id="S01", input_sha256="a" * 64, output_name=private_output.name, output_sha256="b" * 64, duration_ms=12_000, prompt_sha256="c" * 64),),
        voice_master_sha256="d" * 64, ass_sha256="e" * 64, cover_sha256="f" * 64,
        output_sha256="0" * 64, master_duration_ms=48_000, storyboard_sha256="1" * 64,
        semantic_lock_sha256="2" * 64, script_sha256="3" * 64, audio_sha256="4" * 64,
        subtitle_sha256="5" * 64, asr_sha256="6" * 64,
    )
    persisted = manifest.model_dump_json()
    assert str(tmp_path) not in persisted and "output_path" not in persisted


def test_failure_cleanup_preserves_replaced_file_and_allows_a_clean_retry(tmp_path: Path) -> None:
    """Cleanup must use this invocation's recorded hash and remove empty work directories after a failed command."""
    import bv.video.render as module

    victim = _file(tmp_path, "temporary.mp4", b"ours")
    original_hash = _sha(victim)
    victim.write_bytes(b"attacker replacement")
    module._remove_if_hash(victim, original_hash)
    assert victim.read_bytes() == b"attacker replacement"

    inputs = _inputs(tmp_path / "retry")
    failed = lambda argv, *, cwd, timeout: CommandResult(argv=list(argv), returncode=1)
    with pytest.raises(RenderError, match="segment_normalize_failed"):
        module.render_final(inputs, runner=failed)
    assert not inputs.render_root.exists()
    with pytest.raises(RenderError, match="segment_normalize_failed"):
        module.render_final(inputs, runner=failed)


def test_preflight_binds_cover_to_current_book_episode_and_passing_voice(tmp_path: Path) -> None:
    """A valid file hash must not make another episode's cover or failed narration acceptable."""
    inputs = _inputs(tmp_path)
    foreign_cover = inputs.cover.model_copy(update={
        "manifest": inputs.cover.manifest.model_copy(update={"episode_id": "E002"}),
    })
    with pytest.raises(RenderError, match="cover_overlay_identity_invalid"):
        preflight_render(inputs.model_copy(update={"cover": foreign_cover}))

    failed_duration = inputs.voice_manifest.duration.model_copy(update={"level": "fail", "band": "too_short"})
    failed_voice = inputs.voice_manifest.model_copy(update={"duration": failed_duration})
    with pytest.raises(RenderError, match="voice_contract_invalid"):
        preflight_render(inputs.model_copy(update={"voice_manifest": failed_voice}))


def test_malformed_normalized_output_is_removed_so_retry_is_not_poisoned(tmp_path: Path) -> None:
    """A zero/garbage FFmpeg artifact belongs to this attempt and must not block the next attempt."""
    inputs = _inputs(tmp_path)

    def malformed(argv, *, cwd, timeout):
        target = Path(argv[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not an mp4")
        return CommandResult(argv=list(argv), returncode=0)

    with pytest.raises(RenderError, match="normalized_media_invalid"):
        render_final(inputs, runner=malformed)
    assert not inputs.render_root.exists()
    with pytest.raises(RenderError, match="normalized_media_invalid"):
        render_final(inputs, runner=malformed)


def test_render_refuses_temporary_output_changed_after_qc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Publication must rehash the exact temporary bytes after the inspector returns."""
    import bv.video.render as module
    from bv.video.probe import MediaProbeFacts

    inputs = _inputs(tmp_path)

    def successful_writer(argv, *, cwd, timeout):
        target = Path(argv[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((target.name + "-created").encode())
        return CommandResult(argv=list(argv), returncode=0)

    def probe(path, **kwargs):
        duration = 48_000 if Path(path).name == "combined.mp4" else 12_000
        return MediaProbeFacts(
            duration_ms=duration, width=1080, height=1920, frame_rate=30.0,
            video_codec="h264", audio_present=False,
        )

    def swapping_inspector(path, **kwargs):
        Path(path).write_bytes(b"replacement after qc")

    monkeypatch.setattr(module, "probe_media", probe)
    with pytest.raises(RenderError, match="final_output_changed"):
        render_final(inputs, runner=successful_writer, inspector=swapping_inspector)
    assert not inputs.final_path.exists()


def test_qc_runner_is_invoked_once_and_source_failures_are_sanitized(tmp_path: Path) -> None:
    """An internal TypeError must not rerun FFprobe, and missing sources must not leak path errors."""
    inputs = _inputs(tmp_path)
    final = _file(tmp_path, "delivery-once.mp4", b"delivered bytes")
    calls = 0

    def broken_runner(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise TypeError("internal runner bug")

    with pytest.raises(QCError, match="ffprobe_execution_failed"):
        inspect_final(final, inputs=inputs, runner=broken_runner)
    assert calls == 1

    payload = {
        "format": {"duration": "48.000"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "width": 1080, "height": 1920, "r_frame_rate": "30/1"},
            {"codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "48000"},
        ],
    }
    inputs.segments[0].media_path.unlink()
    with pytest.raises(QCError, match="source_hash_mismatch"):
        inspect_final(
            final, inputs=inputs,
            runner=lambda *args, **kwargs: CommandResult(argv=["ffprobe"], returncode=0, stdout=json.dumps(payload)),
        )


def test_qc_rejects_fractional_rotation_instead_of_rounding_it_to_zero(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    final = _file(tmp_path, "fractional-rotation.mp4", b"delivered bytes")
    payload = {
        "format": {"duration": "48.000"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "width": 1080, "height": 1920, "r_frame_rate": "30/1", "tags": {"rotate": "0.4"}},
            {"codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "48000"},
        ],
    }
    with pytest.raises(QCError, match="invalid_ffprobe_output"):
        inspect_final(
            final, inputs=inputs,
            runner=lambda *args, **kwargs: CommandResult(argv=["ffprobe"], returncode=0, stdout=json.dumps(payload)),
        )
