from __future__ import annotations

import hashlib
import json
import wave
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from bv.core.process import CommandResult
from bv.production.profile import ProductionProfile
from bv.subtitles.generate import SubtitleCue, SubtitleManifest
from bv.video.cover import CoverManifest
from bv.video.probe import MediaProbeFacts
from bv.video.qc import QCError, inspect_final
from bv.video.render import (
    CoverOverlay,
    HanddrawnRenderInputs,
    RenderError,
    build_handdrawn_final_argv,
    preflight_handdrawn_render,
)
from bv.voice.processing import NarrationDuration, VoiceManifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(
    tmp_path: Path,
    *,
    duration_ms: int = 15_000,
    include_cover: bool = True,
) -> HanddrawnRenderInputs:
    tmp_path.mkdir(parents=True, exist_ok=True)
    visual = tmp_path / "visual.mp4"
    visual.write_bytes(b"silent handdrawn visual")
    master = tmp_path / "voice_master.wav"
    with wave.open(str(master), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x00\x00" * 48 * duration_ms)
    ass = tmp_path / "subtitles.ass"
    ass.write_text("[Script Info]\n", encoding="utf-8")
    cover = tmp_path / "cover.png"
    cover.write_bytes(b"cover")
    shared = "a" * 64
    voice_sha = _sha(master)
    cues = (SubtitleCue(index=0, start_ms=0, end_ms=duration_ms, text="fixture"),)
    cue_payload = [{"index": 0, "start_ms": 0, "end_ms": duration_ms, "text": "fixture"}]
    cue_hash = hashlib.sha256(
        json.dumps(cue_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return HanddrawnRenderInputs(
        book_id="book-01", episode_id="E001",
        visual_path=visual, visual_sha256=_sha(visual),
        storyboard_sha256=shared, semantic_lock_sha256=shared,
        script_sha256=shared, audio_sha256=voice_sha, subtitle_sha256=cue_hash,
        voice_master_path=master, voice_master_sha256=voice_sha,
        voice_manifest=VoiceManifest(
            voice_id="黑金3", script_sha256=shared, reference_sha256="b" * 64,
            raw_sha256="c" * 64, master_sha256=voice_sha, processing_sha256="d" * 64,
            sample_rate=48_000, channels=1, sample_width=2,
            raw_duration_ms=duration_ms, master_duration_ms=duration_ms,
            qualifying_start_trim_ms=0, qualifying_end_trim_ms=0,
            duration=NarrationDuration(seconds=duration_ms / 1000, level="pass", band="ideal"),
            created_at=datetime.now(UTC),
        ),
        master_duration_ms=duration_ms,
        ass_path=ass, ass_sha256=_sha(ass), asr_sha256="e" * 64,
        subtitle_manifest=SubtitleManifest(
            approved_sha256=shared, audio_sha256=voice_sha, asr_sha256="e" * 64,
            cue_content_sha256=cue_hash, style_template_sha256="f" * 64,
            font_family="fixture", font_sha256="0" * 64, width=1080, height=1920,
            cue_count=1, duration_ms=duration_ms,
        ),
        subtitle_cues=cues,
        cover=CoverOverlay(
            cover_path=cover, cover_sha256=_sha(cover),
            manifest=CoverManifest(
                book_id="book-01", episode_id="E001", source_sha256=_sha(cover),
                copied_sha256=_sha(cover), source_type="user_provided",
                matches_product_version=True, width=40, height=60, image_format="PNG",
            ),
            start_ms=duration_ms - 2_000, end_ms=duration_ms,
            overlay_zone_text="top right",
            overlay_zone_sha256=hashlib.sha256(b"top right").hexdigest(),
            x=760, y=80, width=240, height=360,
        ) if include_cover else None,
        render_root=tmp_path / "render", final_path=tmp_path / "output" / "final.mp4",
    )


def _visual_facts(duration_ms: int) -> MediaProbeFacts:
    return MediaProbeFacts(
        duration_ms=duration_ms, width=1080, height=1920, frame_rate=30.0,
        video_codec="h264", audio_present=False,
    )


def test_handdrawn_final_argv_has_one_visual_voice_cover_and_no_concat(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    argv = build_handdrawn_final_argv(
        inputs, tmp_path / "final.tmp.mp4", ffmpeg_command="ffmpeg"
    )
    text = " ".join(argv)

    assert argv.count("-i") == 3
    assert "concat" not in text and "normalized" not in text
    assert "-map [v] -map 1:a:0" in text
    assert "subtitles=filename=" in text
    assert "-c:v libx264 -pix_fmt yuv420p -r 30 -c:a aac" in text


def test_handdrawn_final_argv_without_product_cover_uses_visual_voice_and_subtitles(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path, include_cover=False)

    preflight_handdrawn_render(
        inputs,
        visual_inspector=lambda *_args, **_kwargs: _visual_facts(15_000),
    )
    argv = build_handdrawn_final_argv(
        inputs, tmp_path / "final.tmp.mp4", ffmpeg_command="ffmpeg"
    )
    text = " ".join(argv)

    assert argv.count("-i") == 2
    assert "-loop" not in argv
    assert "overlay=" not in text
    assert "subtitles=filename=" in text
    assert "-map [v] -map 1:a:0" in text


def test_handdrawn_final_argv_uses_ascii_relative_subtitle_filter_path(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "含中文目录")

    argv = build_handdrawn_final_argv(
        inputs,
        inputs.final_path.parent / "final.tmp.mp4",
        ffmpeg_command="ffmpeg",
    )
    filtergraph = argv[argv.index("-filter_complex") + 1]

    assert "subtitles=filename=../subtitles.ass" in filtergraph
    assert "含中文目录" not in filtergraph


def test_handdrawn_preflight_accepts_one_frame_drift_and_rejects_more(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    accepted = preflight_handdrawn_render(
        inputs, visual_inspector=lambda *_args, **_kwargs: _visual_facts(15_033)
    )
    assert accepted.visual_duration_ms == 15_033

    with pytest.raises(RenderError, match="visual_duration_mismatch"):
        preflight_handdrawn_render(
            inputs, visual_inspector=lambda *_args, **_kwargs: _visual_facts(15_034)
        )


def test_handdrawn_preflight_rejects_audio_or_wrong_geometry(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(RenderError, match="visual_media_invalid"):
        preflight_handdrawn_render(
            inputs,
            visual_inspector=lambda *_args, **_kwargs: _visual_facts(15_000).model_copy(
                update={"audio_present": True}
            ),
        )


def test_handdrawn_final_qc_rejects_more_than_one_frame_duration_drift(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    final = tmp_path / "delivery.mp4"
    final.write_bytes(b"delivery")

    def runner_for(duration: str):
        payload = {
            "format": {"duration": duration},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p",
                 "width": 1080, "height": 1920, "r_frame_rate": "30/1"},
                {"codec_type": "audio", "codec_name": "aac", "channels": 2,
                 "sample_rate": "48000"},
            ],
        }
        return lambda *_args, **_kwargs: CommandResult(
            argv=["ffprobe"], returncode=0, stdout=json.dumps(payload)
        )

    report = inspect_final(final, inputs=inputs, runner=runner_for("15.033"))
    assert report.aesthetic_review_required is True
    with pytest.raises(QCError, match="duration_mismatch"):
        inspect_final(final, inputs=inputs, runner=runner_for("15.034"))


def test_handdrawn_final_qc_applies_episode_duration_profile(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    final = tmp_path / "delivery.mp4"
    final.write_bytes(b"delivery")

    def runner_for(duration: str):
        payload = {
            "format": {"duration": duration},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "width": 1080,
                    "height": 1920,
                    "r_frame_rate": "30/1",
                    "duration": duration,
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "sample_rate": "48000",
                    "duration": duration,
                },
            ],
        }
        return lambda *_args, **_kwargs: CommandResult(
            argv=["ffprobe"], returncode=0, stdout=json.dumps(payload)
        )

    profile = ProductionProfile.living_default()
    accepted = inputs.model_copy(update={"master_duration_ms": 135_000})
    report = inspect_final(
        final,
        inputs=accepted,
        production_profile=profile,
        runner=runner_for("135.000"),
    )
    assert report.facts.duration_ms == 135_000

    too_short = inputs.model_copy(update={"master_duration_ms": 119_900})
    with pytest.raises(QCError, match="duration_profile_out_of_range"):
        inspect_final(
            final,
            inputs=too_short,
            production_profile=profile,
            runner=runner_for("119.900"),
        )


@pytest.mark.parametrize(
    ("include_cover", "manifest_cover_sha256"),
    [(False, "f" * 64), (True, None), (True, "e" * 64)],
)
def test_handdrawn_final_qc_rejects_cover_state_or_hash_mismatch(
    tmp_path: Path,
    include_cover: bool,
    manifest_cover_sha256: str | None,
) -> None:
    inputs = _inputs(tmp_path, include_cover=include_cover)
    final = tmp_path / "delivery.mp4"
    final.write_bytes(b"delivery")
    payload = {
        "format": {"duration": "15.000"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p",
             "width": 1080, "height": 1920, "r_frame_rate": "30/1"},
            {"codec_type": "audio", "codec_name": "aac", "channels": 2,
             "sample_rate": "48000"},
        ],
    }
    manifest = SimpleNamespace(
        output_sha256=_sha(final),
        cover_sha256=manifest_cover_sha256,
    )

    with pytest.raises(QCError, match="cover_identity_mismatch"):
        inspect_final(
            final,
            inputs=inputs,
            render_manifest=manifest,
            runner=lambda *_args, **_kwargs: CommandResult(
                argv=["ffprobe"], returncode=0, stdout=json.dumps(payload)
            ),
        )
