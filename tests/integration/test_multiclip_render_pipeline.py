from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bv.video.render import CoverOverlay, RenderInputs, SegmentRenderInput, render_final
from bv.core.process import run_command
from bv.subtitles.generate import SubtitleCue, SubtitleManifest
from bv.video.cover import CoverManifest
from bv.voice.processing import NarrationDuration, VoiceManifest


def _run(argv: list[str]) -> None:
    result = subprocess.run(argv, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
    assert result.returncode == 0, result.stderr[-1000:]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="local FFmpeg/FFprobe unavailable")
def test_real_ffmpeg_multiclip_pipeline_renders_and_probes_synthetic_assets(tmp_path: Path) -> None:
    """The break caught here is an argv/output change that stops producing a compliant final delivery MP4."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg is not None and ffprobe is not None
    assets = tmp_path / "中文, O'Brien"
    assets.mkdir()
    colours = ("red", "green", "blue", "yellow")
    clips: list[Path] = []
    for index, colour in enumerate(colours, start=1):
        clip = assets / f"S{index:02d}.mp4"
        _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={colour}:s=108x192:r=30:d=12", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)])
        clips.append(clip)
    master = assets / "voice_master.wav"
    _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=48", "-c:a", "pcm_s16le", str(master)])
    cover = assets / "cover.png"
    _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=white:s=40x60", "-frames:v", "1", str(cover)])
    ass = assets / "final.ass"
    ass.write_text("[Script Info]\nScriptType: v4.00+\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\nStyle: Default,Arial,32,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,2,20,20,40,1\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\nDialogue: 0,0:00:00.00,0:00:48.00,Default,,0,0,0,,Synthetic local fixture\n", encoding="utf-8")
    identity = "a" * 64
    master_sha = _sha(master)
    cues = (SubtitleCue(index=0, start_ms=0, end_ms=48_000, text="Synthetic local fixture"),)
    cue_hash = hashlib.sha256(json.dumps([{"index": 0, "start_ms": 0, "end_ms": 48_000, "text": "Synthetic local fixture"}], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    inputs = RenderInputs(
        book_id="book-01", episode_id="E001",
        segments=tuple(SegmentRenderInput(segment_id=f"S{index:02d}", media_path=clip, media_sha256=_sha(clip), prompt_sha256=(str(index) * 64)[:64], storyboard_sha256=identity, semantic_lock_sha256=identity, script_sha256=identity, audio_sha256=master_sha, subtitle_sha256=cue_hash, expected_duration_ms=12_000, source_duration_ms=12_000) for index, clip in enumerate(clips, start=1)),
        voice_master_path=master, voice_master_sha256=master_sha, voice_manifest=VoiceManifest(voice_id="fixture", script_sha256=identity, reference_sha256="b" * 64, raw_sha256="c" * 64, master_sha256=master_sha, processing_sha256="d" * 64, sample_rate=48_000, channels=1, sample_width=2, raw_duration_ms=48_000, master_duration_ms=48_000, qualifying_start_trim_ms=0, qualifying_end_trim_ms=0, duration=NarrationDuration(seconds=48.0, level="pass", band="ideal"), created_at=datetime.now(UTC)), master_duration_ms=48_000,
        ass_path=ass, ass_sha256=_sha(ass), asr_sha256="e" * 64, subtitle_manifest=SubtitleManifest(approved_sha256=identity, audio_sha256=master_sha, asr_sha256="e" * 64, cue_content_sha256=cue_hash, style_template_sha256="f" * 64, font_family="fixture", font_sha256="0" * 64, width=1080, height=1920, cue_count=1, duration_ms=48_000), subtitle_cues=cues,
        cover=CoverOverlay(cover_path=cover, cover_sha256=_sha(cover), manifest=CoverManifest(book_id="book-01", episode_id="E001", source_sha256=_sha(cover), copied_sha256=_sha(cover), source_type="user_provided", matches_product_version=True, width=40, height=60, image_format="PNG"), start_ms=46_000, end_ms=48_000, overlay_zone_text="top right", overlay_zone_sha256=hashlib.sha256(b"top right").hexdigest(), x=760, y=80, width=240, height=360),
        render_root=assets / "render", final_path=assets / "delivery" / "final.mp4",
    )

    diagnostics = []

    def runner(argv, *, cwd, timeout):
        result = run_command(argv, cwd=cwd, timeout=timeout)
        diagnostics.append(result)
        return result

    try:
        result = render_final(
            inputs, ffmpeg_command=ffmpeg, ffprobe_command=ffprobe,
            timeout_seconds=120, runner=runner,
        )
    except Exception as error:
        pytest.fail(f"{error}: {diagnostics[-1].stderr[-1000:]}")

    assert result.final_path.is_file()
    assert result.manifest.output_sha256 == _sha(result.final_path)
