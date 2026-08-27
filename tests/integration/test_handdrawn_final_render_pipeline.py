from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bv.subtitles.generate import SubtitleCue, SubtitleManifest
from bv.video.render import HanddrawnRenderInputs, render_handdrawn_final
from bv.voice.processing import NarrationDuration, VoiceManifest


def _run(argv: list[str]) -> None:
    result = subprocess.run(
        argv, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    assert result.returncode == 0, result.stderr[-1000:]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="local FFmpeg/FFprobe unavailable",
)
def test_real_ffmpeg_composes_handdrawn_visual_voice_and_subtitles_without_product_cover(
    tmp_path: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg is not None and ffprobe is not None
    assets = tmp_path / "含中文目录" / "assets"
    assets.mkdir(parents=True)
    visual = assets / "silent-visual.mp4"
    master = assets / "voice_master.wav"
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=#d6c2a6:s=1080x1920:r=30:d=10", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(visual),
    ])
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=10", "-c:a", "pcm_s16le",
        str(master),
    ])
    ass = assets / "subtitles.ass"
    ass.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,Arial,56,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,80,180,280,1\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        "Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,Local handdrawn fixture\n",
        encoding="utf-8",
    )
    identity = "a" * 64
    voice_sha = _sha(master)
    cues = (SubtitleCue(index=0, start_ms=0, end_ms=10_000, text="Local handdrawn fixture"),)
    cue_payload = [{"index": 0, "start_ms": 0, "end_ms": 10_000, "text": "Local handdrawn fixture"}]
    cue_hash = hashlib.sha256(
        json.dumps(cue_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    inputs = HanddrawnRenderInputs(
        book_id="book-01", episode_id="E001", visual_path=visual,
        visual_sha256=_sha(visual), storyboard_sha256=identity,
        semantic_lock_sha256=identity, script_sha256=identity,
        audio_sha256=voice_sha, subtitle_sha256=cue_hash,
        voice_master_path=master, voice_master_sha256=voice_sha,
        voice_manifest=VoiceManifest(
            voice_id="黑金3", script_sha256=identity, reference_sha256="b" * 64,
            raw_sha256="c" * 64, master_sha256=voice_sha, processing_sha256="d" * 64,
            sample_rate=48_000, channels=1, sample_width=2,
            raw_duration_ms=10_000, master_duration_ms=10_000,
            qualifying_start_trim_ms=0, qualifying_end_trim_ms=0,
            duration=NarrationDuration(seconds=10.0, level="pass", band="edge_short"),
            created_at=datetime.now(UTC),
        ),
        master_duration_ms=10_000, ass_path=ass, ass_sha256=_sha(ass),
        asr_sha256="e" * 64,
        subtitle_manifest=SubtitleManifest(
            approved_sha256=identity, audio_sha256=voice_sha, asr_sha256="e" * 64,
            cue_content_sha256=cue_hash, style_template_sha256="f" * 64,
            font_family="Arial", font_sha256="0" * 64, width=1080, height=1920,
            cue_count=1, duration_ms=10_000,
        ),
        subtitle_cues=cues,
        cover=None,
        render_root=assets / "render", final_path=assets / "delivery" / "final.mp4",
    )

    result = render_handdrawn_final(
        inputs, ffmpeg_command=ffmpeg, ffprobe_command=ffprobe, timeout_seconds=120,
    )

    assert result.final_path.is_file()
    assert result.manifest.output_sha256 == _sha(result.final_path)
    assert result.manifest.cover_sha256 is None

