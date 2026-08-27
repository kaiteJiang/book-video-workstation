from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bv.core.hashing import sha256_file
from bv.illustration.contracts import SafeArea
from bv.state.models import EpisodeState
from bv.state.store import StateStore
from bv.subtitles.generate import SubtitleCue, SubtitleManifest
from bv.video.cover import CoverManifest
from bv.video.render import CoverOverlay, HanddrawnRenderInputs, render_handdrawn_final
from bv.voice.processing import NarrationDuration, VoiceManifest
from bv.workflow.media_runtime import (
    LocalCoverGateway,
    LocalIllustrationGateway,
    MediaProductionService,
)
from bv.workflow.media_stages import VisualRenderStage
from bv.workflow.stages import HANDDRAWN_MEDIA_PREPARE_ORDER, StageContext
from tests.integration.test_handdrawn_media_workflow import _PreparedStage, _manifest


def _run(argv: list[str]) -> None:
    result = subprocess.run(
        argv, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    assert result.returncode == 0, result.stderr[-1200:]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None
    or shutil.which("ffprobe") is None
    or shutil.which("npm") is None,
    reason="local FFmpeg/FFprobe/npm unavailable",
)
def test_fake_providers_reach_real_remotion_and_ffmpeg_delivery(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    npm = shutil.which("npm")
    assert ffmpeg and ffprobe and npm
    workspace = tmp_path / "workspace"
    episode_root = workspace / "books" / "book-demo" / "episodes" / "E001"
    episode_root.mkdir(parents=True)
    storyboard, illustration_manifest = _manifest(episode_root)
    store = StateStore(workspace)
    store.save_episode(EpisodeState(
        book_id="book-demo", episode_id="E001", status="script_approved",
        script_hash="a" * 64,
    ))
    stages = {
        name: _PreparedStage(name, storyboard, illustration_manifest)
        for name in HANDDRAWN_MEDIA_PREPARE_ORDER
    }
    service = MediaProductionService(
        store=store,
        stages=stages,
        images=LocalIllustrationGateway(ffmpeg_command=ffmpeg),
        cover=LocalCoverGateway(),
    )
    generated = tmp_path / "codex-fixture.png"
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=#cf8b6c:s=1080x1920:d=1", "-frames:v", "1", "-y", str(generated),
    ])

    view = service.prepare("book-demo", "E001", mode="technical_sample")
    assert view.missing_scene_ids == ("S01", "S03", "S05")
    for scene_id in view.missing_scene_ids:
        service.import_image("book-demo", "E001", scene_id, generated)
    service.approve_representatives("book-demo", "E001")
    view = service.prepare_batch("book-demo", "E001")
    assert view.missing_scene_ids == ("S02", "S04")
    for scene_id in view.missing_scene_ids:
        view = service.import_image("book-demo", "E001", scene_id, generated)
    assert view.status == "illustrations_ready"
    assert service.import_cover("book-demo", "E001", generated).cover_ready is True

    context = StageContext(
        book_id="book-demo", episode_id="E001", episode_root=episode_root,
        episode_state=store.load_episode("book-demo", "E001"),
    )
    visual_outcome = VisualRenderStage(
        vendor_dir=Path("vendor/story_to_handdrawn_video").resolve(),
        npm_command=npm,
        ffprobe_command=ffprobe,
        safe_area=SafeArea(),
        transition="cross-dissolve",
    ).run(context)
    visual = visual_outcome.outputs["picture_silent"]
    render_storyboard = json.loads(
        visual_outcome.outputs["render_storyboard"].read_text(encoding="utf-8")
    )
    assert render_storyboard["project"]["transition"] == "cross-dissolve"
    assert render_storyboard["project"]["transition_frames"] == 15

    master = episode_root / "media" / "voice" / "voice_master.wav"
    master.parent.mkdir(parents=True, exist_ok=True)
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=15", "-c:a", "pcm_s16le",
        str(master),
    ])
    ass = episode_root / "media" / "subtitles" / "subtitles.ass"
    ass.parent.mkdir(parents=True, exist_ok=True)
    ass.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,Arial,56,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,80,180,280,1\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        "Dialogue: 0,0:00:00.00,0:00:15.00,Default,,0,0,0,,Fake provider acceptance\n",
        encoding="utf-8",
    )
    voice_sha = sha256_file(master)
    cue = SubtitleCue(index=0, start_ms=0, end_ms=15_000, text="Fake provider acceptance")
    cue_hash = hashlib.sha256(json.dumps(
        [{"index": 0, "start_ms": 0, "end_ms": 15_000, "text": cue.text}],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    voice_manifest = VoiceManifest(
        voice_id="黑金3", script_sha256="a" * 64, reference_sha256="b" * 64,
        raw_sha256="c" * 64, master_sha256=voice_sha, processing_sha256="d" * 64,
        sample_rate=48_000, channels=1, sample_width=2, raw_duration_ms=15_000,
        master_duration_ms=15_000, qualifying_start_trim_ms=0, qualifying_end_trim_ms=0,
        duration=NarrationDuration(seconds=15.0, level="pass", band="ideal"),
        created_at=datetime.now(UTC),
    )
    subtitle_manifest = SubtitleManifest(
        approved_sha256="a" * 64, audio_sha256=voice_sha, asr_sha256="e" * 64,
        cue_content_sha256=cue_hash, style_template_sha256="f" * 64,
        font_family="Arial", font_sha256="0" * 64, width=1080, height=1920,
        cue_count=1, duration_ms=15_000,
    )
    cover_root = episode_root / "cover"
    cover_manifest = CoverManifest.model_validate_json(
        (cover_root / "manifest.json").read_text(encoding="utf-8")
    )
    cover_path = next(path for path in cover_root.iterdir() if path.name != "manifest.json")
    overlay = "final cover"
    inputs = HanddrawnRenderInputs(
        book_id="book-demo", episode_id="E001", visual_path=visual,
        visual_sha256=sha256_file(visual), storyboard_sha256=illustration_manifest.storyboard_sha256,
        semantic_lock_sha256="b" * 64, script_sha256="a" * 64,
        audio_sha256=voice_sha, subtitle_sha256=cue_hash,
        voice_master_path=master, voice_master_sha256=voice_sha,
        voice_manifest=voice_manifest, master_duration_ms=15_000,
        ass_path=ass, ass_sha256=sha256_file(ass), asr_sha256="e" * 64,
        subtitle_manifest=subtitle_manifest, subtitle_cues=(cue,),
        cover=CoverOverlay(
            cover_path=cover_path, cover_sha256=cover_manifest.copied_sha256,
            manifest=cover_manifest, start_ms=12_000, end_ms=15_000,
            overlay_zone_text=overlay,
            overlay_zone_sha256=hashlib.sha256(overlay.encode()).hexdigest(),
            x=760, y=100, width=240, height=360,
        ),
        render_root=episode_root / ".private" / "final-render",
        final_path=episode_root / "media" / "final" / "final.mp4",
    )

    result = render_handdrawn_final(
        inputs, ffmpeg_command=ffmpeg, ffprobe_command=ffprobe, timeout_seconds=120,
    )

    assert result.final_path.is_file()
    assert result.manifest.output_sha256 == sha256_file(result.final_path)
