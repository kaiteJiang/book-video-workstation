from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from bv.illustration.assets import import_image_master, prepare_image_jobs
from bv.illustration.contracts import CharacterLock, IllustrationScene, IllustrationStoryboard, StyleDecision
from bv.illustration.style_selector import load_style_catalog


def _canonical(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_real_ffmpeg_imports_color_master_and_derives_grayscale(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("local FFmpeg and FFprobe are required")
    style = next(
        item for item in load_style_catalog(
            Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json")
        )
        if item.style_id == "emotional-watercolor-sketch"
    )
    decision = StyleDecision(
        library_version="1",
        selected_style=style.style_id,
        candidate_styles=(style.style_id, "colored-pencil-diary", "warm-flat-storybook"),
        selection_reasons=("测试",),
        rejected_reasons={"colored-pencil-diary": "测试", "warm-flat-storybook": "测试"},
        confidence=1.0,
        manual_override=False,
        source_script_sha256="a" * 64,
        style_fingerprint="b" * 64,
    )
    character = CharacterLock(
        role="普通读者", age_range="30至39岁", face="椭圆脸", hair="黑色短发",
        body="中等身材", base_clothing="深蓝衬衫", allowed_variations=(),
        color_markers=("深蓝",), personal_objects=("笔记本",), forbidden_changes=("发型",),
        source_script_sha256="a" * 64, style_fingerprint="b" * 64,
    )
    scene = IllustrationScene(
        scene_id="S01", start_ms=0, end_ms=15_000, from_frame=0, to_frame=450,
        narration="测试批准旁白", narration_span=(0, 6), key_line="把生活还给自己",
        visual_purpose="真实生活", setting="餐桌", character_action="停下解释",
        metaphor=None, composition="居中偏下", character_refs=("reader-01",),
        image_prompt="普通读者在餐桌前", negative_constraints=("禁止文字",),
        representative_frame=True, asset_status="planned",
    )
    storyboard = IllustrationStoryboard(
        book_id="book-demo", episode_id="E001", width=1080, height=1920, fps=30,
        master_duration_ms=15_000, total_frames=450, script_sha256="a" * 64,
        audio_sha256="c" * 64, subtitle_sha256="d" * 64,
        style_decision_sha256=_canonical(decision.model_dump(mode="json")),
        character_lock_sha256=_canonical(character.model_dump(mode="json")), scenes=(scene,),
    )
    manifest = prepare_image_jobs(storyboard, character, style, decision, tmp_path / "episode")
    source = tmp_path / "codex-generated.png"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "color=c=#cf8b6c:s=1080x1920:d=1", "-frames:v", "1", "-y", str(source),
        ],
        check=True,
    )

    imported = import_image_master(
        manifest.jobs[0], source, ffmpeg_command=ffmpeg,
    )
    probe = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,pix_fmt", "-of", "json", str(imported.bw_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    facts = json.loads(probe.stdout)["streams"][0]

    assert imported.width == 1080 and imported.height == 1920
    assert imported.master_path.is_file() and imported.bw_path.is_file()
    assert imported.master_sha256 != imported.bw_sha256
    assert facts["width"] == 1080 and facts["height"] == 1920
    assert facts["pix_fmt"] in {"gray", "gray16be", "gray16le"}
