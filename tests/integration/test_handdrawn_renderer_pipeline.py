import json
import shutil
import subprocess
from pathlib import Path

import pymupdf
import pytest

from bv.core.hashing import sha256_file
from bv.illustration.contracts import SilentRenderRequest
from bv.illustration.renderer import render_silent_story
from bv.video.probe import probe_media


def test_real_handdrawn_renderer_produces_15_second_native_vertical_track(
    tmp_path: Path,
) -> None:
    npm = shutil.which("npm")
    ffprobe = shutil.which("ffprobe")
    if npm is None or ffprobe is None:
        pytest.skip("local npm and ffprobe are required for the real render")

    repo_root = Path(__file__).resolve().parents[2]
    vendor = repo_root / "vendor" / "story_to_handdrawn_video"
    if not (vendor / "node_modules").is_dir():
        pytest.skip("run npm ci in the vendored renderer before this integration test")

    episode_root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    storyboard = episode_root / "media" / "illustration" / "storyboard.json"
    storyboard.parent.mkdir(parents=True)
    shutil.copy2(vendor / "storyboard.9x16.fixture.json", storyboard)

    episode_assets = episode_root / "assets"
    episode_assets.mkdir()
    for name in ("02_bw.svg", "02_color.svg", "03_bw.svg", "03_color.svg"):
        shutil.copy2(vendor / "public" / "assets" / name, episode_assets / name)

    request = SilentRenderRequest(
        episode_root=episode_root,
        storyboard_path=storyboard,
        storyboard_sha256=sha256_file(storyboard),
        output_path=episode_root / "media" / "render" / "picture_silent.mp4",
        vendor_dir=vendor,
    )

    result = render_silent_story(
        request,
        npm_command=npm,
        ffprobe_command=ffprobe,
    )
    facts = probe_media(result.output_path, ffprobe_command=ffprobe)

    assert result.width == 1080
    assert result.height == 1920
    assert result.frame_rate == 30.0
    assert result.audio_present is False
    assert result.output_sha256 == sha256_file(result.output_path)
    assert facts.video_codec == "h264"
    assert facts.pixel_format == "yuv420p"
    assert facts.audio_present is False
    assert abs(round(result.duration_ms * 30 / 1000) - 450) <= 1


def test_real_handdrawn_renderer_uses_portrait_art_as_a_full_bleed_background(
    tmp_path: Path,
) -> None:
    npm = shutil.which("npm")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if npm is None or ffmpeg is None or ffprobe is None:
        pytest.skip("local npm, ffmpeg, and ffprobe are required for the real render")

    repo_root = Path(__file__).resolve().parents[2]
    vendor = repo_root / "vendor" / "story_to_handdrawn_video"
    if not (vendor / "node_modules").is_dir():
        pytest.skip("run npm ci in the vendored renderer before this integration test")

    episode_root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    assets = episode_root / "assets"
    assets.mkdir(parents=True)
    portrait = assets / "portrait.png"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "color=c=red:s=1080x1920:d=1", "-frames:v", "1", "-y", str(portrait),
        ],
        check=True,
    )
    storyboard = episode_root / "media" / "illustration" / "storyboard.json"
    storyboard.parent.mkdir(parents=True)
    storyboard.write_text(
        json.dumps(
            {
                "project": {
                    "width": 1080,
                    "height": 1920,
                    "fps": 30,
                    "ratio": "9:16",
                    "total_frames": 30,
                    "transition": "cut",
                    "transition_frames": 0,
                },
                "safe_area": {
                    "top_reserved": 140,
                    "keyline_top": 140,
                    "keyline_bottom": 340,
                    "illustration_top": 340,
                    "illustration_bottom": 1460,
                    "subtitle_top": 1460,
                    "subtitle_bottom": 1660,
                    "bottom_reserved": 260,
                    "right_reserved": 180,
                },
                "scenes": [
                    {
                        "id": "S01",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "from_frame": 0,
                        "to_frame": 30,
                        "key_line": "全屏手绘测试",
                        "narration": "测试旁白。",
                        "assets": {
                            "bw": "assets/portrait.png",
                            "color": "assets/portrait.png",
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    request = SilentRenderRequest(
        episode_root=episode_root,
        storyboard_path=storyboard,
        storyboard_sha256=sha256_file(storyboard),
        output_path=episode_root / "media" / "render" / "picture_silent.mp4",
        vendor_dir=vendor,
    )

    result = render_silent_story(
        request,
        npm_command=npm,
        ffprobe_command=ffprobe,
    )
    frame = episode_root / "frame.png"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", "0.95", "-i",
            str(result.output_path), "-frames:v", "1", "-y", str(frame),
        ],
        check=True,
    )
    pixel = pymupdf.Pixmap(str(frame)).pixel(5, 960)

    assert pixel[0] > 180 and pixel[1] < 80 and pixel[2] < 80


def test_real_handdrawn_renderer_draws_the_top_title_in_white(
    tmp_path: Path,
) -> None:
    npm = shutil.which("npm")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if npm is None or ffmpeg is None or ffprobe is None:
        pytest.skip("local npm, ffmpeg, and ffprobe are required for the real render")

    repo_root = Path(__file__).resolve().parents[2]
    vendor = repo_root / "vendor" / "story_to_handdrawn_video"
    if not (vendor / "node_modules").is_dir():
        pytest.skip("run npm ci in the vendored renderer before this integration test")

    episode_root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    assets = episode_root / "assets"
    assets.mkdir(parents=True)
    portrait = assets / "portrait.png"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "color=c=red:s=1080x1920:d=1", "-frames:v", "1", "-y", str(portrait),
        ],
        check=True,
    )
    storyboard = episode_root / "media" / "illustration" / "storyboard.json"
    storyboard.parent.mkdir(parents=True)
    storyboard.write_text(
        json.dumps(
            {
                "project": {
                    "width": 1080,
                    "height": 1920,
                    "fps": 30,
                    "ratio": "9:16",
                    "total_frames": 30,
                    "transition": "cut",
                    "transition_frames": 0,
                },
                "safe_area": {
                    "top_reserved": 140,
                    "keyline_top": 140,
                    "keyline_bottom": 340,
                    "illustration_top": 340,
                    "illustration_bottom": 1460,
                    "subtitle_top": 1460,
                    "subtitle_bottom": 1660,
                    "bottom_reserved": 260,
                    "right_reserved": 180,
                },
                "scenes": [
                    {
                        "id": "S01",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "from_frame": 0,
                        "to_frame": 30,
                        "key_line": "白色标题测试",
                        "narration": "测试旁白。",
                        "assets": {
                            "bw": "assets/portrait.png",
                            "color": "assets/portrait.png",
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    request = SilentRenderRequest(
        episode_root=episode_root,
        storyboard_path=storyboard,
        storyboard_sha256=sha256_file(storyboard),
        output_path=episode_root / "media" / "render" / "picture_silent.mp4",
        vendor_dir=vendor,
    )

    result = render_silent_story(
        request,
        npm_command=npm,
        ffprobe_command=ffprobe,
    )
    frame = episode_root / "frame.png"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", "0.95", "-i",
            str(result.output_path), "-frames:v", "1", "-y", str(frame),
        ],
        check=True,
    )
    pixmap = pymupdf.Pixmap(str(frame))
    white_pixels = sum(
        1
        for y in range(140, 340, 2)
        for x in range(60, 900, 2)
        if all(channel > 220 for channel in pixmap.pixel(x, y)[:3])
    )

    assert white_pixels > 20


def test_storyboard_validator_accepts_cross_dissolve(tmp_path: Path) -> None:
    npm = shutil.which("npm")
    if npm is None:
        pytest.skip("local npm is required for storyboard validation")

    repo_root = Path(__file__).resolve().parents[2]
    vendor = repo_root / "vendor" / "story_to_handdrawn_video"
    payload = json.loads((vendor / "storyboard.9x16.fixture.json").read_text(encoding="utf-8"))
    payload["project"]["transition"] = "cross-dissolve"
    payload["project"]["transition_frames"] = 15
    storyboard = tmp_path / "cross-dissolve.json"
    storyboard.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    validation = subprocess.run(
        [npm, "run", "validate", "--", "--input", str(storyboard)],
        cwd=vendor,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert validation.returncode == 0, validation.stderr or validation.stdout


def test_real_handdrawn_renderer_cross_dissolves_without_paper_blank_or_timeline_loss(
    tmp_path: Path,
) -> None:
    npm = shutil.which("npm")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if npm is None or ffmpeg is None or ffprobe is None:
        pytest.skip("local npm, ffmpeg, and ffprobe are required for the real render")

    repo_root = Path(__file__).resolve().parents[2]
    vendor = repo_root / "vendor" / "story_to_handdrawn_video"
    if not (vendor / "node_modules").is_dir():
        pytest.skip("run npm ci in the vendored renderer before this integration test")

    episode_root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    assets = episode_root / "assets"
    assets.mkdir(parents=True)
    red = assets / "red.png"
    blue = assets / "blue.png"
    for color, output in (("red", red), ("blue", blue)):
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                f"color=c={color}:s=1080x1920:d=1", "-frames:v", "1", "-y", str(output),
            ],
            check=True,
        )
    storyboard = episode_root / "media" / "illustration" / "storyboard.json"
    storyboard.parent.mkdir(parents=True)
    storyboard.write_text(
        json.dumps(
            {
                "project": {
                    "width": 1080,
                    "height": 1920,
                    "fps": 30,
                    "ratio": "9:16",
                    "total_frames": 60,
                    "transition": "cross-dissolve",
                    "transition_frames": 15,
                },
                "safe_area": {
                    "top_reserved": 140,
                    "keyline_top": 140,
                    "keyline_bottom": 340,
                    "illustration_top": 340,
                    "illustration_bottom": 1460,
                    "subtitle_top": 1460,
                    "subtitle_bottom": 1660,
                    "bottom_reserved": 260,
                    "right_reserved": 180,
                },
                "scenes": [
                    {
                        "id": "S01", "start_ms": 0, "end_ms": 1000,
                        "from_frame": 0, "to_frame": 30,
                        "key_line": "第一幅插画画面", "narration": "第一段。",
                        "assets": {"bw": "assets/red.png", "color": "assets/red.png"},
                    },
                    {
                        "id": "S02", "start_ms": 1000, "end_ms": 2000,
                        "from_frame": 30, "to_frame": 60,
                        "key_line": "第二幅插画画面", "narration": "第二段。",
                        "assets": {"bw": "assets/blue.png", "color": "assets/blue.png"},
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = render_silent_story(
        SilentRenderRequest(
            episode_root=episode_root,
            storyboard_path=storyboard,
            storyboard_sha256=sha256_file(storyboard),
            output_path=episode_root / "media" / "render" / "picture_silent.mp4",
            vendor_dir=vendor,
        ),
        npm_command=npm,
        ffprobe_command=ffprobe,
    )
    frame = episode_root / "transition.png"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", "1.25", "-i",
            str(result.output_path), "-frames:v", "1", "-y", str(frame),
        ],
        check=True,
    )
    pixel = pymupdf.Pixmap(str(frame)).pixel(540, 960)

    assert abs(result.duration_ms - 2_000) <= 34
    assert pixel[0] < 230 or pixel[1] < 230 or pixel[2] < 230
