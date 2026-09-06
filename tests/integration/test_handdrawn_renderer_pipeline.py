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


def _asset_sha256s(episode_root: Path, storyboard: Path) -> dict[str, str]:
    payload = json.loads(storyboard.read_text(encoding="utf-8"))
    return {
        path: sha256_file(episode_root / path)
        for scene in payload["scenes"]
        for path in scene["assets"].values()
    }


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
        asset_sha256s=_asset_sha256s(episode_root, storyboard),
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
        asset_sha256s=_asset_sha256s(episode_root, storyboard),
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

    assert pixel[0] < 80 and pixel[1] < 80 and pixel[2] < 80


def test_real_handdrawn_renderer_keeps_title_author_above_unobscured_illustration(
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
                    "total_frames": 90,
                    "transition": "cut",
                    "transition_frames": 0,
                    "show_key_line": False,
                },
                "title_overlay": {"title": "天幕红尘", "author": "豆豆"},
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
                        "end_ms": 3000,
                        "from_frame": 0,
                        "to_frame": 90,
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
        asset_sha256s=_asset_sha256s(episode_root, storyboard),
    )

    result = render_silent_story(
        request,
        npm_command=npm,
        ffprobe_command=ffprobe,
    )
    frame = episode_root / "frame.png"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", "2.50", "-i",
            str(result.output_path), "-frames:v", "1", "-y", str(frame),
        ],
        check=True,
    )
    pixmap = pymupdf.Pixmap(str(frame))
    title_pixels = sum(
        1
        for y in range(140, 340, 2)
        for x in range(60, 900, 2)
        if (
            pixmap.pixel(x, y)[0] < 100
            and pixmap.pixel(x, y)[1] < 120
            and pixmap.pixel(x, y)[2] < 150
        )
    )

    assert title_pixels > 20
    illustration_pixel = pixmap.pixel(540, 960)[:3]
    assert illustration_pixel[0] > 150 and illustration_pixel[1] < 90


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
            asset_sha256s=_asset_sha256s(episode_root, storyboard),
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


def test_real_handdrawn_renderer_holds_full_bw_for_two_seconds_then_blooms_locally(
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
    bw = assets / "bw.png"
    color = assets / "color.png"
    for value, output in (("black", bw), ("red", color)):
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                f"color=c={value}:s=1080x1920:d=1", "-frames:v", "1", "-y", str(output),
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
                    "total_frames": 240,
                    "transition": "cut",
                    "transition_frames": 0,
                    "show_key_line": False,
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
                        "end_ms": 8000,
                        "from_frame": 0,
                        "to_frame": 240,
                        "key_line": "局部晕染测试",
                        "narration": "测试旁白。",
                        "assets": {"bw": "assets/bw.png", "color": "assets/color.png"},
                    }
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
            asset_sha256s=_asset_sha256s(episode_root, storyboard),
        ),
        npm_command=npm,
        ffprobe_command=ffprobe,
    )
    opening_frame = episode_root / "opening-bw.png"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i",
            str(result.output_path), "-vf", "select=eq(n\\,0)",
            "-frames:v", "1", "-y", str(opening_frame),
        ],
        check=True,
    )
    opening = pymupdf.Pixmap(str(opening_frame))
    assert all(channel < 50 for channel in opening.pixel(540, 960)[:3])

    hold_frame = episode_root / "bw-before-two-seconds.png"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", "1.96", "-i",
            str(result.output_path), "-frames:v", "1", "-y", str(hold_frame),
        ],
        check=True,
    )
    hold = pymupdf.Pixmap(str(hold_frame))

    first_bloom_frame = episode_root / "first-bloom-after-two-seconds.png"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i",
            str(result.output_path), "-vf", "select=eq(n\\,61)",
            "-frames:v", "1", "-y", str(first_bloom_frame),
        ],
        check=True,
    )
    first_bloom = pymupdf.Pixmap(str(first_bloom_frame))

    visibly_changed_pixels = sum(
        1
        for y in range(340, 1460, 4)
        for x in range(0, 1080, 4)
        if max(
            abs(first_bloom.pixel(x, y)[channel] - hold.pixel(x, y)[channel])
            for channel in range(3)
        ) >= 3
    )

    frame = episode_root / "bloom-after-two-seconds.png"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", "3.50", "-i",
            str(result.output_path), "-frames:v", "1", "-y", str(frame),
        ],
        check=True,
    )
    pixmap = pymupdf.Pixmap(str(frame))

    def count_red(image: pymupdf.Pixmap, x_start: int, x_end: int) -> int:
        return sum(
            1
            for y in range(400, 1460, 20)
            for x in range(x_start, x_end, 20)
            if (lambda rgb: rgb[0] > 150 and rgb[1] < 90 and rgb[2] < 90)(
                image.pixel(x, y)[:3]
            )
        )

    black_pixels = sum(
        1
        for y in range(400, 1460, 20)
        for x in range(0, 1080, 20)
        if all(channel < 50 for channel in pixmap.pixel(x, y)[:3])
    )

    assert count_red(hold, 0, 1080) == 0
    assert visibly_changed_pixels > 20
    assert count_red(pixmap, 0, 360) > 20
    assert count_red(pixmap, 720, 1080) > 20
    assert black_pixels > 80


def test_real_handdrawn_renderer_reveals_story_pair_from_semantic_turn(
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
    anchor = assets / "anchor.png"
    continuation = assets / "continuation.png"
    for color, output in (("red", anchor), ("blue", continuation)):
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
                    "total_frames": 240,
                    "transition": "cut",
                    "transition_frames": 0,
                    "ink_reveal_frames": 45,
                    "show_key_line": False,
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
                        "end_ms": 8000,
                        "from_frame": 0,
                        "to_frame": 240,
                        "key_line": "彩色剧情推进测试",
                        "narration": "测试旁白。",
                        "sequence_mode": "color-story-pair",
                        "semantic_turn_frame": 90,
                        "assets": {
                            "anchor": "assets/anchor.png",
                            "continuation": "assets/continuation.png",
                        },
                    }
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
            asset_sha256s=_asset_sha256s(episode_root, storyboard),
        ),
        npm_command=npm,
        ffprobe_command=ffprobe,
    )

    def frame_at(frame_number: int) -> pymupdf.Pixmap:
        output = episode_root / f"frame-{frame_number}.png"
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(result.output_path),
                "-vf", f"select=eq(n\\,{frame_number})", "-frames:v", "1", "-y", str(output),
            ],
            check=True,
        )
        return pymupdf.Pixmap(str(output))

    opening = frame_at(0)
    before = frame_at(89)
    midpoint = frame_at(112)
    after = frame_at(135)
    held = frame_at(136)
    scene_tail = frame_at(224)

    def is_red(pixel: tuple[int, ...]) -> bool:
        return pixel[0] > 150 and pixel[1] < 90 and pixel[2] < 90

    def is_blue(pixel: tuple[int, ...]) -> bool:
        return pixel[2] > 150 and pixel[0] < 90 and pixel[1] < 90

    sample_points = ((80, 420), (360, 740), (540, 1080), (860, 1300), (1000, 1440))
    assert all(is_red(opening.pixel(x, y)[:3]) for x, y in sample_points)
    assert all(is_red(before.pixel(x, y)[:3]) for x, y in sample_points)
    midpoint_pixels = [midpoint.pixel(x, y)[:3] for x, y in sample_points]
    assert any(is_red(pixel) for pixel in midpoint_pixels)
    assert any(is_blue(pixel) for pixel in midpoint_pixels)
    assert all(not all(channel > 245 for channel in pixel) for pixel in midpoint_pixels)
    assert all(is_blue(after.pixel(x, y)[:3]) for x, y in sample_points)
    assert all(is_blue(held.pixel(x, y)[:3]) for x, y in sample_points)
    assert all(is_blue(scene_tail.pixel(x, y)[:3]) for x, y in sample_points)


def test_real_handdrawn_renderer_cross_dissolves_complete_pair_b_to_next_a(
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
    names = {"red": "S01_anchor.png", "blue": "S01_continuation.png", "green": "S02_anchor.png", "yellow": "S02_continuation.png"}
    for color, name in names.items():
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                f"color=c={color}:s=1080x1920:d=1", "-frames:v", "1", "-y", str(assets / name),
            ],
            check=True,
        )
    safe_area = {
        "top_reserved": 140, "keyline_top": 140, "keyline_bottom": 340,
        "illustration_top": 340, "illustration_bottom": 1460, "subtitle_top": 1460,
        "subtitle_bottom": 1660, "bottom_reserved": 260, "right_reserved": 180,
    }
    storyboard = episode_root / "media" / "illustration" / "storyboard.json"
    storyboard.parent.mkdir(parents=True)
    storyboard.write_text(
        json.dumps(
            {
                "project": {
                    "width": 1080, "height": 1920, "fps": 30, "ratio": "9:16",
                    "total_frames": 360, "transition": "cross-dissolve", "transition_frames": 15,
                    "ink_reveal_frames": 45, "show_key_line": False,
                },
                "safe_area": safe_area,
                "scenes": [
                    {
                        "id": "S01", "start_ms": 0, "end_ms": 6000, "from_frame": 0,
                        "to_frame": 180, "key_line": "第一段剧情推进", "narration": "第一段测试旁白。",
                        "sequence_mode": "color-story-pair", "semantic_turn_frame": 45,
                        "assets": {"anchor": "assets/S01_anchor.png", "continuation": "assets/S01_continuation.png"},
                    },
                    {
                        "id": "S02", "start_ms": 6000, "end_ms": 12000, "from_frame": 180,
                        "to_frame": 360, "key_line": "第二段剧情推进", "narration": "第二段测试旁白。",
                        "sequence_mode": "color-story-pair", "semantic_turn_frame": 225,
                        "assets": {"anchor": "assets/S02_anchor.png", "continuation": "assets/S02_continuation.png"},
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
            asset_sha256s=_asset_sha256s(episode_root, storyboard),
        ),
        npm_command=npm,
        ffprobe_command=ffprobe,
    )

    def pixel_at(frame_number: int) -> tuple[int, ...]:
        output = episode_root / f"cross-{frame_number}.png"
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(result.output_path),
                "-vf", f"select=eq(n\\,{frame_number})", "-frames:v", "1", "-y", str(output),
            ],
            check=True,
        )
        return pymupdf.Pixmap(str(output)).pixel(540, 960)[:3]

    before = pixel_at(179)
    midpoint = pixel_at(187)
    after = pixel_at(195)
    assert before[2] > 150 and before[0] < 90 and before[1] < 90
    assert midpoint[1] > 20 and midpoint[2] > 40
    assert not all(channel > 245 for channel in midpoint)
    assert after[1] > 70 and after[0] < 90 and after[2] < 90
