import json
import shutil
import subprocess
from pathlib import Path

import pytest

import bv.illustration.renderer as renderer_module
from bv.core.hashing import sha256_file
from bv.core.process import CommandResult
from bv.illustration.contracts import SilentRenderRequest
from bv.illustration.renderer import (
    HanddrawnRenderError,
    build_handdrawn_render_argv,
    render_silent_story,
)
from bv.video.probe import MediaProbeFacts


def _storyboard_payload(*, asset_path: str = "assets/S01_bw.svg") -> dict[str, object]:
    return {
        "project": {
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "ratio": "9:16",
            "total_frames": 450,
            "transition": "cut",
            "transition_frames": 0,
        },
        "safe_area": {},
        "scenes": [
            {
                "id": "S01",
                "start_ms": 0,
                "end_ms": 15_000,
                "from_frame": 0,
                "to_frame": 450,
                "key_line": "把人生还给自己",
                "narration": "测试旁白。",
                "assets": {
                    "bw": asset_path,
                    "color": "assets/S01_color.svg",
                },
            }
        ],
    }


def _safe_request(
    tmp_path: Path,
    *,
    asset_path: str = "assets/S01_bw.svg",
) -> SilentRenderRequest:
    episode_root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    assets = episode_root / "assets"
    assets.mkdir(parents=True)
    (assets / "S01_bw.svg").write_text("<svg></svg>", encoding="utf-8")
    (assets / "S01_color.svg").write_text("<svg></svg>", encoding="utf-8")
    storyboard = episode_root / "media" / "illustration" / "storyboard.json"
    storyboard.parent.mkdir(parents=True)
    storyboard.write_text(
        json.dumps(_storyboard_payload(asset_path=asset_path)),
        encoding="utf-8",
    )
    vendor = tmp_path / "vendor" / "story_to_handdrawn_video"
    vendor.mkdir(parents=True)
    return SilentRenderRequest(
        episode_root=episode_root,
        storyboard_path=storyboard,
        storyboard_sha256=sha256_file(storyboard),
        output_path=episode_root / "media" / "render" / "picture_silent.mp4",
        vendor_dir=vendor,
        asset_sha256s={
            "assets/S01_bw.svg": sha256_file(assets / "S01_bw.svg"),
            "assets/S01_color.svg": sha256_file(assets / "S01_color.svg"),
        },
    )


def _facts(**updates: object) -> MediaProbeFacts:
    payload: dict[str, object] = {
        "duration_ms": 15_000,
        "width": 1080,
        "height": 1920,
        "frame_rate": 30.0,
        "video_codec": "h264",
        "pixel_format": "yuv420p",
        "audio_present": False,
        "audio_codec": None,
    }
    payload.update(updates)
    return MediaProbeFacts.model_validate(payload)


def _successful_runner(
    argv: list[str],
    *,
    cwd: Path,
    timeout: float,
) -> CommandResult:
    assert cwd.is_dir()
    assert timeout == 900
    output = Path(argv[argv.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"synthetic silent h264")
    return CommandResult(argv=argv, returncode=0)


def _successful_inspector(
    path: Path,
    *,
    ffprobe_command: str,
) -> MediaProbeFacts:
    assert path.is_file()
    assert ffprobe_command == "ffprobe"
    return _facts()


def test_build_render_argv_uses_explicit_props_and_muted_output(
    tmp_path: Path,
) -> None:
    request = _safe_request(tmp_path)

    argv = build_handdrawn_render_argv(request, npm_command="npm")

    assert argv[:3] == ["npm", "run", "render:props"]
    assert argv[3:5] == ["--", "--props"]
    assert str(request.storyboard_path) in argv
    assert str(request.output_path) in argv
    assert argv[argv.index("--public-dir") + 1] == str(request.episode_root)


def test_render_publishes_only_a_valid_silent_track(tmp_path: Path) -> None:
    request = _safe_request(tmp_path)

    result = render_silent_story(
        request,
        runner=_successful_runner,
        inspector=_successful_inspector,
    )

    assert result.output_path == request.output_path
    assert result.output_sha256 == sha256_file(request.output_path)
    assert result.duration_ms == 15_000
    assert result.width == 1080
    assert result.height == 1920
    assert result.frame_rate == 30.0
    assert result.audio_present is False
    assert list(request.output_path.parent.glob(".*.tmp.mp4")) == []


def test_render_accepts_complete_story_pair_assets(tmp_path: Path) -> None:
    request = _safe_request(tmp_path)
    payload = _storyboard_payload()
    payload["project"]["ink_reveal_frames"] = 45
    scene = payload["scenes"][0]
    scene["sequence_mode"] = "color-story-pair"
    scene["semantic_turn_frame"] = 360
    scene["assets"] = {
        "anchor": "assets/S01_bw.svg",
        "continuation": "assets/S01_color.svg",
    }
    request.storyboard_path.write_text(json.dumps(payload), encoding="utf-8")
    request = request.model_copy(
        update={"storyboard_sha256": sha256_file(request.storyboard_path)}
    )

    result = render_silent_story(
        request,
        runner=_successful_runner,
        inspector=_successful_inspector,
    )

    assert result.output_path == request.output_path


@pytest.mark.parametrize(
    "sequence_mode,asset_key",
    [
        ("legacy-monochrome-reveal", "bw"),
        ("legacy-monochrome-reveal", "color"),
        ("color-story-pair", "anchor"),
        ("color-story-pair", "continuation"),
    ],
)
def test_render_rejects_assets_changed_after_manifest_hash_binding(
    tmp_path: Path,
    sequence_mode: str,
    asset_key: str,
) -> None:
    request = _safe_request(tmp_path)
    payload = _storyboard_payload()
    scene = payload["scenes"][0]
    if sequence_mode == "color-story-pair":
        payload["project"]["ink_reveal_frames"] = 45
        scene["sequence_mode"] = sequence_mode
        scene["semantic_turn_frame"] = 360
        scene["assets"] = {
            "anchor": "assets/S01_bw.svg",
            "continuation": "assets/S01_color.svg",
        }
    request.storyboard_path.write_text(json.dumps(payload), encoding="utf-8")
    request = request.model_copy(
        update={
            "storyboard_sha256": sha256_file(request.storyboard_path),
            "asset_sha256s": {
                path: sha256_file(request.episode_root / path)
                for path in scene["assets"].values()
            },
        }
    )
    changed = request.episode_root / scene["assets"][asset_key]
    changed.write_text("<svg>changed after manifest validation</svg>", encoding="utf-8")
    called = False

    def runner(*_args: object, **_kwargs: object) -> CommandResult:
        nonlocal called
        called = True
        raise AssertionError("runner must not be called")

    with pytest.raises(HanddrawnRenderError, match="render_asset_changed"):
        render_silent_story(request, runner=runner)

    assert called is False
    assert not list(request.output_path.parent.glob(".render-public-*"))
    assert not list(request.output_path.parent.glob(".*.tmp.mp4"))


@pytest.mark.parametrize("hashes", [{}, {"assets/S01_bw.svg": "a" * 64, "assets/extra.svg": "b" * 64}])
def test_render_rejects_missing_or_extra_asset_hash_keys(
    tmp_path: Path,
    hashes: dict[str, str],
) -> None:
    request = _safe_request(tmp_path).model_copy(update={"asset_sha256s": hashes})

    with pytest.raises(HanddrawnRenderError, match="render_asset_hashes_invalid"):
        render_silent_story(request, runner=_successful_runner)


def test_render_snapshots_only_legacy_assets_and_cleans_its_public_directory(
    tmp_path: Path,
) -> None:
    request = _safe_request(tmp_path)
    source = request.episode_root / "assets" / "S01_bw.svg"
    observed_public_dir: Path | None = None

    def runner(argv: list[str], *, cwd: Path, timeout: float) -> CommandResult:
        nonlocal observed_public_dir
        observed_public_dir = Path(argv[argv.index("--public-dir") + 1])
        assert observed_public_dir != request.episode_root
        assert (observed_public_dir / "assets" / "S01_bw.svg").read_text(encoding="utf-8") == "<svg></svg>"
        assert (observed_public_dir / "assets" / "S01_color.svg").read_text(encoding="utf-8") == "<svg></svg>"
        assert sorted(path.name for path in (observed_public_dir / "assets").iterdir()) == [
            "S01_bw.svg",
            "S01_color.svg",
        ]
        source.write_text("<svg>changed</svg>", encoding="utf-8")
        assert (observed_public_dir / "assets" / "S01_bw.svg").read_text(encoding="utf-8") == "<svg></svg>"
        return _successful_runner(argv, cwd=cwd, timeout=timeout)

    render_silent_story(
        request,
        runner=runner,
        inspector=_successful_inspector,
    )

    assert observed_public_dir is not None
    assert not observed_public_dir.exists()


def test_render_snapshots_only_pair_assets(tmp_path: Path) -> None:
    request = _safe_request(tmp_path)
    payload = _storyboard_payload()
    payload["project"]["ink_reveal_frames"] = 45
    scene = payload["scenes"][0]
    scene["sequence_mode"] = "color-story-pair"
    scene["semantic_turn_frame"] = 360
    scene["assets"] = {
        "anchor": "assets/S01_bw.svg",
        "continuation": "assets/S01_color.svg",
    }
    request.storyboard_path.write_text(json.dumps(payload), encoding="utf-8")
    request = request.model_copy(
        update={"storyboard_sha256": sha256_file(request.storyboard_path)}
    )
    copied: list[Path] = []

    def runner(argv: list[str], *, cwd: Path, timeout: float) -> CommandResult:
        public_dir = Path(argv[argv.index("--public-dir") + 1])
        copied.extend(path.relative_to(public_dir) for path in public_dir.rglob("*") if path.is_file())
        return _successful_runner(argv, cwd=cwd, timeout=timeout)

    render_silent_story(request, runner=runner, inspector=_successful_inspector)

    assert copied == [Path("assets/S01_bw.svg"), Path("assets/S01_color.svg")]


def test_render_rejects_story_pair_without_30_full_continuation_frames(
    tmp_path: Path,
) -> None:
    request = _safe_request(tmp_path)
    payload = _storyboard_payload()
    payload["project"]["ink_reveal_frames"] = 45
    scene = payload["scenes"][0]
    scene["sequence_mode"] = "color-story-pair"
    scene["semantic_turn_frame"] = 361
    scene["assets"] = {
        "anchor": "assets/S01_bw.svg",
        "continuation": "assets/S01_color.svg",
    }
    request.storyboard_path.write_text(json.dumps(payload), encoding="utf-8")
    request = request.model_copy(
        update={"storyboard_sha256": sha256_file(request.storyboard_path)}
    )

    with pytest.raises(HanddrawnRenderError, match="invalid_storyboard"):
        render_silent_story(request, runner=_successful_runner)


@pytest.mark.parametrize(
    "sequence_mode,assets",
    [
        (
            "color-story-pair",
            {
                "anchor": "assets/S01_bw.svg",
                "continuation": "assets/S01_color.svg",
                "bw": "assets/S01_bw.svg",
            },
        ),
        (
            "legacy-monochrome-reveal",
            {
                "bw": "assets/S01_bw.svg",
                "color": "assets/S01_color.svg",
                "anchor": "assets/S01_bw.svg",
            },
        ),
    ],
    ids=["pair-mixed-with-legacy", "legacy-mixed-with-pair"],
)
def test_render_rejects_mixed_discriminated_asset_keys(
    tmp_path: Path,
    sequence_mode: str,
    assets: dict[str, str],
) -> None:
    request = _safe_request(tmp_path)
    payload = _storyboard_payload()
    scene = payload["scenes"][0]
    scene["sequence_mode"] = sequence_mode
    scene["assets"] = assets
    if sequence_mode == "color-story-pair":
        payload["project"]["ink_reveal_frames"] = 45
        scene["semantic_turn_frame"] = 360
    request.storyboard_path.write_text(json.dumps(payload), encoding="utf-8")
    request = request.model_copy(
        update={"storyboard_sha256": sha256_file(request.storyboard_path)}
    )

    with pytest.raises(HanddrawnRenderError, match="invalid_storyboard"):
        render_silent_story(request, runner=_successful_runner)


@pytest.mark.skipif(renderer_module.os.name != "nt", reason="Windows command shim")
def test_render_resolves_windows_npm_command_shim(tmp_path: Path) -> None:
    request = _safe_request(tmp_path)
    calls: list[list[str]] = []

    def runner(argv: list[str], *, cwd: Path, timeout: float) -> CommandResult:
        calls.append(argv)
        return _successful_runner(argv, cwd=cwd, timeout=timeout)

    render_silent_story(
        request,
        npm_command="npm",
        runner=runner,
        inspector=_successful_inspector,
    )

    assert calls[0][0].lower().endswith("npm.cmd")


def test_render_rejects_output_outside_episode(tmp_path: Path) -> None:
    request = _safe_request(tmp_path).model_copy(
        update={"output_path": tmp_path / "outside.mp4"}
    )

    with pytest.raises(HanddrawnRenderError, match="unsafe_render_path"):
        render_silent_story(
            request,
            runner=_successful_runner,
            inspector=_successful_inspector,
        )


def test_render_refuses_existing_output_without_running(tmp_path: Path) -> None:
    request = _safe_request(tmp_path)
    request.output_path.parent.mkdir(parents=True)
    request.output_path.write_bytes(b"keep me")
    called = False

    def runner(*_args: object, **_kwargs: object) -> CommandResult:
        nonlocal called
        called = True
        raise AssertionError("runner must not be called")

    with pytest.raises(HanddrawnRenderError, match="output_already_exists"):
        render_silent_story(request, runner=runner)

    assert request.output_path.read_bytes() == b"keep me"
    assert called is False


def test_render_rejects_stale_storyboard_hash(tmp_path: Path) -> None:
    request = _safe_request(tmp_path).model_copy(
        update={"storyboard_sha256": "0" * 64}
    )

    with pytest.raises(HanddrawnRenderError, match="storyboard_hash_mismatch"):
        render_silent_story(request, runner=_successful_runner)


def test_render_rejects_storyboard_asset_escape(tmp_path: Path) -> None:
    request = _safe_request(tmp_path, asset_path="../outside.svg")

    with pytest.raises(HanddrawnRenderError, match="unsafe_render_path"):
        render_silent_story(request, runner=_successful_runner)


def test_render_maps_nonzero_npm_exit_to_stable_error(tmp_path: Path) -> None:
    request = _safe_request(tmp_path)

    def failed_runner(
        argv: list[str],
        *,
        cwd: Path,
        timeout: float,
    ) -> CommandResult:
        return CommandResult(argv=argv, returncode=1, stderr="private path")

    with pytest.raises(HanddrawnRenderError, match="handdrawn_render_failed"):
        render_silent_story(request, runner=failed_runner)


@pytest.mark.parametrize(
    "facts",
    [
        _facts(video_codec="hevc"),
        _facts(pixel_format="yuvj420p"),
        _facts(width=720),
        _facts(height=1280),
        _facts(frame_rate=29.97),
        _facts(audio_present=True, audio_codec="aac"),
        _facts(duration_ms=14_900),
    ],
    ids=["codec", "pixel-format", "width", "height", "fps", "audio", "duration"],
)
def test_render_rejects_invalid_silent_media_facts(
    tmp_path: Path,
    facts: MediaProbeFacts,
) -> None:
    request = _safe_request(tmp_path)

    with pytest.raises(HanddrawnRenderError, match="invalid_silent_render"):
        render_silent_story(
            request,
            runner=_successful_runner,
            inspector=lambda *_args, **_kwargs: facts,
        )

    assert not request.output_path.exists()


def test_render_rejects_output_changed_during_probe(tmp_path: Path) -> None:
    request = _safe_request(tmp_path)

    def mutating_inspector(
        path: Path,
        *,
        ffprobe_command: str,
    ) -> MediaProbeFacts:
        path.write_bytes(b"changed during probe")
        return _facts()

    with pytest.raises(HanddrawnRenderError, match="render_output_changed"):
        render_silent_story(
            request,
            runner=_successful_runner,
            inspector=mutating_inspector,
        )

    assert not request.output_path.exists()


def test_render_refuses_publish_race_and_preserves_other_output(
    tmp_path: Path,
) -> None:
    request = _safe_request(tmp_path)

    def racing_inspector(
        path: Path,
        *,
        ffprobe_command: str,
    ) -> MediaProbeFacts:
        request.output_path.write_bytes(b"other process")
        return _facts()

    with pytest.raises(HanddrawnRenderError, match="output_already_exists"):
        render_silent_story(
            request,
            runner=_successful_runner,
            inspector=racing_inspector,
        )

    assert request.output_path.read_bytes() == b"other process"


def test_render_removes_its_published_file_when_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _safe_request(tmp_path)

    def fail_fsync(_path: Path) -> None:
        raise HanddrawnRenderError("render_publish_failed")

    monkeypatch.setattr(renderer_module, "_fsync_file", fail_fsync)

    with pytest.raises(HanddrawnRenderError, match="render_publish_failed"):
        render_silent_story(
            request,
            runner=_successful_runner,
            inspector=_successful_inspector,
        )

    assert not request.output_path.exists()


def test_storyboard_validator_accepts_story_pair_assets_at_the_semantic_turn(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("local Node.js is required for the storyboard validator")
    payload = _storyboard_payload()
    payload["project"]["ink_reveal_frames"] = 45
    payload["safe_area"] = {
        "top_reserved": 140,
        "keyline_top": 140,
        "keyline_bottom": 340,
        "illustration_top": 340,
        "illustration_bottom": 1460,
        "subtitle_top": 1460,
        "subtitle_bottom": 1660,
        "bottom_reserved": 260,
        "right_reserved": 180,
    }
    scene = payload["scenes"][0]
    scene["sequence_mode"] = "color-story-pair"
    scene["semantic_turn_frame"] = 360
    scene["assets"] = {
        "anchor": "assets/02_color.svg",
        "continuation": "assets/03_color.svg",
    }
    fixture = tmp_path / "story-pair.json"
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    validator = (
        Path(__file__).resolve().parents[2]
        / "vendor"
        / "story_to_handdrawn_video"
        / "scripts"
        / "validate-storyboard.mjs"
    )

    result = subprocess.run(
        [node, str(validator), "--input", str(fixture)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "assets,semantic_turn_frame",
    [
        ({"anchor": "assets/02_color.svg"}, 90),
        ({"bw": "assets/02_bw.svg", "color": "assets/02_color.svg"}, 90),
        (
            {
                "anchor": "assets/02_color.svg",
                "continuation": "assets/03_color.svg",
            },
            20,
        ),
        (
            {
                "anchor": "assets/02_color.svg",
                "continuation": "assets/03_color.svg",
            },
            361,
        ),
    ],
    ids=[
        "missing-continuation",
        "legacy-assets",
        "turn-without-anchor-dwell",
        "turn-without-continuation-dwell",
    ],
)
def test_storyboard_validator_rejects_invalid_story_pair_shape(
    tmp_path: Path,
    assets: dict[str, str],
    semantic_turn_frame: int,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("local Node.js is required for the storyboard validator")
    payload = _storyboard_payload()
    payload["project"]["ink_reveal_frames"] = 45
    payload["safe_area"] = {
        "top_reserved": 140,
        "keyline_top": 140,
        "keyline_bottom": 340,
        "illustration_top": 340,
        "illustration_bottom": 1460,
        "subtitle_top": 1460,
        "subtitle_bottom": 1660,
        "bottom_reserved": 260,
        "right_reserved": 180,
    }
    scene = payload["scenes"][0]
    scene["sequence_mode"] = "color-story-pair"
    scene["semantic_turn_frame"] = semantic_turn_frame
    scene["assets"] = assets
    fixture = tmp_path / "invalid-story-pair.json"
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    validator = (
        Path(__file__).resolve().parents[2]
        / "vendor"
        / "story_to_handdrawn_video"
        / "scripts"
        / "validate-storyboard.mjs"
    )

    result = subprocess.run(
        [node, str(validator), "--input", str(fixture)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 1


def test_npm_check_validates_legacy_and_story_pair_fixtures() -> None:
    npm = shutil.which("npm")
    if npm is None:
        pytest.skip("local npm is required for the storyboard validator")
    vendor = Path(__file__).resolve().parents[2] / "vendor" / "story_to_handdrawn_video"

    result = subprocess.run(
        [npm, "run", "check"],
        cwd=vendor,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "✓ storyboard.9x16.fixture.json" in result.stdout
    assert "✓ storyboard.story-pair.fixture.json" in result.stdout
