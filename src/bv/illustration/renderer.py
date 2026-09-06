from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path, PureWindowsPath

from pydantic import ValidationError

from bv.core.hashing import sha256_file
from bv.core.process import CommandResult, run_command
from bv.video.probe import MediaProbeFacts, probe_media

from .contracts import SilentRenderRequest, SilentRenderResult


_STORY_PAIR_INK_REVEAL_FRAMES = 45
_STORY_PAIR_CONTINUATION_SIDE_FRAMES = _STORY_PAIR_INK_REVEAL_FRAMES + 30 + 15


class HanddrawnRenderError(RuntimeError):
    """Stable, privacy-safe error raised by the hand-drawn renderer."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


def _validated_request(request: SilentRenderRequest) -> SilentRenderRequest:
    try:
        return SilentRenderRequest.model_validate(request)
    except ValidationError as exc:
        path_fields = {"episode_root", "storyboard_path", "output_path", "vendor_dir"}
        if any(
            (error.get("loc") and error["loc"][0] in path_fields)
            or "unsafe_" in str(error.get("msg", ""))
            for error in exc.errors()
        ):
            raise HanddrawnRenderError("unsafe_render_path") from None
        raise HanddrawnRenderError("invalid_render_request") from None


def build_handdrawn_render_argv(
    request: SilentRenderRequest,
    *,
    npm_command: str = "npm",
    public_dir: Path | None = None,
) -> list[str]:
    request = _validated_request(request)
    if not isinstance(npm_command, str) or not npm_command.strip():
        raise HanddrawnRenderError("invalid_render_configuration")
    resolved_public_dir = request.episode_root if public_dir is None else public_dir
    return [
        npm_command,
        "run",
        "render:props",
        "--",
        "--props",
        str(request.storyboard_path),
        "--output",
        str(request.output_path),
        "--public-dir",
        str(resolved_public_dir),
    ]


def render_silent_story(
    request: SilentRenderRequest,
    *,
    npm_command: str = "npm",
    ffprobe_command: str = "ffprobe",
    runner: Callable[..., CommandResult] = run_command,
    inspector: Callable[..., MediaProbeFacts] = probe_media,
) -> SilentRenderResult:
    request = _validated_request(request)
    _require_preflight_paths(request)
    if request.output_path.exists():
        raise HanddrawnRenderError("output_already_exists")
    if sha256_file(request.storyboard_path) != request.storyboard_sha256:
        raise HanddrawnRenderError("storyboard_hash_mismatch")
    total_frames, assets = _validate_storyboard_assets(request)
    _ensure_safe_directory(request.output_path.parent)

    temporary = request.output_path.parent / (
        f".{request.output_path.stem}.{uuid.uuid4().hex}.tmp.mp4"
    )
    temporary_request = _validated_request(
        request.model_copy(update={"output_path": temporary})
    )
    output_hash = ""
    published = False
    completed = False
    public_snapshot: Path | None = None
    try:
        public_snapshot = _snapshot_public_assets(request, assets)
        try:
            result = runner(
                build_handdrawn_render_argv(
                    temporary_request,
                    npm_command=(
                        shutil.which(npm_command) or npm_command
                        if os.name == "nt"
                        else npm_command
                    ),
                    public_dir=public_snapshot,
                ),
                cwd=request.vendor_dir,
                timeout=900,
            )
        except Exception:
            raise HanddrawnRenderError("handdrawn_render_failed") from None
        if not isinstance(result, CommandResult) or result.returncode != 0:
            raise HanddrawnRenderError("handdrawn_render_failed")
        _require_safe_file(temporary)
        if sha256_file(request.storyboard_path) != request.storyboard_sha256:
            raise HanddrawnRenderError("storyboard_hash_changed")
        output_hash = sha256_file(temporary)

        try:
            facts = inspector(temporary, ffprobe_command=ffprobe_command)
        except Exception:
            raise HanddrawnRenderError("silent_render_probe_failed") from None
        if not isinstance(facts, MediaProbeFacts):
            raise HanddrawnRenderError("silent_render_probe_failed")
        _validate_silent_media(facts, total_frames=total_frames)

        if (
            sha256_file(temporary) != output_hash
            or sha256_file(request.storyboard_path) != request.storyboard_sha256
        ):
            raise HanddrawnRenderError("render_output_changed")
        if request.output_path.exists() or _redirect_in_existing_chain(
            request.output_path
        ):
            raise HanddrawnRenderError("output_already_exists")
        try:
            os.link(temporary, request.output_path)
        except FileExistsError:
            raise HanddrawnRenderError("output_already_exists") from None
        except OSError:
            raise HanddrawnRenderError("render_publish_failed") from None
        published = True
        _fsync_file(request.output_path)
        if sha256_file(request.output_path) != output_hash:
            raise HanddrawnRenderError("render_output_changed")
        rendered = SilentRenderResult(
            output_path=request.output_path,
            output_sha256=output_hash,
            duration_ms=facts.duration_ms,
            width=1080,
            height=1920,
            frame_rate=30.0,
            audio_present=False,
        )
        completed = True
        return rendered
    finally:
        _remove_owned_file(temporary)
        _remove_owned_directory(public_snapshot)
        if published and not completed:
            _remove_if_hash(request.output_path, output_hash)


def _validate_silent_media(
    facts: MediaProbeFacts,
    *,
    total_frames: int,
) -> None:
    actual_frames = round(facts.duration_ms * 30 / 1000)
    if (
        facts.video_codec != "h264"
        or facts.pixel_format != "yuv420p"
        or (facts.width, facts.height) != (1080, 1920)
        or abs(facts.frame_rate - 30.0) > 0.01
        or facts.audio_present
        or abs(actual_frames - total_frames) > 1
    ):
        raise HanddrawnRenderError("invalid_silent_render")


def _validate_storyboard_assets(
    request: SilentRenderRequest,
) -> tuple[int, tuple[tuple[Path, str], ...]]:
    try:
        payload = json.loads(request.storyboard_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise HanddrawnRenderError("invalid_storyboard") from None
    if not isinstance(payload, Mapping):
        raise HanddrawnRenderError("invalid_storyboard")
    project = payload.get("project")
    scenes = payload.get("scenes")
    if not isinstance(project, Mapping) or not isinstance(scenes, list) or not scenes:
        raise HanddrawnRenderError("invalid_storyboard")
    total_frames = project.get("total_frames")
    if (
        project.get("width") != 1080
        or project.get("height") != 1920
        or project.get("fps") != 30
        or isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames <= 0
    ):
        raise HanddrawnRenderError("invalid_storyboard")
    ink_reveal_frames = project.get("ink_reveal_frames")
    snapshots: dict[Path, str] = {}
    for scene in scenes:
        if not isinstance(scene, Mapping) or not isinstance(scene.get("assets"), Mapping):
            raise HanddrawnRenderError("invalid_storyboard")
        assets = scene["assets"]
        sequence_mode = scene.get("sequence_mode", "legacy-monochrome-reveal")
        if sequence_mode == "color-story-pair":
            semantic_turn_frame = scene.get("semantic_turn_frame")
            from_frame = scene.get("from_frame")
            to_frame = scene.get("to_frame")
            if (
                ink_reveal_frames != _STORY_PAIR_INK_REVEAL_FRAMES
                or isinstance(semantic_turn_frame, bool)
                or not isinstance(semantic_turn_frame, int)
                or isinstance(from_frame, bool)
                or not isinstance(from_frame, int)
                or isinstance(to_frame, bool)
                or not isinstance(to_frame, int)
                or semantic_turn_frame - from_frame < 30
                or to_frame - semantic_turn_frame < _STORY_PAIR_CONTINUATION_SIDE_FRAMES
                or set(assets) != {"anchor", "continuation"}
            ):
                raise HanddrawnRenderError("invalid_storyboard")
            asset_keys = ("anchor", "continuation")
        elif sequence_mode == "legacy-monochrome-reveal":
            if set(assets) != {"bw", "color"}:
                raise HanddrawnRenderError("invalid_storyboard")
            asset_keys = ("bw", "color")
        else:
            raise HanddrawnRenderError("invalid_storyboard")
        for key in asset_keys:
            value = assets.get(key)
            if not isinstance(value, str) or not value:
                raise HanddrawnRenderError("invalid_storyboard")
            windows = PureWindowsPath(value)
            if Path(value).is_absolute() or windows.is_absolute() or windows.drive:
                raise HanddrawnRenderError("unsafe_render_path")
            asset = Path(os.path.abspath(request.episode_root / value))
            if not asset.is_relative_to(request.episode_root):
                raise HanddrawnRenderError("unsafe_render_path")
            _require_safe_file(asset)
            relative_path = asset.relative_to(request.episode_root)
            snapshots[relative_path] = ""
    expected_hashes = request.asset_sha256s
    if {path.as_posix() for path in snapshots} != set(expected_hashes):
        raise HanddrawnRenderError("render_asset_hashes_invalid")
    for relative_path in snapshots:
        expected_hash = expected_hashes[relative_path.as_posix()]
        source = request.episode_root / relative_path
        if sha256_file(source) != expected_hash:
            raise HanddrawnRenderError("render_asset_changed")
        snapshots[relative_path] = expected_hash
    return total_frames, tuple(snapshots.items())


def _snapshot_public_assets(
    request: SilentRenderRequest,
    assets: tuple[tuple[Path, str], ...],
) -> Path:
    try:
        snapshot = Path(
            tempfile.mkdtemp(
                prefix=".render-public-",
                dir=request.output_path.parent,
            )
        )
    except OSError:
        raise HanddrawnRenderError("render_snapshot_failed") from None
    try:
        if _redirect_in_existing_chain(snapshot):
            raise HanddrawnRenderError("unsafe_render_path")
        for relative_path, expected_hash in assets:
            source = request.episode_root / relative_path
            _require_safe_file(source)
            if sha256_file(source) != expected_hash:
                raise HanddrawnRenderError("render_asset_changed")
            target = snapshot / relative_path
            _ensure_safe_directory(target.parent)
            shutil.copyfile(source, target)
            _require_safe_file(target)
            if sha256_file(target) != expected_hash:
                raise HanddrawnRenderError("render_asset_changed")
        return snapshot
    except HanddrawnRenderError:
        _remove_owned_directory(snapshot)
        raise
    except OSError:
        _remove_owned_directory(snapshot)
        raise HanddrawnRenderError("render_snapshot_failed") from None


def _require_preflight_paths(request: SilentRenderRequest) -> None:
    _require_safe_file(request.storyboard_path)
    if not request.vendor_dir.is_dir() or _redirect_in_existing_chain(
        request.vendor_dir
    ):
        raise HanddrawnRenderError("unsafe_render_path")
    if _redirect_in_existing_chain(request.output_path):
        raise HanddrawnRenderError("unsafe_render_path")


def _require_safe_file(path: Path) -> None:
    if not path.is_file() or _redirect_in_existing_chain(path):
        raise HanddrawnRenderError("unsafe_render_path")


def _ensure_safe_directory(path: Path) -> None:
    if _redirect_in_existing_chain(path):
        raise HanddrawnRenderError("unsafe_render_path")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise HanddrawnRenderError("unsafe_render_path") from None
    if not path.is_dir() or _redirect_in_existing_chain(path):
        raise HanddrawnRenderError("unsafe_render_path")


def _redirect_in_existing_chain(path: Path) -> bool:
    candidate = Path(path)
    while True:
        if candidate.exists() or candidate.is_symlink():
            try:
                attributes = candidate.stat(follow_symlinks=False).st_file_attributes
            except (AttributeError, OSError):
                attributes = 0
            marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if candidate.is_symlink() or attributes & marker:
                return True
        if candidate.parent == candidate:
            return False
        candidate = candidate.parent


def _remove_owned_file(path: Path) -> None:
    if _redirect_in_existing_chain(path):
        return
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _remove_owned_directory(path: Path | None) -> None:
    if (
        path is None
        or not path.name.startswith(".render-public-")
        or _redirect_in_existing_chain(path)
    ):
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
    except OSError:
        pass


def _remove_if_hash(path: Path, expected_hash: str) -> None:
    if not expected_hash or _redirect_in_existing_chain(path):
        return
    try:
        if path.is_file() and sha256_file(path) == expected_hash:
            path.unlink()
    except OSError:
        pass


def _fsync_file(path: Path) -> None:
    try:
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
    except OSError:
        raise HanddrawnRenderError("render_publish_failed") from None
