"""Local-only, identity-bound multi-clip final-video rendering."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from bv.core.hashing import sha256_file
from bv.core.process import CommandResult, run_command
from bv.subtitles.generate import SubtitleCue, SubtitleManifest
from bv.video.cover import CoverManifest
from bv.voice.processing import VoiceManifest, inspect_pcm_wav
from bv.video.probe import ProbeError, probe_media


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TIMEOUT_SECONDS = 900.0
_DIAGNOSTIC_LIMIT = 64 * 1024


class RenderError(RuntimeError):
    """Stable privacy-safe renderer error; never includes local media paths."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class SegmentRenderInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    segment_id: str
    media_path: Path
    media_sha256: str
    prompt_sha256: str
    storyboard_sha256: str
    semantic_lock_sha256: str
    script_sha256: str
    audio_sha256: str
    subtitle_sha256: str
    expected_duration_ms: int = Field(ge=1, le=15_000)
    source_duration_ms: int = Field(ge=1, le=15_000)
    warning_codes: tuple[str, ...] = ()


class CoverOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    cover_path: Path
    cover_sha256: str
    manifest: CoverManifest
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    overlay_zone_text: str = Field(min_length=1, max_length=256)
    overlay_zone_sha256: str
    x: int = Field(ge=0, le=1079)
    y: int = Field(ge=0, le=1919)
    width: int = Field(gt=0, le=1080)
    height: int = Field(gt=0, le=1920)


class RenderInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    book_id: str
    episode_id: str
    segments: tuple[SegmentRenderInput, ...]
    voice_master_path: Path
    voice_master_sha256: str
    voice_manifest: VoiceManifest
    master_duration_ms: int = Field(ge=45_000, le=60_000)
    ass_path: Path
    ass_sha256: str
    asr_sha256: str
    subtitle_manifest: SubtitleManifest
    subtitle_cues: tuple[SubtitleCue, ...]
    cover: CoverOverlay
    render_root: Path
    final_path: Path


class HanddrawnRenderInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    book_id: str
    episode_id: str
    visual_path: Path
    visual_sha256: str
    storyboard_sha256: str
    semantic_lock_sha256: str
    script_sha256: str
    audio_sha256: str
    subtitle_sha256: str
    voice_master_path: Path
    voice_master_sha256: str
    voice_manifest: VoiceManifest
    master_duration_ms: int = Field(ge=10_000, le=150_000)
    production_profile_sha256: str | None = None
    ass_path: Path
    ass_sha256: str
    asr_sha256: str
    subtitle_manifest: SubtitleManifest
    subtitle_cues: tuple[SubtitleCue, ...]
    cover: CoverOverlay | None = None
    render_root: Path
    final_path: Path


class HanddrawnRenderPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    visual_duration_ms: int


class HanddrawnRenderManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    visual_sha256: str
    voice_master_sha256: str
    ass_sha256: str
    cover_sha256: str | None = None
    output_sha256: str
    master_duration_ms: int
    storyboard_sha256: str
    semantic_lock_sha256: str
    script_sha256: str
    audio_sha256: str
    subtitle_sha256: str
    asr_sha256: str
    production_profile_sha256: str | None = None


class HanddrawnRenderResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    final_path: Path
    manifest_path: Path
    manifest: HanddrawnRenderManifest


class NormalizedSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    segment_id: str
    input_sha256: str
    output_name: str
    output_sha256: str
    duration_ms: int
    prompt_sha256: str


class RenderPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    normalized_paths: tuple[Path, ...]
    total_duration_ms: int


class ConcatManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    path: Path
    sha256: str


class RenderManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_ids: tuple[str, ...]
    normalized: tuple[NormalizedSegment, ...]
    voice_master_sha256: str
    ass_sha256: str
    cover_sha256: str
    output_sha256: str
    master_duration_ms: int
    storyboard_sha256: str
    semantic_lock_sha256: str
    script_sha256: str
    audio_sha256: str
    subtitle_sha256: str
    asr_sha256: str


class RenderResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    final_path: Path
    manifest_path: Path
    manifest: RenderManifest


Runner = Callable[..., CommandResult]


def preflight_handdrawn_render(
    inputs: HanddrawnRenderInputs,
    *,
    visual_inspector: Callable[..., object] = probe_media,
    ffprobe_command: str | Path = "ffprobe",
) -> HanddrawnRenderPreflight:
    """Validate one silent 9:16 hand-drawn visual against the final audio clock."""
    _require_identifier(inputs.book_id)
    _require_identifier(inputs.episode_id)
    for value in (
        inputs.visual_sha256,
        inputs.storyboard_sha256,
        inputs.semantic_lock_sha256,
        inputs.script_sha256,
        inputs.audio_sha256,
        inputs.subtitle_sha256,
        inputs.voice_master_sha256,
        inputs.ass_sha256,
        inputs.asr_sha256,
    ):
        _require_hash(value)
    required_files = [
        (inputs.visual_path, inputs.visual_sha256, ".mp4"),
        (inputs.voice_master_path, inputs.voice_master_sha256, ".wav"),
        (inputs.ass_path, inputs.ass_sha256, ".ass"),
    ]
    if inputs.cover is not None:
        required_files.append((inputs.cover.cover_path, inputs.cover.cover_sha256, None))
    for path, expected, suffix in required_files:
        _require_safe_file(path, suffix)
        if sha256_file(path) != expected:
            raise RenderError("input_hash_mismatch")
    if inputs.audio_sha256 != inputs.voice_master_sha256:
        raise RenderError("mixed_render_identity")
    _validate_voice_contract(inputs, inputs.script_sha256)
    _validate_subtitle_contract(inputs, inputs.script_sha256, inputs.subtitle_sha256)
    if inputs.cover is not None:
        _validate_handdrawn_cover(
            inputs.cover, inputs.master_duration_ms, inputs.book_id, inputs.episode_id
        )
    _validate_cues(inputs.subtitle_cues, inputs.master_duration_ms)
    try:
        facts = visual_inspector(inputs.visual_path, ffprobe_command=ffprobe_command)
    except Exception:
        raise RenderError("visual_media_invalid") from None
    duration = getattr(facts, "duration_ms", None)
    if not isinstance(duration, int) or abs(duration - inputs.master_duration_ms) > 33:
        raise RenderError("visual_duration_mismatch")
    if (
        getattr(facts, "width", None) != 1080
        or getattr(facts, "height", None) != 1920
        or abs(float(getattr(facts, "frame_rate", 0.0)) - 30.0) > 0.01
        or getattr(facts, "video_codec", None) not in {"h264", "avc1"}
        or getattr(facts, "audio_present", None) is not False
    ):
        raise RenderError("visual_media_invalid")
    _require_safe_target(inputs.render_root)
    _require_safe_target(inputs.final_path.parent)
    if (
        inputs.final_path.name != "final.mp4"
        or inputs.final_path.exists()
        or inputs.final_path.with_suffix(".render.json").exists()
        or _redirect_in_existing_chain(inputs.final_path)
        or _redirect_in_existing_chain(inputs.final_path.with_suffix(".render.json"))
    ):
        raise RenderError("output_already_exists")
    return HanddrawnRenderPreflight(visual_duration_ms=duration)


def build_handdrawn_final_argv(
    inputs: HanddrawnRenderInputs,
    temporary_path: Path,
    *,
    ffmpeg_command: str | Path = "ffmpeg",
) -> list[str]:
    """Build the shell-free final composition command with an optional product cover."""
    _require_safe_target(temporary_path.parent)
    cover = inputs.cover
    ass = _relative_filter_path(inputs.ass_path, inputs.render_root)
    argv = [
        str(ffmpeg_command), "-hide_banner", "-nostdin", "-n",
        "-i", str(inputs.visual_path),
        "-i", str(inputs.voice_master_path),
    ]
    if cover is None:
        filtergraph = f"[0:v]subtitles=filename={ass}[v]"
    else:
        argv.extend(["-loop", "1", "-i", str(cover.cover_path)])
        filtergraph = (
            f"[2:v]scale=w={cover.width}:h={cover.height}:force_original_aspect_ratio=decrease,"
            f"pad={cover.width}:{cover.height}:(ow-iw)/2:(oh-ih)/2[cover];"
            f"[0:v]subtitles=filename={ass}[sub];[sub][cover]overlay={cover.x}:{cover.y}:"
            f"enable='between(t,{cover.start_ms / 1000:.3f},{cover.end_ms / 1000:.3f})'[v]"
        )
    argv.extend([
        "-filter_complex", filtergraph,
        "-map", "[v]", "-map", "1:a:0",
        "-t", f"{inputs.master_duration_ms / 1000:.3f}",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-ac", "2", "-ar", "48000",
        "-movflags", "+faststart", "-preset", "medium", "-shortest",
        str(temporary_path),
    ])
    return argv


def render_handdrawn_final(
    inputs: HanddrawnRenderInputs,
    *,
    ffmpeg_command: str | Path = "ffmpeg",
    runner: Runner = run_command,
    ffprobe_command: str | Path = "ffprobe",
    timeout_seconds: float = 600.0,
    inspector: Callable[..., object] | None = None,
) -> HanddrawnRenderResult:
    """Compose one verified silent visual with narration, subtitles, and optional cover."""
    if not 0 < timeout_seconds <= _MAX_TIMEOUT_SECONDS:
        raise RenderError("invalid_render_configuration")
    preflight_handdrawn_render(inputs, ffprobe_command=ffprobe_command)
    root = inputs.render_root
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise RenderError("render_workspace_not_empty")
    _ensure_safe_directory(root)
    created: list[tuple[Path, str]] = []
    temporary: Path | None = None
    try:
        _ensure_safe_directory(inputs.final_path.parent)
        temporary = inputs.final_path.parent / f".{uuid.uuid4().hex}.tmp.mp4"
        _run(
            build_handdrawn_final_argv(
                inputs,
                temporary,
                ffmpeg_command=ffmpeg_command,
            ),
            root,
            runner,
            timeout_seconds,
            "final_render_failed",
        )
        output_hash = _record_created_file(temporary, ".mp4", created)
        _check_handdrawn_inputs_unchanged(inputs)
        manifest = HanddrawnRenderManifest(
            visual_sha256=inputs.visual_sha256,
            voice_master_sha256=inputs.voice_master_sha256,
            ass_sha256=inputs.ass_sha256,
            cover_sha256=inputs.cover.cover_sha256 if inputs.cover is not None else None,
            output_sha256=output_hash,
            master_duration_ms=inputs.master_duration_ms,
            storyboard_sha256=inputs.storyboard_sha256,
            semantic_lock_sha256=inputs.semantic_lock_sha256,
            script_sha256=inputs.script_sha256,
            audio_sha256=inputs.audio_sha256,
            subtitle_sha256=inputs.subtitle_sha256,
            asr_sha256=inputs.asr_sha256,
            production_profile_sha256=inputs.production_profile_sha256,
        )
        if inspector is None:
            from bv.video.qc import inspect_final
            try:
                inspect_final(
                    temporary,
                    inputs=inputs,
                    render_manifest=manifest,
                    ffprobe_command=ffprobe_command,
                )
            except Exception:
                raise RenderError("final_qc_failed") from None
        else:
            try:
                inspector(temporary, inputs=inputs, render_manifest=manifest)
            except Exception:
                raise RenderError("final_qc_failed") from None
        _check_handdrawn_inputs_unchanged(inputs)
        _require_file_hash(temporary, output_hash, "final_output_changed")
        manifest_path = inputs.final_path.with_suffix(".render.json")
        if (
            inputs.final_path.exists()
            or manifest_path.exists()
            or _redirect_in_existing_chain(inputs.final_path)
            or _redirect_in_existing_chain(manifest_path)
        ):
            raise RenderError("output_already_exists")
        os.link(temporary, inputs.final_path)
        _fsync_file(inputs.final_path)
        try:
            _atomic_new_text(manifest_path, manifest.model_dump_json(indent=None))
        except RenderError:
            _remove_if_hash(inputs.final_path, output_hash)
            raise
        _remove_if_hash(temporary, output_hash)
        return HanddrawnRenderResult(
            final_path=inputs.final_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    except RenderError:
        raise
    except OSError:
        raise RenderError("render_publish_failed") from None
    finally:
        for path, expected_hash in reversed(created):
            _remove_if_hash(path, expected_hash)
        try:
            if root.is_dir() and not _redirect_in_existing_chain(root) and not any(root.iterdir()):
                root.rmdir()
        except OSError:
            pass


def preflight_render(inputs: RenderInputs) -> RenderPreflight:
    """Validate the complete current identity graph without modifying media."""
    _require_identifier(inputs.book_id)
    _require_identifier(inputs.episode_id)
    values = tuple(inputs.segments)
    expected_ids = tuple(f"S{index:02d}" for index in range(1, len(values) + 1))
    if len(values) not in {4, 5} or tuple(item.segment_id for item in values) != expected_ids:
        raise RenderError("invalid_segment_set")
    if any(not _is_sha256(getattr(item, field)) for item in values for field in _SEGMENT_HASHES):
        raise RenderError("invalid_render_identity")
    shared = ("storyboard_sha256", "semantic_lock_sha256", "script_sha256", "audio_sha256", "subtitle_sha256")
    if any(len({getattr(item, field) for item in values}) != 1 for field in shared):
        raise RenderError("mixed_render_identity")
    if len({item.prompt_sha256 for item in values}) != len(values):
        raise RenderError("invalid_render_identity")
    total = sum(item.expected_duration_ms for item in values)
    if not 45_000 <= total <= 60_000 or abs(total - inputs.master_duration_ms) > 100:
        raise RenderError("invalid_render_duration")
    if any(item.audio_sha256 != inputs.voice_master_sha256 for item in values):
        raise RenderError("mixed_render_identity")
    _require_hash(inputs.voice_master_sha256)
    _require_hash(inputs.ass_sha256)
    for item in values:
        _require_safe_file(item.media_path, ".mp4")
        if sha256_file(item.media_path) != item.media_sha256:
            raise RenderError("input_hash_mismatch")
        shortfall = item.expected_duration_ms - item.source_duration_ms
        allowed = shortfall == 0 or (
            1 <= shortfall <= 300 and item.warning_codes == ("duration_shortfall_repair_required",)
        )
        if not allowed:
            raise RenderError("invalid_segment_duration")
    for path, expected, suffix in ((inputs.voice_master_path, inputs.voice_master_sha256, ".wav"), (inputs.ass_path, inputs.ass_sha256, ".ass"), (inputs.cover.cover_path, inputs.cover.cover_sha256, None)):
        _require_safe_file(path, suffix)
        if sha256_file(path) != expected:
            raise RenderError("input_hash_mismatch")
    _validate_voice_contract(inputs, values[0].script_sha256)
    _validate_subtitle_contract(inputs, values[0].script_sha256, values[0].subtitle_sha256)
    _validate_cover(
        inputs.cover, inputs.master_duration_ms, values[-1],
        book_id=inputs.book_id, episode_id=inputs.episode_id,
    )
    _validate_cues(inputs.subtitle_cues, inputs.master_duration_ms)
    _require_safe_target(inputs.render_root)
    _require_safe_target(inputs.final_path.parent)
    if (
        inputs.final_path.name != "final.mp4" or inputs.final_path.exists()
        or inputs.final_path.with_suffix(".render.json").exists()
        or _redirect_in_existing_chain(inputs.final_path)
        or _redirect_in_existing_chain(inputs.final_path.with_suffix(".render.json"))
    ):
        raise RenderError("output_already_exists")
    normalized = tuple(inputs.render_root / "normalized" / f"{item.segment_id}.mp4" for item in values)
    return RenderPreflight(normalized_paths=normalized, total_duration_ms=total)


_SEGMENT_HASHES = (
    "media_sha256", "prompt_sha256", "storyboard_sha256", "semantic_lock_sha256", "script_sha256", "audio_sha256", "subtitle_sha256",
)


def build_segment_normalize_argv(segment: SegmentRenderInput, output_path: Path, *, ffmpeg_command: str | Path = "ffmpeg") -> list[str]:
    """Return the shell-free per-segment normalization command."""
    _require_safe_file(segment.media_path, ".mp4")
    _require_safe_target(output_path.parent)
    shortfall = segment.expected_duration_ms - segment.source_duration_ms
    if shortfall < 0 or shortfall > 300 or (shortfall and segment.warning_codes != ("duration_shortfall_repair_required",)):
        raise RenderError("invalid_segment_duration")
    filters = ["scale=1080:1920:force_original_aspect_ratio=increase", "crop=1080:1920", "fps=30", "setsar=1"]
    if shortfall:
        filters.append(f"tpad=stop_mode=clone:stop_duration={shortfall / 1000:.3f}")
    filters.extend((f"trim=duration={segment.expected_duration_ms / 1000:.3f}", "setpts=PTS-STARTPTS"))
    return [
        str(ffmpeg_command), "-hide_banner", "-nostdin", "-n", "-i", str(segment.media_path),
        "-map", "0:v:0", "-an", "-vf", ",".join(filters), "-r", "30", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-movflags", "+faststart", "-metadata:s:v:0", "rotate=0", str(output_path),
    ]


def build_concat_manifest(paths: Sequence[Path], manifest_path: Path) -> ConcatManifest:
    """Write a fresh UTF-8 FFmpeg concat list in the supplied exact order."""
    values = tuple(Path(path) for path in paths)
    if not values:
        raise RenderError("invalid_segment_set")
    _require_safe_target(Path(manifest_path).parent)
    lines: list[str] = []
    for path in values:
        _require_safe_file(path, ".mp4")
        raw = str(path)
        if any(ord(char) < 32 for char in raw) or "\n" in raw or "\r" in raw:
            raise RenderError("unsafe_render_path")
        lines.append("file '" + raw.replace("\\", "/").replace("'", "'\\''") + "'")
    target = Path(manifest_path)
    if target.exists() or _redirect_in_existing_chain(target):
        raise RenderError("output_already_exists")
    _atomic_new_text(target, "\n".join(lines) + "\n")
    return ConcatManifest(path=target, sha256=sha256_file(target))


def build_render_argv(inputs: RenderInputs, concat_path: Path, temporary_path: Path, *, ffmpeg_command: str | Path = "ffmpeg") -> list[str]:
    """Return the final delivery command using only approved fixed-layout data."""
    if Path(concat_path).suffix.lower() != ".mp4" or any(ord(char) < 32 for char in str(concat_path)):
        raise RenderError("unsafe_render_path")
    _require_safe_target(temporary_path.parent)
    cover = inputs.cover
    ass = _filter_path(inputs.ass_path)
    cover_path = _filter_path(cover.cover_path)
    overlay = (
        f"[2:v]scale=w={cover.width}:h={cover.height}:force_original_aspect_ratio=decrease,"
        f"pad={cover.width}:{cover.height}:(ow-iw)/2:(oh-ih)/2[cover];"
        f"[0:v]subtitles=filename={ass}[sub];[sub][cover]overlay={cover.x}:{cover.y}:"
        f"enable='between(t,{cover.start_ms / 1000:.3f},{cover.end_ms / 1000:.3f})'[v]"
    )
    return [
        str(ffmpeg_command), "-hide_banner", "-nostdin", "-n", "-i", str(concat_path),
        "-i", str(inputs.voice_master_path), "-loop", "1", "-i", str(inputs.cover.cover_path), "-filter_complex", overlay,
        "-map", "[v]", "-map", "1:a:0", "-t", f"{inputs.master_duration_ms / 1000:.3f}",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-ac", "2", "-ar", "48000", "-movflags", "+faststart", "-preset", "medium", "-shortest", str(temporary_path),
    ]


def render_final(
    inputs: RenderInputs, *, ffmpeg_command: str | Path = "ffmpeg", runner: Runner = run_command,
    ffprobe_command: str | Path = "ffprobe", timeout_seconds: float = 600.0,
    inspector: Callable[..., object] | None = None,
) -> RenderResult:
    """Normalize, concat, render, inspect, then no-overwrite publish one MP4."""
    if not 0 < timeout_seconds <= _MAX_TIMEOUT_SECONDS:
        raise RenderError("invalid_render_configuration")
    preflight = preflight_render(inputs)
    root = inputs.render_root
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise RenderError("render_workspace_not_empty")
    _ensure_safe_directory(root)
    normalized_root = root / "normalized"
    _ensure_safe_directory(normalized_root)
    created: list[tuple[Path, str]] = []
    try:
        normalized: list[NormalizedSegment] = []
        for segment, target in zip(inputs.segments, preflight.normalized_paths, strict=True):
            _run(build_segment_normalize_argv(segment, target, ffmpeg_command=ffmpeg_command), root, runner, timeout_seconds, "segment_normalize_failed")
            target_hash = _record_created_file(target, ".mp4", created)
            _require_normalized_media(target, segment.expected_duration_ms, ffprobe_command)
            _check_inputs_unchanged(inputs)
            _require_file_hash(target, target_hash, "normalized_output_changed")
            normalized.append(NormalizedSegment(segment_id=segment.segment_id, input_sha256=segment.media_sha256, output_name=target.name, output_sha256=target_hash, duration_ms=segment.expected_duration_ms, prompt_sha256=segment.prompt_sha256))
        concat = build_concat_manifest(preflight.normalized_paths, root / "concat.txt")
        created.append((concat.path, concat.sha256))
        combined = root / "combined.mp4"
        _run([str(ffmpeg_command), "-hide_banner", "-nostdin", "-n", "-f", "concat", "-safe", "0", "-i", str(concat.path), "-map", "0:v:0", "-an", "-c", "copy", str(combined)], root, runner, timeout_seconds, "concat_failed")
        combined_hash = _record_created_file(combined, ".mp4", created)
        _require_normalized_media(combined, inputs.master_duration_ms, ffprobe_command)
        _require_file_hash(combined, combined_hash, "combined_output_changed")
        _ensure_safe_directory(inputs.final_path.parent)
        temporary = inputs.final_path.parent / f".{uuid.uuid4().hex}.tmp.mp4"
        _run(build_render_argv(inputs, combined, temporary, ffmpeg_command=ffmpeg_command), root, runner, timeout_seconds, "final_render_failed")
        temporary_hash = _record_created_file(temporary, ".mp4", created)
        _check_inputs_unchanged(inputs)
        output_hash = temporary_hash
        first = inputs.segments[0]
        manifest = RenderManifest(segment_ids=tuple(item.segment_id for item in inputs.segments), normalized=tuple(normalized), voice_master_sha256=inputs.voice_master_sha256, ass_sha256=inputs.ass_sha256, cover_sha256=inputs.cover.cover_sha256, output_sha256=output_hash, master_duration_ms=inputs.master_duration_ms, storyboard_sha256=first.storyboard_sha256, semantic_lock_sha256=first.semantic_lock_sha256, script_sha256=first.script_sha256, audio_sha256=first.audio_sha256, subtitle_sha256=first.subtitle_sha256, asr_sha256=inputs.asr_sha256)
        if inspector is None:
            from bv.video.qc import inspect_final
            try:
                inspect_final(
                    temporary, inputs=inputs, render_manifest=manifest,
                    ffprobe_command=ffprobe_command,
                )
            except Exception:
                raise RenderError("final_qc_failed") from None
        else:
            try:
                inspector(temporary, inputs=inputs, render_manifest=manifest)
            except Exception:
                raise RenderError("final_qc_failed") from None
        _check_inputs_unchanged(inputs)
        _require_file_hash(temporary, output_hash, "final_output_changed")
        if inputs.final_path.exists() or inputs.final_path.with_suffix(".render.json").exists() or _redirect_in_existing_chain(inputs.final_path):
            raise RenderError("output_already_exists")
        os.link(temporary, inputs.final_path)
        _fsync_file(inputs.final_path)
        try:
            _atomic_new_text(inputs.final_path.with_suffix(".render.json"), manifest.model_dump_json(indent=None))
        except RenderError:
            _remove_if_hash(inputs.final_path, output_hash)
            raise
        _remove_if_hash(temporary, output_hash)
        return RenderResult(final_path=inputs.final_path, manifest_path=inputs.final_path.with_suffix(".render.json"), manifest=manifest)
    except RenderError:
        raise
    except OSError:
        raise RenderError("render_publish_failed") from None
    finally:
        for path, expected_hash in reversed(created):
            _remove_if_hash(path, expected_hash)
        for directory in (normalized_root, root):
            try:
                if directory.is_dir() and not _redirect_in_existing_chain(directory) and not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                pass


def _validate_cover(
    cover: CoverOverlay, master_duration: int, final_segment: SegmentRenderInput,
    *, book_id: str, episode_id: str,
) -> None:
    _require_hash(cover.cover_sha256)
    _require_hash(cover.overlay_zone_sha256)
    if hashlib.sha256(cover.overlay_zone_text.encode("utf-8")).hexdigest() != cover.overlay_zone_sha256:
        raise RenderError("cover_overlay_identity_invalid")
    if (
        cover.manifest.copied_sha256 != cover.cover_sha256 or cover.manifest.source_sha256 != cover.cover_sha256
        or cover.manifest.book_id != book_id or cover.manifest.episode_id != episode_id
        or cover.manifest.source_type not in {"user_provided", "ebook_extracted", "official_product_image"}
        or cover.manifest.matches_product_version is not True or cover.manifest.width <= 0 or cover.manifest.height <= 0
    ):
        raise RenderError("cover_overlay_identity_invalid")
    final_start = master_duration - final_segment.expected_duration_ms
    if not (final_start <= cover.start_ms < cover.end_ms <= master_duration) or cover.x + cover.width > 1080 or cover.y + cover.height > 1920:
        raise RenderError("invalid_cover_overlay")


def _validate_handdrawn_cover(
    cover: CoverOverlay,
    master_duration: int,
    book_id: str,
    episode_id: str,
) -> None:
    _require_hash(cover.cover_sha256)
    _require_hash(cover.overlay_zone_sha256)
    if hashlib.sha256(cover.overlay_zone_text.encode("utf-8")).hexdigest() != cover.overlay_zone_sha256:
        raise RenderError("cover_overlay_identity_invalid")
    if (
        cover.manifest.copied_sha256 != cover.cover_sha256
        or cover.manifest.source_sha256 != cover.cover_sha256
        or cover.manifest.book_id != book_id
        or cover.manifest.episode_id != episode_id
        or cover.manifest.source_type not in {
            "user_provided", "ebook_extracted", "official_product_image"
        }
        or cover.manifest.matches_product_version is not True
        or cover.manifest.width <= 0
        or cover.manifest.height <= 0
    ):
        raise RenderError("cover_overlay_identity_invalid")
    final_window_start = max(0, master_duration - 15_000)
    if (
        not (final_window_start <= cover.start_ms < cover.end_ms <= master_duration)
        or cover.x + cover.width > 1080
        or cover.y + cover.height > 1920
    ):
        raise RenderError("invalid_cover_overlay")


def _validate_cues(cues: tuple[SubtitleCue, ...], duration: int) -> None:
    if (
        not cues or tuple(cue.index for cue in cues) != tuple(range(len(cues)))
        or any(cue.start_ms < 0 or cue.end_ms <= cue.start_ms or cue.end_ms > duration for cue in cues)
        or any(right.start_ms < left.end_ms for left, right in zip(cues, cues[1:]))
    ):
        raise RenderError("invalid_subtitle_timing")


def _validate_voice_contract(inputs: RenderInputs, script_sha256: str) -> None:
    manifest = inputs.voice_manifest
    try:
        facts = inspect_pcm_wav(inputs.voice_master_path)
    except Exception:
        raise RenderError("voice_contract_invalid") from None
    if (
        manifest.master_sha256 != inputs.voice_master_sha256 or manifest.script_sha256 != script_sha256
        or manifest.master_duration_ms != inputs.master_duration_ms or manifest.sample_rate != 48_000
        or manifest.channels != 1 or manifest.sample_width != 2
        or manifest.duration.level != "pass"
        or round(manifest.duration.seconds * 1000) != inputs.master_duration_ms
        or (facts.sample_rate, facts.channels, facts.sample_width, round(facts.duration_seconds * 1000)) != (48_000, 1, 2, inputs.master_duration_ms)
    ):
        raise RenderError("voice_contract_invalid")


def _validate_subtitle_contract(inputs: RenderInputs, script_sha256: str, subtitle_sha256: str) -> None:
    manifest = inputs.subtitle_manifest
    payload = [{"index": cue.index, "start_ms": cue.start_ms, "end_ms": cue.end_ms, "text": cue.text} for cue in inputs.subtitle_cues]
    cue_hash = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if (
        manifest.approved_sha256 != script_sha256 or manifest.audio_sha256 != inputs.voice_master_sha256
        or manifest.asr_sha256 != inputs.asr_sha256 or manifest.cue_content_sha256 != cue_hash
        or subtitle_sha256 != cue_hash or manifest.duration_ms != inputs.master_duration_ms
        or (manifest.width, manifest.height, manifest.cue_count) != (1080, 1920, len(inputs.subtitle_cues))
    ):
        raise RenderError("subtitle_contract_invalid")


def _check_inputs_unchanged(inputs: RenderInputs) -> None:
    for path, expected in [(item.media_path, item.media_sha256) for item in inputs.segments] + [(inputs.voice_master_path, inputs.voice_master_sha256), (inputs.ass_path, inputs.ass_sha256), (inputs.cover.cover_path, inputs.cover.cover_sha256)]:
        _require_safe_file(path, None)
        if sha256_file(path) != expected:
            raise RenderError("input_hash_changed")


def _check_handdrawn_inputs_unchanged(inputs: HanddrawnRenderInputs) -> None:
    required_files = [
        (inputs.visual_path, inputs.visual_sha256),
        (inputs.voice_master_path, inputs.voice_master_sha256),
        (inputs.ass_path, inputs.ass_sha256),
    ]
    if inputs.cover is not None:
        required_files.append((inputs.cover.cover_path, inputs.cover.cover_sha256))
    for path, expected in required_files:
        _require_safe_file(path, None)
        if sha256_file(path) != expected:
            raise RenderError("input_hash_changed")


def _require_normalized_media(
    path: Path, expected_duration_ms: int, ffprobe_command: str | Path,
) -> None:
    try:
        facts = probe_media(path, ffprobe_command=str(ffprobe_command))
    except ProbeError:
        raise RenderError("normalized_media_invalid") from None
    if abs(facts.duration_ms - expected_duration_ms) > 100:
        raise RenderError("normalized_duration_mismatch")
    if (
        (facts.width, facts.height) != (1080, 1920)
        or abs(facts.frame_rate - 30.0) > 0.01
        or facts.video_codec not in {"h264", "avc1"}
        or facts.audio_present
    ):
        raise RenderError("normalized_media_invalid")


def _record_created_file(
    path: Path, suffix: str | None, created: list[tuple[Path, str]],
) -> str:
    _require_safe_file(path, suffix)
    try:
        digest = sha256_file(path)
    except OSError:
        raise RenderError("render_output_invalid") from None
    created.append((path, digest))
    return digest


def _require_file_hash(path: Path, expected: str, error_code: str) -> None:
    try:
        _require_safe_file(path, None)
        if sha256_file(path) != expected:
            raise RenderError(error_code)
    except OSError:
        raise RenderError(error_code) from None


def _run(argv: list[str], cwd: Path, runner: Runner, timeout: float, error_code: str) -> None:
    try:
        result = runner(argv, cwd=cwd, timeout=timeout)
    except Exception:
        raise RenderError(error_code) from None
    if not isinstance(result, CommandResult) or result.returncode != 0:
        raise RenderError(error_code)


def _filter_path(path: Path) -> str:
    value = str(path).replace("\\", "/")
    if any(ord(char) < 32 for char in value):
        raise RenderError("unsafe_render_path")
    # FFmpeg parses a filter option and then the surrounding filtergraph. With
    # shell=False those two parsers still require two escapes for a drive colon
    # and three before a literal apostrophe; outer quotes would consume it.
    return value.replace(":", "\\\\:").replace("'", "\\\\\\'").replace(",", "\\,")


def _relative_filter_path(path: Path, cwd: Path) -> str:
    try:
        relative = Path(os.path.relpath(path, cwd))
    except (OSError, ValueError):
        raise RenderError("unsafe_render_path") from None
    if relative.is_absolute():
        raise RenderError("unsafe_render_path")
    return _filter_path(relative)


def _atomic_new_text(path: Path, text: str) -> None:
    _require_safe_target(path.parent)
    if path.exists() or _redirect_in_existing_chain(path):
        raise RenderError("output_already_exists")
    temporary: Path | None = None
    temporary_hash = ""
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_hash = sha256_file(temporary)
        os.link(temporary, path)
    except OSError:
        raise RenderError("manifest_write_failed") from None
    finally:
        _remove_if_hash(temporary, temporary_hash)


def _require_identifier(value: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise RenderError("unsafe_identifier")


def _require_hash(value: str) -> None:
    if not _is_sha256(value):
        raise RenderError("invalid_render_identity")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require_safe_file(path: Path, suffix: str | None) -> None:
    value = Path(path)
    if (suffix is not None and value.suffix.lower() != suffix) or not value.is_file() or _redirect_in_existing_chain(value):
        raise RenderError("unsafe_render_path")


def _require_safe_target(path: Path) -> None:
    if _redirect_in_existing_chain(path):
        raise RenderError("unsafe_render_path")


def _ensure_safe_directory(path: Path) -> None:
    _require_safe_target(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise RenderError("unsafe_render_path") from None
    if not path.is_dir() or _redirect_in_existing_chain(path):
        raise RenderError("unsafe_render_path")


def _redirect_in_existing_chain(path: Path) -> bool:
    candidate = Path(path)
    while True:
        if candidate.exists() or candidate.is_symlink():
            try:
                attributes = candidate.stat(follow_symlinks=False).st_file_attributes
            except (AttributeError, OSError):
                attributes = 0
            if candidate.is_symlink() or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                return True
        if candidate.parent == candidate:
            return False
        candidate = candidate.parent


def _remove_if_hash(path: Path | None, expected: str) -> None:
    if path is None or not expected or _redirect_in_existing_chain(path):
        return
    try:
        if path.is_file() and sha256_file(path) == expected:
            path.unlink()
    except OSError:
        pass


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())
