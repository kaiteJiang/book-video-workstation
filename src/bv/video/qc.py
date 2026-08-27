"""Bounded technical inspection for a completed local final MP4."""

from __future__ import annotations

import json
import hashlib
import math
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bv.core.hashing import sha256_file
from bv.core.process import CommandResult, run_command
from bv.production.profile import ProductionProfile, production_profile_sha256


_MAX_PROBE_BYTES = 256 * 1024


class QCError(RuntimeError):
    """Stable local technical-QC error without paths or raw tool output."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class FinalMediaFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str
    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: float = Field(gt=0)
    video_codec: str
    pixel_format: str
    audio_codec: str
    channels: int = Field(gt=0)
    sample_rate: int = Field(gt=0)
    rotation: int
    video_duration_ms: int | None = None
    audio_duration_ms: int | None = None


class VideoQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    facts: FinalMediaFacts
    aesthetic_review_required: Literal[True] = True
    loudness_status: str = "NOT_MEASURED"
    face_character_consistency: str = "NOT_PERFORMED"
    hand_artifact_check: str = "NOT_PERFORMED"
    generated_text_artifact_check: str = "NOT_PERFORMED"
    aesthetic_check: str = "NOT_PERFORMED"
    emotion_check: str = "NOT_PERFORMED"
    narrative_semantic_check: str = "NOT_PERFORMED"
    cover_product_visual_match: str = "NOT_PERFORMED"
    platform_policy_check: str = "NOT_PERFORMED"
    production_profile_sha256: str | None = None


def inspect_final(
    final_path: Path, *, inputs: object, render_manifest: object | None = None,
    production_profile: ProductionProfile | None = None,
    episode_root: Path | None = None,
    runner: Callable[..., CommandResult] | None = None, ffprobe_command: str | Path = "ffprobe", timeout_seconds: float = 15.0,
) -> VideoQualityReport:
    """Verify only measurable container/codec/timing/identity facts."""
    path = Path(final_path)
    if not path.is_file() or _redirect_in_existing_chain(path) or timeout_seconds <= 0:
        raise QCError("unsafe_final_path")
    argv = [str(ffprobe_command), "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,channels,sample_rate,duration:stream_tags=rotate:stream_side_data=rotation", "-of", "json", str(path)]
    execute = runner or _runner
    try:
        result = execute(argv, timeout_seconds=timeout_seconds, output_limit=_MAX_PROBE_BYTES)
    except Exception:
        raise QCError("ffprobe_execution_failed") from None
    if not isinstance(result, CommandResult) or result.returncode != 0:
        raise QCError("ffprobe_failed")
    if len(result.stdout.encode("utf-8", errors="replace")) > _MAX_PROBE_BYTES:
        raise QCError("ffprobe_output_too_large")
    try:
        payload = json.loads(result.stdout)
        facts = _facts(payload, _safe_sha256(path, "unsafe_final_path"))
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        raise QCError("invalid_ffprobe_output") from None
    _validate_contract(facts, inputs, render_manifest, production_profile)
    profile_hash = None
    if production_profile is not None:
        if episode_root is None:
            canonical = production_profile.model_dump_json(exclude_none=False)
            profile_hash = hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()
        else:
            profile_hash = production_profile_sha256(episode_root)
    return VideoQualityReport(
        facts=facts,
        production_profile_sha256=profile_hash,
    )


def _runner(argv: list[str], *, timeout_seconds: float, output_limit: int) -> CommandResult:
    result = run_command(argv, timeout=timeout_seconds)
    return CommandResult(argv=result.argv, returncode=result.returncode, stdout=result.stdout[: output_limit + 1], stderr=result.stderr[: output_limit + 1])


def _facts(payload: object, digest: str) -> FinalMediaFacts:
    if not isinstance(payload, dict) or not isinstance(payload.get("streams"), list) or not isinstance(payload.get("format"), dict):
        raise ValueError
    streams = payload["streams"]
    videos = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"]
    audios = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise QCError("unexpected_stream_layout")
    video, audio = videos[0], audios[0]
    duration = float(payload["format"]["duration"])
    rate = _rate(video["r_frame_rate"])
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(rate):
        raise ValueError
    rotation = _rotation(video)
    return FinalMediaFacts(sha256=digest, duration_ms=round(duration * 1000), width=_integer(video["width"]), height=_integer(video["height"]), frame_rate=rate, video_codec=_text(video["codec_name"]), pixel_format=_text(video["pix_fmt"]), audio_codec=_text(audio["codec_name"]), channels=_integer(audio["channels"]), sample_rate=_integer(audio["sample_rate"]), rotation=rotation, video_duration_ms=_optional_duration_ms(video.get("duration")), audio_duration_ms=_optional_duration_ms(audio.get("duration")))


def _validate_contract(
    facts: FinalMediaFacts,
    inputs: object,
    manifest: object | None,
    production_profile: ProductionProfile | None,
) -> None:
    master = getattr(inputs, "master_duration_ms", None)
    is_handdrawn = isinstance(getattr(inputs, "visual_path", None), Path)
    duration_tolerance_ms = 33 if is_handdrawn else 100
    if not isinstance(master, int) or abs(facts.duration_ms - master) > duration_tolerance_ms:
        raise QCError("duration_mismatch")
    if production_profile is not None:
        duration = facts.duration_ms / 1000
        if not (
            production_profile.duration.hard_min_seconds
            <= duration
            <= production_profile.duration.hard_max_seconds
        ):
            raise QCError("duration_profile_out_of_range")
        if facts.video_duration_ms is None or facts.audio_duration_ms is None:
            raise QCError("stream_duration_unavailable")
        if abs(facts.video_duration_ms - facts.audio_duration_ms) > round(1000 / 30):
            raise QCError("audio_video_end_mismatch")
    if (facts.width, facts.height) != (1080, 1920) or abs(facts.frame_rate - 30.0) > 0.01:
        raise QCError("video_geometry_invalid")
    if facts.video_codec not in {"h264", "avc1"} or facts.pixel_format != "yuv420p" or facts.audio_codec != "aac" or facts.channels != 2 or facts.sample_rate != 48_000 or facts.rotation != 0:
        raise QCError("delivery_format_invalid")
    expected = getattr(manifest, "output_sha256", None) if manifest is not None else None
    if expected is not None and facts.sha256 != expected:
        raise QCError("final_hash_mismatch")
    cover = getattr(inputs, "cover", None)
    if manifest is not None:
        input_cover_sha256 = getattr(cover, "cover_sha256", None)
        manifest_cover_sha256 = getattr(manifest, "cover_sha256", object())
        if manifest_cover_sha256 != input_cover_sha256:
            raise QCError("cover_identity_mismatch")
    for item in tuple(getattr(inputs, "segments", ())):
        if not _source_hash_matches(item.media_path, item.media_sha256):
            raise QCError("source_hash_mismatch")
    if is_handdrawn and not _source_hash_matches(
        getattr(inputs, "visual_path", None), getattr(inputs, "visual_sha256", None)
    ):
        raise QCError("source_hash_mismatch")
    source_files = [
        (getattr(inputs, "voice_master_path", None), getattr(inputs, "voice_master_sha256", None)),
        (getattr(inputs, "ass_path", None), getattr(inputs, "ass_sha256", None)),
    ]
    if cover is not None:
        source_files.append((getattr(cover, "cover_path", None), getattr(cover, "cover_sha256", None)))
    for path, expected_hash in source_files:
        if not _source_hash_matches(path, expected_hash):
            raise QCError("source_hash_mismatch")
    cues = tuple(getattr(inputs, "subtitle_cues", ()))
    if any(getattr(cue, "start_ms", -1) < 0 or getattr(cue, "end_ms", 0) <= getattr(cue, "start_ms", -1) or getattr(cue, "end_ms", 0) > facts.duration_ms for cue in cues):
        raise QCError("subtitle_timing_invalid")
    if master > facts.duration_ms + duration_tolerance_ms:
        raise QCError("voice_exceeds_final_duration")


def _integer(value: object) -> int:
    if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, (int, str)):
        raise ValueError
    result = int(value)
    if result <= 0 or str(result) != str(value):
        raise ValueError
    return result


def _optional_duration_ms(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError
    return round(number * 1000)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError
    return value


def _rate(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError
    numerator, slash, denominator = value.partition("/")
    return float(numerator) / float(denominator) if slash else float(numerator)


def _rotation(video: dict[str, object]) -> int:
    values: list[object] = []
    tags = video.get("tags")
    if isinstance(tags, dict):
        values.append(tags.get("rotate", 0))
    side_data = video.get("side_data_list")
    if isinstance(side_data, list):
        values.extend(item.get("rotation", 0) for item in side_data if isinstance(item, dict))
    if not values:
        return 0
    rotations: set[int] = set()
    for value in values:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError
        rotations.add(int(number))
    if len(rotations) != 1:
        raise ValueError
    return rotations.pop()


def _safe_sha256(path: Path, error_code: str) -> str:
    try:
        if not path.is_file() or _redirect_in_existing_chain(path):
            raise QCError(error_code)
        return sha256_file(path)
    except QCError:
        raise
    except OSError:
        raise QCError(error_code) from None


def _source_hash_matches(path: object, expected_hash: object) -> bool:
    if not isinstance(path, Path) or not isinstance(expected_hash, str):
        return False
    try:
        return path.is_file() and not _redirect_in_existing_chain(path) and sha256_file(path) == expected_hash
    except OSError:
        return False


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
