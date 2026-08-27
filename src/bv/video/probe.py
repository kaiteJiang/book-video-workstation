"""Bounded, local-only media facts for manually supplied H3 clips."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from bv.core.process import CommandResult, run_command


_MAX_PROBE_OUTPUT_BYTES = 256 * 1024
_DEFAULT_TIMEOUT_SECONDS = 10.0


class ProbeError(RuntimeError):
    """A stable probe failure that never reveals media paths or probe output."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class MediaProbeFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: float = Field(gt=0)
    video_codec: str = Field(min_length=1)
    pixel_format: str | None = None
    audio_present: bool
    audio_codec: str | None = None


ProbeRunner = Callable[[list[str]], CommandResult]


def probe_media(
    media_path: object,
    *,
    runner: Callable[..., CommandResult] | None = None,
    ffprobe_command: str | Path = "ffprobe",
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> MediaProbeFacts:
    """Run FFprobe with a finite argv and accept only bounded, useful facts."""
    if not isinstance(ffprobe_command, (str, Path)) or not str(ffprobe_command) or timeout_seconds <= 0:
        raise ProbeError("invalid_probe_configuration")
    command = str(ffprobe_command)
    argv = [
        command, "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate",
        "-of", "json", str(media_path),
    ]
    execute = runner or _default_runner
    try:
        result = execute(argv, timeout_seconds=timeout_seconds, output_limit=_MAX_PROBE_OUTPUT_BYTES)
    except TypeError:
        try:
            result = execute(argv)
        except Exception:
            raise ProbeError("probe_execution_failed") from None
    except Exception:
        raise ProbeError("probe_execution_failed") from None
    if not isinstance(result, CommandResult):
        raise ProbeError("probe_execution_failed")
    if result.returncode != 0:
        raise ProbeError("probe_failed")
    encoded = result.stdout.encode("utf-8", errors="replace")
    if len(encoded) > _MAX_PROBE_OUTPUT_BYTES:
        raise ProbeError("probe_output_too_large")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError):
        raise ProbeError("invalid_probe_output") from None
    return _parse_facts(payload)


def _default_runner(
    argv: list[str], *, timeout_seconds: float, output_limit: int,
) -> CommandResult:
    """Use the project's shell-free process helper and truncate untrusted diagnostics."""
    result = run_command(argv, timeout=timeout_seconds)
    return CommandResult(
        argv=result.argv,
        returncode=result.returncode,
        stdout=result.stdout[: output_limit + 1],
        stderr=result.stderr[: output_limit + 1],
    )


def _parse_facts(payload: object) -> MediaProbeFacts:
    if not isinstance(payload, dict):
        raise ProbeError("invalid_probe_output")
    streams = payload.get("streams")
    format_data = payload.get("format")
    if not isinstance(streams, list) or not isinstance(format_data, dict):
        raise ProbeError("invalid_probe_output")
    video_streams = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"]
    if len(video_streams) != 1:
        raise ProbeError("no_decodable_video")
    video = video_streams[0]
    try:
        duration_seconds = float(format_data["duration"])
        width = _positive_int(video["width"])
        height = _positive_int(video["height"])
        frame_rate = _frame_rate(video["r_frame_rate"])
        codec = _codec(video["codec_name"])
        pixel_format = _optional_codec(video.get("pix_fmt"))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise ProbeError("invalid_probe_facts") from None
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ProbeError("invalid_probe_facts")
    duration_ms = round(duration_seconds * 1_000)
    if duration_ms <= 0:
        raise ProbeError("invalid_probe_facts")
    audio = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"), None)
    audio_codec = None
    if audio is not None:
        try:
            audio_codec = _codec(audio.get("codec_name"))
        except ValueError:
            raise ProbeError("invalid_probe_facts") from None
    return MediaProbeFacts(
        duration_ms=duration_ms, width=width, height=height, frame_rate=frame_rate,
        video_codec=codec, pixel_format=pixel_format,
        audio_present=audio is not None, audio_codec=audio_codec,
    )


def _positive_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError
    result = int(value)
    if result <= 0 or str(result) != str(value):
        raise ValueError
    return result


def _frame_rate(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError
    numerator, separator, denominator = value.partition("/")
    if not separator:
        result = float(numerator)
    else:
        result = float(numerator) / float(denominator)
    if not math.isfinite(result) or result <= 0:
        raise ValueError
    return result


def _codec(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError
    return value.strip()


def _optional_codec(value: object) -> str | None:
    if value is None:
        return None
    return _codec(value)
