from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from bv.core.hashing import sha256_file
from bv.core.process import CommandResult, run_command
from bv.video.contracts import (
    VideoGenerationRequest,
    VideoGenerationResult,
    _is_contained_under,
    _is_redirected,
    _redirect_in_chain,
    _require_regular_file,
)
from bv.video.probe import ProbeError, probe_media

_MAX_RESPONSE_CHARS = 65_536
_MAX_OUTPUT_BYTES = 500 * 1024 * 1024
_AUTH_TOKEN_PHRASE = re.compile(
    r"(?:authorization|access|api)\s+token\s+(?:rejected|invalid|expired)"
)
_AUTH_PHRASES = (
    "not authenticated",
    "unauthenticated",
    "unauthorized",
    "not logged in",
    "credentials expired",
    "session expired",
)
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_VIDEO_SUFFIX = ".mp4"
_ASPECT = 9 / 16
_ASPECT_TOLERANCE = 0.01


class GrokVideoError(RuntimeError):
    """A safe, stable failure from the Grok video adapter."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _GrokVideoResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    segment_id: str
    reference_image_paths: tuple[Path, ...]
    video_path: Path


def build_grok_video_argv(
    prompt_path: Path,
    json_schema: str,
    request_dir: Path,
    command: str = "grok",
) -> list[str]:
    return [
        command,
        "--prompt-file",
        str(prompt_path),
        "--json-schema",
        json_schema,
        "--output-format",
        "json",
        "--tools",
        "image_gen,reference_to_video",
        "--permission-mode",
        "dontAsk",
        "--no-memory",
        "--no-subagents",
        "--disable-web-search",
        "--cwd",
        str(request_dir),
    ]


def _safe_absolute_path(value: Path) -> bool:
    raw = str(value)
    candidate = Path(raw)
    if not candidate.is_absolute():
        return False
    return ".." not in candidate.parts


def _load_payload(stdout: str) -> object:
    if len(stdout) > _MAX_RESPONSE_CHARS:
        raise GrokVideoError("video_output_invalid")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        raise GrokVideoError("video_output_invalid") from None
    if not isinstance(payload, dict):
        raise GrokVideoError("video_output_invalid")
    if "text" in payload:
        text = payload["text"]
        if not isinstance(text, str) or len(text) > _MAX_RESPONSE_CHARS:
            raise GrokVideoError("video_output_invalid")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            raise GrokVideoError("video_output_invalid") from None
    return payload


def parse_grok_video_response(
    stdout: str, *, expected_segment_id: str
) -> _GrokVideoResponse:
    payload = _load_payload(stdout)
    try:
        parsed = _GrokVideoResponse.model_validate(payload)
    except ValidationError:
        raise GrokVideoError("video_output_invalid") from None
    if parsed.segment_id != expected_segment_id:
        raise GrokVideoError("video_output_invalid")
    paths = (*parsed.reference_image_paths, parsed.video_path)
    if any(not _safe_absolute_path(path) for path in paths):
        raise GrokVideoError("video_output_invalid")
    return parsed


def map_grok_video_process_result(result: CommandResult) -> None:
    if result.returncode == 0:
        return None
    diagnostics = f"{result.stdout}\n{result.stderr}".casefold()
    if result.returncode == -9 or "timed out" in diagnostics:
        raise GrokVideoError("video_generation_timeout")
    if "authentication" in diagnostics or "login" in diagnostics:
        raise GrokVideoError("video_generation_auth_failed")
    if any(phrase in diagnostics for phrase in _AUTH_PHRASES):
        raise GrokVideoError("video_generation_auth_failed")
    if _AUTH_TOKEN_PHRASE.search(diagnostics):
        raise GrokVideoError("video_generation_auth_failed")
    raise GrokVideoError("video_generation_failed")


class GrokCliVideoProvider:
    def __init__(
        self,
        *,
        command: str = "grok",
        timeout_seconds: float = 900,
        ffprobe_command: str = "ffprobe",
        runner: Callable[..., CommandResult] | None = None,
        ffprobe_runner: Callable[..., CommandResult] | None = None,
    ) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.ffprobe_command = ffprobe_command
        self.runner = runner if runner is not None else run_command
        self.ffprobe_runner = ffprobe_runner

    def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        try:
            return self._generate(request)
        except GrokVideoError:
            raise
        except ValidationError:
            raise GrokVideoError("video_output_unsafe") from None
        except ProbeError:
            raise GrokVideoError("video_output_invalid") from None
        except Exception:
            raise GrokVideoError("video_generation_failed") from None

    def _generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        if request.provider != "grok_cli":
            raise GrokVideoError("video_generation_failed")
        request.verify_reference_images()

        request_dir = _create_request_directory(request.episode_root)
        reserved: set[Path] = set()
        input_paths = _snapshot_inputs(request, request_dir)
        reserved.update(input_paths)
        prompt_path = request_dir / "prompt.md"
        prompt_path.write_text(_build_prompt(request, input_paths), encoding="utf-8")
        reserved.add(prompt_path)
        prompt_sha256 = sha256_file(prompt_path)

        schema = json.dumps(
            _GrokVideoResponse.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
        )
        argv = build_grok_video_argv(
            prompt_path, schema, request_dir, command=self.command
        )
        request.verify_reference_images()
        result = self._run_grok(argv, request_dir, request.prompt)
        map_grok_video_process_result(result)
        _require_prompt_unchanged(prompt_path, prompt_sha256)
        _require_immutable_input_snapshots(input_paths, request.reference_image_sha256s)
        parsed = parse_grok_video_response(
            result.stdout, expected_segment_id=request.segment_id
        )
        generated_refs, video_path = _require_generated_outputs(
            parsed, request, request_dir, reserved
        )
        _require_request_tree(request_dir, reserved, (*generated_refs, video_path))
        return _publish_accepted(
            request_dir,
            request,
            generated_refs,
            video_path,
            self.ffprobe_runner,
            self.ffprobe_command,
        )

    def _run_grok(
        self, argv: list[str], request_dir: Path, prompt: str
    ) -> CommandResult:
        try:
            result = self.runner(
                argv,
                cwd=request_dir,
                timeout=self.timeout_seconds,
                secrets=(prompt,),
            )
        except GrokVideoError:
            raise
        except Exception:
            raise GrokVideoError("video_generation_failed") from None
        if not isinstance(result, CommandResult):
            raise GrokVideoError("video_generation_failed")
        return result


def _create_request_directory(episode_root: Path) -> Path:
    private_root = episode_root / ".private" / "grok-video-requests"
    if _redirect_in_chain(private_root):
        raise GrokVideoError("video_output_unsafe")
    private_root.mkdir(parents=True, exist_ok=True)
    if _redirect_in_chain(private_root):
        raise GrokVideoError("video_output_unsafe")
    request_dir = private_root / uuid.uuid4().hex
    request_dir.mkdir()
    if _redirect_in_chain(request_dir) or not _is_contained_under(request_dir, private_root):
        raise GrokVideoError("video_output_unsafe")
    return request_dir


def _snapshot_inputs(
    request: VideoGenerationRequest, request_dir: Path
) -> tuple[Path, ...]:
    if not request.reference_images:
        return ()
    inputs_dir = request_dir / "inputs"
    inputs_dir.mkdir()
    copied: list[Path] = []
    for index, (source, expected) in enumerate(
        zip(request.reference_images, request.reference_image_sha256s, strict=True)
    ):
        destination = inputs_dir / f"ref-{index}{Path(source).suffix}"
        _stream_copy(source, destination)
        if sha256_file(destination) != expected:
            raise GrokVideoError("video_output_unsafe")
        copied.append(destination)
    return tuple(copied)


def _build_prompt(request: VideoGenerationRequest, input_paths: tuple[Path, ...]) -> str:
    lines = [
        f"Segment {request.segment_id}",
        f"Duration {request.expected_duration_ms} milliseconds",
        f"Resolution {request.resolution}",
        "Aspect 9:16",
        "",
        request.prompt,
        "",
        "Create each of the following exactly once:",
    ]
    for spec in request.reference_image_specs:
        lines.append(f"- image_gen: {spec}")
    lines.extend(
        [
            "",
            "Then call reference_to_video exactly once using the generated images",
            "and any request-local input files listed below.",
        ]
    )
    if input_paths:
        lines.extend(
            [
                "",
                "Copied request-local images are identity and character-continuity",
                "anchors that each image action must use.",
            ]
        )
    for path in input_paths:
        lines.append(f"- input: {path.as_posix()}")
    lines.extend(
        [
            "",
            "Prohibitions: no dialogue, no lip-sync, no lip sync, no narration,",
            "no music, no subtitle, no readable text, no logo, no watermark.",
        ]
    )
    return "\n".join(lines) + "\n"


def _require_generated_outputs(
    parsed: _GrokVideoResponse,
    request: VideoGenerationRequest,
    request_dir: Path,
    reserved: set[Path],
) -> tuple[tuple[Path, ...], Path]:
    refs = tuple(Path(path) for path in parsed.reference_image_paths)
    video = Path(parsed.video_path)
    seen: set[Path] = set()
    for path in (*refs, video):
        if not _is_contained_under(path, request_dir) or ".." in path.parts:
            raise GrokVideoError("video_output_unsafe")
        if not path.exists() and not path.is_symlink():
            raise GrokVideoError("video_output_missing")
        try:
            _require_regular_file(path)
        except ValueError:
            raise GrokVideoError("video_output_unsafe") from None
        if path in reserved or path in seen:
            raise GrokVideoError("video_output_unsafe")
        seen.add(path)
    if len(refs) != len(request.reference_image_specs):
        raise GrokVideoError("video_output_invalid")
    for path in refs:
        if path.suffix.casefold() not in _IMAGE_SUFFIXES:
            raise GrokVideoError("video_output_invalid")
    if video.suffix.casefold() != _VIDEO_SUFFIX:
        raise GrokVideoError("video_output_invalid")
    for path in (*refs, video):
        try:
            size = path.stat().st_size
        except OSError:
            raise GrokVideoError("video_output_unsafe") from None
        if size > _MAX_OUTPUT_BYTES:
            raise GrokVideoError("video_output_unsafe")
    return refs, video


def _require_request_tree(
    request_dir: Path,
    reserved: set[Path],
    generated: tuple[Path, ...],
) -> None:
    allowed = {path for path in (*reserved, *generated)}
    for path in request_dir.rglob("*"):
        if _is_redirected(path) or _redirect_in_chain(path):
            raise GrokVideoError("video_output_unsafe")
        if path.is_dir():
            continue
        if path not in allowed:
            raise GrokVideoError("video_output_unsafe")


def _expected_width(resolution: str) -> int:
    if resolution == "480p":
        return 480
    if resolution == "720p":
        return 720
    raise GrokVideoError("video_output_invalid")


def _aspect_ok(width: int, height: int) -> bool:
    if width <= 0 or height <= width:
        return False
    ratio = width / height
    if abs(ratio - _ASPECT) <= _ASPECT_TOLERANCE:
        return True
    expected = width * 16 / 9
    return abs(height - expected) < 1.0


def _duration_ok(duration_ms: int, expected_ms: int) -> bool:
    return expected_ms - 300 <= duration_ms <= expected_ms + 50


def _file_size_and_hash(path: Path) -> tuple[int, str]:
    try:
        _require_regular_file(path)
    except ValueError:
        raise GrokVideoError("video_output_unsafe") from None
    try:
        size = path.stat().st_size
    except OSError:
        raise GrokVideoError("video_output_unsafe") from None
    if size > _MAX_OUTPUT_BYTES:
        raise GrokVideoError("video_output_unsafe")
    return size, sha256_file(path)


def _require_prompt_unchanged(prompt_path: Path, expected_sha256: str) -> None:
    try:
        _require_regular_file(prompt_path)
    except ValueError:
        raise GrokVideoError("video_output_unsafe") from None
    if sha256_file(prompt_path) != expected_sha256:
        raise GrokVideoError("video_output_unsafe")


def _require_probe(
    video_path: Path,
    request: VideoGenerationRequest,
    runner: Callable[..., CommandResult] | None,
    ffprobe_command: str,
) -> None:
    facts = probe_media(
        video_path,
        runner=runner,
        ffprobe_command=ffprobe_command,
    )
    if facts.width != _expected_width(request.resolution):
        raise GrokVideoError("video_output_invalid")
    if not _aspect_ok(facts.width, facts.height):
        raise GrokVideoError("video_output_invalid")
    if not _duration_ok(facts.duration_ms, request.expected_duration_ms):
        raise GrokVideoError("video_output_invalid")


def _require_immutable_input_snapshots(
    snapshots: tuple[Path, ...], expected_hashes: tuple[str, ...]
) -> None:
    if len(snapshots) != len(expected_hashes):
        raise GrokVideoError("video_output_unsafe")
    for snapshot, expected in zip(snapshots, expected_hashes, strict=True):
        if sha256_file(snapshot) != expected:
            raise GrokVideoError("video_output_unsafe")


def _copy_raw_once(source: Path, destination: Path) -> tuple[int, str]:
    before = _file_size_and_hash(source)
    _stream_copy(source, destination)
    after = _file_size_and_hash(source)
    staged = _file_size_and_hash(destination)
    if before != after or after != staged:
        raise GrokVideoError("video_output_unsafe")
    return staged


def _require_staging_unchanged(
    paths: tuple[Path, ...], expected: tuple[tuple[int, str], ...]
) -> None:
    for path, snapshot in zip(paths, expected, strict=True):
        if _file_size_and_hash(path) != snapshot:
            raise GrokVideoError("video_output_unsafe")


def _publish_accepted(
    request_dir: Path,
    request: VideoGenerationRequest,
    refs: tuple[Path, ...],
    video: Path,
    ffprobe_runner: Callable[..., CommandResult] | None,
    ffprobe_command: str,
) -> VideoGenerationResult:
    accepted = request_dir / "accepted"
    if accepted.exists() or accepted.is_symlink():
        raise GrokVideoError("video_output_unsafe")
    staging = request_dir / f"publish-{uuid.uuid4().hex}"
    staging.mkdir()
    copied_refs: list[Path] = []
    ref_snapshots: list[tuple[int, str]] = []
    for index, source in enumerate(refs):
        destination = staging / f"ref-{index}{source.suffix}"
        snapshot = _copy_raw_once(source, destination)
        copied_refs.append(destination)
        ref_snapshots.append(snapshot)
    staging_video = staging / "video.mp4"
    video_snapshot = _copy_raw_once(video, staging_video)
    staged_paths = (*copied_refs, staging_video)
    staged_snapshots = (*ref_snapshots, video_snapshot)
    provisional = VideoGenerationResult(
        provider="grok_cli",
        segment_id=request.segment_id,
        source_path=staging_video,
        source_sha256=video_snapshot[1],
        reference_images=tuple(copied_refs),
        reference_image_sha256s=tuple(item[1] for item in ref_snapshots),
    )
    _require_probe(
        staging_video,
        request,
        ffprobe_runner,
        ffprobe_command,
    )
    _require_staging_unchanged(staged_paths, staged_snapshots)
    if accepted.exists() or accepted.is_symlink():
        raise GrokVideoError("video_output_unsafe")
    staging.rename(accepted)
    accepted_refs = tuple(accepted / path.name for path in copied_refs)
    accepted_video = accepted / staging_video.name
    return VideoGenerationResult(
        provider="grok_cli",
        segment_id=request.segment_id,
        source_path=accepted_video,
        source_sha256=provisional.source_sha256,
        reference_images=accepted_refs,
        reference_image_sha256s=provisional.reference_image_sha256s,
    )


def _stream_copy(source: Path, destination: Path) -> None:
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
