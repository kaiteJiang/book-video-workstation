"""Copy-only, identity-bound import of manual H3 segment media."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bv.core.process import CommandResult
from bv.storyboard.generate import H3PromptArtifact
from bv.video.contracts import VideoProviderName
from bv.video.probe import MediaProbeFacts, ProbeError, probe_media


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DURATION_METADATA_TOLERANCE_MS = 50
_MAX_SHORTFALL_MS = 300
_MAX_H3_SOURCE_BYTES = 500 * 1024 * 1024
_ASPECT_RATIO = 9 / 16
_ASPECT_TOLERANCE = 0.03


class H3ImportError(RuntimeError):
    """A stable local-import error that never includes source paths."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class RequiredSegmentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_id: str
    prompt_sha256: str
    storyboard_sha256: str
    semantic_lock_sha256: str
    approved_script_sha256: str
    audio_sha256: str
    subtitle_sha256: str
    expected_duration_ms: int


class H3SegmentManifest(RequiredSegmentIdentity):
    provider: VideoProviderName = "h3_manual"
    source_sha256: str
    copied_sha256: str
    probe: MediaProbeFacts
    warning_codes: tuple[str, ...]
    imported_at: str
    event_id: str


class ImportedSegmentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_id: str
    source_sha256: str
    copied_sha256: str
    manifest_sha256: str


class H3AggregateManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: VideoProviderName = "h3_manual"
    book_id: str
    episode_id: str
    required_segments: tuple[RequiredSegmentIdentity, ...]
    imported_segments: tuple[ImportedSegmentIdentity, ...]
    present_segment_ids: tuple[str, ...]
    missing_segment_ids: tuple[str, ...]
    status: str


class H3ImportResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    provider: VideoProviderName
    status: str
    present_segment_ids: tuple[str, ...]
    missing_segment_ids: tuple[str, ...]
    manifest_path: Path
    aggregate_manifest_path: Path
    idempotent: bool


class H3SegmentSetStatus(BaseModel):
    """Read-only, freshly revalidated status for the current Task 18 prompt set."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    provider: VideoProviderName
    status: str
    present_segment_ids: tuple[str, ...]
    missing_segment_ids: tuple[str, ...]
    aggregate_manifest_path: Path


VideoImportError = H3ImportError
VideoSegmentManifest = H3SegmentManifest
VideoAggregateManifest = H3AggregateManifest
VideoImportResult = H3ImportResult
VideoSegmentSetStatus = H3SegmentSetStatus


def _status_from_ids(present: Sequence[str], missing: Sequence[str]) -> str:
    if not present:
        return "awaiting_video_generation"
    if not missing:
        return "video_imported"
    if tuple(present) == ("S01",):
        return "visual_sample_ready"
    return "video_partial"


def inspect_video_segments(
    *,
    book_id: str,
    episode_id: str,
    prompt_artifacts: Sequence[H3PromptArtifact],
    episode_media_root: Path,
    provider: VideoProviderName,
) -> H3SegmentSetStatus:
    """Revalidate current segment media/manifests without importing or writing."""
    _require_provider(provider)
    _require_identifier(book_id)
    _require_identifier(episode_id)
    required = _required_segments(prompt_artifacts)
    root = Path(episode_media_root)
    h3_root = root / "h3"
    _require_child(root, h3_root)
    if _redirect_in_existing_chain(root):
        raise H3ImportError("unsafe_h3_path")
    existing = _collect_existing(
        h3_root=h3_root, required=required, book_id=book_id, episode_id=episode_id,
        provider=provider,
    )
    required_ids = tuple(item.segment_id for item in required)
    present = tuple(item for item in required_ids if item in existing)
    missing = tuple(item for item in required_ids if item not in existing)
    return H3SegmentSetStatus(
        provider=provider,
        status=_status_from_ids(present, missing),
        present_segment_ids=present,
        missing_segment_ids=missing,
        aggregate_manifest_path=h3_root / "manifest.json",
    )


def inspect_h3_segments(
    *,
    book_id: str,
    episode_id: str,
    prompt_artifacts: Sequence[H3PromptArtifact],
    episode_media_root: Path,
) -> H3SegmentSetStatus:
    """Revalidate current segment media/manifests without importing or writing."""
    return inspect_video_segments(
        book_id=book_id,
        episode_id=episode_id,
        prompt_artifacts=prompt_artifacts,
        episode_media_root=episode_media_root,
        provider="h3_manual",
    )


def all_required_video_segments_present(
    *,
    book_id: str,
    episode_id: str,
    prompt_artifacts: Sequence[H3PromptArtifact],
    episode_media_root: Path,
    provider: VideoProviderName,
) -> bool:
    """Return true only after read-only validation of every current required segment."""
    _require_provider(provider)
    return inspect_video_segments(
        book_id=book_id,
        episode_id=episode_id,
        prompt_artifacts=prompt_artifacts,
        episode_media_root=episode_media_root,
        provider=provider,
    ).status == "video_imported"


def all_required_segments_present(
    *,
    book_id: str,
    episode_id: str,
    prompt_artifacts: Sequence[H3PromptArtifact],
    episode_media_root: Path,
) -> bool:
    """Return true only after read-only validation of every current required segment."""
    return all_required_video_segments_present(
        book_id=book_id,
        episode_id=episode_id,
        prompt_artifacts=prompt_artifacts,
        episode_media_root=episode_media_root,
        provider="h3_manual",
    )


def import_video_segment(
    *,
    book_id: str,
    episode_id: str,
    segment_id: str,
    prompt_artifacts: Sequence[H3PromptArtifact],
    source_path: Path,
    episode_media_root: Path,
    provider: VideoProviderName,
    ffprobe_runner: Callable[..., CommandResult] | None = None,
) -> H3ImportResult:
    """Validate one current Task 18 segment, snapshot it, then publish atomically."""
    _require_provider(provider)
    _require_identifier(book_id)
    _require_identifier(episode_id)
    required = _required_segments(prompt_artifacts)
    if segment_id not in {item.segment_id for item in required}:
        raise H3ImportError("invalid_segment_id")
    source = Path(source_path)
    root = Path(episode_media_root)
    _require_safe_source(source)
    snapshot, source_sha = _snapshot_source(source)
    try:
        try:
            facts = probe_media(snapshot, runner=ffprobe_runner)
        except ProbeError as error:
            raise H3ImportError(error.error_code) from None
        target_identity = next(item for item in required if item.segment_id == segment_id)
        warnings = _validate_probe(target_identity.expected_duration_ms, facts)
        if _sha256_file(source) != source_sha or _redirect_in_existing_chain(source):
            raise H3ImportError("source_changed_during_import")
        h3_root = root / "h3"
        _require_child(root, h3_root)
        existing = _collect_existing(
            h3_root=h3_root, required=required, book_id=book_id, episode_id=episode_id,
            provider=provider,
        )
        existing_manifest = existing.get(segment_id)
        if existing_manifest is not None:
            if existing_manifest.source_sha256 != source_sha:
                raise H3ImportError("existing_segment_mismatch")
            return _result(
                h3_root=h3_root, required=required, existing=existing, segment_id=segment_id,
                book_id=book_id, episode_id=episode_id, provider=provider, idempotent=True,
            )
        _ensure_safe_directory(h3_root)
        segment_root = h3_root / segment_id
        _require_child(root, segment_root)
        _ensure_safe_directory(segment_root)
        target = _publish_snapshot(snapshot, segment_root, source_sha)
        copied_sha = _sha256_file(target)
        manifest = H3SegmentManifest(
            **target_identity.model_dump(), provider=provider,
            source_sha256=source_sha, copied_sha256=copied_sha,
            probe=facts, warning_codes=warnings,
            imported_at=datetime.now(timezone.utc).isoformat(), event_id=str(uuid.uuid4()),
        )
        manifest_path = segment_root / "manifest.json"
        try:
            _write_new_manifest(manifest_path, manifest)
        except H3ImportError:
            _rollback_new_segment(segment_root, target, manifest_path, source_sha)
            raise
        existing = _collect_existing(
            h3_root=h3_root, required=required, book_id=book_id, episode_id=episode_id,
            provider=provider, allow_missing_aggregate=True, allow_stale_aggregate=True,
        )
        try:
            return _result(
                h3_root=h3_root, required=required, existing=existing, segment_id=segment_id,
                book_id=book_id, episode_id=episode_id, provider=provider, idempotent=False,
            )
        except H3ImportError:
            _rollback_new_segment(segment_root, target, manifest_path, source_sha)
            raise
    finally:
        _remove_temp(snapshot)


def import_h3_segment(
    *,
    book_id: str,
    episode_id: str,
    segment_id: str,
    prompt_artifacts: Sequence[H3PromptArtifact],
    source_path: Path,
    episode_media_root: Path,
    ffprobe_runner: Callable[..., CommandResult] | None = None,
) -> H3ImportResult:
    """Validate one current Task 18 segment, snapshot it, then publish atomically."""
    return import_video_segment(
        book_id=book_id,
        episode_id=episode_id,
        segment_id=segment_id,
        prompt_artifacts=prompt_artifacts,
        source_path=source_path,
        episode_media_root=episode_media_root,
        provider="h3_manual",
        ffprobe_runner=ffprobe_runner,
    )


def _required_segments(artifacts: Sequence[H3PromptArtifact]) -> tuple[RequiredSegmentIdentity, ...]:
    values = tuple(artifacts)
    expected_ids = tuple(f"S{index:02d}" for index in range(1, len(values) + 1))
    if len(values) not in {4, 5} or tuple(item.segment_id for item in values) != expected_ids:
        raise H3ImportError("invalid_prompt_artifacts")
    if any(
        item.shot_id != item.segment_id or not 1 <= item.required_duration_ms <= 15_000
        or hashlib.sha256(item.prompt.encode("utf-8")).hexdigest() != item.prompt_sha256
        for item in values
    ):
        raise H3ImportError("invalid_prompt_artifacts")
    fields = (
        "prompt_sha256", "storyboard_sha256", "semantic_lock_sha256", "approved_script_sha256",
        "audio_sha256", "subtitle_sha256",
    )
    if any(not all(_is_sha256(getattr(item, field)) for field in fields) for item in values):
        raise H3ImportError("invalid_prompt_artifacts")
    if any(
        len({getattr(item, field) for item in values}) != 1
        for field in ("storyboard_sha256", "semantic_lock_sha256", "approved_script_sha256", "audio_sha256", "subtitle_sha256")
    ) or not 45_000 <= sum(item.required_duration_ms for item in values) <= 60_000:
        raise H3ImportError("invalid_prompt_artifacts")
    return tuple(RequiredSegmentIdentity(
        segment_id=item.segment_id, prompt_sha256=item.prompt_sha256,
        storyboard_sha256=item.storyboard_sha256, semantic_lock_sha256=item.semantic_lock_sha256,
        approved_script_sha256=item.approved_script_sha256, audio_sha256=item.audio_sha256,
        subtitle_sha256=item.subtitle_sha256, expected_duration_ms=item.required_duration_ms,
    ) for item in values)


def _validate_probe(expected_duration_ms: int, facts: MediaProbeFacts) -> tuple[str, ...]:
    if abs((facts.width / facts.height) - _ASPECT_RATIO) > _ASPECT_TOLERANCE:
        raise H3ImportError("invalid_video_aspect")
    if facts.duration_ms > expected_duration_ms + _DURATION_METADATA_TOLERANCE_MS:
        raise H3ImportError("segment_too_long")
    shortfall = expected_duration_ms - facts.duration_ms
    if shortfall > _MAX_SHORTFALL_MS:
        raise H3ImportError("segment_too_short")
    return ("duration_shortfall_repair_required",) if shortfall > 0 else ()


def _collect_existing(
    *, h3_root: Path, required: tuple[RequiredSegmentIdentity, ...], book_id: str, episode_id: str,
    provider: VideoProviderName,
    allow_missing_aggregate: bool = False, allow_stale_aggregate: bool = False,
) -> dict[str, H3SegmentManifest]:
    if not h3_root.exists():
        return {}
    if not h3_root.is_dir() or _redirect_in_existing_chain(h3_root):
        raise H3ImportError("aggregate_integrity_invalid")
    allowed_ids = {item.segment_id for item in required}
    try:
        entries = {entry.name: entry for entry in h3_root.iterdir()}
    except OSError:
        raise H3ImportError("aggregate_integrity_invalid") from None
    if set(entries) - (allowed_ids | {"manifest.json"}):
        raise H3ImportError("aggregate_integrity_invalid")
    manifests: dict[str, H3SegmentManifest] = {}
    for identity in required:
        segment_root = entries.get(identity.segment_id)
        if segment_root is None:
            continue
        manifests[identity.segment_id] = _read_segment_manifest(
            segment_root, identity, book_id=book_id, episode_id=episode_id, provider=provider,
        )
    aggregate_path = h3_root / "manifest.json"
    if aggregate_path.exists():
        expected = _aggregate(book_id, episode_id, required, manifests, h3_root, provider=provider)
        try:
            actual = H3AggregateManifest.model_validate_json(aggregate_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError):
            raise H3ImportError("aggregate_integrity_invalid") from None
        actual = actual.model_copy(update={
            "status": _status_from_ids(actual.present_segment_ids, actual.missing_segment_ids),
        })
        if (actual != expected and not allow_stale_aggregate) or _redirect_in_existing_chain(aggregate_path):
            raise H3ImportError("aggregate_integrity_invalid")
    elif manifests and not allow_missing_aggregate:
        raise H3ImportError("aggregate_integrity_invalid")
    return manifests


def _read_segment_manifest(
    segment_root: Path, identity: RequiredSegmentIdentity, *, book_id: str, episode_id: str,
    provider: VideoProviderName,
) -> H3SegmentManifest:
    manifest_path = segment_root / "manifest.json"
    if (
        not segment_root.is_dir() or _redirect_in_existing_chain(segment_root)
        or not manifest_path.is_file() or _redirect_in_existing_chain(manifest_path)
    ):
        raise H3ImportError("existing_segment_integrity_invalid")
    try:
        manifest = H3SegmentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        datetime.fromisoformat(manifest.imported_at)
        uuid.UUID(manifest.event_id)
    except (OSError, ValidationError, ValueError):
        raise H3ImportError("existing_segment_integrity_invalid") from None
    if any(not _is_sha256(getattr(manifest, field)) for field in (
        "prompt_sha256", "storyboard_sha256", "semantic_lock_sha256", "approved_script_sha256",
        "audio_sha256", "subtitle_sha256", "source_sha256", "copied_sha256",
    )):
        raise H3ImportError("existing_segment_integrity_invalid")
    expected_media = segment_root / f"original-{manifest.copied_sha256}.mp4"
    try:
        entries = set(segment_root.iterdir())
    except OSError:
        raise H3ImportError("existing_segment_integrity_invalid") from None
    if entries != {manifest_path, expected_media} or not expected_media.is_file() or _redirect_in_existing_chain(expected_media):
        raise H3ImportError("existing_segment_integrity_invalid")
    if _sha256_file(expected_media) != manifest.copied_sha256 or manifest.source_sha256 != manifest.copied_sha256:
        raise H3ImportError("existing_segment_integrity_invalid")
    if manifest.model_dump(exclude={
        "provider", "source_sha256", "copied_sha256", "probe", "warning_codes", "imported_at", "event_id",
    }) != identity.model_dump():
        raise H3ImportError("existing_segment_integrity_invalid")
    if manifest.provider != provider:
        raise H3ImportError("existing_segment_integrity_invalid")
    return manifest


def _result(
    *, h3_root: Path, required: tuple[RequiredSegmentIdentity, ...], existing: dict[str, H3SegmentManifest],
    segment_id: str, book_id: str, episode_id: str, provider: VideoProviderName, idempotent: bool,
) -> H3ImportResult:
    required_ids = tuple(item.segment_id for item in required)
    present = tuple(item for item in required_ids if item in existing)
    missing = tuple(item for item in required_ids if item not in existing)
    aggregate = _aggregate(book_id, episode_id, required, existing, h3_root, provider=provider)
    aggregate_path = h3_root / "manifest.json"
    _write_or_replace_aggregate(aggregate_path, aggregate)
    return H3ImportResult(
        provider=provider, status=_status_from_ids(present, missing),
        present_segment_ids=present, missing_segment_ids=missing,
        manifest_path=h3_root / segment_id / "manifest.json", aggregate_manifest_path=aggregate_path,
        idempotent=idempotent,
    )


def _aggregate(
    book_id: str, episode_id: str, required: tuple[RequiredSegmentIdentity, ...],
    existing: dict[str, H3SegmentManifest], h3_root: Path, *, provider: VideoProviderName,
) -> H3AggregateManifest:
    required_ids = tuple(item.segment_id for item in required)
    present = tuple(item for item in required_ids if item in existing)
    missing = tuple(item for item in required_ids if item not in existing)
    imported = tuple(ImportedSegmentIdentity(
        segment_id=segment_id, source_sha256=existing[segment_id].source_sha256,
        copied_sha256=existing[segment_id].copied_sha256,
        manifest_sha256=_sha256_file(h3_root / segment_id / "manifest.json"),
    ) for segment_id in present)
    return H3AggregateManifest(
        provider=provider, book_id=book_id, episode_id=episode_id, required_segments=required,
        imported_segments=imported, present_segment_ids=present, missing_segment_ids=missing,
        status=_status_from_ids(present, missing),
    )


def _snapshot_source(source: Path) -> tuple[Path, str]:
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{uuid.uuid4().hex}.", suffix=".tmp", dir=source.parent)
        temporary = Path(name)
        digest = hashlib.sha256()
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            for chunk in iter(lambda: input_stream.read(1_048_576), b""):
                digest.update(chunk)
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        value = digest.hexdigest()
        if _sha256_file(temporary) != value or _sha256_file(source) != value:
            raise H3ImportError("source_changed_during_import")
        return temporary, value
    except H3ImportError:
        _remove_temp(temporary)
        raise
    except OSError:
        _remove_temp(temporary)
        raise H3ImportError("source_snapshot_failed") from None


def _publish_snapshot(snapshot: Path, segment_root: Path, source_sha: str) -> Path:
    target = segment_root / f"original-{source_sha}.mp4"
    if target.exists():
        raise H3ImportError("existing_segment_integrity_invalid")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{source_sha}.", suffix=".tmp", dir=segment_root)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as output_stream, snapshot.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        copied_sha = _sha256_file(temporary)
        if copied_sha != source_sha or _redirect_in_existing_chain(target):
            raise H3ImportError("source_changed_during_import")
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise H3ImportError("existing_segment_integrity_invalid") from None
        temporary.unlink()
        if not target.is_file() or _sha256_file(target) != source_sha:
            raise H3ImportError("media_publish_failed")
        return target
    except H3ImportError:
        raise
    except OSError:
        raise H3ImportError("media_publish_failed") from None
    finally:
        _remove_temp(temporary)


def _write_new_manifest(path: Path, manifest: H3SegmentManifest) -> None:
    if path.exists() or _redirect_in_existing_chain(path):
        raise H3ImportError("existing_segment_integrity_invalid")
    _atomic_json(path, manifest.model_dump(mode="json"), replace=False, error_code="manifest_write_failed")
    try:
        if H3SegmentManifest.model_validate_json(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError
    except (OSError, ValidationError, ValueError):
        raise H3ImportError("manifest_write_failed") from None


def _write_or_replace_aggregate(path: Path, manifest: H3AggregateManifest) -> None:
    _atomic_json(path, manifest.model_dump(mode="json"), replace=True, error_code="aggregate_write_failed")
    try:
        if H3AggregateManifest.model_validate_json(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError
    except (OSError, ValidationError, ValueError):
        raise H3ImportError("aggregate_write_failed") from None


def _atomic_json(path: Path, value: dict[str, object], *, replace: bool, error_code: str) -> None:
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        if _redirect_in_existing_chain(path):
            raise H3ImportError("unsafe_h3_path")
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    except H3ImportError:
        raise
    except (OSError, ValueError):
        raise H3ImportError(error_code) from None
    finally:
        _remove_temp(temporary)


def _require_identifier(value: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise H3ImportError("unsafe_identifier")


def _require_provider(provider: object) -> None:
    if provider not in {"grok_cli", "grok_manual", "h3_manual"}:
        raise H3ImportError("invalid_video_provider")


def _require_safe_source(path: Path) -> None:
    try:
        too_large = path.stat().st_size > _MAX_H3_SOURCE_BYTES
    except OSError:
        too_large = False
    if too_large:
        raise H3ImportError("h3_source_too_large")
    if path.suffix.lower() != ".mp4" or not path.is_file() or _redirect_in_existing_chain(path):
        raise H3ImportError("unsafe_h3_path")


def _require_child(root: Path, path: Path) -> None:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        raise H3ImportError("unsafe_h3_path") from None


def _ensure_safe_directory(path: Path) -> None:
    if _redirect_in_existing_chain(path):
        raise H3ImportError("unsafe_h3_path")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise H3ImportError("unsafe_h3_path") from None
    if not path.is_dir() or _redirect_in_existing_chain(path):
        raise H3ImportError("unsafe_h3_path")


def _redirect_in_existing_chain(path: Path) -> bool:
    candidate = Path(path)
    while True:
        if candidate.exists() or candidate.is_symlink():
            if _is_redirected(candidate):
                return True
        parent = candidate.parent
        if parent == candidate:
            return False
        candidate = parent


def _is_redirected(path: Path) -> bool:
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        attributes = 0
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.is_file() and not path.is_symlink():
            path.unlink()
    except OSError:
        pass


def _rollback_new_segment(segment_root: Path, target: Path, manifest_path: Path, expected_sha: str) -> None:
    """Remove only files created by this failed import; never follow a redirect."""
    if _redirect_in_existing_chain(segment_root):
        return
    try:
        if manifest_path.is_file() and not manifest_path.is_symlink():
            manifest_path.unlink()
        if target.is_file() and not target.is_symlink() and _sha256_file(target) == expected_sha:
            target.unlink()
        if segment_root.is_dir() and not any(segment_root.iterdir()):
            segment_root.rmdir()
    except OSError:
        pass
