from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bv.core.hashing import sha256_file
from bv.state.locks import EpisodeLock, LockHeldError


_SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_VERSION = re.compile(r"-V(\d+)\.")


class DeliveryError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _DeliveryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DeliveryRequest(_DeliveryModel):
    episode_root: Path
    book_id: str = Field(min_length=1, max_length=128)
    episode_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    author: str = Field(min_length=1, max_length=200)
    script_path: Path
    voice_path: Path
    subtitle_path: Path
    video_path: Path
    cover_path: Path
    production_profile_path: Path
    voice_manifest_path: Path
    render_manifest_path: Path
    qc_manifest_path: Path
    social_cover_manifest_path: Path


class DeliveryBundle(_DeliveryModel):
    root: Path
    version: int = Field(ge=1)
    manifest_path: Path
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[Path, ...]
    video_path: Path
    cover_path: Path


class ApprovalRecord(_DeliveryModel):
    delivery_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cover_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_at: datetime
    status: Literal["final_approved"] = "final_approved"
    path: Path


class DeliveryExporter:
    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self.clock = clock or (lambda: datetime.now(tz=_SHANGHAI))

    def export_candidate(self, request: DeliveryRequest) -> DeliveryBundle:
        try:
            request = DeliveryRequest.model_validate(request)
        except ValidationError:
            raise DeliveryError("delivery_request_invalid") from None
        root = Path(request.episode_root).absolute()
        _require_safe_directory(root)
        safe_title = _safe_title(request.title)
        sources = {
            "script": Path(request.script_path).absolute(),
            "voice": Path(request.voice_path).absolute(),
            "subtitles": Path(request.subtitle_path).absolute(),
            "video": Path(request.video_path).absolute(),
            "cover": Path(request.cover_path).absolute(),
            "production_profile": Path(request.production_profile_path).absolute(),
            "voice_manifest": Path(request.voice_manifest_path).absolute(),
            "render_manifest": Path(request.render_manifest_path).absolute(),
            "qc_manifest": Path(request.qc_manifest_path).absolute(),
            "social_cover_manifest": Path(request.social_cover_manifest_path).absolute(),
        }
        for path in sources.values():
            _require_episode_source(root, path)
        now = self.clock()
        if now.tzinfo is None:
            raise DeliveryError("delivery_clock_invalid")
        date = now.astimezone(_SHANGHAI).date().isoformat()
        deliveries = root / "deliveries"
        _require_safe_destination(root, deliveries)
        deliveries.mkdir(parents=True, exist_ok=True)
        _require_safe_directory(deliveries)
        try:
            with EpisodeLock(root / ".delivery.lock"):
                return self._export_locked(
                    request=request,
                    root=root,
                    deliveries=deliveries,
                    sources=sources,
                    safe_title=safe_title,
                    date=date,
                )
        except LockHeldError:
            raise DeliveryError("delivery_locked") from None

    def _export_locked(
        self,
        *,
        request: DeliveryRequest,
        root: Path,
        deliveries: Path,
        sources: dict[str, Path],
        safe_title: str,
        date: str,
    ) -> DeliveryBundle:
        date_root = deliveries / date
        version = _next_version(date_root)
        suffix = "" if version == 1 else f"-V{version}"
        names = {
            "video": f"{date}-{safe_title}-成片{suffix}.mp4",
            "cover": f"{date}-{safe_title}-作品封面{suffix}.png",
            "script": f"{date}-{safe_title}-文稿{suffix}.txt",
            "voice": f"{date}-{safe_title}-旁白{suffix}.wav",
            "subtitles": f"{date}-{safe_title}-字幕{suffix}.ass",
            "manifest": f"{date}-{safe_title}-交付清单{suffix}.json",
        }
        targets = {key: date_root / value for key, value in names.items()}
        for target in targets.values():
            _require_safe_destination(root, target)
            if target.exists() or target.is_symlink():
                raise DeliveryError("delivery_target_exists")

        staging = deliveries / f".staging-delivery-{uuid.uuid4().hex}"
        staging.mkdir(exist_ok=False)
        staged: dict[str, Path] = {}
        source_evidence: dict[str, dict[str, object]] = {}
        published: list[tuple[Path, str]] = []
        try:
            for key in ("video", "cover", "script", "voice", "subtitles"):
                staged_path = staging / names[key]
                evidence = _copy_exact_source(sources[key], staged_path)
                staged[key] = staged_path
                source_evidence[key] = evidence
            dependency_hashes = {
                key: sha256_file(path)
                for key, path in sources.items()
                if key not in {"video", "cover", "script", "voice", "subtitles"}
            }
            voice_payload = _read_small_json(sources["voice_manifest"])
            manifest_payload = {
                "schema_version": 1,
                "state": "awaiting_final_review",
                "book_id": request.book_id,
                "episode_id": request.episode_id,
                "title": request.title,
                "author": request.author,
                "shanghai_date": date,
                "timezone": "Asia/Shanghai",
                "version": version,
                "tts_provider": voice_payload.get("provider", "indextts2"),
                "voice_id": voice_payload.get("provider_voice_id", voice_payload.get("voice_id")),
                "dependencies": dependency_hashes,
                "artifacts": {
                    key: {
                        "path": names[key],
                        "size_bytes": source_evidence[key]["size_bytes"],
                        "sha256": source_evidence[key]["sha256"],
                    }
                    for key in ("video", "cover", "script", "voice", "subtitles")
                },
            }
            manifest_stage = staging / names["manifest"]
            _write_json_exclusive(manifest_stage, manifest_payload)
            staged["manifest"] = manifest_stage

            date_root.mkdir(parents=True, exist_ok=True)
            _require_safe_directory(date_root)
            for key in ("video", "cover", "script", "voice", "subtitles", "manifest"):
                if key == "manifest" and any(
                    sha256_file(sources[name]) != expected
                    for name, expected in dependency_hashes.items()
                ):
                    raise DeliveryError("delivery_source_changed")
                digest = sha256_file(staged[key])
                _publish_new(staged[key], targets[key], digest)
                published.append((targets[key], digest))
            files = tuple(targets[key] for key in (
                "video", "cover", "script", "voice", "subtitles", "manifest"
            ))
            return DeliveryBundle(
                root=date_root,
                version=version,
                manifest_path=targets["manifest"],
                manifest_sha256=sha256_file(targets["manifest"]),
                files=files,
                video_path=targets["video"],
                cover_path=targets["cover"],
            )
        except DeliveryError:
            _recover_partial(deliveries, published)
            raise
        except Exception:
            _recover_partial(deliveries, published)
            raise DeliveryError("delivery_export_failed") from None
        finally:
            _remove_staging(staging)

    def record_final_approval(
        self,
        bundle: DeliveryBundle,
        *,
        approved_at: datetime,
    ) -> ApprovalRecord:
        try:
            bundle = DeliveryBundle.model_validate(bundle)
        except ValidationError:
            raise DeliveryError("delivery_bundle_invalid") from None
        if approved_at.tzinfo is None:
            raise DeliveryError("approval_time_invalid")
        root = Path(bundle.root).absolute()
        manifest_path = Path(bundle.manifest_path).absolute()
        video_path = Path(bundle.video_path).absolute()
        cover_path = Path(bundle.cover_path).absolute()
        try:
            _require_safe_directory(root)
            if (
                manifest_path.parent != root
                or video_path.parent != root
                or cover_path.parent != root
                or manifest_path not in tuple(Path(path).absolute() for path in bundle.files)
            ):
                raise DeliveryError("unsafe_approval_bundle")
            for path in bundle.files:
                candidate = Path(path).absolute()
                if candidate.parent != root:
                    raise DeliveryError("unsafe_approval_bundle")
                _require_safe_regular(candidate)
        except DeliveryError as error:
            if error.error_code == "unsafe_approval_bundle":
                raise
            raise DeliveryError("unsafe_approval_bundle") from None
        if sha256_file(manifest_path) != bundle.manifest_sha256:
            raise DeliveryError("delivery_manifest_stale")
        try:
            payload = _read_small_json(manifest_path)
            artifacts = payload["artifacts"]
            shanghai_date = payload["shanghai_date"]
            original_title = payload["title"]
            if (
                payload.get("state") != "awaiting_final_review"
                or payload.get("version") != bundle.version
                or not isinstance(shanghai_date, str)
                or not isinstance(original_title, str)
                or not isinstance(artifacts, dict)
            ):
                raise ValueError
            expected_paths: dict[str, Path] = {}
            for key in ("video", "cover", "script", "voice", "subtitles"):
                record = artifacts[key]
                if not isinstance(record, dict):
                    raise ValueError
                relative = record.get("path")
                digest = record.get("sha256")
                size = record.get("size_bytes")
                if (
                    not isinstance(relative, str)
                    or not isinstance(digest, str)
                    or not isinstance(size, int)
                ):
                    raise ValueError
                candidate = (root / relative).absolute()
                if candidate.parent != root:
                    raise ValueError
                _require_safe_regular(candidate)
                if candidate.stat().st_size != size or sha256_file(candidate) != digest:
                    raise DeliveryError("delivery_artifact_stale")
                expected_paths[key] = candidate
            if set(Path(path).absolute() for path in bundle.files) != {
                *expected_paths.values(),
                manifest_path,
            }:
                raise DeliveryError("unsafe_approval_bundle")
            video_record = artifacts["video"]
            cover_record = artifacts["cover"]
        except (KeyError, TypeError, ValueError):
            raise DeliveryError("delivery_manifest_invalid") from None
        if (
            video_path != expected_paths["video"]
            or cover_path != expected_paths["cover"]
        ):
            raise DeliveryError("delivery_artifact_stale")
        suffix = "" if bundle.version == 1 else f"-V{bundle.version}"
        approval_path = root / (
            f"{shanghai_date}-{_safe_title(original_title)}"
            f"-最终批准{suffix}.json"
        )
        if approval_path.exists() or approval_path.is_symlink():
            raise DeliveryError("approval_exists")
        _require_safe_destination(root, approval_path)
        record = ApprovalRecord(
            delivery_manifest_sha256=bundle.manifest_sha256,
            video_sha256=video_record["sha256"],
            cover_sha256=cover_record["sha256"],
            approved_at=approved_at.astimezone(_SHANGHAI),
            path=approval_path,
        )
        _write_json_atomic_new(approval_path, record.model_dump(mode="json"))
        return record


def _safe_title(value: object) -> str:
    if not isinstance(value, str):
        raise DeliveryError("delivery_title_invalid")
    sanitized = "".join(
        "_" if ord(char) < 32 or char in '<>:"/\\|?*' else char
        for char in value
    ).strip().rstrip(" .")
    if not sanitized or sanitized.split(".", 1)[0].upper() in _RESERVED:
        raise DeliveryError("delivery_title_invalid")
    return sanitized


def _next_version(date_root: Path) -> int:
    if not date_root.exists() and not date_root.is_symlink():
        return 1
    _require_safe_directory(date_root)
    versions = {1}
    for path in date_root.iterdir():
        match = _VERSION.search(path.name)
        if match is not None:
            versions.add(int(match.group(1)))
    return max(versions) + 1


def _require_episode_source(root: Path, path: Path) -> None:
    if not path.is_relative_to(root):
        raise DeliveryError("unsafe_delivery_source")
    try:
        _require_safe_regular(path)
    except DeliveryError:
        raise DeliveryError("unsafe_delivery_source") from None


def _require_safe_regular(path: Path) -> None:
    try:
        if _redirect_in_existing_chain(path):
            raise DeliveryError("unsafe_delivery_path")
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise DeliveryError("unsafe_delivery_path")
    except DeliveryError:
        raise
    except OSError:
        raise DeliveryError("unsafe_delivery_path") from None


def _require_safe_directory(path: Path) -> None:
    try:
        if _redirect_in_existing_chain(path) or not path.is_dir():
            raise DeliveryError("unsafe_delivery_path")
    except OSError:
        raise DeliveryError("unsafe_delivery_path") from None


def _require_safe_destination(root: Path, path: Path) -> None:
    if not path.absolute().is_relative_to(root) or _redirect_in_existing_chain(path):
        raise DeliveryError("unsafe_delivery_path")


def _redirect_in_existing_chain(path: Path) -> bool:
    candidate = Path(path)
    while True:
        if candidate.exists() or candidate.is_symlink():
            attributes = getattr(candidate.stat(follow_symlinks=False), "st_file_attributes", 0)
            if candidate.is_symlink() or attributes & 0x400:
                return True
        if candidate.parent == candidate:
            return False
        candidate = candidate.parent


def _copy_exact_source(source: Path, target: Path) -> dict[str, object]:
    before = source.stat(follow_symlinks=False)
    expected_hash = sha256_file(source)
    digest = hashlib.sha256()
    total = 0
    with source.open("rb") as input_stream, target.open("xb") as output_stream:
        for chunk in iter(lambda: input_stream.read(1_048_576), b""):
            digest.update(chunk)
            total += len(chunk)
            output_stream.write(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    after = source.stat(follow_symlinks=False)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if (
        identity_before != identity_after
        or digest.hexdigest() != expected_hash
        or sha256_file(source) != expected_hash
        or sha256_file(target) != expected_hash
        or _redirect_in_existing_chain(source)
    ):
        raise DeliveryError("delivery_source_changed")
    return {"sha256": expected_hash, "size_bytes": total}


def _publish_new(source: Path, target: Path, expected_hash: str) -> None:
    try:
        os.link(source, target)
        if sha256_file(target) != expected_hash:
            raise OSError
    except FileExistsError:
        raise DeliveryError("delivery_target_exists") from None
    except OSError:
        _remove_if_hash(target, expected_hash)
        raise DeliveryError("delivery_publish_failed") from None


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise DeliveryError("delivery_target_exists") from None


def _write_json_atomic_new(path: Path, payload: dict[str, object]) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_json_exclusive(temporary, payload)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise DeliveryError("approval_exists") from None
        except OSError:
            raise DeliveryError("approval_publish_failed") from None
    finally:
        try:
            if temporary.is_file() and not temporary.is_symlink():
                temporary.unlink()
        except OSError:
            pass


def _read_small_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError
        return payload
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise DeliveryError("delivery_manifest_invalid") from None


def _recover_partial(deliveries: Path, published: list[tuple[Path, str]]) -> None:
    if not published:
        return
    recovery = deliveries / ".recovery" / f"delivery-{uuid.uuid4().hex}"
    recovery.mkdir(parents=True, exist_ok=False)
    for path, expected_hash in published:
        if path.is_file() and not path.is_symlink() and sha256_file(path) == expected_hash:
            os.replace(path, recovery / path.name)


def _remove_if_hash(path: Path, expected_hash: str) -> None:
    try:
        if path.is_file() and not path.is_symlink() and sha256_file(path) == expected_hash:
            path.unlink()
    except OSError:
        pass


def _remove_staging(path: Path) -> None:
    try:
        if not path.is_dir() or path.is_symlink() or _redirect_in_existing_chain(path):
            return
        for child in path.iterdir():
            if child.is_file() and not child.is_symlink():
                child.unlink()
            else:
                return
        path.rmdir()
    except OSError:
        pass
