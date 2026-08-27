"""Approve one real local product cover without accepting generated substitutes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path

try:
    import pymupdf
except ModuleNotFoundError:  # pragma: no cover - compatibility for older local environments
    import fitz as pymupdf
from pydantic import BaseModel, ConfigDict, Field, ValidationError


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_SOURCE_TYPES = frozenset({"user_provided", "ebook_extracted", "official_product_image"})
_ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_MAX_COVER_BYTES = 20 * 1024 * 1024
_MAX_COVER_PIXELS = 40_000_000
_MAX_COVER_DIMENSION = 10_000


class CoverApprovalError(RuntimeError):
    """A stable product-cover error that never reveals private media paths."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class CoverImageFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    image_format: str = Field(min_length=1)


class CoverManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    book_id: str
    episode_id: str
    source_sha256: str
    copied_sha256: str
    source_type: str
    matches_product_version: bool
    width: int
    height: int
    image_format: str


class CoverApprovalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    cover_path: Path
    manifest_path: Path
    idempotent: bool


def approve_cover(
    *,
    book_id: str,
    episode_id: str,
    source_path: Path,
    episode_cover_root: Path,
    source_type: str,
    matches_product_version: bool,
    decoder: Callable[..., CoverImageFacts] | None = None,
) -> CoverApprovalResult:
    """Snapshot, decode, and atomically approve a user-selected real cover."""
    _require_identifier(book_id)
    _require_identifier(episode_id)
    if source_type not in _ALLOWED_SOURCE_TYPES:
        raise CoverApprovalError("invalid_cover_source_type")
    if matches_product_version is not True:
        raise CoverApprovalError("product_version_not_confirmed")
    source = Path(source_path)
    root = Path(episode_cover_root)
    _require_safe_source(source)
    snapshot, source_sha = _snapshot_source(source)
    try:
        raw = _read_bounded(snapshot)
        try:
            facts = (decoder or _decode_with_pymupdf)(raw, source.suffix.lower(), max_pixels=_MAX_COVER_PIXELS)
        except Exception:
            raise CoverApprovalError("invalid_cover_image") from None
        if not isinstance(facts, CoverImageFacts):
            raise CoverApprovalError("invalid_cover_image")
        if (
            facts.width > _MAX_COVER_DIMENSION or facts.height > _MAX_COVER_DIMENSION
            or facts.width * facts.height > _MAX_COVER_PIXELS
        ):
            raise CoverApprovalError("cover_dimensions_out_of_bounds")
        if _sha256_file(source) != source_sha or _redirect_in_existing_chain(source):
            raise CoverApprovalError("cover_source_changed_during_import")
        _ensure_safe_directory(root)
        extension = source.suffix.lower()
        target = root / f"original-{source_sha}{extension}"
        manifest_path = root / "manifest.json"
        existing = _existing_if_identical(
            target, manifest_path, book_id=book_id, episode_id=episode_id, source_sha=source_sha,
            source_type=source_type, facts=facts,
        )
        if existing:
            return CoverApprovalResult(cover_path=target, manifest_path=manifest_path, idempotent=True)
        _publish_snapshot(snapshot, target, source_sha)
        manifest = CoverManifest(
            book_id=book_id, episode_id=episode_id, source_sha256=source_sha, copied_sha256=source_sha,
            source_type=source_type, matches_product_version=True, width=facts.width,
            height=facts.height, image_format=facts.image_format,
        )
        try:
            _write_new_manifest(manifest_path, manifest)
        except CoverApprovalError:
            _rollback_new_cover(root, target, manifest_path, source_sha)
            raise
        return CoverApprovalResult(cover_path=target, manifest_path=manifest_path, idempotent=False)
    finally:
        _remove_temp(snapshot)


def _decode_with_pymupdf(raw: bytes, suffix: str, *, max_pixels: int) -> CoverImageFacts:
    document = pymupdf.open(stream=raw, filetype=suffix.removeprefix("."))
    try:
        if document.page_count != 1:
            raise ValueError
        page = document[0]
        image_info = page.get_image_info(xrefs=True)
        if len(image_info) != 1:
            raise ValueError
        width = image_info[0].get("width")
        height = image_info[0].get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0 or width * height > max_pixels:
            raise ValueError
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(0.1, 0.1), alpha=False)
        if pixmap.width <= 0 or pixmap.height <= 0:
            raise ValueError
        return CoverImageFacts(width=width, height=height, image_format=suffix.removeprefix(".").upper())
    finally:
        document.close()


def _existing_if_identical(
    target: Path, manifest_path: Path, *, book_id: str, episode_id: str, source_sha: str,
    source_type: str, facts: CoverImageFacts,
) -> bool:
    try:
        entries = set(target.parent.iterdir())
    except OSError:
        raise CoverApprovalError("existing_cover_integrity_invalid") from None
    if entries not in (set(), {target, manifest_path}):
        raise CoverApprovalError("existing_cover_integrity_invalid")
    if not target.exists() and not manifest_path.exists():
        return False
    if (
        not target.is_file() or not manifest_path.is_file() or _redirect_in_existing_chain(target)
        or _redirect_in_existing_chain(manifest_path)
    ):
        raise CoverApprovalError("existing_cover_integrity_invalid")
    try:
        manifest = CoverManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        raise CoverApprovalError("existing_cover_integrity_invalid") from None
    if (
        not _is_sha256(manifest.source_sha256) or not _is_sha256(manifest.copied_sha256)
        or _sha256_file(target) != manifest.copied_sha256
    ):
        raise CoverApprovalError("existing_cover_integrity_invalid")
    expected = CoverManifest(
        book_id=book_id, episode_id=episode_id, source_sha256=source_sha, copied_sha256=source_sha,
        source_type=source_type, matches_product_version=True, width=facts.width,
        height=facts.height, image_format=facts.image_format,
    )
    if manifest != expected:
        raise CoverApprovalError("existing_cover_integrity_invalid")
    return True


def _read_bounded(path: Path) -> bytes:
    try:
        if path.stat().st_size > _MAX_COVER_BYTES:
            raise CoverApprovalError("cover_source_too_large")
        with path.open("rb") as stream:
            raw = stream.read(_MAX_COVER_BYTES + 1)
    except CoverApprovalError:
        raise
    except OSError:
        raise CoverApprovalError("cover_snapshot_failed") from None
    if len(raw) > _MAX_COVER_BYTES:
        raise CoverApprovalError("cover_source_too_large")
    return raw


def _snapshot_source(source: Path) -> tuple[Path, str]:
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".cover-", suffix=".tmp", dir=source.parent)
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
            raise CoverApprovalError("cover_source_changed_during_import")
        return temporary, value
    except CoverApprovalError:
        _remove_temp(temporary)
        raise
    except OSError:
        _remove_temp(temporary)
        raise CoverApprovalError("cover_snapshot_failed") from None


def _publish_snapshot(snapshot: Path, target: Path, expected_sha: str) -> None:
    if target.exists() or _redirect_in_existing_chain(target):
        raise CoverApprovalError("existing_cover_integrity_invalid")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".cover-copy-", suffix=".tmp", dir=target.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as output_stream, snapshot.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if _sha256_file(temporary) != expected_sha:
            raise CoverApprovalError("cover_publish_failed")
        os.link(temporary, target)
        temporary.unlink()
        if not target.is_file() or _sha256_file(target) != expected_sha:
            raise CoverApprovalError("cover_publish_failed")
    except CoverApprovalError:
        raise
    except OSError:
        raise CoverApprovalError("cover_publish_failed") from None
    finally:
        _remove_temp(temporary)


def _write_new_manifest(path: Path, manifest: CoverManifest) -> None:
    if path.exists() or _redirect_in_existing_chain(path):
        raise CoverApprovalError("existing_cover_integrity_invalid")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest.model_dump(mode="json"), stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        os.link(temporary, path)
        temporary.unlink()
        if CoverManifest.model_validate_json(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError
    except (OSError, ValidationError, ValueError):
        raise CoverApprovalError("manifest_write_failed") from None
    finally:
        _remove_temp(temporary)


def _require_identifier(value: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise CoverApprovalError("unsafe_identifier")


def _require_safe_source(path: Path) -> None:
    if path.suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise CoverApprovalError("unsupported_cover_extension")
    if not path.is_file() or _redirect_in_existing_chain(path):
        raise CoverApprovalError("unsafe_cover_path")


def _ensure_safe_directory(path: Path) -> None:
    if _redirect_in_existing_chain(path):
        raise CoverApprovalError("unsafe_cover_path")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise CoverApprovalError("unsafe_cover_path") from None
    if not path.is_dir() or _redirect_in_existing_chain(path):
        raise CoverApprovalError("unsafe_cover_path")


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


def _rollback_new_cover(root: Path, target: Path, manifest_path: Path, expected_sha: str) -> None:
    """Remove only this call's verified target after manifest publication fails."""
    if _redirect_in_existing_chain(root):
        return
    try:
        if manifest_path.is_file() and not manifest_path.is_symlink():
            manifest_path.unlink()
        if target.is_file() and not target.is_symlink() and _sha256_file(target) == expected_sha:
            target.unlink()
        if root.is_dir() and not any(root.iterdir()):
            root.rmdir()
    except OSError:
        pass
