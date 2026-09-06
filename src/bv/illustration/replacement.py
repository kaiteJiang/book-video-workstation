from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bv.core.hashing import sha256_file
from bv.core.process import CommandResult, run_command
from bv.state.invalidation import invalidate_from
from bv.state.models import ArtifactRef, EpisodeState, StageManifest
from bv.state.store import StateStore

from .assets import (
    ImageFacts,
    ImageJob,
    IllustrationAssetError,
    IllustrationManifest,
    ImportedImage,
    bind_job_references,
    import_image_asset,
    record_imported_image,
    validate_generated_image_job,
)
from .prompts import canonical_model_sha256, image_prompt_identity_sha256
from .reviews import RepresentativeApproval, RepresentativeReview


_ASSET_ID = re.compile(r"^S[0-9]+-[AB]$")
_REPLACEMENT_STAGE = "illustration_replacement"
_REPLACEMENT_CONFIG_SHA256 = hashlib.sha256(
    b"bv.illustration.replacement.transaction.v5"
).hexdigest()
_DOWNSTREAM_STAGES = (
    "representative_review",
    "illustration_images",
    "visual_render",
    "render",
    "qc",
    "delivery",
    "final_approval",
)
_EPISODE_LOCKS_GUARD = threading.Lock()
_EPISODE_LOCKS: dict[str, threading.RLock] = {}


class IllustrationReplacementError(RuntimeError):
    """Stable failure for the explicit illustration replacement transaction."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _ReplacementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class ReplacementAuthorization(_ReplacementModel):
    schema_version: Literal[3] = 3
    transaction_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]+$")
    nonce: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]+$")
    book_id: str
    episode_id: str
    asset_id: str
    phase: Literal["anchor", "continuation"]
    reason: str = Field(min_length=1)
    rejected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    representative_review_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    storyboard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    style_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    character_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_attempt: int = Field(ge=1)
    archive_relative_path: str
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    job_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_at_ns: int = Field(gt=0)
    status: Literal["authorized", "consumed"] = "authorized"
    replacement_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    replacement_size_bytes: int | None = Field(default=None, ge=0)


class ReplacementPreparation(_ReplacementModel):
    transaction_id: str
    nonce: str
    history_dir: Path
    authorization_paths: tuple[Path, ...]
    reset_asset_ids: tuple[str, ...]
    status: str


class ReplacementImportResult(_ReplacementModel):
    transaction_id: str
    imported: ImportedImage
    authorization_path: Path
    status: str


class ReplacementArchiveEntry(_ReplacementModel):
    source: str
    archive: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ReplacementReceipt(_ReplacementModel):
    schema_version: Literal[4] = 4
    transaction_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]+$")
    nonce: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]+$")
    book_id: str
    episode_id: str
    rejected_asset_id: str
    reason: str = Field(min_length=1)
    expected_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    representative_review_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    representative_approval_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    history_relative_path: str
    reset_asset_ids: tuple[str, ...]
    authorization_asset_ids: tuple[str, ...]
    reset_job_projection_sha256s: dict[str, str]
    archived_files: tuple[ReplacementArchiveEntry, ...]
    state_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    affected_stage_manifests_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_at_ns: int = Field(gt=0)


@dataclass(frozen=True)
class _ReplacementClosure:
    stage: StageManifest
    receipt: ReplacementReceipt
    receipt_path: Path
    authorizations: dict[str, ReplacementAuthorization]
    authorization_paths: dict[str, Path]
    authorization_bytes: dict[str, bytes]


def prepare_illustration_replacement(
    *,
    store: StateStore,
    book_id: str,
    episode_id: str,
    asset_id: str,
    expected_asset_sha256: str,
    reason: str,
    catalog_path: Path | None = None,
    expected_representative_review_sha256: str | None = None,
) -> ReplacementPreparation:
    root = _episode_root(store, book_id, episode_id)
    with _episode_lock(root):
        return _prepare_illustration_replacement_unlocked(
            store=store,
            book_id=book_id,
            episode_id=episode_id,
            asset_id=asset_id,
            expected_asset_sha256=expected_asset_sha256,
            reason=reason,
            catalog_path=catalog_path,
            expected_representative_review_sha256=(
                expected_representative_review_sha256
            ),
        )


def _prepare_illustration_replacement_unlocked(
    *,
    store: StateStore,
    book_id: str,
    episode_id: str,
    asset_id: str,
    expected_asset_sha256: str,
    reason: str,
    catalog_path: Path | None = None,
    expected_representative_review_sha256: str | None = None,
) -> ReplacementPreparation:
    """Archive a rejected pair asset and durably authorize its replacement.

    This is deliberately not an import side effect.  Every validation completes
    before the first file move; the returned authorization is then required by
    :func:`import_authorized_replacement`.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise IllustrationReplacementError("replacement_reason_required")
    if not _ASSET_ID.fullmatch(asset_id):
        raise IllustrationReplacementError("replacement_asset_invalid")
    root = _episode_root(store, book_id, episode_id)
    state_path = root / "episode.json"
    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"
    try:
        _require_safe_episode_root(root)
        state_bytes = _read_safe_file(root, state_path)
        manifest_bytes = _read_safe_file(root, manifest_path)
        state = EpisodeState.model_validate_json(state_bytes)
        manifest = IllustrationManifest.model_validate_json(manifest_bytes)
    except IllustrationReplacementError:
        raise
    except Exception:
        raise IllustrationReplacementError("replacement_state_invalid") from None
    if state.book_id != book_id or state.episode_id != episode_id:
        raise IllustrationReplacementError("replacement_state_invalid")
    if state.status == "final_approved":
        raise IllustrationReplacementError("replacement_final_approved")
    if (
        manifest.book_id != book_id
        or manifest.episode_id != episode_id
        or Path(os.path.abspath(manifest.episode_root)) != root
    ):
        raise IllustrationReplacementError("replacement_state_invalid")
    target = _one_job(manifest, asset_id)
    if target.phase not in {"anchor", "continuation"}:
        raise IllustrationReplacementError("replacement_pair_only")
    try:
        from .restyle import (
            IllustrationRestyleError,
            validate_current_illustration_provenance,
        )

        validate_current_illustration_provenance(
            episode=state,
            episode_root=root,
            catalog_path=catalog_path,
        )
    except IllustrationRestyleError as error:
        raise IllustrationReplacementError(error.error_code) from None
    try:
        validate_generated_image_job(
            target, manifest=manifest if target.phase == "continuation" else None
        )
    except IllustrationAssetError:
        raise IllustrationReplacementError("replacement_evidence_invalid") from None
    if (
        target.master_sha256 is None
        or target.master_sha256 != expected_asset_sha256
        or sha256_file(target.output_master) != expected_asset_sha256
    ):
        raise IllustrationReplacementError("replacement_evidence_mismatch")

    review_path = root / "media" / "illustration" / "representative_review.json"
    review_hash: str | None = None
    if review_path.exists():
        _require_safe_file(root, review_path)
        review_hash = sha256_file(review_path)
        if expected_representative_review_sha256 != review_hash:
            raise IllustrationReplacementError("replacement_review_mismatch")
    elif expected_representative_review_sha256 is not None:
        raise IllustrationReplacementError("replacement_review_mismatch")
    approval_path = root / "media" / "illustration" / "representative_approval.json"
    approval_hash: str | None = None
    if approval_path.exists() or os.path.lexists(approval_path):
        _require_safe_file(root, approval_path)
        approval_hash = sha256_file(approval_path)

    history_parent = root / ".private" / "history"
    replacements_root = root / ".private" / "replacements"
    _require_safe_destination(root, history_parent)
    _require_safe_destination(root, replacements_root)
    if history_parent.exists():
        _require_safe_tree(root, history_parent)
    if replacements_root.exists():
        _require_safe_tree(root, replacements_root)

    reset_jobs = _reset_jobs_for(target, manifest)
    old_jobs = tuple(job for job in reset_jobs if job.master_sha256 is not None)
    authorization_paths = tuple(
        replacements_root / f"{job.asset_id}.json" for job in old_jobs
    )
    previous_replacement = state.stage_manifests.get(_REPLACEMENT_STAGE)
    reusable_authorizations: set[Path] = set()
    if previous_replacement is not None:
        previous_closure = _validate_replacement_closure(
            root=root, state=state, manifest=manifest
        )
        for previous_authorization in previous_closure.authorizations.values():
            if previous_authorization.status != "consumed":
                raise IllustrationReplacementError("replacement_already_authorized")
        reusable_authorizations.update(previous_closure.authorization_paths.values())
    for path in authorization_paths:
        if (path.exists() or os.path.lexists(path)) and path not in reusable_authorizations:
            raise IllustrationReplacementError("replacement_already_authorized")
    history_dir = history_parent / _new_history_name(asset_id)
    if history_dir.exists() or os.path.lexists(history_dir):
        raise IllustrationReplacementError("replacement_history_collision")
    _require_safe_destination(root, history_dir)

    updated_manifest = _reset_manifest(manifest, reset_jobs)
    updated_state = state.model_copy(deep=True)
    invalidate_from(
        updated_state,
        "illustration_anchor" if target.phase == "anchor" else "illustration_continuation",
    )
    updated_state.status = _first_incomplete_status(updated_manifest, approval_present=False)
    updated_state.stage_manifests.pop(_REPLACEMENT_STAGE, None)
    updated_state.completed_stages = [
        name for name in updated_state.completed_stages if name != _REPLACEMENT_STAGE
    ]
    updated_state.stale_stages = [
        name for name in updated_state.stale_stages if name != _REPLACEMENT_STAGE
    ]

    source_projection = _expected_archive_sources(
        root=root,
        state=state,
        manifest=manifest,
        reset_jobs=reset_jobs,
        representative_review_sha256=review_hash,
        representative_approval_sha256=approval_hash,
        evidence_files={
            path.relative_to(root).as_posix(): path
            for path in (review_path, approval_path)
            if path.exists()
        },
    )
    sources = tuple(
        (_resolve_relative(root, source), expected_hash)
        for source, (_, expected_hash, _) in source_projection.items()
    )
    for source, expected_hash in sources:
        _require_safe_file(root, source)
        if sha256_file(source) != expected_hash:
            raise IllustrationReplacementError("replacement_archive_hash_mismatch")

    transaction_id = uuid.uuid4().hex
    nonce = uuid.uuid4().hex
    prepared_at_ns = time.time_ns()
    history_relative = history_dir.relative_to(root).as_posix()
    reset_projection_sha256s = {
        job.asset_id or "": _job_projection_sha256(_one_job(updated_manifest, job.asset_id or ""))
        for job in reset_jobs
    }
    moved: list[tuple[Path, Path, str]] = []
    try:
        history_dir.mkdir(parents=True, exist_ok=False)
        _require_safe_tree(root, history_dir)
        _write_new_bytes(history_dir / "episode.json", state_bytes)
        _write_new_bytes(history_dir / "illustration_manifest.json", manifest_bytes)
        affected_payload = {
            name: state.stage_manifests[name].model_dump(mode="json")
            for name in _DOWNSTREAM_STAGES
            if name in state.stage_manifests
        }
        if previous_replacement is not None:
            affected_payload[_REPLACEMENT_STAGE] = previous_replacement.model_dump(mode="json")
        affected_path = history_dir / "affected_stage_manifests.json"
        _write_new_json(affected_path, affected_payload)
        affected_stage_manifests_sha256 = sha256_file(affected_path)
        for source, expected_hash in sources:
            destination = history_dir / "files" / f"{len(moved):03d}-{source.name}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            _require_safe_destination(root, destination)
            os.replace(source, destination)
            moved.append((source, destination, expected_hash))
            if sha256_file(destination) != expected_hash:
                raise IllustrationReplacementError("replacement_archive_hash_mismatch")
        archive_entries = [
            ReplacementArchiveEntry(
                source="episode.json",
                archive="episode.json",
                sha256=_sha256_bytes(state_bytes),
                size_bytes=len(state_bytes),
            ),
            ReplacementArchiveEntry(
                source=manifest_path.relative_to(root).as_posix(),
                archive="illustration_manifest.json",
                sha256=_sha256_bytes(manifest_bytes),
                size_bytes=len(manifest_bytes),
            ),
            ReplacementArchiveEntry(
                source=".transaction/affected_stage_manifests.json",
                archive="affected_stage_manifests.json",
                sha256=affected_stage_manifests_sha256,
                size_bytes=affected_path.stat().st_size,
            ),
        ]
        archive_entries.extend(
            ReplacementArchiveEntry(
                source=source.relative_to(root).as_posix(),
                archive=destination.relative_to(history_dir).as_posix(),
                sha256=expected_hash,
                size_bytes=destination.stat().st_size,
            )
            for source, destination, expected_hash in moved
        )
        receipt_path = history_dir / "replacement_receipt.json"
        receipt = ReplacementReceipt(
            transaction_id=transaction_id,
            nonce=nonce,
            book_id=book_id,
            episode_id=episode_id,
            rejected_asset_id=asset_id,
            reason=reason.strip(),
            expected_asset_sha256=expected_asset_sha256,
            representative_review_sha256=review_hash,
            representative_approval_sha256=approval_hash,
            history_relative_path=history_relative,
            reset_asset_ids=tuple(job.asset_id or "" for job in reset_jobs),
            authorization_asset_ids=tuple(job.asset_id or "" for job in old_jobs),
            reset_job_projection_sha256s=reset_projection_sha256s,
            archived_files=tuple(archive_entries),
            state_snapshot_sha256=_sha256_bytes(state_bytes),
            manifest_snapshot_sha256=_sha256_bytes(manifest_bytes),
            affected_stage_manifests_sha256=affected_stage_manifests_sha256,
            prepared_at_ns=prepared_at_ns,
        )
        _write_new_json(receipt_path, receipt.model_dump(mode="json"))
        receipt_sha256 = sha256_file(receipt_path)
        authorizations = tuple(
            ReplacementAuthorization(
                transaction_id=transaction_id,
                nonce=nonce,
                book_id=book_id,
                episode_id=episode_id,
                asset_id=job.asset_id or "",
                phase=job.phase,
                reason=reason.strip(),
                rejected_sha256=job.master_sha256 or "",
                representative_review_sha256=review_hash,
                storyboard_sha256=manifest.storyboard_sha256,
                style_fingerprint=manifest.style_fingerprint,
                character_lock_sha256=manifest.character_lock_sha256,
                prepared_attempt=job.attempts,
                archive_relative_path=history_relative,
                receipt_sha256=receipt_sha256,
                state_snapshot_sha256=receipt.state_snapshot_sha256,
                manifest_snapshot_sha256=receipt.manifest_snapshot_sha256,
                job_projection_sha256=reset_projection_sha256s[job.asset_id or ""],
                prepared_at_ns=prepared_at_ns,
            )
            for job in old_jobs
        )
        replacements_root.mkdir(parents=True, exist_ok=True)
        for path, authorization in zip(authorization_paths, authorizations, strict=True):
            _write_new_json(path, authorization.model_dump(mode="json"))
        _atomic_write_json(manifest_path, updated_manifest.model_dump(mode="json"))
        replacement_outputs = {
            f"authorization_{authorization.asset_id}": _artifact_ref(path)
            for path, authorization in zip(authorization_paths, authorizations, strict=True)
        }
        replacement_outputs["replacement_receipt"] = _artifact_ref(receipt_path)
        updated_state.stage_manifests[_REPLACEMENT_STAGE] = StageManifest(
            stage=_REPLACEMENT_STAGE,
            status="completed",
            inputs={
                "transaction_id": transaction_id,
                "nonce": nonce,
                "rejected_asset_id": asset_id,
                "reason": reason.strip(),
                "history_relative_path": history_relative,
                "representative_review_sha256": review_hash or "absent",
                "representative_approval_sha256": approval_hash or "absent",
                "state_snapshot_sha256": receipt.state_snapshot_sha256,
                "manifest_snapshot_sha256": receipt.manifest_snapshot_sha256,
                "affected_stage_manifests_sha256": (
                    receipt.affected_stage_manifests_sha256
                ),
                "receipt_sha256": receipt_sha256,
                "reset_job_projection_sha256": _canonical_sha256(
                    reset_projection_sha256s
                ),
            },
            outputs=replacement_outputs,
            config_sha256=_REPLACEMENT_CONFIG_SHA256,
        )
        if _REPLACEMENT_STAGE not in updated_state.completed_stages:
            updated_state.completed_stages.append(_REPLACEMENT_STAGE)
        store.save_episode(updated_state)
    except Exception as error:
        _rollback_prepare(
            root=root,
            state_path=state_path,
            state_bytes=state_bytes,
            manifest_path=manifest_path,
            manifest_bytes=manifest_bytes,
            moved=moved,
            authorization_paths=authorization_paths,
            history_dir=history_dir,
        )
        if isinstance(error, IllustrationReplacementError):
            raise
        raise IllustrationReplacementError("replacement_transaction_failed") from None
    return ReplacementPreparation(
        transaction_id=transaction_id,
        nonce=nonce,
        history_dir=history_dir,
        authorization_paths=authorization_paths,
        reset_asset_ids=tuple(job.asset_id or "" for job in reset_jobs),
        status=updated_state.status,
    )


def import_authorized_replacement(
    *,
    store: StateStore,
    book_id: str,
    episode_id: str,
    asset_id: str,
    nonce: str,
    source_path: Path,
    ffmpeg_command: str | Path,
    inspector: Callable[[Path], ImageFacts] | None = None,
    runner: Callable[..., CommandResult] = run_command,
) -> ReplacementImportResult:
    root = _episode_root(store, book_id, episode_id)
    with _episode_lock(root):
        return _import_authorized_replacement_unlocked(
            store=store,
            book_id=book_id,
            episode_id=episode_id,
            asset_id=asset_id,
            nonce=nonce,
            source_path=source_path,
            ffmpeg_command=ffmpeg_command,
            inspector=inspector,
            runner=runner,
        )


def _import_authorized_replacement_unlocked(
    *,
    store: StateStore,
    book_id: str,
    episode_id: str,
    asset_id: str,
    nonce: str,
    source_path: Path,
    ffmpeg_command: str | Path,
    inspector: Callable[[Path], ImageFacts] | None = None,
    runner: Callable[..., CommandResult] = run_command,
) -> ReplacementImportResult:
    """Import one replacement only under a current, state-bound authorization."""
    if not _ASSET_ID.fullmatch(asset_id):
        raise IllustrationReplacementError("replacement_asset_invalid")
    root = _episode_root(store, book_id, episode_id)
    state_path = root / "episode.json"
    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"
    try:
        _require_safe_episode_root(root)
        state_bytes = _read_safe_file(root, state_path)
        manifest_bytes = _read_safe_file(root, manifest_path)
        state = EpisodeState.model_validate_json(state_bytes)
        manifest = IllustrationManifest.model_validate_json(manifest_bytes)
        closure = _validate_replacement_closure(
            root=root, state=state, manifest=manifest
        )
    except IllustrationReplacementError:
        raise
    except Exception:
        raise IllustrationReplacementError("replacement_authorization_invalid") from None
    if state.status == "final_approved":
        raise IllustrationReplacementError("replacement_final_approved")
    authorization = closure.authorizations.get(asset_id)
    if authorization is None or authorization.status != "authorized" or nonce != authorization.nonce:
        raise IllustrationReplacementError("replacement_authorization_invalid")
    authorization_path = closure.authorization_paths[asset_id]
    job = _one_job(manifest, asset_id)
    pending = next((item for item in manifest.jobs if item.status == "planned"), None)
    if pending is None or pending.asset_id != asset_id:
        raise IllustrationReplacementError("replacement_import_order_invalid")
    try:
        source_hash = sha256_file(Path(source_path))
    except OSError:
        raise IllustrationReplacementError("replacement_import_failed") from None
    if source_hash == authorization.rejected_sha256:
        raise IllustrationReplacementError("replacement_same_sha256")

    imported: ImportedImage | None = None
    try:
        imported = import_image_asset(
            job,
            Path(source_path),
            manifest=manifest,
            inspector=inspector,
            ffmpeg_command=ffmpeg_command,
            runner=runner,
        )
        updated_manifest = record_imported_image(manifest, imported)
        updated_authorizations = dict(closure.authorizations)
        changed_authorization_ids = {asset_id}
        updated_job = _one_job(updated_manifest, asset_id)
        updated_authorizations[asset_id] = authorization.model_copy(
            update={
                "status": "consumed",
                "replacement_sha256": imported.master_sha256,
                "replacement_size_bytes": imported.master_path.stat().st_size,
                "job_projection_sha256": _job_projection_sha256(updated_job),
            }
        )
        if authorization.phase == "anchor":
            updated_manifest, rebound_ids = _rebind_authorized_dependents(
                updated_manifest,
                updated_authorizations,
                changed_anchor=updated_job,
            )
            for dependent_asset_id in rebound_ids:
                dependent_authorization = updated_authorizations[dependent_asset_id]
                updated_authorizations[dependent_asset_id] = (
                    dependent_authorization.model_copy(
                        update={
                            "job_projection_sha256": _job_projection_sha256(
                                _one_job(updated_manifest, dependent_asset_id)
                            )
                        }
                    )
                )
                changed_authorization_ids.add(dependent_asset_id)
        _atomic_write_json(manifest_path, updated_manifest.model_dump(mode="json"))
        for current_asset_id in sorted(changed_authorization_ids):
            updated_authorization = updated_authorizations[current_asset_id]
            _atomic_write_json(
                closure.authorization_paths[current_asset_id],
                updated_authorization.model_dump(mode="json"),
            )
        updated_state = state.model_copy(deep=True)
        updated_stage = updated_state.stage_manifests[_REPLACEMENT_STAGE]
        updated_outputs = dict(updated_stage.outputs)
        for current_asset_id in sorted(changed_authorization_ids):
            path = closure.authorization_paths[current_asset_id]
            updated_outputs[f"authorization_{current_asset_id}"] = _artifact_ref(path)
        updated_state.stage_manifests[_REPLACEMENT_STAGE] = updated_stage.model_copy(
            update={"outputs": updated_outputs}
        )
        approval_present = (
            root / "media" / "illustration" / "representative_approval.json"
        ).is_file()
        updated_state.status = _first_incomplete_status(
            updated_manifest, approval_present=approval_present
        )
        result = ReplacementImportResult(
            transaction_id=authorization.transaction_id,
            imported=imported,
            authorization_path=authorization_path,
            status=updated_state.status,
        )
        store.save_episode(updated_state)
    except Exception as error:
        _rollback_import(
            state_path=state_path,
            state_bytes=state_bytes,
            manifest_path=manifest_path,
            manifest_bytes=manifest_bytes,
            authorization_bytes=closure.authorization_bytes,
            authorization_paths=closure.authorization_paths,
            imported=imported,
        )
        if isinstance(error, IllustrationReplacementError):
            raise
        raise IllustrationReplacementError(
            "replacement_import_failed"
            if isinstance(error, IllustrationAssetError)
            else "replacement_transaction_failed"
        ) from None
    return result


def ensure_ordinary_import_allowed(
    *,
    episode_root: Path,
    state: EpisodeState | None,
    manifest: IllustrationManifest,
    asset_id: str,
) -> None:
    """Fail closed when an asset is controlled by a replacement transaction."""
    root = Path(os.path.abspath(episode_root))
    authorization_path = root / ".private" / "replacements" / f"{asset_id}.json"
    stage = state.stage_manifests.get(_REPLACEMENT_STAGE) if state is not None else None
    label = f"authorization_{asset_id}"
    if not (
        authorization_path.exists()
        or os.path.lexists(authorization_path)
        or (stage is not None and label in stage.outputs)
    ):
        return
    if state is None:
        raise IllustrationReplacementError("replacement_authorization_invalid")
    closure = _validate_replacement_closure(root=root, state=state, manifest=manifest)
    authorization = closure.authorizations.get(asset_id)
    if authorization is None:
        raise IllustrationReplacementError("replacement_authorization_invalid")
    if authorization.status == "authorized":
        raise IllustrationReplacementError("replacement_import_requires_authorization")


def validated_replacement_authorized_asset_ids(
    *,
    episode_root: Path,
    state: EpisodeState,
    manifest: IllustrationManifest,
) -> frozenset[str]:
    """Return pending asset IDs only after validating the complete transaction."""
    closure = _validate_replacement_closure(
        root=Path(os.path.abspath(episode_root)),
        state=state,
        manifest=manifest,
    )
    return frozenset(
        asset_id
        for asset_id, authorization in closure.authorizations.items()
        if authorization.status == "authorized"
    )


def _validate_replacement_closure(
    *, root: Path, state: EpisodeState, manifest: IllustrationManifest
) -> _ReplacementClosure:
    try:
        _require_safe_episode_root(root)
        if (
            state.book_id != manifest.book_id
            or state.episode_id != manifest.episode_id
            or Path(os.path.abspath(manifest.episode_root)) != root
        ):
            raise ValueError
        stage = state.stage_manifests.get(_REPLACEMENT_STAGE)
        if (
            stage is None
            or stage.stage != _REPLACEMENT_STAGE
            or stage.status != "completed"
            or stage.config_sha256 != _REPLACEMENT_CONFIG_SHA256
            or _REPLACEMENT_STAGE not in state.completed_stages
            or _REPLACEMENT_STAGE in state.stale_stages
        ):
            raise ValueError
        history_relative = stage.inputs.get("history_relative_path", "")
        history_dir = _resolve_relative(root, history_relative)
        expected_history_parent = root / ".private" / "history"
        if (
            history_dir.parent != expected_history_parent
            or re.fullmatch(
                r"[0-9]+-replace-S[0-9]+-[AB]-[0-9a-f]{32}", history_dir.name
            )
            is None
        ):
            raise ValueError
        _require_safe_tree(root, history_dir)
        receipt_path = history_dir / "replacement_receipt.json"
        receipt_bytes = _read_safe_file(root, receipt_path)
        receipt = ReplacementReceipt.model_validate_json(receipt_bytes)
        receipt_sha256 = _sha256_bytes(receipt_bytes)
        receipt_artifact = stage.outputs.get("replacement_receipt")
        if (
            receipt_artifact is None
            or Path(os.path.abspath(receipt_artifact.path)) != receipt_path
            or receipt_artifact.sha256 != receipt_sha256
            or receipt_artifact.size_bytes != len(receipt_bytes)
            or receipt.history_relative_path != history_relative
            or receipt.book_id != state.book_id
            or receipt.episode_id != state.episode_id
        ):
            raise ValueError

        state_snapshot = _read_safe_file(root, history_dir / "episode.json")
        manifest_snapshot = _read_safe_file(
            root, history_dir / "illustration_manifest.json"
        )
        if (
            _sha256_bytes(state_snapshot) != receipt.state_snapshot_sha256
            or _sha256_bytes(manifest_snapshot) != receipt.manifest_snapshot_sha256
        ):
            raise ValueError
        snapshot_state = EpisodeState.model_validate_json(state_snapshot)
        snapshot_manifest = IllustrationManifest.model_validate_json(manifest_snapshot)
        expected_affected_payload = {
            name: snapshot_state.stage_manifests[name].model_dump(mode="json")
            for name in _DOWNSTREAM_STAGES
            if name in snapshot_state.stage_manifests
        }
        if _REPLACEMENT_STAGE in snapshot_state.stage_manifests:
            expected_affected_payload[_REPLACEMENT_STAGE] = (
                snapshot_state.stage_manifests[_REPLACEMENT_STAGE].model_dump(mode="json")
            )
        affected_stage_manifests = _read_safe_file(
            root, history_dir / "affected_stage_manifests.json"
        )
        if (
            affected_stage_manifests != _json_bytes(expected_affected_payload)
            or _sha256_bytes(affected_stage_manifests)
            != receipt.affected_stage_manifests_sha256
        ):
            raise ValueError
        snapshot_target = _one_job(snapshot_manifest, receipt.rejected_asset_id)
        snapshot_reset_jobs = _reset_jobs_for(
            snapshot_target, snapshot_manifest, validate_files=False
        )
        expected_reset_ids = tuple(job.asset_id or "" for job in snapshot_reset_jobs)
        expected_authorization_ids = tuple(
            job.asset_id or "" for job in snapshot_reset_jobs if job.master_sha256 is not None
        )
        reset_manifest = _reset_manifest(snapshot_manifest, snapshot_reset_jobs)
        expected_reset_projections = {
            asset: _job_projection_sha256(_one_job(reset_manifest, asset))
            for asset in expected_reset_ids
        }
        if (
            receipt.expected_asset_sha256 != snapshot_target.master_sha256
            or receipt.reset_asset_ids != expected_reset_ids
            or receipt.authorization_asset_ids != expected_authorization_ids
            or receipt.reset_job_projection_sha256s != expected_reset_projections
            or len(set(receipt.reset_asset_ids)) != len(receipt.reset_asset_ids)
        ):
            raise ValueError
        expected_inputs = {
            "transaction_id": receipt.transaction_id,
            "nonce": receipt.nonce,
            "rejected_asset_id": receipt.rejected_asset_id,
            "reason": receipt.reason,
            "history_relative_path": receipt.history_relative_path,
            "representative_review_sha256": (
                receipt.representative_review_sha256 or "absent"
            ),
            "representative_approval_sha256": (
                receipt.representative_approval_sha256 or "absent"
            ),
            "state_snapshot_sha256": receipt.state_snapshot_sha256,
            "manifest_snapshot_sha256": receipt.manifest_snapshot_sha256,
            "affected_stage_manifests_sha256": (
                receipt.affected_stage_manifests_sha256
            ),
            "receipt_sha256": receipt_sha256,
            "reset_job_projection_sha256": _canonical_sha256(
                receipt.reset_job_projection_sha256s
            ),
        }
        if stage.inputs != expected_inputs:
            raise ValueError

        evidence_sources = {
            "media/illustration/representative_review.json",
            "media/illustration/representative_approval.json",
        }
        evidence_files: dict[str, Path] = {}
        for source in evidence_sources:
            matches = [entry for entry in receipt.archived_files if entry.source == source]
            if len(matches) > 1:
                raise ValueError
            if matches:
                evidence_files[source] = _resolve_relative(
                    history_dir, matches[0].archive
                )
        expected_archive_sources = _expected_archive_sources(
            root=root,
            state=snapshot_state,
            manifest=snapshot_manifest,
            reset_jobs=snapshot_reset_jobs,
            representative_review_sha256=receipt.representative_review_sha256,
            representative_approval_sha256=receipt.representative_approval_sha256,
            evidence_files=evidence_files,
        )
        expected_archive_sources.update(
            {
                "episode.json": (
                    "episode.json",
                    receipt.state_snapshot_sha256,
                    len(state_snapshot),
                ),
                "media/illustration/illustration_manifest.json": (
                    "illustration_manifest.json",
                    receipt.manifest_snapshot_sha256,
                    len(manifest_snapshot),
                ),
                ".transaction/affected_stage_manifests.json": (
                    "affected_stage_manifests.json",
                    receipt.affected_stage_manifests_sha256,
                    len(affected_stage_manifests),
                ),
            }
        )
        if {entry.source for entry in receipt.archived_files} != set(
            expected_archive_sources
        ) or len(receipt.archived_files) != len(expected_archive_sources):
            raise ValueError
        seen_sources: set[str] = set()
        seen_archives: set[str] = set()
        for entry in receipt.archived_files:
            source_path = _resolve_relative(root, entry.source)
            archive_relative = PurePosixPath(entry.archive)
            expected_archive, expected_hash, expected_size = expected_archive_sources[
                entry.source
            ]
            if (
                archive_relative.is_absolute()
                or ".." in archive_relative.parts
                or not archive_relative.parts
                or entry.source in seen_sources
                or entry.archive in seen_archives
                or entry.archive != expected_archive
                or entry.sha256 != expected_hash
                or (expected_size is not None and entry.size_bytes != expected_size)
            ):
                raise ValueError
            archive_path = _resolve_relative(history_dir, entry.archive)
            _require_safe_file(root, archive_path)
            if (
                archive_path.stat().st_size != entry.size_bytes
                or sha256_file(archive_path) != entry.sha256
                or not source_path.is_relative_to(root)
            ):
                raise ValueError
            seen_sources.add(entry.source)
            seen_archives.add(entry.archive)

        for sibling_receipt in expected_history_parent.glob(
            "*/replacement_receipt.json"
        ):
            if sibling_receipt == receipt_path:
                continue
            _require_safe_file(root, sibling_receipt)
            try:
                sibling = json.loads(sibling_receipt.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(sibling, dict) and (
                sibling.get("transaction_id") == receipt.transaction_id
                or sibling.get("nonce") == receipt.nonce
            ):
                raise ValueError

        expected_output_labels = {"replacement_receipt"} | {
            f"authorization_{asset}" for asset in receipt.authorization_asset_ids
        }
        if set(stage.outputs) != expected_output_labels:
            raise ValueError
        authorizations: dict[str, ReplacementAuthorization] = {}
        authorization_paths: dict[str, Path] = {}
        authorization_bytes: dict[str, bytes] = {}
        for asset in receipt.authorization_asset_ids:
            path = root / ".private" / "replacements" / f"{asset}.json"
            payload = _read_safe_file(root, path)
            authorization = ReplacementAuthorization.model_validate_json(payload)
            artifact = stage.outputs[f"authorization_{asset}"]
            snapshot_job = _one_job(snapshot_manifest, asset)
            if (
                Path(os.path.abspath(artifact.path)) != path
                or artifact.sha256 != _sha256_bytes(payload)
                or artifact.size_bytes != len(payload)
                or authorization.transaction_id != receipt.transaction_id
                or authorization.nonce != receipt.nonce
                or authorization.book_id != receipt.book_id
                or authorization.episode_id != receipt.episode_id
                or authorization.asset_id != asset
                or authorization.phase != snapshot_job.phase
                or authorization.reason != receipt.reason
                or authorization.rejected_sha256 != snapshot_job.master_sha256
                or authorization.representative_review_sha256
                != receipt.representative_review_sha256
                or authorization.storyboard_sha256 != snapshot_manifest.storyboard_sha256
                or authorization.style_fingerprint != snapshot_manifest.style_fingerprint
                or authorization.character_lock_sha256
                != snapshot_manifest.character_lock_sha256
                or authorization.prepared_attempt != snapshot_job.attempts
                or authorization.archive_relative_path != receipt.history_relative_path
                or authorization.receipt_sha256 != receipt_sha256
                or authorization.state_snapshot_sha256
                != receipt.state_snapshot_sha256
                or authorization.manifest_snapshot_sha256
                != receipt.manifest_snapshot_sha256
                or authorization.prepared_at_ns != receipt.prepared_at_ns
                or (
                    authorization.status == "authorized"
                    and (
                        authorization.replacement_sha256 is not None
                        or authorization.replacement_size_bytes is not None
                    )
                )
                or (
                    authorization.status == "consumed"
                    and (
                        authorization.replacement_sha256 is None
                        or authorization.replacement_size_bytes is None
                    )
                )
            ):
                raise ValueError
            current_job = _one_job(manifest, asset)
            if authorization.status == "authorized":
                if (
                    current_job.phase != authorization.phase
                    or current_job.status != "planned"
                    or current_job.master_sha256 is not None
                    or current_job.bw_sha256 is not None
                    or _job_projection_sha256(current_job)
                    != authorization.job_projection_sha256
                ):
                    raise ValueError
            else:
                _require_safe_file(root, current_job.output_master)
                if (
                    current_job.phase != authorization.phase
                    or current_job.status != "generated"
                    or current_job.master_sha256 != authorization.replacement_sha256
                    or current_job.bw_sha256 is not None
                    or current_job.output_master.stat().st_size
                    != authorization.replacement_size_bytes
                    or sha256_file(current_job.output_master)
                    != authorization.replacement_sha256
                    or _job_projection_sha256(current_job)
                    != authorization.job_projection_sha256
                ):
                    raise ValueError
            authorizations[asset] = authorization
            authorization_paths[asset] = path
            authorization_bytes[asset] = payload
        if any(
            authorization.status == "authorized"
            for authorization in authorizations.values()
        ) and any(
            _one_job(manifest, expected.asset_id or "") != expected
            for expected in reset_manifest.jobs
            if (
                (expected.asset_id or "") not in expected_authorization_ids
                and expected
                != _one_job(snapshot_manifest, expected.asset_id or "")
            )
        ):
            raise ValueError
        return _ReplacementClosure(
            stage=stage,
            receipt=receipt,
            receipt_path=receipt_path,
            authorizations=authorizations,
            authorization_paths=authorization_paths,
            authorization_bytes=authorization_bytes,
        )
    except Exception:
        raise IllustrationReplacementError("replacement_authorization_invalid") from None


def _episode_root(store: StateStore, book_id: str, episode_id: str) -> Path:
    return Path(os.path.abspath(store.root / "books" / book_id / "episodes" / episode_id))


def _episode_lock(root: Path) -> threading.RLock:
    key = os.path.normcase(str(Path(os.path.abspath(root))))
    with _EPISODE_LOCKS_GUARD:
        return _EPISODE_LOCKS.setdefault(key, threading.RLock())


def _one_job(manifest: IllustrationManifest, asset_id: str) -> ImageJob:
    matches = tuple(job for job in manifest.jobs if job.asset_id == asset_id)
    if len(matches) != 1:
        if any(job.phase == "legacy" and job.scene_id == asset_id.removesuffix("-A") for job in manifest.jobs):
            raise IllustrationReplacementError("replacement_pair_only")
        raise IllustrationReplacementError("replacement_asset_invalid")
    return matches[0]


def _reset_jobs_for(
    target: ImageJob,
    manifest: IllustrationManifest,
    *,
    validate_files: bool = True,
) -> tuple[ImageJob, ...]:
    dependencies = _trusted_reference_dependencies(
        manifest, validate_files=validate_files
    )
    affected = {target.asset_id or ""}
    if target.phase == "anchor":
        continuation = tuple(
            job
            for job in manifest.jobs
            if job.scene_id == target.scene_id and job.phase == "continuation"
        )
        if len(continuation) != 1:
            raise IllustrationReplacementError("replacement_pair_invalid")
        affected.add(continuation[0].asset_id or "")
    changed = True
    while changed:
        changed = False
        for job in manifest.jobs:
            asset_id = job.asset_id or ""
            if (
                asset_id not in affected
                and job.master_sha256 is not None
                and dependencies[asset_id].intersection(affected)
            ):
                affected.add(asset_id)
                changed = True
    return tuple(job for job in manifest.jobs if (job.asset_id or "") in affected)


def _reset_manifest(
    manifest: IllustrationManifest, reset_jobs: tuple[ImageJob, ...]
) -> IllustrationManifest:
    reset_ids = {job.asset_id for job in reset_jobs}
    target_phase = reset_jobs[0].phase
    changed_hashes = {
        job.master_sha256 for job in reset_jobs if job.master_sha256 is not None
    }
    updated_jobs: list[ImageJob] = []
    for job in manifest.jobs:
        if job.asset_id not in reset_ids:
            if (
                job.status == "planned"
                and changed_hashes.intersection(job.reference_image_sha256s)
            ):
                updated_jobs.append(
                    job.model_copy(
                        update={
                            "anchor_sha256": None,
                            "reference_image_sha256s": (),
                            "prompt_sha256": image_prompt_identity_sha256(
                                prompt=job.prompt,
                                style_fingerprint=job.style_fingerprint,
                                character_lock_sha256=job.character_lock_sha256,
                                reference_image_sha256s=(),
                            ),
                        }
                    )
                )
            else:
                updated_jobs.append(job)
            continue
        changes: dict[str, object] = {
            "status": "planned",
            "master_sha256": None,
            "bw_sha256": None,
        }
        if target_phase == "anchor":
            changes.update(
                {
                    "anchor_sha256": None,
                    "reference_image_sha256s": (),
                    "prompt_sha256": image_prompt_identity_sha256(
                        prompt=job.prompt,
                        style_fingerprint=job.style_fingerprint,
                        character_lock_sha256=job.character_lock_sha256,
                        reference_image_sha256s=(),
                    ),
                }
            )
        updated_jobs.append(job.model_copy(update=changes))
    updated = manifest.model_copy(update={"jobs": tuple(updated_jobs)})
    if target_phase != "continuation":
        return updated
    continuation = next(job for job in updated.jobs if job.asset_id in reset_ids)
    cleared = continuation.model_copy(
        update={"anchor_sha256": None, "reference_image_sha256s": ()}
    )
    interim = updated.model_copy(
        update={
            "jobs": tuple(
                cleared if job.asset_id == cleared.asset_id else job for job in updated.jobs
            )
        }
    )
    try:
        rebound = bind_job_references(cleared, interim)
    except IllustrationAssetError:
        raise IllustrationReplacementError("replacement_pair_invalid") from None
    return interim.model_copy(
        update={
            "jobs": tuple(
                rebound if job.asset_id == rebound.asset_id else job for job in interim.jobs
            )
        }
    )


def _trusted_reference_dependencies(
    manifest: IllustrationManifest,
    *,
    validate_files: bool,
) -> dict[str, set[str]]:
    anchors: dict[str, ImageJob] = {}
    for job in manifest.jobs:
        if job.phase == "anchor":
            if job.scene_id in anchors:
                raise IllustrationReplacementError("replacement_evidence_invalid")
            anchors[job.scene_id] = job
    dependencies = {job.asset_id or "": set() for job in manifest.jobs}
    try:
        for job in manifest.jobs:
            if not job.reference_image_sha256s:
                if job.phase == "continuation" and job.master_sha256 is not None:
                    raise IllustrationAssetError("continuation_anchor_binding_invalid")
                continue
            if (
                len(job.reference_scene_ids) != len(job.reference_image_sha256s)
                or len(set(job.reference_scene_ids)) != len(job.reference_scene_ids)
            ):
                raise IllustrationAssetError("reference_image_not_ready")
            for scene_id, expected_hash in zip(
                job.reference_scene_ids,
                job.reference_image_sha256s,
                strict=True,
            ):
                reference = anchors.get(scene_id)
                if (
                    reference is None
                    or reference.status not in {"generated", "approved"}
                    or reference.master_sha256 is None
                ):
                    raise IllustrationAssetError("reference_image_not_ready")
                if validate_files:
                    validate_generated_image_job(reference)
                if reference.master_sha256 != expected_hash:
                    raise IllustrationAssetError("reference_image_not_ready")
                dependencies[job.asset_id or ""].add(reference.asset_id or "")
            if validate_files and job.master_sha256 is not None:
                validate_generated_image_job(
                    job,
                    manifest=manifest if job.phase == "continuation" else None,
                )
        return dependencies
    except IllustrationAssetError:
        raise IllustrationReplacementError("replacement_evidence_invalid") from None


def _rebind_authorized_dependents(
    manifest: IllustrationManifest,
    authorizations: dict[str, ReplacementAuthorization],
    *,
    changed_anchor: ImageJob,
) -> tuple[IllustrationManifest, tuple[str, ...]]:
    updated = manifest
    rebound_ids: list[str] = []
    for asset_id, authorization in authorizations.items():
        if authorization.status != "authorized":
            continue
        candidate = _one_job(updated, asset_id)
        if changed_anchor.scene_id not in candidate.reference_scene_ids:
            continue
        try:
            rebound = bind_job_references(candidate, updated)
        except IllustrationAssetError:
            raise IllustrationReplacementError("replacement_pair_invalid") from None
        updated = updated.model_copy(
            update={
                "jobs": tuple(
                    rebound if job.asset_id == asset_id else job
                    for job in updated.jobs
                )
            }
        )
        rebound_ids.append(asset_id)
    return updated, tuple(rebound_ids)


def _expected_archive_sources(
    *,
    root: Path,
    state: EpisodeState,
    manifest: IllustrationManifest,
    reset_jobs: tuple[ImageJob, ...],
    representative_review_sha256: str | None,
    representative_approval_sha256: str | None,
    evidence_files: dict[str, Path],
) -> dict[str, tuple[str, str, int | None]]:
    found: dict[str, tuple[Path, str, int | None]] = {}

    def remember(path: Path, expected_hash: str, expected_size: int | None = None) -> None:
        canonical = Path(os.path.abspath(path))
        key = os.path.normcase(str(canonical))
        previous = found.get(key)
        if previous is not None:
            if previous[1] != expected_hash:
                raise ValueError
            if (
                previous[2] is not None
                and expected_size is not None
                and previous[2] != expected_size
            ):
                raise ValueError
            expected_size = previous[2] if expected_size is None else expected_size
        found[key] = (canonical, expected_hash, expected_size)

    reset_ids = {job.asset_id for job in reset_jobs}
    protected = {
        job.output_master for job in manifest.jobs if job.asset_id not in reset_ids
    }
    protected.update(_stage_output_paths(state, ("tts", "subtitles", "social_cover")))
    protected_keys = {
        os.path.normcase(str(Path(os.path.abspath(path)))) for path in protected
    }
    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"
    for job in reset_jobs:
        if job.master_sha256 is not None:
            remember(job.output_master, job.master_sha256)
    for name in _DOWNSTREAM_STAGES:
        stage = state.stage_manifests.get(name)
        if stage is None or stage.status != "completed":
            continue
        for artifact in stage.outputs.values():
            path = Path(os.path.abspath(artifact.path))
            if path == manifest_path:
                continue
            if os.path.normcase(str(path)) in protected_keys:
                raise ValueError
            remember(path, artifact.sha256, artifact.size_bytes)
    for source, (path, expected_hash, expected_size) in (
        _validated_representative_evidence(
            root=root,
            state=state,
            manifest=manifest,
            representative_review_sha256=representative_review_sha256,
            representative_approval_sha256=representative_approval_sha256,
            evidence_files=evidence_files,
        ).items()
    ):
        remember(root / source, expected_hash, expected_size)
    previous_replacement = state.stage_manifests.get(_REPLACEMENT_STAGE)
    if previous_replacement is not None:
        for label, artifact in previous_replacement.outputs.items():
            if label == "replacement_receipt":
                continue
            remember(Path(artifact.path), artifact.sha256, artifact.size_bytes)

    result: dict[str, tuple[str, str, int | None]] = {}
    ordered = sorted(found.values(), key=lambda item: item[0].as_posix())
    for index, (path, expected_hash, expected_size) in enumerate(ordered):
        source = path.relative_to(root).as_posix()
        result[source] = (
            f"files/{index:03d}-{path.name}",
            expected_hash,
            expected_size,
        )
    return result


def _validated_representative_evidence(
    *,
    root: Path,
    state: EpisodeState,
    manifest: IllustrationManifest,
    representative_review_sha256: str | None,
    representative_approval_sha256: str | None,
    evidence_files: dict[str, Path],
) -> dict[str, tuple[Path, str, int]]:
    review_source = "media/illustration/representative_review.json"
    approval_source = "media/illustration/representative_approval.json"
    review_path = evidence_files.get(review_source)
    approval_path = evidence_files.get(approval_source)
    approval_required = state.status not in {
        "representative_generation_running",
        "awaiting_representative_review",
    }
    if (
        (review_path is None) != (representative_review_sha256 is None)
        or (approval_path is None) != (representative_approval_sha256 is None)
        or (approval_required and (review_path is None or approval_path is None))
        or (approval_path is not None and review_path is None)
    ):
        raise IllustrationReplacementError("replacement_evidence_invalid")
    if review_path is None:
        return {}
    try:
        review_bytes = _read_safe_file(root, review_path)
        if _sha256_bytes(review_bytes) != representative_review_sha256:
            raise ValueError
        review = RepresentativeReview.model_validate_json(review_bytes)
        representatives = tuple(job for job in manifest.jobs if job.representative)
        representative_asset_ids = tuple(job.asset_id or "" for job in representatives)
        representative_image_sha256s = tuple(
            job.master_sha256 or "" for job in representatives
        )
        representative_scene_ids = tuple(
            dict.fromkeys(job.scene_id for job in representatives)
        )
        all_scene_ids = tuple(dict.fromkeys(job.scene_id for job in manifest.jobs))
        if (
            not representatives
            or any(not sha for sha in representative_image_sha256s)
            or review.asset_ids != representative_asset_ids
            or review.image_sha256s != representative_image_sha256s
            or review.scene_ids != representative_scene_ids
            or review.all_scene_ids is None
            or len(review.all_scene_ids) != len(all_scene_ids)
            or set(review.all_scene_ids) != set(all_scene_ids)
            or review.storyboard_sha256 != manifest.storyboard_sha256
            or review.style_fingerprint != manifest.style_fingerprint
            or review.character_lock_sha256 != manifest.character_lock_sha256
        ):
            raise ValueError
        result = {
            review_source: (
                review_path,
                representative_review_sha256,
                len(review_bytes),
            )
        }
        if approval_path is None:
            return result
        approval_bytes = _read_safe_file(root, approval_path)
        if _sha256_bytes(approval_bytes) != representative_approval_sha256:
            raise ValueError
        approval = RepresentativeApproval.model_validate_json(approval_bytes)
        if (
            approval.review_sha256 != canonical_model_sha256(review)
            or approval.image_sha256s != review.image_sha256s
            or approval.storyboard_sha256 != review.storyboard_sha256
            or approval.style_fingerprint != review.style_fingerprint
            or approval.character_lock_sha256 != review.character_lock_sha256
            or approval.asset_ids != review.asset_ids
            or approval.all_scene_ids != review.all_scene_ids
            or approval.reviewer != "user"
        ):
            raise ValueError
        result[approval_source] = (
            approval_path,
            representative_approval_sha256,
            len(approval_bytes),
        )
        return result
    except IllustrationReplacementError:
        raise
    except Exception:
        raise IllustrationReplacementError("replacement_evidence_invalid") from None


def _stage_output_paths(state: EpisodeState, names: tuple[str, ...]) -> set[Path]:
    return {
        Path(artifact.path)
        for name in names
        for artifact in (
            state.stage_manifests[name].outputs.values()
            if name in state.stage_manifests
            else ()
        )
    }


def _first_incomplete_status(
    manifest: IllustrationManifest, *, approval_present: bool
) -> str:
    if any(job.representative and job.status == "planned" for job in manifest.jobs):
        return "representative_generation_running"
    if not approval_present:
        return "awaiting_representative_review"
    if any(job.status == "planned" for job in manifest.jobs):
        return "illustration_batch_running"
    return "illustrations_ready"


def _artifact_ref(path: Path) -> ArtifactRef:
    return ArtifactRef(
        path=str(path), sha256=sha256_file(path), size_bytes=path.stat().st_size
    )


def _new_history_name(asset_id: str) -> str:
    return f"{time.time_ns()}-replace-{asset_id}-{uuid.uuid4().hex}"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(serialized)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _job_projection_sha256(job: ImageJob) -> str:
    return _canonical_sha256(job.model_dump(mode="json"))


def _resolve_relative(parent: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError
    candidate = Path(os.path.abspath(parent.joinpath(*pure.parts)))
    canonical_parent = Path(os.path.abspath(parent))
    if candidate == canonical_parent or not candidate.is_relative_to(canonical_parent):
        raise ValueError
    return candidate


def _read_safe_file(root: Path, path: Path) -> bytes:
    _require_safe_file(root, path)
    try:
        return path.read_bytes()
    except OSError:
        raise IllustrationReplacementError("replacement_state_invalid") from None


def _atomic_write_json(path: Path, payload: object) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(path, data)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _write_new_json(path: Path, payload: object) -> None:
    _write_new_bytes(path, _json_bytes(payload))


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if sha256_file(path) != _sha256_bytes(payload):
        raise IllustrationReplacementError("replacement_archive_hash_mismatch")


def _require_safe_episode_root(root: Path) -> None:
    if not root.is_dir() or _redirect_in_existing_chain(root):
        raise IllustrationReplacementError("replacement_unsafe_path")


def _require_safe_file(root: Path, path: Path) -> None:
    canonical = Path(os.path.abspath(path))
    if (
        canonical == root
        or not canonical.is_relative_to(root)
        or not canonical.is_file()
        or _redirect_in_existing_chain(canonical)
    ):
        raise IllustrationReplacementError("replacement_unsafe_path")


def _require_safe_destination(root: Path, path: Path) -> None:
    canonical = Path(os.path.abspath(path))
    if canonical == root or not canonical.is_relative_to(root) or _redirect_in_existing_chain(canonical):
        raise IllustrationReplacementError("replacement_unsafe_path")


def _require_safe_tree(root: Path, tree: Path) -> None:
    _require_safe_destination(root, tree)
    if not tree.is_dir():
        raise IllustrationReplacementError("replacement_unsafe_path")
    pending = [tree]
    while pending:
        directory = pending.pop()
        if _redirect_entry(directory):
            raise IllustrationReplacementError("replacement_unsafe_path")
        try:
            entries = tuple(os.scandir(directory))
        except OSError:
            raise IllustrationReplacementError("replacement_unsafe_path") from None
        for entry in entries:
            child = Path(entry.path)
            if _redirect_entry(child):
                raise IllustrationReplacementError("replacement_unsafe_path")
            if entry.is_dir(follow_symlinks=False):
                pending.append(child)


def _redirect_entry(path: Path) -> bool:
    try:
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
        return path.is_symlink() or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    except OSError:
        return True


def _redirect_in_existing_chain(path: Path) -> bool:
    current = Path(path)
    while True:
        if os.path.lexists(current) and _redirect_entry(current):
            return True
        if current.parent == current:
            return False
        current = current.parent


def _rollback_prepare(
    *,
    root: Path,
    state_path: Path,
    state_bytes: bytes,
    manifest_path: Path,
    manifest_bytes: bytes,
    moved: list[tuple[Path, Path, str]],
    authorization_paths: tuple[Path, ...],
    history_dir: Path,
) -> None:
    for path in authorization_paths:
        try:
            if path.is_file() and not _redirect_entry(path):
                path.unlink()
        except OSError:
            pass
    for source, destination, expected_hash in reversed(moved):
        try:
            if destination.is_file() and sha256_file(destination) == expected_hash:
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, source)
        except OSError:
            pass
    try:
        _atomic_write_bytes(manifest_path, manifest_bytes)
        _atomic_write_bytes(state_path, state_bytes)
    except OSError:
        pass
    _remove_owned_tree(history_dir)
    _prune_empty(root / ".private" / "replacements", root)
    _prune_empty(root / ".private" / "history", root)
    _prune_empty(root / ".private", root)


def _rollback_import(
    *,
    state_path: Path,
    state_bytes: bytes,
    manifest_path: Path,
    manifest_bytes: bytes,
    authorization_paths: dict[str, Path],
    authorization_bytes: dict[str, bytes],
    imported: ImportedImage | None,
) -> None:
    if imported is not None:
        try:
            if (
                imported.master_path.is_file()
                and not _redirect_entry(imported.master_path)
                and sha256_file(imported.master_path) == imported.master_sha256
            ):
                imported.master_path.unlink()
        except OSError:
            pass
    restores = [(manifest_path, manifest_bytes)]
    restores.extend(
        (authorization_paths[asset], payload)
        for asset, payload in authorization_bytes.items()
    )
    restores.append((state_path, state_bytes))
    for path, payload in restores:
        try:
            _atomic_write_bytes(path, payload)
        except OSError:
            pass


def _remove_owned_tree(root: Path) -> None:
    if not root.exists() or _redirect_entry(root):
        return
    try:
        paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
        for path in paths:
            if _redirect_entry(path):
                return
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        root.rmdir()
    except OSError:
        pass


def _prune_empty(path: Path, stop: Path) -> None:
    current = path
    while current != stop and current.is_relative_to(stop):
        try:
            if not current.exists() or _redirect_entry(current):
                return
            current.rmdir()
        except OSError:
            return
        current = current.parent
