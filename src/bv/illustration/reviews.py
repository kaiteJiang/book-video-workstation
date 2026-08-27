from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .assets import (
    ImageJob,
    IllustrationAssetError,
    IllustrationManifest,
    bind_job_references,
    validate_generated_image_job,
)
from .contracts import IllustrationStoryboard
from .prompts import canonical_model_sha256


class RepresentativeReviewError(RuntimeError):
    """Stable failure for the three-image human review gate."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class RepresentativeReview(_ReviewModel):
    scene_ids: tuple[str, str, str]
    image_sha256s: tuple[str, str, str]
    storyboard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    style_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    character_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RepresentativeApproval(_ReviewModel):
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_sha256s: tuple[str, str, str]
    storyboard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    style_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    character_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: Literal["user"]
    approved_at: datetime


def build_representative_review(
    storyboard: IllustrationStoryboard,
    manifest: IllustrationManifest,
) -> RepresentativeReview:
    try:
        storyboard = IllustrationStoryboard.model_validate(storyboard)
        manifest = IllustrationManifest.model_validate(manifest)
    except ValidationError:
        raise RepresentativeReviewError("representative_contract_invalid") from None
    if (
        canonical_model_sha256(storyboard) != manifest.storyboard_sha256
        or storyboard.book_id != manifest.book_id
        or storyboard.episode_id != manifest.episode_id
        or storyboard.character_lock_sha256 != manifest.character_lock_sha256
    ):
        raise RepresentativeReviewError("representative_dependency_mismatch")
    expected_ids = tuple(
        scene.scene_id for scene in storyboard.scenes if scene.representative_frame
    )
    if len(expected_ids) != 3:
        raise RepresentativeReviewError("representative_count_invalid")
    review = _review_from_manifest(manifest)
    if review.scene_ids != expected_ids:
        raise RepresentativeReviewError("representative_dependency_mismatch")
    return review


def approve_representatives(
    review: RepresentativeReview,
    *,
    approved_image_sha256s: tuple[str, str, str],
    reviewer: Literal["user"],
) -> RepresentativeApproval:
    try:
        review = RepresentativeReview.model_validate(review)
    except ValidationError:
        raise RepresentativeReviewError("representative_contract_invalid") from None
    if approved_image_sha256s != review.image_sha256s:
        raise RepresentativeReviewError("representative_hash_mismatch")
    try:
        return RepresentativeApproval(
            review_sha256=canonical_model_sha256(review),
            image_sha256s=review.image_sha256s,
            storyboard_sha256=review.storyboard_sha256,
            style_fingerprint=review.style_fingerprint,
            character_lock_sha256=review.character_lock_sha256,
            reviewer=reviewer,
            approved_at=datetime.now(UTC),
        )
    except ValidationError:
        raise RepresentativeReviewError("representative_contract_invalid") from None


def unlock_batch_jobs(
    manifest: IllustrationManifest,
    approval: RepresentativeApproval | None,
) -> tuple[ImageJob, ...]:
    if approval is None:
        raise RepresentativeReviewError("representatives_not_approved")
    try:
        manifest = IllustrationManifest.model_validate(manifest)
        approval = RepresentativeApproval.model_validate(approval)
    except ValidationError:
        raise RepresentativeReviewError("representative_approval_stale") from None
    current = _review_from_manifest(manifest)
    if (
        approval.review_sha256 != canonical_model_sha256(current)
        or approval.image_sha256s != current.image_sha256s
        or approval.storyboard_sha256 != manifest.storyboard_sha256
        or approval.style_fingerprint != manifest.style_fingerprint
        or approval.character_lock_sha256 != manifest.character_lock_sha256
        or approval.reviewer != "user"
    ):
        raise RepresentativeReviewError("representative_approval_stale")
    try:
        return tuple(
            bind_job_references(job, manifest)
            for job in manifest.jobs
            if not job.representative
        )
    except IllustrationAssetError:
        raise RepresentativeReviewError("representative_asset_invalid") from None


def _review_from_manifest(manifest: IllustrationManifest) -> RepresentativeReview:
    representatives = tuple(job for job in manifest.jobs if job.representative)
    if len(representatives) != 3:
        raise RepresentativeReviewError("representative_count_invalid")
    image_hashes: list[str] = []
    for job in representatives:
        try:
            validate_generated_image_job(job)
        except IllustrationAssetError:
            raise RepresentativeReviewError("representative_asset_invalid")
        assert job.master_sha256 is not None
        image_hashes.append(job.master_sha256)
    try:
        return RepresentativeReview(
            scene_ids=tuple(job.scene_id for job in representatives),
            image_sha256s=tuple(image_hashes),
            storyboard_sha256=manifest.storyboard_sha256,
            style_fingerprint=manifest.style_fingerprint,
            character_lock_sha256=manifest.character_lock_sha256,
        )
    except ValidationError:
        raise RepresentativeReviewError("representative_contract_invalid") from None
