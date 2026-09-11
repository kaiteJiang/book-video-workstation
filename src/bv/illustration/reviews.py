from __future__ import annotations

from datetime import UTC, datetime
import json
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
    scene_ids: tuple[str, ...]
    image_sha256s: tuple[str, ...]
    asset_ids: tuple[str, ...] | None = None
    all_scene_ids: tuple[str, ...] | None = None
    storyboard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    style_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    character_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
        payload = super().model_dump(*args, **kwargs)
        if self.asset_ids is None:
            payload.pop("asset_ids", None)
            payload.pop("all_scene_ids", None)
        return payload

    def model_dump_json(self, *args: object, **kwargs: object) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


class RepresentativeApproval(_ReviewModel):
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_sha256s: tuple[str, ...]
    storyboard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    style_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    character_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_ids: tuple[str, ...] | None = None
    all_scene_ids: tuple[str, ...] | None = None
    reviewer: Literal["user"]
    approved_at: datetime

    def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
        payload = super().model_dump(*args, **kwargs)
        if self.asset_ids is None:
            payload.pop("asset_ids", None)
            payload.pop("all_scene_ids", None)
        return payload

    def model_dump_json(self, *args: object, **kwargs: object) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


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
    review = _review_from_manifest(
        manifest,
        sequence_mode=storyboard.sequence_mode,
        expected_scene_ids=expected_ids,
        all_scene_ids=tuple(scene.scene_id for scene in storyboard.scenes),
    )
    if review.scene_ids != expected_ids:
        raise RepresentativeReviewError("representative_dependency_mismatch")
    return review


def approve_representatives(
    review: RepresentativeReview,
    *,
    approved_image_sha256s: tuple[str, ...],
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
            asset_ids=review.asset_ids,
            all_scene_ids=review.all_scene_ids,
            reviewer=reviewer,
            approved_at=datetime.now(UTC),
        )
    except ValidationError:
        raise RepresentativeReviewError("representative_contract_invalid") from None


def unlock_batch_jobs(
    storyboard: IllustrationStoryboard,
    manifest: IllustrationManifest,
    approval: RepresentativeApproval | None,
) -> tuple[ImageJob, ...]:
    if approval is None:
        raise RepresentativeReviewError("representatives_not_approved")
    try:
        storyboard = IllustrationStoryboard.model_validate(storyboard)
        manifest = IllustrationManifest.model_validate(manifest)
        approval = RepresentativeApproval.model_validate(approval)
    except ValidationError:
        raise RepresentativeReviewError("representative_approval_stale") from None
    if (
        canonical_model_sha256(storyboard) != manifest.storyboard_sha256
        or storyboard.book_id != manifest.book_id
        or storyboard.episode_id != manifest.episode_id
        or storyboard.character_lock_sha256 != manifest.character_lock_sha256
    ):
        raise RepresentativeReviewError("representative_dependency_mismatch")
    current = _review_from_manifest(
        manifest,
        sequence_mode=storyboard.sequence_mode,
        expected_scene_ids=tuple(
            scene.scene_id for scene in storyboard.scenes if scene.representative_frame
        ),
        all_scene_ids=tuple(scene.scene_id for scene in storyboard.scenes),
    )
    if (
        approval.review_sha256 != canonical_model_sha256(current)
        or approval.image_sha256s != current.image_sha256s
        or approval.storyboard_sha256 != manifest.storyboard_sha256
        or approval.style_fingerprint != manifest.style_fingerprint
        or approval.character_lock_sha256 != manifest.character_lock_sha256
        or approval.asset_ids != current.asset_ids
        or approval.all_scene_ids != current.all_scene_ids
        or approval.reviewer != "user"
    ):
        raise RepresentativeReviewError("representative_approval_stale")
    try:
        return tuple(
            job if job.phase == "continuation" else bind_job_references(job, manifest)
            for job in manifest.jobs
            if not job.representative
        )
    except IllustrationAssetError:
        raise RepresentativeReviewError("representative_asset_invalid") from None


def _review_from_manifest(
    manifest: IllustrationManifest,
    *,
    sequence_mode: Literal["legacy-monochrome-reveal", "color-story-pair"],
    expected_scene_ids: tuple[str, ...] | None = None,
    all_scene_ids: tuple[str, ...] | None = None,
) -> RepresentativeReview:
    representatives = tuple(job for job in manifest.jobs if job.representative)
    pair_mode = sequence_mode == "color-story-pair"
    if pair_mode:
        if (
            expected_scene_ids is None
            or all_scene_ids is None
            or not 3 <= len(all_scene_ids) <= 48
            or len(set(all_scene_ids)) != len(all_scene_ids)
            or len(expected_scene_ids) != 3
        ):
            raise RepresentativeReviewError("representative_contract_invalid")
        if len(representatives) != 6:
            raise RepresentativeReviewError("representative_count_invalid")
        scene_ids = tuple(dict.fromkeys(job.scene_id for job in representatives))
        if len(scene_ids) != 3:
            raise RepresentativeReviewError("representative_count_invalid")
        scene_ids = expected_scene_ids
        expected_asset_ids = tuple(
            asset_id
            for scene_id in scene_ids
            for asset_id in (f"{scene_id}-A", f"{scene_id}-B")
        )
        expected_identities = tuple(
            (scene_id, phase, asset_id)
            for scene_id in scene_ids
            for phase, asset_id in (
                ("anchor", f"{scene_id}-A"),
                ("continuation", f"{scene_id}-B"),
            )
        )
        identities = tuple(
            (job.scene_id, job.phase, job.asset_id) for job in representatives
        )
        all_identities = tuple(
            (job.scene_id, job.phase, job.asset_id) for job in manifest.jobs
        )
        ordered_scene_ids = (*scene_ids, *(
            scene_id for scene_id in all_scene_ids if scene_id not in scene_ids
        ))
        expected_all_identities = tuple(
            (scene_id, phase, asset_id)
            for scene_id in ordered_scene_ids
            for phase, asset_id in (
                ("anchor", f"{scene_id}-A"),
                ("continuation", f"{scene_id}-B"),
            )
        )
        if (
            identities != expected_identities
            or all_identities != expected_all_identities
            or len(manifest.jobs) != len(expected_all_identities)
            or len(set(all_identities)) != len(all_identities)
            or len({job.asset_id for job in manifest.jobs}) != len(manifest.jobs)
            or len({job.output_master for job in manifest.jobs}) != len(manifest.jobs)
        ):
            raise RepresentativeReviewError("representative_asset_invalid")
    else:
        if len(representatives) != 3:
            raise RepresentativeReviewError("representative_count_invalid")
        scene_ids = tuple(job.scene_id for job in representatives)
        expected_asset_ids = None
        all_scene_ids = None
    image_hashes: list[str] = []
    for job in representatives:
        try:
            validate_generated_image_job(job, manifest=manifest if pair_mode else None)
        except IllustrationAssetError:
            raise RepresentativeReviewError("representative_asset_invalid")
        assert job.master_sha256 is not None
        image_hashes.append(job.master_sha256)
    if pair_mode and len(set(image_hashes)) != 6:
        raise RepresentativeReviewError("representative_asset_invalid")
    try:
        return RepresentativeReview(
            scene_ids=scene_ids,
            image_sha256s=tuple(image_hashes),
            asset_ids=expected_asset_ids,
            all_scene_ids=all_scene_ids,
            storyboard_sha256=manifest.storyboard_sha256,
            style_fingerprint=manifest.style_fingerprint,
            character_lock_sha256=manifest.character_lock_sha256,
        )
    except ValidationError:
        raise RepresentativeReviewError("representative_contract_invalid") from None
