from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bv.illustration.assets import ImageJob, IllustrationManifest
from bv.illustration.contracts import IllustrationScene, IllustrationStoryboard
from bv.illustration.prompts import canonical_model_sha256
from bv.illustration.reviews import (
    RepresentativeReview,
    RepresentativeReviewError,
    approve_representatives,
    build_representative_review,
    unlock_batch_jobs,
)


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _storyboard() -> IllustrationStoryboard:
    scenes = tuple(
        IllustrationScene(
            scene_id=f"S{index + 1:02d}",
            start_ms=index * 5_000,
            end_ms=(index + 1) * 5_000,
            from_frame=index * 150,
            to_frame=(index + 1) * 150,
            narration=f"第{index + 1}段批准旁白",
            narration_span=(index * 8, (index + 1) * 8),
            key_line="把生活还给自己",
            visual_purpose="把价值放回生活",
            setting="餐桌",
            character_action="停下解释",
            metaphor=None,
            composition="居中偏下",
            character_refs=("reader-01",),
            image_prompt="普通成年人在餐桌前",
            negative_constraints=("禁止文字",),
            representative_frame=index in {0, 2, 3},
            asset_status="planned",
        )
        for index in range(4)
    )
    return IllustrationStoryboard(
        book_id="book-demo", episode_id="E001", width=1080, height=1920, fps=30,
        master_duration_ms=20_000, total_frames=600, script_sha256="a" * 64,
        audio_sha256="b" * 64, subtitle_sha256="c" * 64,
        style_decision_sha256="d" * 64, character_lock_sha256="1" * 64,
        scenes=scenes,
    )


def _manifest(tmp_path: Path) -> tuple[IllustrationStoryboard, IllustrationManifest]:
    storyboard = _storyboard()
    root = tmp_path / "episode"
    images = root / "media" / "illustration" / "images"
    images.mkdir(parents=True)
    representative_ids = {"S01", "S03", "S04"}
    jobs: list[ImageJob] = []
    for scene_id in ("S01", "S03", "S04", "S02"):
        master = images / f"{scene_id}_master.png"
        bw = images / f"{scene_id}_bw.png"
        is_representative = scene_id in representative_ids
        if is_representative:
            master.write_bytes(f"{scene_id}-master".encode())
            bw.write_bytes(f"{scene_id}-bw".encode())
        jobs.append(
            ImageJob(
                episode_root=root,
                scene_id=scene_id,
                representative=is_representative,
                prompt=f"prompt-{scene_id}",
                prompt_sha256=_hash(f"prompt-{scene_id}".encode()),
                style_fingerprint="f" * 64,
                character_lock_sha256="1" * 64,
                reference_scene_ids=() if scene_id == "S01" else ("S01",),
                reference_image_sha256s=(),
                output_master=master,
                output_bw=bw,
                master_sha256=_hash(master.read_bytes()) if is_representative else None,
                bw_sha256=_hash(bw.read_bytes()) if is_representative else None,
                attempts=1 if is_representative else 0,
                status="generated" if is_representative else "planned",
            )
        )
    return storyboard, IllustrationManifest(
        episode_root=root,
        book_id="book-demo",
        episode_id="E001",
        storyboard_sha256=canonical_model_sha256(storyboard),
        style_fingerprint="f" * 64,
        character_lock_sha256="1" * 64,
        jobs=tuple(jobs),
    )


def _story_pair_manifest(
    tmp_path: Path,
    *,
    scene_count: int = 4,
) -> tuple[IllustrationStoryboard, IllustrationManifest]:
    representative_indexes = {0, scene_count // 2, scene_count - 1}
    scenes = tuple(
        IllustrationScene(
            scene_id=f"S{index + 1:02d}",
            start_ms=index * 5_000,
            end_ms=(index + 1) * 5_000,
            from_frame=index * 150,
            to_frame=(index + 1) * 150,
            narration=f"第{index + 1}段批准旁白",
            narration_span=(index * 8, (index + 1) * 8),
            key_line="把生活还给自己",
            visual_purpose="把价值放回生活",
            setting="餐桌",
            character_action="停下解释",
            metaphor=None,
            composition="居中偏下",
            character_refs=("reader-01",),
            image_prompt="普通成年人在餐桌前",
            negative_constraints=("禁止文字",),
            representative_frame=index in representative_indexes,
            asset_status="planned",
            semantic_turn_span=(index * 8, index * 8 + 2),
            semantic_turn_ms=index * 5_000 + 1_000,
            semantic_turn_frame=index * 150 + 30,
            continuation_action="主人公放下杯子，抬头看向家人",
            continuation_prompt="同一餐桌同一机位，主人公放下杯子并抬头",
            continuity_constraints=(
                "same character",
                "same clothing",
                "same setting",
                "same camera direction",
            ),
        )
        for index in range(scene_count)
    )
    storyboard = IllustrationStoryboard(
        book_id="book-demo",
        episode_id="E001",
        width=1080,
        height=1920,
        fps=30,
        master_duration_ms=scene_count * 5_000,
        total_frames=scene_count * 150,
        script_sha256="a" * 64,
        audio_sha256="b" * 64,
        subtitle_sha256="c" * 64,
        style_decision_sha256="d" * 64,
        character_lock_sha256="1" * 64,
        sequence_mode="color-story-pair",
        scenes=scenes,
    )
    root = tmp_path / "pair-episode"
    images = root / "media" / "illustration" / "images"
    images.mkdir(parents=True)
    jobs: list[ImageJob] = []
    ordered_scenes = tuple(scene for scene in scenes if scene.representative_frame) + tuple(
        scene for scene in scenes if not scene.representative_frame
    )
    for index, scene in enumerate(ordered_scenes):
        anchor = images / f"{scene.scene_id}_anchor.png"
        for phase, suffix in (("anchor", "A"), ("continuation", "B")):
            output = images / f"{scene.scene_id}_{phase}.png"
            representative = scene.representative_frame
            if representative:
                output.write_bytes(f"{scene.scene_id}-{suffix}".encode())
            digest = _hash(output.read_bytes()) if representative else None
            jobs.append(
                ImageJob(
                    episode_root=root,
                    scene_id=scene.scene_id,
                    representative=representative,
                    prompt=f"prompt-{scene.scene_id}-{suffix}",
                    prompt_sha256=_hash(f"prompt-{scene.scene_id}-{suffix}".encode()),
                    style_fingerprint="f" * 64,
                    character_lock_sha256="1" * 64,
                    reference_scene_ids=(scene.scene_id,) if phase == "continuation" else (),
                    reference_image_sha256s=(
                        (_hash(anchor.read_bytes()),)
                        if phase == "continuation" and representative
                        else ()
                    ),
                    phase=phase,
                    asset_id=f"{scene.scene_id}-{suffix}",
                    anchor_sha256=(
                        _hash(anchor.read_bytes())
                        if phase == "continuation" and representative
                        else None
                    ),
                    output_master=output,
                    master_sha256=digest,
                    attempts=1 if representative else 0,
                    status="generated" if representative else "planned",
                )
            )
    return storyboard, IllustrationManifest(
        episode_root=root,
        book_id="book-demo",
        episode_id="E001",
        storyboard_sha256=canonical_model_sha256(storyboard),
        style_fingerprint="f" * 64,
        character_lock_sha256="1" * 64,
        jobs=tuple(jobs),
    )


def test_batch_jobs_remain_locked_without_exact_representative_approval(tmp_path: Path) -> None:
    storyboard, manifest = _manifest(tmp_path)

    with pytest.raises(RepresentativeReviewError, match="representatives_not_approved"):
        unlock_batch_jobs(storyboard, manifest, approval=None)


def test_legacy_review_hash_omits_pair_only_asset_ids() -> None:
    review = RepresentativeReview(
        scene_ids=("S01", "S03", "S04"),
        image_sha256s=("a" * 64, "b" * 64, "c" * 64),
        storyboard_sha256="d" * 64,
        style_fingerprint="e" * 64,
        character_lock_sha256="f" * 64,
    )
    baseline = {
        "scene_ids": ["S01", "S03", "S04"],
        "image_sha256s": ["a" * 64, "b" * 64, "c" * 64],
        "storyboard_sha256": "d" * 64,
        "style_fingerprint": "e" * 64,
        "character_lock_sha256": "f" * 64,
    }
    expected = hashlib.sha256(
        json.dumps(baseline, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert canonical_model_sha256(review) == expected


def test_review_and_approval_bind_exact_three_images(tmp_path: Path) -> None:
    storyboard, manifest = _manifest(tmp_path)
    review = build_representative_review(storyboard, manifest)
    approval = approve_representatives(
        review,
        approved_image_sha256s=review.image_sha256s,
        reviewer="user",
    )

    unlocked = unlock_batch_jobs(storyboard, manifest, approval)

    assert review.scene_ids == ("S01", "S03", "S04")
    assert approval.image_sha256s == review.image_sha256s
    assert [job.scene_id for job in unlocked] == ["S02"]
    assert unlocked[0].reference_image_sha256s == (review.image_sha256s[0],)


def test_story_pair_review_binds_six_hashes_in_opening_middle_final_order(
    tmp_path: Path,
) -> None:
    storyboard, manifest = _story_pair_manifest(tmp_path)

    review = build_representative_review(storyboard, manifest)
    approval = approve_representatives(
        review,
        approved_image_sha256s=review.image_sha256s,
        reviewer="user",
    )
    unlocked = unlock_batch_jobs(storyboard, manifest, approval)

    assert review.scene_ids == ("S01", "S03", "S04")
    assert review.asset_ids == ("S01-A", "S01-B", "S03-A", "S03-B", "S04-A", "S04-B")
    assert len(review.image_sha256s) == 6
    assert [(job.scene_id, job.phase) for job in unlocked] == [
        ("S02", "anchor"),
        ("S02", "continuation"),
    ]


def test_story_pair_review_rejects_continuation_bound_to_stale_anchor(tmp_path: Path) -> None:
    storyboard, manifest = _story_pair_manifest(tmp_path)
    stale = manifest.jobs[1].model_copy(update={"anchor_sha256": "0" * 64})
    manifest = manifest.model_copy(update={"jobs": (manifest.jobs[0], stale, *manifest.jobs[2:])})

    with pytest.raises(RepresentativeReviewError, match="representative_asset_invalid"):
        build_representative_review(storyboard, manifest)


def test_story_pair_review_rejects_duplicate_hash_or_forged_representative_identity(
    tmp_path: Path,
) -> None:
    storyboard, manifest = _story_pair_manifest(tmp_path)
    duplicate = manifest.jobs[1]
    duplicate.output_master.write_bytes(manifest.jobs[0].output_master.read_bytes())
    duplicate = duplicate.model_copy(update={"master_sha256": manifest.jobs[0].master_sha256})
    duplicated = manifest.model_copy(update={"jobs": (manifest.jobs[0], duplicate, *manifest.jobs[2:])})

    with pytest.raises(RepresentativeReviewError, match="representative_asset_invalid"):
        build_representative_review(storyboard, duplicated)

    forged = manifest.jobs[1].model_copy(update={"asset_id": "S01-A"})
    forged_manifest = manifest.model_copy(update={"jobs": (manifest.jobs[0], forged, *manifest.jobs[2:])})
    with pytest.raises(RepresentativeReviewError, match="representative_asset_invalid"):
        build_representative_review(storyboard, forged_manifest)


@pytest.mark.parametrize("mutation", ["all_legacy", "mixed", "missing", "extra"])
def test_pair_storyboard_rejects_non_pair_or_incomplete_manifest(
    tmp_path: Path,
    mutation: str,
) -> None:
    storyboard, manifest = _story_pair_manifest(tmp_path)
    if mutation == "all_legacy":
        jobs = tuple(
            job.model_copy(update={"phase": "legacy", "asset_id": None, "anchor_sha256": None})
            for job in manifest.jobs
        )
    elif mutation == "mixed":
        jobs = (manifest.jobs[0].model_copy(update={"phase": "legacy"}), *manifest.jobs[1:])
    elif mutation == "missing":
        jobs = tuple(job for job in manifest.jobs if job.asset_id != "S02-B")
    else:
        jobs = (*manifest.jobs, manifest.jobs[-1])
    damaged = manifest.model_copy(update={"jobs": jobs})

    with pytest.raises(RepresentativeReviewError, match="representative_(asset|count)_invalid"):
        build_representative_review(storyboard, damaged)


def test_pair_approval_cannot_unlock_a_manifest_rewritten_as_legacy(tmp_path: Path) -> None:
    storyboard, manifest = _story_pair_manifest(tmp_path)
    review = build_representative_review(storyboard, manifest)
    approval = approve_representatives(
        review, approved_image_sha256s=review.image_sha256s, reviewer="user"
    )
    rewritten = manifest.model_copy(
        update={
            "jobs": tuple(
                job.model_copy(update={"phase": "legacy", "asset_id": None, "anchor_sha256": None})
                for job in manifest.jobs
            )
        }
    )

    with pytest.raises(RepresentativeReviewError, match="representative_(asset|count)_invalid"):
        unlock_batch_jobs(storyboard, rewritten, approval)


def test_pair_storyboard_rejects_legacy_shaped_approval_and_clean_legacy_jobs(
    tmp_path: Path,
) -> None:
    pair_storyboard, pair_manifest = _story_pair_manifest(tmp_path)
    _, legacy_manifest = _manifest(tmp_path)
    legacy_manifest = legacy_manifest.model_copy(
        update={"storyboard_sha256": pair_manifest.storyboard_sha256}
    )
    legacy_review = RepresentativeReview(
        scene_ids=("S01", "S03", "S04"),
        image_sha256s=tuple(job.master_sha256 for job in legacy_manifest.jobs[:3]),
        storyboard_sha256=pair_manifest.storyboard_sha256,
        style_fingerprint=legacy_manifest.style_fingerprint,
        character_lock_sha256=legacy_manifest.character_lock_sha256,
    )
    approval = approve_representatives(
        legacy_review, approved_image_sha256s=legacy_review.image_sha256s, reviewer="user"
    )

    with pytest.raises(RepresentativeReviewError, match="representative_(asset|count)_invalid"):
        unlock_batch_jobs(pair_storyboard, legacy_manifest, approval)


@pytest.mark.parametrize("mutation", ["batch_early", "batch_b_before_a"])
def test_pair_review_rejects_any_nonrepresentative_order_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    storyboard, manifest = _story_pair_manifest(tmp_path)
    if mutation == "batch_early":
        jobs = (*manifest.jobs[:2], *manifest.jobs[-2:], *manifest.jobs[2:-2])
    else:
        jobs = (*manifest.jobs[:-2], manifest.jobs[-1], manifest.jobs[-2])
    damaged = manifest.model_copy(update={"jobs": jobs})

    with pytest.raises(RepresentativeReviewError, match="representative_asset_invalid"):
        build_representative_review(storyboard, damaged)


def test_three_story_pairs_have_no_batch_jobs_after_representative_approval(
    tmp_path: Path,
) -> None:
    storyboard, manifest = _story_pair_manifest(tmp_path, scene_count=3)
    review = build_representative_review(storyboard, manifest)
    approval = approve_representatives(
        review,
        approved_image_sha256s=review.image_sha256s,
        reviewer="user",
    )

    assert unlock_batch_jobs(storyboard, manifest, approval) == ()


def test_user_cannot_approve_different_image_hashes(tmp_path: Path) -> None:
    storyboard, manifest = _manifest(tmp_path)
    review = build_representative_review(storyboard, manifest)

    with pytest.raises(RepresentativeReviewError, match="representative_hash_mismatch"):
        approve_representatives(
            review,
            approved_image_sha256s=("0" * 64, *review.image_sha256s[1:]),
            reviewer="user",
        )


def test_style_change_makes_old_approval_stale(tmp_path: Path) -> None:
    storyboard, manifest = _manifest(tmp_path)
    review = build_representative_review(storyboard, manifest)
    approval = approve_representatives(
        review,
        approved_image_sha256s=review.image_sha256s,
        reviewer="user",
    )
    changed = manifest.model_copy(update={"style_fingerprint": "2" * 64})

    with pytest.raises(RepresentativeReviewError, match="representative_approval_stale"):
        unlock_batch_jobs(storyboard, changed, approval)


def test_missing_representative_file_blocks_review(tmp_path: Path) -> None:
    storyboard, manifest = _manifest(tmp_path)
    manifest.jobs[0].output_master.unlink()

    with pytest.raises(RepresentativeReviewError, match="representative_asset_invalid"):
        build_representative_review(storyboard, manifest)


def test_representative_manifest_cannot_redirect_image_outside_episode(tmp_path: Path) -> None:
    storyboard, manifest = _manifest(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(manifest.jobs[0].output_master.read_bytes())
    forged = manifest.jobs[0].model_copy(update={"output_master": outside})
    manifest = manifest.model_copy(update={"jobs": (forged, *manifest.jobs[1:])})

    with pytest.raises(RepresentativeReviewError, match="representative_asset_invalid"):
        build_representative_review(storyboard, manifest)
