from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bv.illustration.assets import ImageJob, IllustrationManifest
from bv.illustration.contracts import IllustrationScene, IllustrationStoryboard
from bv.illustration.prompts import canonical_model_sha256
from bv.illustration.reviews import (
    RepresentativeReviewError,
    approve_representatives,
    build_representative_review,
    unlock_batch_jobs,
)
from bv.core.atomic import atomic_write_json
from bv.workflow.media_runtime import LocalIllustrationGateway


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_representative_gate_unlocks_batch_then_detects_image_tampering(tmp_path: Path) -> None:
    root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    image_root = root / "media" / "illustration" / "images"
    image_root.mkdir(parents=True)
    representative_ids = {"S01", "S03", "S04"}
    scenes = tuple(
        IllustrationScene(
            scene_id=f"S{index + 1:02d}", start_ms=index * 5_000,
            end_ms=(index + 1) * 5_000, from_frame=index * 150,
            to_frame=(index + 1) * 150, narration=f"旁白{index + 1}",
            narration_span=(index * 3, (index + 1) * 3), key_line="把生活还给自己",
            visual_purpose="生活连接", setting="餐桌", character_action="停下解释",
            metaphor=None, composition="居中偏下", character_refs=("reader-01",),
            image_prompt="普通成年人在餐桌前", negative_constraints=("禁止文字",),
            representative_frame=f"S{index + 1:02d}" in representative_ids,
            asset_status="planned",
        )
        for index in range(4)
    )
    storyboard = IllustrationStoryboard(
        book_id="book-demo", episode_id="E001", width=1080, height=1920, fps=30,
        master_duration_ms=20_000, total_frames=600, script_sha256="a" * 64,
        audio_sha256="b" * 64, subtitle_sha256="c" * 64,
        style_decision_sha256="d" * 64, character_lock_sha256="e" * 64,
        scenes=scenes,
    )
    jobs: list[ImageJob] = []
    for scene_id in ("S01", "S03", "S04", "S02"):
        representative = scene_id in representative_ids
        master = image_root / f"{scene_id}_master.png"
        bw = image_root / f"{scene_id}_bw.png"
        if representative:
            master.write_bytes(f"{scene_id}-master".encode())
            bw.write_bytes(f"{scene_id}-bw".encode())
        jobs.append(
            ImageJob(
                episode_root=root, scene_id=scene_id, representative=representative,
                prompt=f"prompt-{scene_id}", prompt_sha256=_sha(f"prompt-{scene_id}".encode()),
                style_fingerprint="f" * 64, character_lock_sha256="e" * 64,
                reference_scene_ids=() if scene_id == "S01" else ("S01",),
                reference_image_sha256s=(), output_master=master, output_bw=bw,
                master_sha256=_sha(master.read_bytes()) if representative else None,
                bw_sha256=_sha(bw.read_bytes()) if representative else None,
                attempts=1 if representative else 0,
                status="generated" if representative else "planned",
            )
        )
    manifest = IllustrationManifest(
        episode_root=root, book_id="book-demo", episode_id="E001",
        storyboard_sha256=canonical_model_sha256(storyboard), style_fingerprint="f" * 64,
        character_lock_sha256="e" * 64, jobs=tuple(jobs),
    )

    review = build_representative_review(storyboard, manifest)
    approval = approve_representatives(
        review, approved_image_sha256s=review.image_sha256s, reviewer="user"
    )
    unlocked = unlock_batch_jobs(storyboard, manifest, approval)

    assert [job.scene_id for job in unlocked] == ["S02"]
    assert unlocked[0].reference_image_sha256s == (review.image_sha256s[0],)

    manifest.jobs[0].output_master.write_bytes(b"tampered after approval")
    with pytest.raises(RepresentativeReviewError, match="representative_asset_invalid"):
        unlock_batch_jobs(storyboard, manifest, approval)


def test_pair_gate_binds_six_hashes_then_unlocks_s02_by_asset_id(tmp_path: Path) -> None:
    root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    image_root = root / "media" / "illustration" / "images"
    image_root.mkdir(parents=True)
    representative_ids = {"S01", "S03", "S04"}
    scenes = tuple(
        IllustrationScene(
            scene_id=f"S{index + 1:02d}", start_ms=index * 5_000,
            end_ms=(index + 1) * 5_000, from_frame=index * 150,
            to_frame=(index + 1) * 150, narration=f"旁白{index + 1}",
            narration_span=(index * 3, (index + 1) * 3), key_line="把生活还给自己",
            visual_purpose="生活连接", setting="餐桌", character_action="停下解释",
            metaphor=None, composition="居中偏下", character_refs=("reader-01",),
            image_prompt="普通成年人在餐桌前", negative_constraints=("禁止文字",),
            representative_frame=f"S{index + 1:02d}" in representative_ids,
            asset_status="planned", semantic_turn_span=(index * 3, index * 3 + 1),
            semantic_turn_ms=index * 5_000 + 1_000, semantic_turn_frame=index * 150 + 30,
            continuation_action="放下杯子，抬头看向家人",
            continuation_prompt="同一餐桌同一机位，人物放下杯子并抬头",
            continuity_constraints=("same character", "same clothing", "same setting", "same camera direction"),
        )
        for index in range(4)
    )
    storyboard = IllustrationStoryboard(
        book_id="book-demo", episode_id="E001", width=1080, height=1920, fps=30,
        master_duration_ms=20_000, total_frames=600, script_sha256="a" * 64,
        audio_sha256="b" * 64, subtitle_sha256="c" * 64,
        style_decision_sha256="d" * 64, character_lock_sha256="e" * 64,
        sequence_mode="color-story-pair", scenes=scenes,
    )
    jobs: list[ImageJob] = []
    for scene_id in ("S01", "S03", "S04", "S02"):
        representative = scene_id in representative_ids
        anchor = image_root / f"{scene_id}_anchor.png"
        if representative:
            anchor.write_bytes(f"{scene_id}-A".encode())
        anchor_sha = _sha(anchor.read_bytes()) if representative else None
        jobs.append(
            ImageJob(
                episode_root=root, scene_id=scene_id, representative=representative,
                prompt=f"prompt-{scene_id}-A", prompt_sha256=_sha(f"{scene_id}-A".encode()),
                style_fingerprint="f" * 64, character_lock_sha256="e" * 64,
                reference_scene_ids=(), reference_image_sha256s=(), phase="anchor",
                asset_id=f"{scene_id}-A", output_master=anchor, master_sha256=anchor_sha,
                attempts=1 if representative else 0, status="generated" if representative else "planned",
            )
        )
        continuation = image_root / f"{scene_id}_continuation.png"
        if representative:
            continuation.write_bytes(f"{scene_id}-B".encode())
        jobs.append(
            ImageJob(
                episode_root=root, scene_id=scene_id, representative=representative,
                prompt=f"prompt-{scene_id}-B", prompt_sha256=_sha(f"{scene_id}-B".encode()),
                style_fingerprint="f" * 64, character_lock_sha256="e" * 64,
                reference_scene_ids=(scene_id,),
                reference_image_sha256s=(anchor_sha,) if anchor_sha is not None else (),
                phase="continuation", asset_id=f"{scene_id}-B", anchor_sha256=anchor_sha,
                output_master=continuation,
                master_sha256=_sha(continuation.read_bytes()) if representative else None,
                attempts=1 if representative else 0, status="generated" if representative else "planned",
            )
        )
    manifest = IllustrationManifest(
        episode_root=root, book_id="book-demo", episode_id="E001",
        storyboard_sha256=canonical_model_sha256(storyboard), style_fingerprint="f" * 64,
        character_lock_sha256="e" * 64, jobs=tuple(jobs),
    )

    with pytest.raises(RepresentativeReviewError, match="representatives_not_approved"):
        unlock_batch_jobs(storyboard, manifest, None)
    missing_b = manifest.model_copy(update={"jobs": (*manifest.jobs[:1], manifest.jobs[1].model_copy(update={"status": "planned", "master_sha256": None}), *manifest.jobs[2:])})
    with pytest.raises(RepresentativeReviewError, match="representative_asset_invalid"):
        build_representative_review(storyboard, missing_b)
    stale_b = manifest.model_copy(update={"jobs": (*manifest.jobs[:1], manifest.jobs[1].model_copy(update={"anchor_sha256": "0" * 64}), *manifest.jobs[2:])})
    with pytest.raises(RepresentativeReviewError, match="representative_asset_invalid"):
        build_representative_review(storyboard, stale_b)

    review = build_representative_review(storyboard, manifest)
    approval = approve_representatives(review, approved_image_sha256s=review.image_sha256s, reviewer="user")
    unlocked = unlock_batch_jobs(storyboard, manifest, approval)
    atomic_write_json(
        root / "media" / "illustration" / "illustration_manifest.json",
        manifest.model_dump(mode="json"),
    )
    context = type("Context", (), {"episode_root": root, "book_id": "book-demo", "episode_id": "E001"})()
    present, missing = LocalIllustrationGateway(ffmpeg_command="ffmpeg").batch_status(context)

    assert review.asset_ids == ("S01-A", "S01-B", "S03-A", "S03-B", "S04-A", "S04-B")
    assert [job.asset_id for job in unlocked] == ["S02-A", "S02-B"]
    assert all(job.output_bw is None for job in manifest.jobs)
    assert present == review.asset_ids and missing == ("S02-A", "S02-B")
