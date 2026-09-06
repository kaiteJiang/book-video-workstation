from __future__ import annotations

from pathlib import Path

import pytest

from bv.illustration.assets import IllustrationAssetError, IllustrationManifest
from bv.illustration.contracts import SafeArea
from bv.illustration.prompts import canonical_model_sha256
from bv.workflow.media_stages import (
    _book_title_overlay,
    build_renderer_asset_sha256s,
    build_renderer_storyboard,
    MediaStageError,
)
from tests.integration.test_handdrawn_media_workflow import _manifest


def test_renderer_storyboard_includes_book_title_and_author_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    storyboard, manifest = _manifest(episode_root)
    monkeypatch.setattr(
        "bv.workflow.media_stages.validate_generated_image_job",
        lambda *_args, **_kwargs: None,
    )

    payload = build_renderer_storyboard(
        storyboard,
        manifest,
        safe_area=SafeArea(),
        transition="cross-dissolve",
        title_overlay={"title": "沧浪之水", "author": "阎真"},
    )

    assert payload["title_overlay"] == {"title": "沧浪之水", "author": "阎真"}
    assert payload["project"]["show_key_line"] is False
    assert set(payload["scenes"][0]["assets"]) == {"bw", "color"}


def test_book_title_overlay_keeps_every_author(tmp_path: Path) -> None:
    episode_root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    episode_root.mkdir(parents=True)
    (episode_root.parents[1] / "book.json").write_text(
        '{"title":"被讨厌的勇气","authors":["岸见一郎","古贺史健"]}',
        encoding="utf-8",
    )

    assert _book_title_overlay(episode_root) == {
        "title": "被讨厌的勇气",
        "author": "岸见一郎、古贺史健",
    }


def test_renderer_storyboard_emits_story_pair_assets_at_the_semantic_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    legacy_storyboard, legacy_manifest = _manifest(episode_root)
    pair_scenes = tuple(
        scene.model_copy(
            update={
                "start_ms": index * 6_000,
                "end_ms": (index + 1) * 6_000,
                "from_frame": index * 180,
                "to_frame": (index + 1) * 180,
                "narration_span": (index * 12, (index + 1) * 12),
                "semantic_turn_span": (index * 12, index * 12 + 4),
                "semantic_turn_ms": index * 6_000 + 2_000,
                "semantic_turn_frame": index * 180 + 60,
                "continuation_action": "读者抬头望向窗外",
                "continuation_prompt": "同一餐桌同一机位，读者抬头望向窗外",
                "continuity_constraints": ("人物一致", "服装一致", "场所一致", "镜头方向一致"),
            }
        )
        for index, scene in enumerate(legacy_storyboard.scenes[:3])
    )
    storyboard = legacy_storyboard.model_copy(
        update={
            "sequence_mode": "color-story-pair",
            "master_duration_ms": 18_000,
            "total_frames": 540,
            "scenes": pair_scenes,
        }
    )
    jobs = tuple(
        job.model_copy(
            update={
                "scene_id": scene.scene_id,
                "phase": phase,
                "asset_id": f"{scene.scene_id}-{'A' if phase == 'anchor' else 'B'}",
                "output_master": (
                    episode_root
                    / "media"
                    / "illustration"
                    / "images"
                    / f"{scene.scene_id}_{phase}.png"
                ),
                "output_bw": None,
                "master_sha256": "a" * 64,
                "anchor_sha256": "a" * 64 if phase == "continuation" else None,
                "reference_image_sha256s": ("a" * 64,) if phase == "continuation" else (),
            }
        )
        for scene in storyboard.scenes
        for phase, job in (
            ("anchor", legacy_manifest.jobs[0]),
            ("continuation", legacy_manifest.jobs[0]),
        )
    )
    manifest = IllustrationManifest(
        episode_root=episode_root,
        book_id=legacy_manifest.book_id,
        episode_id=legacy_manifest.episode_id,
        storyboard_sha256=canonical_model_sha256(storyboard),
        style_fingerprint=legacy_manifest.style_fingerprint,
        character_lock_sha256=legacy_manifest.character_lock_sha256,
        jobs=jobs,
    )
    monkeypatch.setattr(
        "bv.workflow.media_stages.validate_generated_image_job",
        lambda *_args, **_kwargs: None,
    )

    payload = build_renderer_storyboard(
        storyboard,
        manifest,
        safe_area=SafeArea(),
        transition="cross-dissolve",
    )

    assert payload["scenes"][0]["sequence_mode"] == "color-story-pair"
    assert payload["scenes"][0]["assets"] == {
        "anchor": "media/illustration/images/S01_anchor.png",
        "continuation": "media/illustration/images/S01_continuation.png",
    }
    assert payload["scenes"][0]["semantic_turn_frame"] == 60
    assert payload["project"]["ink_reveal_frames"] == 45
    assert build_renderer_asset_sha256s(storyboard, manifest) == {
        "media/illustration/images/S01_anchor.png": "a" * 64,
        "media/illustration/images/S01_continuation.png": "a" * 64,
        "media/illustration/images/S02_anchor.png": "a" * 64,
        "media/illustration/images/S02_continuation.png": "a" * 64,
        "media/illustration/images/S03_anchor.png": "a" * 64,
        "media/illustration/images/S03_continuation.png": "a" * 64,
    }


@pytest.mark.parametrize(
    "sequence_mode,asset_error",
    [
        ("legacy-monochrome-reveal", "anchor_image_not_ready"),
        ("legacy-monochrome-reveal", "anchor_image_stale"),
        ("legacy-monochrome-reveal", "recorded_image_invalid"),
        ("color-story-pair", "anchor_image_not_ready"),
        ("color-story-pair", "anchor_image_stale"),
        ("color-story-pair", "recorded_image_invalid"),
    ],
)
def test_renderer_storyboard_maps_invalid_assets_to_stable_media_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sequence_mode: str,
    asset_error: str,
) -> None:
    episode_root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    storyboard, manifest = _manifest(episode_root)
    if sequence_mode == "color-story-pair":
        storyboard = storyboard.model_copy(update={"sequence_mode": sequence_mode})
        pair_jobs = tuple(
            job.model_copy(
                update={
                    "phase": phase,
                    "asset_id": f"{job.scene_id}-{'A' if phase == 'anchor' else 'B'}",
                    "output_bw": None,
                }
            )
            for job in manifest.jobs
            for phase in ("anchor", "continuation")
        )
        manifest = manifest.model_copy(
            update={
                "storyboard_sha256": canonical_model_sha256(storyboard),
                "jobs": pair_jobs,
            }
        )
    monkeypatch.setattr(
        "bv.workflow.media_stages.validate_generated_image_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            IllustrationAssetError(asset_error)
        ),
    )

    with pytest.raises(MediaStageError, match="illustration_asset_invalid"):
        build_renderer_storyboard(
            storyboard,
            manifest,
            safe_area=SafeArea(),
            transition="cut",
        )
