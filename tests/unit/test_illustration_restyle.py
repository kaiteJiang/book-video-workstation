from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from bv.core.atomic import atomic_write_json
from bv.core.hashing import sha256_file
from bv.illustration.assets import IllustrationManifest, prepare_image_jobs
from bv.illustration.contracts import (
    CharacterBible,
    CharacterLock,
    IllustrationScene,
    IllustrationStoryboard,
    StyleDecision,
)
from bv.illustration.prompts import canonical_model_sha256
import bv.illustration.restyle as restyle_module
from bv.illustration.restyle import IllustrationRestyleError, restyle_illustrations
from bv.illustration.style_selector import load_style_catalog
from bv.state.models import ArtifactRef, EpisodeState, StageManifest


CATALOG = Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json")


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _scene(index: int) -> IllustrationScene:
    start = index * 5_000
    return IllustrationScene(
        scene_id=f"S{index + 1:02d}", start_ms=start, end_ms=start + 5_000,
        from_frame=index * 150, to_frame=(index + 1) * 150,
        narration=f"旁白{index + 1}", narration_span=(index * 3, index * 3 + 3),
        key_line="把生活还给自己", visual_purpose="生活连接", setting="餐桌",
        character_action="停下解释", metaphor=None, composition="居中偏下",
        character_refs=("reader-01",), image_prompt=f"scene-{index + 1}",
        negative_constraints=("禁止文字",), representative_frame=index in {0, 2, 3},
        asset_status="planned", semantic_turn_span=(index * 3, index * 3 + 1),
        semantic_turn_ms=start + 1_000, semantic_turn_frame=index * 150 + 30,
        continuation_action="放下杯子，抬头看向家人",
        continuation_prompt="同一餐桌同一机位，人物放下杯子并抬头",
        continuity_constraints=("same character", "same clothing", "same setting", "same camera direction"),
    )


def _episode_fixture(tmp_path: Path, *, scene_count: int = 4) -> tuple[Path, EpisodeState, IllustrationStoryboard, CharacterBible]:
    root = tmp_path / "books" / "book-demo" / "episodes" / "E001"
    illustration = root / "media" / "illustration"
    images = illustration / "images"
    images.mkdir(parents=True)
    old_style = StyleDecision(
        library_version="1", selected_style="retro-gouache-concept",
        candidate_styles=("retro-gouache-concept", "warm-flat-storybook", "colored-pencil-diary"),
        selection_reasons=("旧风格",),
        rejected_reasons={"warm-flat-storybook": "旧", "colored-pencil-diary": "旧"},
        confidence=1.0, manual_override=False, source_script_sha256="a" * 64,
        style_fingerprint="b" * 64,
    )
    character = CharacterLock(
        role="普通读者", age_range="30至39岁", face="椭圆脸", hair="黑色短发",
        body="中等身材", base_clothing="深蓝衬衫", allowed_variations=(),
        color_markers=("深蓝",), personal_objects=("笔记本",), forbidden_changes=("发型",),
        source_script_sha256="a" * 64, style_fingerprint="b" * 64,
    )
    bible = CharacterBible(source_script_sha256="a" * 64, style_fingerprint="b" * 64, characters=(character,))
    storyboard = IllustrationStoryboard(
        book_id="book-demo", episode_id="E001", width=1080, height=1920, fps=30,
        master_duration_ms=scene_count * 5_000, total_frames=scene_count * 150, script_sha256="a" * 64,
        audio_sha256="c" * 64, subtitle_sha256="d" * 64,
        style_decision_sha256=canonical_model_sha256(old_style),
        character_lock_sha256=canonical_model_sha256(bible),
        character_bible_sha256=canonical_model_sha256(bible),
        sequence_mode="color-story-pair", scenes=tuple(_scene(index).model_copy(update={"representative_frame": index in {0, scene_count // 2, scene_count - 1}}) for index in range(scene_count)),
    )
    style = next(item for item in load_style_catalog(CATALOG) if item.style_id == old_style.selected_style)
    manifest = prepare_image_jobs(storyboard, bible, style, old_style, root)
    jobs = tuple(
        job.model_copy(update={"status": "generated", "master_sha256": _sha(job.asset_id)})
        for job in manifest.jobs
    )
    for job in jobs:
        job.output_master.write_bytes(job.asset_id.encode("utf-8"))
    atomic_write_json(illustration / "style_decision.json", old_style.model_dump(mode="json"))
    atomic_write_json(illustration / "character_bible.json", bible.model_dump(mode="json"))
    atomic_write_json(illustration / "illustration_storyboard.json", storyboard.model_dump(mode="json"))
    atomic_write_json(illustration / "illustration_manifest.json", manifest.model_copy(update={"jobs": jobs}).model_dump(mode="json"))
    (root / "media" / "render").mkdir(parents=True)
    (root / "media" / "render" / "picture_silent.mp4").write_bytes(b"old visual")
    (root / "media" / "final").mkdir(parents=True)
    (root / "media" / "final" / "final.mp4").write_bytes(b"old final")
    (root / "media" / "qc").mkdir(parents=True)
    (root / "media" / "qc" / "report.json").write_bytes(b"old qc")
    approved = root / "script" / "approved.txt"
    approved.parent.mkdir(parents=True)
    approved_text = "".join(scene.narration for scene in storyboard.scenes)
    approved.write_text(approved_text, encoding="utf-8")
    breaks = approved.parent / "subtitle_breaks.txt"
    breaks.write_text(
        "\n".join(scene.narration for scene in storyboard.scenes) + "\n",
        encoding="utf-8",
    )
    episode = EpisodeState(
        book_id="book-demo", episode_id="E001", status="awaiting_final_review",
        script_hash="a" * 64,
        completed_stages=[
            "tts", "asr", "subtitles", "select_style", "plan_illustrations",
            "prepare_representatives", "representative_review", "illustration_images",
            "social_cover", "visual_render", "render", "qc", "delivery", "final_approval",
        ],
        stage_manifests={
            name: StageManifest(
                stage=name,
                status="completed",
                inputs={"media_mode": "full"}
                if name in {"tts", "asr", "subtitles", "select_style", "plan_illustrations", "prepare_representatives"}
                else {},
                outputs={},
                config_sha256="0" * 64,
            )
            for name in (
                "tts", "asr", "subtitles",
                "select_style", "plan_illustrations", "prepare_representatives",
                "representative_review", "illustration_images", "social_cover",
                "visual_render", "render", "qc", "delivery", "final_approval",
            )
        },
    )
    episode.stage_manifests["subtitles"].inputs.update(
        {
            "subtitle_sequence_mode": "color-story-pair",
            "subtitle_breaks_presence": "present",
            "subtitle_breaks_sha256": sha256_file(breaks),
        }
    )
    return root, episode, storyboard, bible


def _semantic(storyboard: IllustrationStoryboard) -> dict[str, object]:
    payload = storyboard.model_dump(mode="json")
    for key in ("style_decision_sha256", "character_lock_sha256", "character_bible_sha256"):
        payload.pop(key)
    return payload


def _media_files(root: Path) -> dict[str, bytes]:
    media = root / "media"
    return {
        path.relative_to(media).as_posix(): path.read_bytes()
        for path in media.rglob("*")
        if path.is_file()
    }


def _current_social_cover(root: Path, episode: EpisodeState) -> Path:
    social = root / "media" / "final" / "social_cover.png"
    social.write_bytes(b"current social cover")
    episode.stage_manifests["social_cover"].outputs = {
        "social_cover": ArtifactRef(
            path=str(social), sha256=sha256_file(social), size_bytes=social.stat().st_size
        )
    }
    return social


def test_restyle_replaces_only_style_dependencies_and_preserves_semantics(tmp_path: Path) -> None:
    root, episode, old_storyboard, _ = _episode_fixture(tmp_path)

    result = restyle_illustrations(
        episode=episode, episode_root=root, catalog_path=CATALOG,
        requested_style_id="warm-flat-storybook",
    )

    updated_storyboard = IllustrationStoryboard.model_validate_json(
        (root / "media" / "illustration" / "illustration_storyboard.json").read_text(encoding="utf-8")
    )
    updated_manifest = IllustrationManifest.model_validate_json(
        (root / "media" / "illustration" / "illustration_manifest.json").read_text(encoding="utf-8")
    )
    assert result.selected_style == "warm-flat-storybook"
    assert _semantic(updated_storyboard) == _semantic(old_storyboard)
    assert updated_storyboard.style_decision_sha256 != old_storyboard.style_decision_sha256
    assert updated_storyboard.character_lock_sha256 != old_storyboard.character_lock_sha256
    assert len(updated_manifest.jobs) == 8
    assert all(
        job.status == "planned"
        and job.master_sha256 is None
        and job.anchor_sha256 is None
        and job.reference_image_sha256s == ()
        for job in updated_manifest.jobs
    )
    assert all(job.style_fingerprint == result.style_fingerprint for job in updated_manifest.jobs)
    assert not (root / "media" / "illustration" / "images").exists()
    recovery = root / ".recovery" / "style-restyle"
    assert len(tuple(recovery.iterdir())) == 1
    archive = next(recovery.iterdir())
    assert (archive / "illustration" / "images" / "S01_anchor.png").is_file()
    assert (archive / "render" / "picture_silent.mp4").read_bytes() == b"old visual"
    assert (archive / "final" / "final.mp4").read_bytes() == b"old final"
    assert (archive / "qc" / "report.json").read_bytes() == b"old qc"


def test_restyle_reads_protected_social_cover_paths_once_before_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    _current_social_cover(root, episode)
    original_paths = restyle_module._social_cover_paths
    calls = 0

    def paths_once(state: EpisodeState, episode_root: Path) -> tuple[Path, ...]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise IllustrationRestyleError("injected_post_install_failure")
        return original_paths(state, episode_root)

    monkeypatch.setattr(restyle_module, "_social_cover_paths", paths_once)
    result = restyle_illustrations(
        episode=episode, episode_root=root, catalog_path=CATALOG,
        requested_style_id="warm-flat-storybook",
    )

    assert calls == 1
    assert result.protected_final_relpaths == (Path("social_cover.png"),)
    assert (root / "media" / "final" / "social_cover.png").read_bytes() == b"current social cover"


@pytest.mark.parametrize(
    "requested_style_id",
    ["legacy-monochrome-reveal", "unknown-style", "retro-gouache-concept"],
)
def test_restyle_rejects_invalid_style_without_writing(tmp_path: Path, requested_style_id: str) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    before = (root / "media" / "illustration" / "style_decision.json").read_bytes()

    with pytest.raises(IllustrationRestyleError):
        restyle_illustrations(
            episode=episode, episode_root=root, catalog_path=CATALOG,
            requested_style_id=requested_style_id,
        )

    assert (root / "media" / "illustration" / "style_decision.json").read_bytes() == before


def test_restyle_rejects_legacy_storyboard_without_writing(tmp_path: Path) -> None:
    root, episode, storyboard, _ = _episode_fixture(tmp_path)
    legacy = storyboard.model_copy(update={"sequence_mode": "legacy-monochrome-reveal"})
    storyboard_path = root / "media" / "illustration" / "illustration_storyboard.json"
    atomic_write_json(storyboard_path, legacy.model_dump(mode="json"))
    before = storyboard_path.read_bytes()

    with pytest.raises(IllustrationRestyleError, match="restyle_legacy_episode"):
        restyle_illustrations(
            episode=episode, episode_root=root, catalog_path=CATALOG,
            requested_style_id="warm-flat-storybook",
        )

    assert storyboard_path.read_bytes() == before


def test_restyle_rejects_final_approved_without_writing(tmp_path: Path) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    episode.status = "final_approved"
    before = (root / "media" / "illustration" / "style_decision.json").read_bytes()

    with pytest.raises(IllustrationRestyleError, match="restyle_final_approved"):
        restyle_illustrations(
            episode=episode, episode_root=root, catalog_path=CATALOG,
            requested_style_id="warm-flat-storybook",
        )

    assert (root / "media" / "illustration" / "style_decision.json").read_bytes() == before


def test_restyle_rejects_internal_windows_junction(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction contract")
    root, episode, _, _ = _episode_fixture(tmp_path)
    illustration = root / "media" / "illustration"
    target = root / "junction-target"
    os.replace(illustration, target)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(illustration), str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("junction privilege unavailable")

    with pytest.raises(IllustrationRestyleError, match="unsafe_restyle_recovery"):
        restyle_illustrations(
            episode=episode, episode_root=root, catalog_path=CATALOG,
            requested_style_id="warm-flat-storybook",
        )


@pytest.mark.parametrize("failure_point", range(1, 7))
def test_restyle_install_failure_restores_every_old_media_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: int
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    social = _current_social_cover(root, episode)
    before = _media_files(root)
    original_replace = os.replace
    calls = 0

    def fail_once(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        nonlocal calls
        source_path = Path(source)
        destination_path = Path(destination)
        is_install_move = (
            ".recovery" in destination_path.parts
            or destination_path == social
            or (
                destination_path == root / "media" / "illustration"
                and ".private" in source_path.parts
            )
        )
        if is_install_move:
            calls += 1
        if is_install_move and calls == failure_point:
            raise OSError("injected install failure")
        original_replace(source, destination)

    monkeypatch.setattr("bv.illustration.restyle.os.replace", fail_once)
    with pytest.raises(IllustrationRestyleError, match="restyle_recovery_failed"):
        restyle_illustrations(
            episode=episode, episode_root=root, catalog_path=CATALOG,
            requested_style_id="warm-flat-storybook",
        )

    assert _media_files(root) == before
    assert social.read_bytes() == b"current social cover"
    staged = tuple((root / ".private").glob("style-restyle-staging-*"))
    assert len(staged) == 1
    assert (staged[0] / "illustration" / "style_decision.json").is_file()


def test_restyle_rejects_nested_windows_junction_without_touching_outside(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction contract")
    root, episode, _, _ = _episode_fixture(tmp_path)
    images = root / "media" / "illustration" / "images"
    outside = tmp_path / "outside-images"
    os.replace(images, outside)
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(images), str(outside)],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("junction privilege unavailable")
    before = {path.name: path.read_bytes() for path in outside.iterdir()}

    with pytest.raises(IllustrationRestyleError, match="unsafe_restyle_recovery"):
        restyle_illustrations(
            episode=episode, episode_root=root, catalog_path=CATALOG,
            requested_style_id="warm-flat-storybook",
        )

    assert {path.name: path.read_bytes() for path in outside.iterdir()} == before


def test_restyle_rejects_social_cover_parent_junction_without_touching_outside(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction contract")
    root, episode, _, _ = _episode_fixture(tmp_path)
    parent = root / "media" / "final" / "social"
    outside = tmp_path / "outside-social"
    outside.mkdir()
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(parent), str(outside)],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("junction privilege unavailable")
    social = parent / "cover.png"
    social.write_bytes(b"outside cover")
    episode.stage_manifests["social_cover"].outputs = {
        "social_cover": ArtifactRef(
            path=str(social), sha256=sha256_file(social), size_bytes=social.stat().st_size
        )
    }

    with pytest.raises(IllustrationRestyleError, match="unsafe_restyle_recovery"):
        restyle_illustrations(
            episode=episode, episode_root=root, catalog_path=CATALOG,
            requested_style_id="warm-flat-storybook",
        )

    assert (outside / "cover.png").read_bytes() == b"outside cover"


@pytest.mark.parametrize("scene_count", [16, 30])
def test_dynamic_story_restyle_retains_all_pairs(tmp_path: Path, scene_count: int) -> None:
    root, episode, old_storyboard, _ = _episode_fixture(tmp_path, scene_count=scene_count)
    result = restyle_illustrations(episode=episode, episode_root=root, catalog_path=CATALOG, requested_style_id="warm-flat-storybook")
    manifest = IllustrationManifest.model_validate_json((root / "media/illustration/illustration_manifest.json").read_text(encoding="utf-8"))
    storyboard = IllustrationStoryboard.model_validate_json((root / "media/illustration/illustration_storyboard.json").read_text(encoding="utf-8"))
    assert len(manifest.jobs) == scene_count * 2
    assert _semantic(storyboard) == _semantic(old_storyboard)
    assert len(result.missing_asset_ids) == 6
    episode.status = "representative_generation_running"
    recovered = restyle_module.validate_restyle_provenance(
        episode=episode, episode_root=root, catalog_path=CATALOG,
        expected_override_sha256=result.override_sha256,
        expected_requested_style_id=result.selected_style,
        expected_style_fingerprint=result.style_fingerprint,
        expected_catalog_sha256=result.catalog_sha256,
        expected_plan_sha256=result.plan_sha256,
    )
    assert len(recovered.storyboard.scenes) == scene_count
