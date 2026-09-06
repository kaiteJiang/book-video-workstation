from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from bv.core.atomic import atomic_write_json
from bv.core.hashing import sha256_file
from bv.config import AppConfig
from bv.illustration.assets import IllustrationManifest, prepare_image_jobs
from bv.illustration.contracts import (
    IllustrationStoryboard,
    StyleDecision,
    load_character_bible,
)
from bv.illustration.replacement import (
    IllustrationReplacementError,
    import_authorized_replacement,
    prepare_illustration_replacement,
)
from bv.illustration.prompts import image_prompt_identity_sha256
from bv.illustration.restyle import restyle_plan_sha256
from bv.illustration.style_selector import load_style_catalog
from bv.state.models import ArtifactRef, EpisodeState, StageManifest
from bv.state.store import StateStore
from bv.workflow.media_runtime import (
    LocalIllustrationGateway,
    MediaProductionService,
    MediaWorkflowError,
)
from bv.workflow.runtime import (
    RuntimeAuthorization,
    _media_runtime_config_sha256,
    build_media_runtime_bindings,
)
from bv.workflow.stages import StageOutcome

from tests.unit.test_illustration_restyle import CATALOG, _episode_fixture, _media_files
from tests.integration.test_handdrawn_end_to_end_fake import _write_color_sources


STYLE_STAGES = ("select_style", "plan_illustrations", "prepare_representatives")
UPSTREAM_STAGES = ("tts", "asr", "subtitles")
PAIR_IDS = ("S01-A", "S01-B", "S03-A", "S03-B", "S04-A", "S04-B")


class _Images:
    def __init__(self, ids: tuple[str, ...] = PAIR_IDS) -> None:
        self.ids = ids

    def status(self, _context):
        return (), self.ids


class _Trap:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def run(self, _context):
        self.calls.append(self.name)
        raise AssertionError(f"stage unexpectedly invoked: {self.name}")


class _ExistingArtifactStage:
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, context):
        path = context.episode_root / "media" / "stage-fixtures" / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stage": self.name}), encoding="utf-8")
        inputs = {}
        if self.name == "subtitles":
            breaks = context.episode_root / "script" / "subtitle_breaks.txt"
            inputs = {
                "subtitle_sequence_mode": "color-story-pair",
                "subtitle_breaks_presence": "present",
                "subtitle_breaks_sha256": sha256_file(breaks),
            }
        return StageOutcome(outputs={self.name: path}, inputs=inputs)


class _PlanningArtifactStage:
    def __init__(self, name: str, root: Path, calls: list[str]) -> None:
        self.name = name
        self.root = root
        self.calls = calls

    def run(self, _context):
        self.calls.append(self.name)
        illustration = self.root / "media" / "illustration"
        outputs = {
            "select_style": {
                "style_decision": illustration / "style_decision.json",
            },
            "plan_illustrations": {
                "character_bible": illustration / "character_bible.json",
                "illustration_storyboard": illustration
                / "illustration_storyboard.json",
            },
            "prepare_representatives": {
                "illustration_manifest": illustration / "illustration_manifest.json",
            },
        }
        inputs: dict[str, str] = {}
        if self.name == "prepare_representatives":
            manifest = IllustrationManifest.model_validate_json(
                outputs[self.name]["illustration_manifest"].read_text(
                    encoding="utf-8"
                )
            )
            inputs = {
                "storyboard_sha256": manifest.storyboard_sha256,
                "style_fingerprint": manifest.style_fingerprint,
                "character_lock_sha256": manifest.character_lock_sha256,
            }
        return StageOutcome(outputs=outputs[self.name], inputs=inputs)


def _stage_hashes(*, style: str) -> dict[str, str]:
    return {
        "tts": "0" * 64,
        "asr": "0" * 64,
        "subtitles": "0" * 64,
        "select_style": style,
        "plan_illustrations": style,
        "prepare_representatives": style,
    }


def _restyled(tmp_path: Path, *, catalog: Path = CATALOG):
    root, episode, storyboard, _ = _episode_fixture(tmp_path)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = MediaProductionService(
        store=store,
        stages={},
        images=_Images(),
        configured_mode="full",
        stage_config_sha256s=_stage_hashes(style="1" * 64),
        restyle_catalog_path=catalog,
    )
    service.restyle("book-demo", "E001", style_id="warm-flat-storybook")
    return root, store, storyboard


def _three_scene_pair_fixture(tmp_path: Path):
    root, episode, storyboard, bible = _episode_fixture(tmp_path)
    scenes = tuple(
        scene.model_copy(update={"representative_frame": True})
        for scene in storyboard.scenes[:3]
    )
    storyboard = storyboard.model_copy(
        update={"master_duration_ms": 15_000, "total_frames": 450, "scenes": scenes}
    )
    decision = StyleDecision.model_validate_json(
        (root / "media" / "illustration" / "style_decision.json").read_text(
            encoding="utf-8"
        )
    )
    style = next(
        item
        for item in load_style_catalog(CATALOG)
        if item.style_id == decision.selected_style
    )
    manifest = prepare_image_jobs(storyboard, bible, style, decision, root)
    illustration = root / "media" / "illustration"
    atomic_write_json(
        illustration / "illustration_storyboard.json",
        storyboard.model_dump(mode="json"),
    )
    atomic_write_json(
        illustration / "illustration_manifest.json",
        manifest.model_dump(mode="json"),
    )
    return root, episode, storyboard


def _local_restyle_service(
    store: StateStore,
    *,
    calls: list[str] | None = None,
) -> MediaProductionService:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("local FFmpeg is required")
    recorded = [] if calls is None else calls
    return MediaProductionService(
        store=store,
        stages={
            name: _Trap(name, recorded)
            for name in (*UPSTREAM_STAGES, *STYLE_STAGES)
        },
        images=LocalIllustrationGateway(ffmpeg_command=ffmpeg),
        legacy_config_sha256="0" * 64,
        configured_mode="full",
        stage_config_sha256s=_stage_hashes(style="1" * 64),
        restyle_catalog_path=CATALOG,
    )


def _bind_original_planning_provenance(
    root: Path,
    episode: EpisodeState,
) -> IllustrationManifest:
    illustration = root / "media" / "illustration"
    decision_path = illustration / "style_decision.json"
    bible_path = illustration / "character_bible.json"
    storyboard_path = illustration / "illustration_storyboard.json"
    manifest_path = illustration / "illustration_manifest.json"
    decision = StyleDecision.model_validate_json(
        decision_path.read_text(encoding="utf-8")
    )
    bible = load_character_bible(json.loads(bible_path.read_text(encoding="utf-8")))
    storyboard = IllustrationStoryboard.model_validate_json(
        storyboard_path.read_text(encoding="utf-8")
    )
    style = next(
        item
        for item in load_style_catalog(CATALOG)
        if item.style_id == decision.selected_style
    )
    manifest = prepare_image_jobs(storyboard, bible, style, decision, root)
    for path in (illustration / "images").glob("*.png"):
        path.unlink()
    atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    outputs = {
        "select_style": {"style_decision": decision_path},
        "plan_illustrations": {
            "character_bible": bible_path,
            "illustration_storyboard": storyboard_path,
        },
        "prepare_representatives": {"illustration_manifest": manifest_path},
    }
    for name in STYLE_STAGES:
        episode.stage_manifests[name] = StageManifest(
            stage=name,
            status="completed",
            inputs=(
                {
                    "media_mode": "full",
                    "storyboard_sha256": manifest.storyboard_sha256,
                    "style_fingerprint": manifest.style_fingerprint,
                    "character_lock_sha256": manifest.character_lock_sha256,
                }
                if name == "prepare_representatives"
                else {"media_mode": "full"}
            ),
            outputs={
                label: ArtifactRef(
                    path=str(path),
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
                for label, path in outputs[name].items()
            },
            config_sha256=f"stage-v1:{'1' * 64}",
        )
    episode.status = "representative_generation_running"
    return manifest


def test_restyle_binds_exact_override_digest_into_all_style_stage_manifests(
    tmp_path: Path,
) -> None:
    root, store, _ = _restyled(tmp_path)

    override = root / "script" / "illustration_style_override.json"
    state = store.load_episode("book-demo", "E001")
    recorded = {
        state.stage_manifests[name].inputs["restyle_override_sha256"]
        for name in STYLE_STAGES
    }

    assert recorded == {sha256_file(override)}
    assert {
        state.stage_manifests[name].inputs["requested_style_id"]
        for name in STYLE_STAGES
    } == {"warm-flat-storybook"}
    storyboard = IllustrationStoryboard.model_validate_json(
        (root / "media" / "illustration" / "illustration_storyboard.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = IllustrationManifest.model_validate_json(
        (root / "media" / "illustration" / "illustration_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        state.stage_manifests[name].inputs["restyle_plan_sha256"]
        for name in STYLE_STAGES
    } == {restyle_plan_sha256(storyboard, manifest)}


@pytest.mark.parametrize("scene_count", [3, 4])
def test_real_local_restyle_ledger_survives_every_restart_through_batch(
    tmp_path: Path,
    scene_count: int,
) -> None:
    if scene_count == 3:
        root, episode, _ = _three_scene_pair_fixture(tmp_path)
    else:
        root, episode, _, _ = _episode_fixture(tmp_path)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    calls: list[str] = []
    service = _local_restyle_service(store, calls=calls)
    view = service.restyle(
        "book-demo", "E001", style_id="warm-flat-storybook"
    )
    expected_representatives = tuple(view.missing_scene_ids)
    sources = _write_color_sources(root, scene_count * 2)

    for asset_id, source in zip(
        expected_representatives, sources[:6], strict=True
    ):
        view = service.import_image("book-demo", "E001", asset_id, source)
        restarted = _local_restyle_service(store, calls=calls)
        assert restarted.status("book-demo", "E001").present_scene_ids == view.present_scene_ids
        prepared = restarted.prepare("book-demo", "E001", mode="full")
        assert prepared.present_scene_ids == view.present_scene_ids
        service = restarted

    approved = service.approve_representatives("book-demo", "E001")
    assert approved.status == "representatives_approved"
    service = _local_restyle_service(store, calls=calls)
    assert service.status("book-demo", "E001").status == "representatives_approved"
    batch = service.prepare_batch("book-demo", "E001")
    if scene_count == 3:
        assert batch.status == "illustrations_ready"
        assert batch.missing_scene_ids == ()
    else:
        assert batch.missing_scene_ids == ("S02-A", "S02-B")
        for asset_id, source in zip(batch.missing_scene_ids, sources[6:], strict=True):
            batch = service.import_image("book-demo", "E001", asset_id, source)
            service = _local_restyle_service(store, calls=calls)
            assert service.status("book-demo", "E001").status == batch.status
        assert batch.status == "illustrations_ready"
    assert _local_restyle_service(store, calls=calls).status(
        "book-demo", "E001"
    ).status == "illustrations_ready"
    assert calls == []


@pytest.mark.parametrize(
    ("scene_count", "expected_reset_ids"),
    (
        (3, ("S01-A", "S01-B", "S02-B", "S03-B")),
        (4, ("S01-A", "S01-B", "S03-B", "S04-B")),
    ),
)
def test_real_restyle_shared_anchor_replacement_closes_cross_scene_references(
    tmp_path: Path,
    scene_count: int,
    expected_reset_ids: tuple[str, ...],
) -> None:
    if scene_count == 3:
        root, episode, _ = _three_scene_pair_fixture(tmp_path)
    else:
        root, episode, _, _ = _episode_fixture(tmp_path)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = _local_restyle_service(store)
    view = service.restyle("book-demo", "E001", style_id="warm-flat-storybook")
    sources = _write_color_sources(root, 10)
    for asset_id, source in zip(view.missing_scene_ids, sources[:6], strict=True):
        view = service.import_image("book-demo", "E001", asset_id, source)
    service.approve_representatives("book-demo", "E001")

    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"
    before = IllustrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    old_anchor = next(job for job in before.jobs if job.asset_id == "S01-A")
    preserved_anchors = {
        job.asset_id: (job.master_sha256, job.output_master.read_bytes())
        for job in before.jobs
        if (
            job.phase == "anchor"
            and job.asset_id != "S01-A"
            and job.status == "generated"
        )
    }
    prepared = prepare_illustration_replacement(
        store=store,
        book_id="book-demo",
        episode_id="E001",
        asset_id="S01-A",
        expected_asset_sha256=old_anchor.master_sha256 or "",
        catalog_path=CATALOG,
        expected_representative_review_sha256=sha256_file(
            root / "media" / "illustration" / "representative_review.json"
        ),
        reason="共享人物锚点需要重画",
    )
    assert prepared.reset_asset_ids == expected_reset_ids
    assert _local_restyle_service(store).status(
        "book-demo", "E001"
    ).missing_scene_ids == expected_reset_ids
    receipt = json.loads(
        (prepared.history_dir / "replacement_receipt.json").read_text(encoding="utf-8")
    )
    assert tuple(receipt["authorization_asset_ids"]) == expected_reset_ids
    assert {
        entry["source"]
        for entry in receipt["archived_files"]
        if str(entry["source"]).endswith(".png")
    }.issuperset(
        {
            next(job for job in before.jobs if job.asset_id == asset_id)
            .output_master.relative_to(root)
            .as_posix()
            for asset_id in expected_reset_ids
        }
    )
    with pytest.raises(MediaWorkflowError, match="replacement_import_requires_authorization"):
        _local_restyle_service(store).import_image(
            "book-demo", "E001", expected_reset_ids[-1], sources[6]
        )

    for asset_id, source in zip(expected_reset_ids, sources[6:], strict=True):
        result = import_authorized_replacement(
            store=store,
            book_id="book-demo",
            episode_id="E001",
            asset_id=asset_id,
            nonce=prepared.nonce,
            source_path=source,
            ffmpeg_command=shutil.which("ffmpeg") or "ffmpeg",
        )
        restarted = _local_restyle_service(store)
        assert restarted.status("book-demo", "E001").status == result.status
        restarted.prepare("book-demo", "E001", mode="full")

    current = IllustrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    new_anchor = next(job for job in current.jobs if job.asset_id == "S01-A")
    assert new_anchor.master_sha256 != old_anchor.master_sha256
    for asset_id in expected_reset_ids[1:]:
        dependent = next(job for job in current.jobs if job.asset_id == asset_id)
        if "S01" in dependent.reference_scene_ids:
            index = dependent.reference_scene_ids.index("S01")
            assert dependent.reference_image_sha256s[index] == new_anchor.master_sha256
    assert {
        job.asset_id: (job.master_sha256, job.output_master.read_bytes())
        for job in current.jobs
        if job.asset_id in preserved_anchors
    } == preserved_anchors

    service = _local_restyle_service(store)
    service.approve_representatives("book-demo", "E001")
    batch = service.prepare_batch("book-demo", "E001")
    if scene_count == 4:
        for asset_id, source in zip(batch.missing_scene_ids, sources[:2], strict=True):
            batch = service.import_image("book-demo", "E001", asset_id, source)
    assert batch.status == "illustrations_ready"
    assert _local_restyle_service(store).status(
        "book-demo", "E001"
    ).status == "illustrations_ready"


@pytest.mark.parametrize("tamper", ("reference_hash", "reference_scene"))
def test_replacement_rejects_cross_scene_reference_graph_tamper_before_writes(
    tmp_path: Path, tamper: str
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = _local_restyle_service(store)
    view = service.restyle("book-demo", "E001", style_id="warm-flat-storybook")
    sources = _write_color_sources(root, 6)
    for asset_id, source in zip(view.missing_scene_ids, sources, strict=True):
        service.import_image("book-demo", "E001", asset_id, source)
    service.approve_representatives("book-demo", "E001")

    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"
    manifest = IllustrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    target = next(job for job in manifest.jobs if job.asset_id == "S01-A")
    dependent = next(job for job in manifest.jobs if job.asset_id == "S03-B")
    if tamper == "reference_hash":
        forged = dependent.model_copy(
            update={
                "reference_image_sha256s": (
                    dependent.reference_image_sha256s[0],
                    "f" * 64,
                )
            }
        )
    else:
        forged = dependent.model_copy(
            update={"reference_scene_ids": (dependent.reference_scene_ids[0], "S04")}
        )
    atomic_write_json(
        manifest_path,
        manifest.model_copy(
            update={
                "jobs": tuple(
                    forged if job.asset_id == forged.asset_id else job
                    for job in manifest.jobs
                )
            }
        ).model_dump(mode="json"),
    )
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(IllustrationReplacementError, match="restyle_provenance_invalid"):
        prepare_illustration_replacement(
            store=store,
            book_id="book-demo",
            episode_id="E001",
            asset_id="S01-A",
            expected_asset_sha256=target.master_sha256 or "",
            catalog_path=CATALOG,
            expected_representative_review_sha256=sha256_file(
                root / "media" / "illustration" / "representative_review.json"
            ),
            reason="拒绝被篡改的跨场引用图",
        )
    assert {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before


@pytest.mark.parametrize(
    "tamper",
    (
        "coordinated_swap",
        "coordinated_add",
        "coordinated_remove",
        "hash_only",
        "prompt_only",
        "stage_refresh",
    ),
)
def test_public_rejection_rejects_immutable_plan_tamper_before_writes(
    tmp_path: Path,
    tamper: str,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = _local_restyle_service(store)
    view = service.restyle("book-demo", "E001", style_id="warm-flat-storybook")
    sources = _write_color_sources(root, 6)
    for asset_id, source in zip(view.missing_scene_ids, sources, strict=True):
        service.import_image("book-demo", "E001", asset_id, source)
    service.approve_representatives("book-demo", "E001")

    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"
    manifest = IllustrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    target = next(job for job in manifest.jobs if job.asset_id == "S01-A")
    dependent = next(job for job in manifest.jobs if job.asset_id == "S03-B")
    other_anchor = next(job for job in manifest.jobs if job.asset_id == "S04-A")
    if tamper in {"coordinated_swap", "stage_refresh"}:
        forged_scene_ids = (dependent.reference_scene_ids[0], "S04")
        forged_hashes = (
            dependent.reference_image_sha256s[0],
            other_anchor.master_sha256 or "",
        )
        update = {
            "reference_scene_ids": forged_scene_ids,
            "reference_image_sha256s": forged_hashes,
        }
    elif tamper == "coordinated_add":
        forged_scene_ids = (*dependent.reference_scene_ids, "S04")
        forged_hashes = (
            *dependent.reference_image_sha256s,
            other_anchor.master_sha256 or "",
        )
        update = {
            "reference_scene_ids": forged_scene_ids,
            "reference_image_sha256s": forged_hashes,
        }
    elif tamper == "coordinated_remove":
        forged_hashes = dependent.reference_image_sha256s[:1]
        update = {
            "reference_scene_ids": dependent.reference_scene_ids[:1],
            "reference_image_sha256s": forged_hashes,
        }
    elif tamper == "hash_only":
        forged_hashes = (
            dependent.reference_image_sha256s[0],
            other_anchor.master_sha256 or "",
        )
        update = {"reference_image_sha256s": forged_hashes}
    else:
        forged_hashes = dependent.reference_image_sha256s
        update = {"prompt": f"{dependent.prompt} forged"}
    forged_prompt = str(update.get("prompt", dependent.prompt))
    update["prompt_sha256"] = image_prompt_identity_sha256(
        prompt=forged_prompt,
        style_fingerprint=dependent.style_fingerprint,
        character_lock_sha256=dependent.character_lock_sha256,
        reference_image_sha256s=forged_hashes,
    )
    forged = dependent.model_copy(update=update)
    forged_manifest = manifest.model_copy(
        update={
            "jobs": tuple(
                forged if job.asset_id == forged.asset_id else job
                for job in manifest.jobs
            )
        }
    )
    atomic_write_json(
        manifest_path,
        forged_manifest.model_dump(mode="json"),
    )
    if tamper == "stage_refresh":
        storyboard = IllustrationStoryboard.model_validate_json(
            (root / "media" / "illustration" / "illustration_storyboard.json").read_text(
                encoding="utf-8"
            )
        )
        state = store.load_episode("book-demo", "E001")
        refreshed_plan_sha256 = restyle_plan_sha256(storyboard, forged_manifest)
        for name in STYLE_STAGES:
            state.stage_manifests[name].inputs[
                "restyle_plan_sha256"
            ] = refreshed_plan_sha256
        state.stage_manifests["prepare_representatives"].outputs[
            "illustration_manifest"
        ] = ArtifactRef(
            path=str(manifest_path),
            sha256=sha256_file(manifest_path),
            size_bytes=manifest_path.stat().st_size,
        )
        store.save_episode(state)
    before_directories = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    )
    before_files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(MediaWorkflowError, match="restyle_provenance_invalid"):
        service.reject_illustration(
            "book-demo",
            "E001",
            asset_id="S01-A",
            expected_asset_sha256=target.master_sha256 or "",
            expected_representative_review_sha256=sha256_file(
                root / "media" / "illustration" / "representative_review.json"
            ),
            reason="拒绝协调篡改后的共享引用图",
        )
    assert tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    ) == before_directories
    assert {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before_files


def test_original_pair_rejection_uses_bound_immutable_planning_projection(
    tmp_path: Path,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    _bind_original_planning_provenance(root, episode)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = _local_restyle_service(store)
    sources = _write_color_sources(root, 6)
    for asset_id, source in zip(PAIR_IDS, sources, strict=True):
        service.import_image("book-demo", "E001", asset_id, source)
    service.approve_representatives("book-demo", "E001")

    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"
    manifest = IllustrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    target = next(job for job in manifest.jobs if job.asset_id == "S01-A")
    dependent = next(job for job in manifest.jobs if job.asset_id == "S03-B")
    other_anchor = next(job for job in manifest.jobs if job.asset_id == "S04-A")
    forged_hashes = (
        dependent.reference_image_sha256s[0],
        other_anchor.master_sha256 or "",
    )
    forged = dependent.model_copy(
        update={
            "reference_scene_ids": (dependent.reference_scene_ids[0], "S04"),
            "reference_image_sha256s": forged_hashes,
            "prompt_sha256": image_prompt_identity_sha256(
                prompt=dependent.prompt,
                style_fingerprint=dependent.style_fingerprint,
                character_lock_sha256=dependent.character_lock_sha256,
                reference_image_sha256s=forged_hashes,
            ),
        }
    )
    atomic_write_json(
        manifest_path,
        manifest.model_copy(
            update={
                "jobs": tuple(
                    forged if job.asset_id == forged.asset_id else job
                    for job in manifest.jobs
                )
            }
        ).model_dump(mode="json"),
    )
    state = store.load_episode("book-demo", "E001")
    state.stage_manifests["prepare_representatives"].outputs[
        "illustration_manifest"
    ] = ArtifactRef(
        path=str(manifest_path),
        sha256=sha256_file(manifest_path),
        size_bytes=manifest_path.stat().st_size,
    )
    store.save_episode(state)
    before_directories = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    )
    before_files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(MediaWorkflowError, match="illustration_provenance_invalid"):
        service.reject_illustration(
            "book-demo",
            "E001",
            asset_id="S01-A",
            expected_asset_sha256=target.master_sha256 or "",
            expected_representative_review_sha256=sha256_file(
                root / "media" / "illustration" / "representative_review.json"
            ),
            reason="拒绝被改写的原始共享引用图",
        )
    assert tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    ) == before_directories
    assert {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before_files


@pytest.mark.parametrize("scene_count", (3, 4))
@pytest.mark.parametrize(
    "tamper",
    (
        "coordinated_swap",
        "coordinated_add",
        "coordinated_remove",
        "hash_only",
        "prompt_only",
        "stage_refresh",
    ),
)
def test_prepare_validates_complete_original_pair_planning_projection(
    tmp_path: Path,
    scene_count: int,
    tamper: str,
) -> None:
    if scene_count == 3:
        root, episode, _ = _three_scene_pair_fixture(tmp_path)
    else:
        root, episode, _, _ = _episode_fixture(tmp_path)
    _bind_original_planning_provenance(root, episode)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    calls: list[str] = []
    service = _local_restyle_service(store, calls=calls)
    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"
    planned = IllustrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    representative_ids = tuple(
        job.asset_id or "" for job in planned.jobs if job.representative
    )
    for asset_id, source in zip(
        representative_ids,
        _write_color_sources(root, len(representative_ids)),
        strict=True,
    ):
        service.import_image("book-demo", "E001", asset_id, source)
    service.approve_representatives("book-demo", "E001")
    manifest = IllustrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    dependent = next(
        job
        for job in manifest.jobs
        if job.phase == "continuation" and len(job.reference_scene_ids) > 1
    )
    other_anchor = next(
        job
        for job in manifest.jobs
        if job.phase == "anchor" and job.scene_id not in dependent.reference_scene_ids
    )
    if tamper in {"coordinated_swap", "stage_refresh"}:
        forged_hashes = (
            dependent.reference_image_sha256s[0],
            other_anchor.master_sha256 or "",
        )
        update: dict[str, object] = {
            "reference_scene_ids": (
                dependent.reference_scene_ids[0],
                other_anchor.scene_id,
            ),
            "reference_image_sha256s": forged_hashes,
        }
    elif tamper == "coordinated_add":
        forged_hashes = (
            *dependent.reference_image_sha256s,
            other_anchor.master_sha256 or "",
        )
        update = {
            "reference_scene_ids": (
                *dependent.reference_scene_ids,
                other_anchor.scene_id,
            ),
            "reference_image_sha256s": forged_hashes,
        }
    elif tamper == "coordinated_remove":
        forged_hashes = dependent.reference_image_sha256s[:1]
        update = {
            "reference_scene_ids": dependent.reference_scene_ids[:1],
            "reference_image_sha256s": forged_hashes,
        }
    elif tamper == "hash_only":
        forged_hashes = (
            dependent.reference_image_sha256s[0],
            other_anchor.master_sha256 or "",
        )
        update = {"reference_image_sha256s": forged_hashes}
    else:
        forged_hashes = dependent.reference_image_sha256s
        update = {"prompt": f"{dependent.prompt} forged"}
    forged_prompt = str(update.get("prompt", dependent.prompt))
    update["prompt_sha256"] = image_prompt_identity_sha256(
        prompt=forged_prompt,
        style_fingerprint=dependent.style_fingerprint,
        character_lock_sha256=dependent.character_lock_sha256,
        reference_image_sha256s=forged_hashes,
    )
    forged = dependent.model_copy(update=update)
    forged_manifest = manifest.model_copy(
        update={
            "jobs": tuple(
                forged if job.asset_id == forged.asset_id else job
                for job in manifest.jobs
            )
        }
    )
    atomic_write_json(manifest_path, forged_manifest.model_dump(mode="json"))
    if tamper == "stage_refresh":
        state = store.load_episode("book-demo", "E001")
        state.stage_manifests["prepare_representatives"].outputs[
            "illustration_manifest"
        ] = ArtifactRef(
            path=str(manifest_path),
            sha256=sha256_file(manifest_path),
            size_bytes=manifest_path.stat().st_size,
        )
        store.save_episode(state)
    before_directories = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    )
    before_files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(MediaWorkflowError, match="illustration_provenance_invalid"):
        service.prepare("book-demo", "E001", mode="full")

    assert calls == []
    assert tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    ) == before_directories
    assert {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before_files


@pytest.mark.parametrize("first_missing", (*STYLE_STAGES, None))
def test_prepare_rebuilds_only_clean_original_pair_planning_suffix(
    tmp_path: Path,
    first_missing: str | None,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    _bind_original_planning_provenance(root, episode)
    first_index = (
        len(STYLE_STAGES)
        if first_missing is None
        else STYLE_STAGES.index(first_missing)
    )
    expected_calls = list(STYLE_STAGES[first_index:])
    for name in expected_calls:
        episode.stage_manifests.pop(name)
        episode.completed_stages.remove(name)
    episode.status = "illustration_planning"
    store = StateStore(tmp_path)
    store.save_episode(episode)
    calls: list[str] = []
    service = _local_restyle_service(store, calls=calls)
    for name in STYLE_STAGES:
        service.stages[name] = _PlanningArtifactStage(name, root, calls)

    view = service.prepare("book-demo", "E001", mode="full")

    assert calls == expected_calls
    assert view.status == "representative_generation_running"
    service.status("book-demo", "E001")


@pytest.mark.parametrize(
    "tamper",
    (
        "manifest_without_marker",
        "missing_manifest_with_marker",
        "stale",
        "missing_middle",
        "invalid_outputs",
    ),
)
def test_prepare_rejects_contradictory_early_original_planning_state(
    tmp_path: Path,
    tamper: str,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    _bind_original_planning_provenance(root, episode)
    episode.status = "illustration_planning"
    if tamper == "manifest_without_marker":
        episode.completed_stages.remove("plan_illustrations")
    elif tamper == "missing_manifest_with_marker":
        episode.stage_manifests.pop("plan_illustrations")
    elif tamper == "stale":
        episode.stale_stages.append("plan_illustrations")
    elif tamper == "missing_middle":
        episode.stage_manifests.pop("plan_illustrations")
        episode.completed_stages.remove("plan_illustrations")
    else:
        episode.stage_manifests["plan_illustrations"].outputs.pop(
            "character_bible"
        )
    store = StateStore(tmp_path)
    store.save_episode(episode)
    calls: list[str] = []
    service = _local_restyle_service(store, calls=calls)
    before_directories = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    )
    before_files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(MediaWorkflowError, match="illustration_provenance_invalid"):
        service.prepare("book-demo", "E001", mode="full")

    assert calls == []
    assert tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    ) == before_directories
    assert {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before_files


@pytest.mark.parametrize("scene_count", (3, 4))
@pytest.mark.parametrize("entrypoint", ("status", "reject", "prepare"))
@pytest.mark.parametrize(
    "tamper",
    (
        "all_missing",
        "one_missing",
        "two_missing",
        "completed_marker_missing",
        "stale",
        "outputs",
        "stage_identity",
    ),
)
def test_original_pair_requires_complete_current_planning_provenance(
    tmp_path: Path,
    scene_count: int,
    entrypoint: str,
    tamper: str,
) -> None:
    if scene_count == 3:
        root, episode, _ = _three_scene_pair_fixture(tmp_path)
    else:
        root, episode, _, _ = _episode_fixture(tmp_path)
    _bind_original_planning_provenance(root, episode)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = _local_restyle_service(store)
    manifest = IllustrationManifest.model_validate_json(
        (root / "media" / "illustration" / "illustration_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    representative_ids = tuple(
        job.asset_id or "" for job in manifest.jobs if job.representative
    )
    sources = _write_color_sources(root, 6)
    for asset_id, source in zip(representative_ids, sources, strict=True):
        service.import_image("book-demo", "E001", asset_id, source)
    service.approve_representatives("book-demo", "E001")

    state = store.load_episode("book-demo", "E001")
    if tamper == "all_missing":
        removed = STYLE_STAGES
    elif tamper == "one_missing":
        removed = ("select_style",)
    elif tamper == "two_missing":
        removed = ("select_style", "plan_illustrations")
    else:
        removed = ()
    for name in removed:
        state.stage_manifests.pop(name, None)
        state.completed_stages = [
            stage for stage in state.completed_stages if stage != name
        ]
        state.stale_stages = [stage for stage in state.stale_stages if stage != name]
    if tamper == "completed_marker_missing":
        state.completed_stages = [
            stage for stage in state.completed_stages if stage != "plan_illustrations"
        ]
    elif tamper == "stale":
        state.stale_stages.append("plan_illustrations")
    elif tamper == "outputs":
        state.stage_manifests["select_style"].outputs["extra"] = next(
            iter(state.stage_manifests["select_style"].outputs.values())
        )
    elif tamper == "stage_identity":
        state.stage_manifests["plan_illustrations"] = state.stage_manifests[
            "plan_illustrations"
        ].model_copy(update={"stage": "select_style"})
    store.save_episode(state)
    current = IllustrationManifest.model_validate_json(
        (root / "media" / "illustration" / "illustration_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    target = next(job for job in current.jobs if job.asset_id == "S01-A")
    before_directories = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    )
    before_files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(MediaWorkflowError, match="illustration_provenance_invalid"):
        if entrypoint == "status":
            service.status("book-demo", "E001")
        elif entrypoint == "reject":
            service.reject_illustration(
                "book-demo",
                "E001",
                asset_id="S01-A",
                expected_asset_sha256=target.master_sha256 or "",
                expected_representative_review_sha256=sha256_file(
                    root / "media" / "illustration" / "representative_review.json"
                ),
                reason="缺失原始 pair planning provenance",
            )
        else:
            service.prepare("book-demo", "E001", mode="full")
    assert tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    ) == before_directories
    assert {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before_files


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("master_sha256", "f" * 64),
        ("anchor_sha256", "e" * 64),
        ("reference_image_sha256s", ("d" * 64,)),
        ("prompt", "forged mutable prompt"),
    ],
)
def test_restyle_mutable_ledger_tamper_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root, store, _ = _restyled(tmp_path)
    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"
    manifest = IllustrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    target = manifest.jobs[1]
    forged = target.model_copy(update={field: value})
    atomic_write_json(
        manifest_path,
        manifest.model_copy(
            update={
                "jobs": tuple(forged if job is target else job for job in manifest.jobs)
            }
        ).model_dump(mode="json"),
    )

    with pytest.raises(MediaWorkflowError, match="restyle_provenance_invalid"):
        _local_restyle_service(store).status("book-demo", "E001")


@pytest.mark.parametrize("tamper", ["generated_file", "continuation_binding"])
def test_restyle_generated_ledger_tamper_fails_closed(
    tmp_path: Path,
    tamper: str,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = _local_restyle_service(store)
    view = service.restyle(
        "book-demo", "E001", style_id="warm-flat-storybook"
    )
    source = _write_color_sources(root, 1)[0]
    service.import_image("book-demo", "E001", view.missing_scene_ids[0], source)
    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"
    manifest = IllustrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if tamper == "generated_file":
        manifest.jobs[0].output_master.write_bytes(b"forged generated image")
    else:
        continuation = manifest.jobs[1].model_copy(
            update={"anchor_sha256": "f" * 64}
        )
        atomic_write_json(
            manifest_path,
            manifest.model_copy(
                update={"jobs": (manifest.jobs[0], continuation, *manifest.jobs[2:])}
            ).model_dump(mode="json"),
        )

    with pytest.raises(MediaWorkflowError, match="restyle_provenance_invalid"):
        _local_restyle_service(store).status("book-demo", "E001")


def test_generated_to_approved_status_tamper_fails_at_every_restart_phase(
    tmp_path: Path,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    calls: list[str] = []
    service = _local_restyle_service(store, calls=calls)
    view = service.restyle(
        "book-demo", "E001", style_id="warm-flat-storybook"
    )
    sources = _write_color_sources(root, 8)
    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"

    def assert_status_tamper_rejected(*, also_prepare: bool = False) -> None:
        original = manifest_path.read_bytes()
        manifest = IllustrationManifest.model_validate_json(original)
        generated = next(job for job in manifest.jobs if job.status == "generated")
        forged = generated.model_copy(update={"status": "approved"})
        atomic_write_json(
            manifest_path,
            manifest.model_copy(
                update={
                    "jobs": tuple(
                        forged if job.asset_id == generated.asset_id else job
                        for job in manifest.jobs
                    )
                }
            ).model_dump(mode="json"),
        )
        before = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
        restarted = _local_restyle_service(store, calls=calls)
        with pytest.raises(MediaWorkflowError, match="restyle_provenance_invalid"):
            restarted.status("book-demo", "E001")
        assert {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        } == before
        if also_prepare:
            with pytest.raises(
                MediaWorkflowError, match="restyle_provenance_invalid"
            ):
                restarted.prepare("book-demo", "E001", mode="full")
            assert {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            } == before
        manifest_path.write_bytes(original)

    service.import_image("book-demo", "E001", view.missing_scene_ids[0], sources[0])
    assert_status_tamper_rejected(also_prepare=True)
    for asset_id, source in zip(view.missing_scene_ids[1:], sources[1:6], strict=True):
        service.import_image("book-demo", "E001", asset_id, source)
    service.approve_representatives("book-demo", "E001")
    assert_status_tamper_rejected()
    batch = service.prepare_batch("book-demo", "E001")
    assert batch.missing_scene_ids == ("S02-A", "S02-B")
    assert_status_tamper_rejected()
    for asset_id, source in zip(batch.missing_scene_ids, sources[6:], strict=True):
        service.import_image("book-demo", "E001", asset_id, source)
    assert store.load_episode("book-demo", "E001").status == "illustrations_ready"
    assert_status_tamper_rejected()
    assert calls == []


def test_restyle_representative_approval_hash_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = _local_restyle_service(store)
    view = service.restyle(
        "book-demo", "E001", style_id="warm-flat-storybook"
    )
    for asset_id, source in zip(
        view.missing_scene_ids, _write_color_sources(root, 6), strict=True
    ):
        service.import_image("book-demo", "E001", asset_id, source)
    service.approve_representatives("book-demo", "E001")
    approval_path = root / "media" / "illustration" / "representative_approval.json"
    payload = json.loads(approval_path.read_text(encoding="utf-8"))
    payload["review_sha256"] = "f" * 64
    atomic_write_json(approval_path, payload)

    with pytest.raises(MediaWorkflowError, match="restyle_provenance_invalid"):
        _local_restyle_service(store).status("book-demo", "E001")


@pytest.mark.parametrize("tamper", ["missing", "valid-shaped"])
def test_restyle_override_loss_or_valid_shaped_tamper_fails_before_any_stage_or_write(
    tmp_path: Path,
    tamper: str,
) -> None:
    root, store, _ = _restyled(tmp_path)
    override = root / "script" / "illustration_style_override.json"
    if tamper == "missing":
        override.unlink()
    else:
        atomic_write_json(
            override,
            {
                "requested_style_id": "forged-style",
                "style_fingerprint": "f" * 64,
                "catalog_sha256": "e" * 64,
            },
        )
    before_media = _media_files(root)
    before_state = (root / "episode.json").read_bytes()
    before_override = override.read_bytes() if override.exists() else None
    calls: list[str] = []
    restarted = MediaProductionService(
        store=store,
        stages={name: _Trap(name, calls) for name in (*UPSTREAM_STAGES, *STYLE_STAGES)},
        images=_Images(),
        legacy_config_sha256="0" * 64,
        configured_mode="full",
        stage_config_sha256s=_stage_hashes(style="2" * 64),
        restyle_catalog_path=CATALOG,
    )

    with pytest.raises(MediaWorkflowError, match="restyle_override_(missing|digest_mismatch)"):
        restarted.prepare("book-demo", "E001", mode="full")

    assert calls == []
    assert _media_files(root) == before_media
    assert (root / "episode.json").read_bytes() == before_state
    assert (override.read_bytes() if override.exists() else None) == before_override


def test_changed_catalog_rebases_explicit_style_without_any_prepare_stage(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes(CATALOG.read_bytes())
    root, store, old_storyboard = _restyled(tmp_path, catalog=catalog)
    old_decision = StyleDecision.model_validate_json(
        (root / "media" / "illustration" / "style_decision.json").read_text(
            encoding="utf-8"
        )
    )
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    selected = next(
        item for item in payload["styles"] if item["id"] == "warm-flat-storybook"
    )
    selected["summary"] += " 配置更新"
    atomic_write_json(catalog, payload)
    calls: list[str] = []
    restarted = MediaProductionService(
        store=store,
        stages={name: _Trap(name, calls) for name in (*UPSTREAM_STAGES, *STYLE_STAGES)},
        images=_Images(),
        legacy_config_sha256="0" * 64,
        configured_mode="full",
        stage_config_sha256s=_stage_hashes(style="2" * 64),
        restyle_catalog_path=catalog,
    )

    view = restarted.prepare("book-demo", "E001", mode="full")

    illustration = root / "media" / "illustration"
    decision = StyleDecision.model_validate_json(
        (illustration / "style_decision.json").read_text(encoding="utf-8")
    )
    storyboard = IllustrationStoryboard.model_validate_json(
        (illustration / "illustration_storyboard.json").read_text(encoding="utf-8")
    )
    manifest = IllustrationManifest.model_validate_json(
        (illustration / "illustration_manifest.json").read_text(encoding="utf-8")
    )
    state = store.load_episode("book-demo", "E001")
    override_sha = sha256_file(root / "script" / "illustration_style_override.json")

    assert view.status == "representative_generation_running"
    assert calls == []
    assert decision.selected_style == "warm-flat-storybook"
    assert decision.style_fingerprint != old_decision.style_fingerprint
    assert storyboard.scenes == old_storyboard.scenes
    assert len(manifest.jobs) == 8
    assert all(
        job.status == "planned"
        and job.master_sha256 is None
        and job.anchor_sha256 is None
        and not job.reference_image_sha256s
        for job in manifest.jobs
    )
    assert {
        state.stage_manifests[name].inputs["restyle_override_sha256"]
        for name in STYLE_STAGES
    } == {override_sha}
    assert {
        state.stage_manifests[name].config_sha256 for name in UPSTREAM_STAGES
    } == {"0" * 64}
    assert {
        state.stage_manifests[name].config_sha256 for name in STYLE_STAGES
    } == {f"stage-v1:{'2' * 64}"}


def test_real_runtime_changed_catalog_keeps_upstream_and_calls_no_prepare_runner(
    tmp_path: Path,
) -> None:
    vendor = tmp_path / "vendor"
    catalog = vendor / "references" / "handdrawn-style-library.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes(CATALOG.read_bytes())
    root, episode, old_storyboard, _ = _episode_fixture(tmp_path)
    episode.status = "script_approved"
    for name in STYLE_STAGES:
        episode.stage_manifests.pop(name)
        episode.completed_stages.remove(name)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    config = AppConfig(
        workspace_dir=tmp_path,
        subtitle_font_path=tmp_path / "font.ttf",
        handdrawn={"vendor_dir": vendor},
    )
    first = build_media_runtime_bindings(
        config, RuntimeAuthorization(allow_external=False), mode="full"
    ).service
    first.images = _Images()
    first.stages = {
        name: _ExistingArtifactStage(name)
        for name in (*UPSTREAM_STAGES, *STYLE_STAGES)
    }
    first.prepare("book-demo", "E001", mode="full")
    first.restyle("book-demo", "E001", style_id="warm-flat-storybook")
    assert {
        store.load_episode("book-demo", "E001")
        .stage_manifests[name]
        .config_sha256.split(":", 1)[0]
        for name in (*UPSTREAM_STAGES, *STYLE_STAGES)
    } == {"stage-v1"}
    upstream_before = {
        name: store.load_episode("book-demo", "E001").stage_manifests[name].model_dump()
        for name in UPSTREAM_STAGES
    }
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    selected = next(
        item for item in payload["styles"] if item["id"] == "warm-flat-storybook"
    )
    selected["summary"] += " production refresh"
    atomic_write_json(catalog, payload)
    calls: list[str] = []
    restarted = build_media_runtime_bindings(
        config, RuntimeAuthorization(allow_external=False), mode="full"
    ).service
    restarted.images = _Images()
    restarted.stages = {
        name: _Trap(name, calls) for name in (*UPSTREAM_STAGES, *STYLE_STAGES)
    }

    view = restarted.prepare("book-demo", "E001", mode="full")

    state = store.load_episode("book-demo", "E001")
    storyboard = IllustrationStoryboard.model_validate_json(
        (root / "media" / "illustration" / "illustration_storyboard.json").read_text(
            encoding="utf-8"
        )
    )
    assert view.status == "representative_generation_running"
    assert calls == []
    assert storyboard.scenes == old_storyboard.scenes
    assert {
        name: state.stage_manifests[name].model_dump() for name in UPSTREAM_STAGES
    } == upstream_before


@pytest.mark.parametrize("sequence_mode", ["color-story-pair", "legacy-monochrome-reveal"])
def test_real_runtime_accepts_exact_current_unversioned_global_hash_without_stage_calls(
    tmp_path: Path,
    sequence_mode: str,
) -> None:
    vendor = tmp_path / "vendor"
    catalog = vendor / "references" / "handdrawn-style-library.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes(CATALOG.read_bytes())
    workspace = tmp_path / "workspace"
    root, episode, storyboard, bible = _episode_fixture(workspace)
    if sequence_mode == "legacy-monochrome-reveal":
        storyboard = storyboard.model_copy(
            update={"sequence_mode": "legacy-monochrome-reveal"}
        )
        decision = StyleDecision.model_validate_json(
            (root / "media" / "illustration" / "style_decision.json").read_text(
                encoding="utf-8"
            )
        )
        style = next(
            item
            for item in load_style_catalog(catalog)
            if item.style_id == decision.selected_style
        )
        manifest = prepare_image_jobs(storyboard, bible, style, decision, root)
        illustration = root / "media" / "illustration"
        atomic_write_json(
            illustration / "illustration_storyboard.json",
            storyboard.model_dump(mode="json"),
        )
        atomic_write_json(
            illustration / "illustration_manifest.json",
            manifest.model_dump(mode="json"),
        )
    config = AppConfig(
        workspace_dir=workspace,
        subtitle_font_path=tmp_path / "font.ttf",
        handdrawn={"vendor_dir": vendor},
    )
    prompt_root = Path("prompts/illustration")
    old_global = _media_runtime_config_sha256(
        config,
        catalog_path=catalog,
        prompt_paths=(
            prompt_root / "style_select.md",
            prompt_root / "storyboard.md",
        ),
    )
    for name in (*UPSTREAM_STAGES, *STYLE_STAGES):
        artifact = root / "media" / "baseline" / f"{name}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({"stage": name}), encoding="utf-8")
        inputs = {"media_mode": "full"}
        if name == "subtitles":
            inputs.update(
                {
                    "subtitle_sequence_mode": sequence_mode,
                    "subtitle_breaks_presence": "present",
                    "subtitle_breaks_sha256": sha256_file(
                        root / "script" / "subtitle_breaks.txt"
                    ),
                }
            )
        episode.stage_manifests[name] = StageManifest(
            stage=name,
            status="completed",
            inputs=inputs,
            outputs={
                name: ArtifactRef(
                    path=str(artifact),
                    sha256=sha256_file(artifact),
                    size_bytes=artifact.stat().st_size,
                )
            },
            config_sha256=old_global,
        )
    if sequence_mode == "color-story-pair":
        _bind_original_planning_provenance(root, episode)
        for name in STYLE_STAGES:
            episode.stage_manifests[name].config_sha256 = old_global
    episode.status = "representative_generation_running"
    store = StateStore(workspace)
    store.save_episode(episode)
    calls: list[str] = []
    service = build_media_runtime_bindings(
        config,
        RuntimeAuthorization(allow_external=False),
        mode="full",
    ).service
    service.stages = {
        name: _Trap(name, calls) for name in (*UPSTREAM_STAGES, *STYLE_STAGES)
    }

    view = service.prepare("book-demo", "E001", mode="full")

    assert view.status in {
        "representative_generation_running",
        "awaiting_representative_review",
    }
    assert calls == []
    current = store.load_episode("book-demo", "E001")
    assert all(
        current.stage_manifests[name].config_sha256 == old_global
        for name in (*UPSTREAM_STAGES, *STYLE_STAGES)
    )

    current.status = (
        "final_approved"
        if sequence_mode == "legacy-monochrome-reveal"
        else "representative_generation_running"
    )
    store.save_episode(current)
    before = (root / "episode.json").read_bytes()
    service.status("book-demo", "E001")
    assert (root / "episode.json").read_bytes() == before


def test_unversioned_global_hash_is_rejected_after_real_catalog_drift(
    tmp_path: Path,
) -> None:
    vendor = tmp_path / "vendor"
    catalog = vendor / "references" / "handdrawn-style-library.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes(CATALOG.read_bytes())
    workspace = tmp_path / "workspace"
    root, episode, _, _ = _episode_fixture(workspace)
    config = AppConfig(
        workspace_dir=workspace,
        subtitle_font_path=tmp_path / "font.ttf",
        handdrawn={"vendor_dir": vendor},
    )
    prompt_root = Path("prompts/illustration")
    old_global = _media_runtime_config_sha256(
        config,
        catalog_path=catalog,
        prompt_paths=(
            prompt_root / "style_select.md",
            prompt_root / "storyboard.md",
        ),
    )
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["styles"][0]["summary"] += " actual drift"
    atomic_write_json(catalog, payload)
    service = build_media_runtime_bindings(
        config,
        RuntimeAuthorization(allow_external=False),
        mode="full",
    ).service
    artifact = root / "media" / "old.json"
    artifact.write_text("{}", encoding="utf-8")
    episode.stage_manifests["tts"] = StageManifest(
        stage="tts",
        status="completed",
        inputs={"media_mode": "full"},
        outputs={
            "tts": ArtifactRef(
                path=str(artifact),
                sha256=sha256_file(artifact),
                size_bytes=artifact.stat().st_size,
            )
        },
        config_sha256=old_global,
    )

    assert service._manifest_current(episode, "tts", mode="full") is False


def test_unknown_style_after_catalog_removal_fails_closed_without_stage_calls(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes(CATALOG.read_bytes())
    root, store, _ = _restyled(tmp_path, catalog=catalog)
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["styles"] = [
        item for item in payload["styles"] if item["id"] != "warm-flat-storybook"
    ]
    atomic_write_json(catalog, payload)
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    calls: list[str] = []
    restarted = MediaProductionService(
        store=store,
        stages={name: _Trap(name, calls) for name in (*UPSTREAM_STAGES, *STYLE_STAGES)},
        images=_Images(),
        configured_mode="full",
        stage_config_sha256s=_stage_hashes(style="2" * 64),
        restyle_catalog_path=catalog,
    )

    with pytest.raises(MediaWorkflowError, match="restyle_style_not_in_catalog"):
        restarted.prepare("book-demo", "E001", mode="full")

    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert calls == []
    assert after == before


@pytest.mark.parametrize("field", ["selected_style", "style_fingerprint"])
def test_style_decision_requested_or_fingerprint_mismatch_fails_closed(
    tmp_path: Path,
    field: str,
) -> None:
    root, store, _ = _restyled(tmp_path)
    decision_path = root / "media" / "illustration" / "style_decision.json"
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    payload[field] = "retro-gouache-concept" if field == "selected_style" else "f" * 64
    atomic_write_json(decision_path, payload)
    state = store.load_episode("book-demo", "E001")
    artifact = state.stage_manifests["select_style"].outputs["style_decision"]
    state.stage_manifests["select_style"].outputs["style_decision"] = artifact.model_copy(
        update={
            "sha256": sha256_file(decision_path),
            "size_bytes": decision_path.stat().st_size,
        }
    )
    store.save_episode(state)
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    calls: list[str] = []
    restarted = MediaProductionService(
        store=store,
        stages={name: _Trap(name, calls) for name in (*UPSTREAM_STAGES, *STYLE_STAGES)},
        images=_Images(),
        legacy_config_sha256="0" * 64,
        configured_mode="full",
        stage_config_sha256s=_stage_hashes(style="2" * 64),
        restyle_catalog_path=CATALOG,
    )

    with pytest.raises(MediaWorkflowError, match="restyle_provenance_invalid"):
        restarted.prepare("book-demo", "E001", mode="full")

    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert calls == []
    assert after == before


def test_restyle_postwrite_state_save_failure_restores_state_media_and_override(
    tmp_path: Path,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)

    class _SaveThenFail(StateStore):
        armed = False

        def save_episode(self, state):
            super().save_episode(state)
            if self.armed:
                raise OSError("injected postwrite failure")

    store = _SaveThenFail(tmp_path)
    store.save_episode(episode)
    store.armed = True
    before_state = (root / "episode.json").read_bytes()
    before_media = _media_files(root)
    service = MediaProductionService(
        store=store,
        stages={},
        images=_Images(),
        configured_mode="full",
        stage_config_sha256s=_stage_hashes(style="1" * 64),
        restyle_catalog_path=CATALOG,
    )

    with pytest.raises(MediaWorkflowError, match="restyle_state_commit_failed"):
        service.restyle("book-demo", "E001", style_id="warm-flat-storybook")

    assert (root / "episode.json").read_bytes() == before_state
    assert _media_files(root) == before_media
    assert not (root / "script" / "illustration_style_override.json").exists()
    assert tuple((root / ".recovery" / "style-restyle").glob("failed-new-*"))


def test_rebase_postwrite_state_save_failure_restores_last_stable_tuple(
    tmp_path: Path,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)

    class _SaveThenFail(StateStore):
        armed = False

        def save_episode(self, state):
            super().save_episode(state)
            if self.armed:
                raise OSError("injected rebase postwrite failure")

    store = _SaveThenFail(tmp_path)
    store.save_episode(episode)
    first = MediaProductionService(
        store=store,
        stages={},
        images=_Images(),
        configured_mode="full",
        stage_config_sha256s=_stage_hashes(style="1" * 64),
        restyle_catalog_path=CATALOG,
    )
    first.restyle("book-demo", "E001", style_id="warm-flat-storybook")
    before_state = (root / "episode.json").read_bytes()
    before_media = _media_files(root)
    override = root / "script" / "illustration_style_override.json"
    before_override = override.read_bytes()
    store.armed = True
    calls: list[str] = []
    restarted = MediaProductionService(
        store=store,
        stages={name: _Trap(name, calls) for name in (*UPSTREAM_STAGES, *STYLE_STAGES)},
        images=_Images(),
        legacy_config_sha256="0" * 64,
        configured_mode="full",
        stage_config_sha256s=_stage_hashes(style="2" * 64),
        restyle_catalog_path=CATALOG,
    )

    with pytest.raises(MediaWorkflowError, match="restyle_state_commit_failed"):
        restarted.prepare("book-demo", "E001", mode="full")

    assert calls == []
    assert (root / "episode.json").read_bytes() == before_state
    assert _media_files(root) == before_media
    assert override.read_bytes() == before_override
    assert tuple((root / ".recovery" / "style-restyle").glob("failed-new-*"))


@pytest.mark.parametrize(
    "bad_ids",
    [
        ("S01-A", "S01-B", "S03-A", "S03-B", "S04-A"),
        ("S01-B", "S01-A", "S03-A", "S03-B", "S04-A", "S04-B"),
        ("S01-A", "S01-A", "S03-A", "S03-B", "S04-A", "S04-B"),
    ],
)
def test_public_status_rejects_wrong_pair_count_order_or_duplicates(
    tmp_path: Path,
    bad_ids: tuple[str, ...],
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    _bind_original_planning_provenance(root, episode)
    episode.status = "awaiting_representative_review"
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = MediaProductionService(
        store=store,
        stages={},
        images=_Images(bad_ids),
        restyle_catalog_path=CATALOG,
    )

    with pytest.raises(MediaWorkflowError, match="representative_surface_invalid"):
        service.status("book-demo", "E001")


def test_public_status_derives_valid_three_scene_pair_ids_in_storyboard_order(
    tmp_path: Path,
) -> None:
    root, episode, storyboard = _three_scene_pair_fixture(tmp_path)
    _bind_original_planning_provenance(root, episode)
    episode.status = "awaiting_representative_review"
    ids = tuple(
        asset_id
        for scene in storyboard.scenes
        if scene.representative_frame
        for asset_id in (f"{scene.scene_id}-A", f"{scene.scene_id}-B")
    )
    store = StateStore(tmp_path)
    store.save_episode(episode)

    view = MediaProductionService(
        store=store,
        stages={},
        images=_Images(ids),
        restyle_catalog_path=CATALOG,
    ).status("book-demo", "E001")

    assert ids == ("S01-A", "S01-B", "S02-A", "S02-B", "S03-A", "S03-B")
    assert view.missing_scene_ids == ids
    assert "六张" in view.next_action


def test_public_status_rejects_wrong_three_scene_pair_order(tmp_path: Path) -> None:
    root, episode, _ = _three_scene_pair_fixture(tmp_path)
    _bind_original_planning_provenance(root, episode)
    episode.status = "awaiting_representative_review"
    store = StateStore(tmp_path)
    store.save_episode(episode)
    wrong = ("S01-A", "S01-B", "S03-A", "S03-B", "S02-A", "S02-B")

    with pytest.raises(MediaWorkflowError, match="representative_surface_invalid"):
        MediaProductionService(
            store=store,
            stages={},
            images=_Images(wrong),
            restyle_catalog_path=CATALOG,
        ).status("book-demo", "E001")


def test_normal_style_fingerprint_input_is_not_mistaken_for_restyle_provenance(
    tmp_path: Path,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    _bind_original_planning_provenance(root, episode)
    episode.status = "awaiting_representative_review"
    episode.stage_manifests["prepare_representatives"].inputs[
        "style_fingerprint"
    ] = "b" * 64
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = MediaProductionService(
        store=store,
        stages={},
        images=_Images(),
        restyle_catalog_path=CATALOG,
    )

    view = service.status("book-demo", "E001")

    assert view.missing_scene_ids == PAIR_IDS
    assert "六张" in view.next_action


@pytest.mark.parametrize("missing", ["storyboard", "manifest"])
def test_public_status_rejects_missing_joint_mode_evidence(
    tmp_path: Path,
    missing: str,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    episode.status = "awaiting_representative_review"
    store = StateStore(tmp_path)
    store.save_episode(episode)
    (root / "media" / "illustration" / f"illustration_{missing}.json").unlink()
    service = MediaProductionService(store=store, stages={}, images=_Images())

    with pytest.raises(MediaWorkflowError, match="illustration_provenance_invalid"):
        service.status("book-demo", "E001")


@pytest.mark.parametrize("corrupt", ["storyboard", "manifest"])
def test_public_status_rejects_corrupt_joint_mode_evidence(
    tmp_path: Path,
    corrupt: str,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    episode.status = "awaiting_representative_review"
    store = StateStore(tmp_path)
    store.save_episode(episode)
    (root / "media" / "illustration" / f"illustration_{corrupt}.json").write_text(
        "{not-json", encoding="utf-8"
    )
    service = MediaProductionService(store=store, stages={}, images=_Images())

    with pytest.raises(MediaWorkflowError, match="illustration_provenance_invalid"):
        service.status("book-demo", "E001")


def test_public_status_rejects_cross_mode_storyboard_and_manifest(
    tmp_path: Path,
) -> None:
    root, episode, storyboard, _ = _episode_fixture(tmp_path)
    legacy_storyboard = storyboard.model_copy(
        update={"sequence_mode": "legacy-monochrome-reveal"}
    )
    atomic_write_json(
        root / "media" / "illustration" / "illustration_storyboard.json",
        legacy_storyboard.model_dump(mode="json"),
    )
    episode.status = "awaiting_representative_review"
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = MediaProductionService(store=store, stages={}, images=_Images())

    with pytest.raises(MediaWorkflowError, match="illustration_provenance_invalid"):
        service.status("book-demo", "E001")


def test_public_status_validates_legacy_three_ids_and_uses_legacy_wording(
    tmp_path: Path,
) -> None:
    root, episode, pair_storyboard, bible = _episode_fixture(tmp_path)
    legacy = pair_storyboard.model_copy(update={"sequence_mode": "legacy-monochrome-reveal"})
    decision = StyleDecision.model_validate_json(
        (root / "media" / "illustration" / "style_decision.json").read_text(
            encoding="utf-8"
        )
    )
    style = next(
        item
        for item in load_style_catalog(CATALOG)
        if item.style_id == decision.selected_style
    )
    manifest = prepare_image_jobs(legacy, bible, style, decision, root)
    illustration = root / "media" / "illustration"
    atomic_write_json(
        illustration / "illustration_storyboard.json", legacy.model_dump(mode="json")
    )
    atomic_write_json(
        illustration / "illustration_manifest.json", manifest.model_dump(mode="json")
    )
    legacy_ids = tuple(job.scene_id for job in manifest.jobs if job.representative)
    episode.status = "awaiting_representative_review"
    for name in STYLE_STAGES:
        episode.stage_manifests.pop(name, None)
        episode.completed_stages = [
            stage for stage in episode.completed_stages if stage != name
        ]
        episode.stale_stages = [
            stage for stage in episode.stale_stages if stage != name
        ]
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = MediaProductionService(
        store=store, stages={}, images=_Images(legacy_ids)
    )

    view = service.status("book-demo", "E001")

    assert view.missing_scene_ids == legacy_ids
    assert len(legacy_ids) == 3
    assert "三张代表图" in view.next_action
    assert "六张" not in view.next_action


def test_original_pair_without_trusted_catalog_reports_configuration_error(
    tmp_path: Path,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    _bind_original_planning_provenance(root, episode)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = MediaProductionService(store=store, stages={}, images=_Images())

    with pytest.raises(
        MediaWorkflowError, match="illustration_catalog_not_configured"
    ):
        service.status("book-demo", "E001")


def test_override_leaf_symlink_is_rejected_without_following_target(
    tmp_path: Path,
) -> None:
    root, store, _ = _restyled(tmp_path)
    override = root / "script" / "illustration_style_override.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(override.read_bytes())
    override.unlink()
    try:
        os.symlink(outside, override)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    before_outside = outside.read_bytes()
    calls: list[str] = []
    restarted = MediaProductionService(
        store=store,
        stages={name: _Trap(name, calls) for name in (*UPSTREAM_STAGES, *STYLE_STAGES)},
        images=_Images(),
        configured_mode="full",
        stage_config_sha256s=_stage_hashes(style="2" * 64),
        restyle_catalog_path=CATALOG,
    )

    with pytest.raises(MediaWorkflowError, match="unsafe_restyle_recovery"):
        restarted.prepare("book-demo", "E001", mode="full")

    assert calls == []
    assert outside.read_bytes() == before_outside


def test_override_parent_junction_is_rejected_without_following_target(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction contract")
    root, store, _ = _restyled(tmp_path)
    script = root / "script"
    outside = tmp_path / "outside-script"
    outside.mkdir()
    override_bytes = (script / "illustration_style_override.json").read_bytes()
    (outside / "illustration_style_override.json").write_bytes(override_bytes)
    for child in script.iterdir():
        child.unlink()
    script.rmdir()
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(script), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip("junction unavailable")
    calls: list[str] = []
    restarted = MediaProductionService(
        store=store,
        stages={name: _Trap(name, calls) for name in (*UPSTREAM_STAGES, *STYLE_STAGES)},
        images=_Images(),
        configured_mode="full",
        stage_config_sha256s=_stage_hashes(style="2" * 64),
        restyle_catalog_path=CATALOG,
    )

    with pytest.raises(MediaWorkflowError, match="unsafe_restyle_recovery"):
        restarted.prepare("book-demo", "E001", mode="full")

    assert calls == []
    assert (outside / "illustration_style_override.json").read_bytes() == override_bytes
