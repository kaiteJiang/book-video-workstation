from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import bv.illustration.replacement as replacement_module
import bv.illustration.restyle as restyle_module
from bv.core.hashing import sha256_file
from bv.core.process import CommandResult
from bv.illustration.assets import (
    ImageFacts,
    ImageJob,
    IllustrationManifest,
    bind_job_references,
    import_image_asset,
    prepare_image_jobs,
    record_imported_image,
)
from bv.illustration.contracts import (
    CharacterBible,
    CharacterLock,
    IllustrationScene,
    IllustrationStoryboard,
)
from bv.illustration.prompts import canonical_model_sha256
from bv.illustration.replacement import (
    IllustrationReplacementError,
    ReplacementAuthorization,
    ReplacementImportResult,
    import_authorized_replacement,
    prepare_illustration_replacement as _prepare_illustration_replacement,
)
from bv.illustration.reviews import (
    RepresentativeApproval,
    approve_representatives,
    build_representative_review,
    unlock_batch_jobs,
)
from bv.illustration.style_selector import load_style_catalog, manual_style_decision
from bv.state.models import ArtifactRef, EpisodeState, StageManifest
from bv.state.store import StateStore


BOOK_ID = "book-demo"
EPISODE_ID = "E001"
CATALOG = Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json")


def prepare_illustration_replacement(**kwargs: object):
    kwargs.setdefault("catalog_path", CATALOG)
    return _prepare_illustration_replacement(**kwargs)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact(path: Path) -> ArtifactRef:
    return ArtifactRef(
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _job(root: Path, scene_id: str, phase: str, *, representative: bool) -> ImageJob:
    suffix = "A" if phase == "anchor" else "B"
    output = root / "media" / "illustration" / "images" / f"{scene_id}_{phase}.png"
    payload = f"old-{scene_id}-{suffix}".encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    anchor_hash = None
    references: tuple[str, ...] = ()
    if phase == "continuation":
        anchor = root / "media" / "illustration" / "images" / f"{scene_id}_anchor.png"
        anchor_hash = sha256_file(anchor)
        references = (anchor_hash,)
    return ImageJob(
        episode_root=root,
        scene_id=scene_id,
        representative=representative,
        prompt=f"prompt-{scene_id}-{suffix}",
        prompt_sha256=_sha_bytes(f"prompt-{scene_id}-{suffix}".encode()),
        style_fingerprint="1" * 64,
        character_lock_sha256="2" * 64,
        reference_scene_ids=(scene_id,) if phase == "continuation" else (),
        reference_image_sha256s=references,
        phase=phase,
        asset_id=f"{scene_id}-{suffix}",
        anchor_sha256=anchor_hash,
        output_master=output,
        master_sha256=_sha_bytes(payload),
        attempts=1,
        status="generated",
    )


def _storyboard() -> IllustrationStoryboard:
    scenes = tuple(
        IllustrationScene(
            scene_id=f"S{index + 1:02d}",
            start_ms=index * 5_000,
            end_ms=(index + 1) * 5_000,
            from_frame=index * 150,
            to_frame=(index + 1) * 150,
            narration=f"第{index + 1}段批准旁白内容",
            narration_span=(index * 10, (index + 1) * 10),
            key_line="把生活还给自己",
            visual_purpose="把书的价值放回真实生活",
            setting="通勤车厢",
            character_action="主人公停下解释，打开笔记本",
            metaphor=None,
            composition="主体居中偏下，安全留白",
            character_refs=("reader-01",),
            image_prompt="普通成年读者停下解释的生活场景",
            negative_constraints=("禁止画内文字",),
            representative_frame=index in {0, 2, 3},
            asset_status="planned",
            semantic_turn_span=(index * 10, index * 10 + 2),
            semantic_turn_ms=index * 5_000 + 1_000,
            semantic_turn_frame=index * 150 + 30,
            continuation_action="主人公合上笔记本，抬头看向车窗",
            continuation_prompt="同一车厢同一机位，主人公合上笔记本并抬头",
            continuity_constraints=(
                "same character", "same clothing", "same setting", "same camera direction"
            ),
        )
        for index in range(4)
    )
    return IllustrationStoryboard(
        book_id=BOOK_ID,
        episode_id=EPISODE_ID,
        width=1080,
        height=1920,
        fps=30,
        master_duration_ms=20_000,
        total_frames=600,
        script_sha256="5" * 64,
        audio_sha256="6" * 64,
        subtitle_sha256="7" * 64,
        style_decision_sha256="8" * 64,
        character_lock_sha256="2" * 64,
        sequence_mode="color-story-pair",
        scenes=scenes,
    )


def _build_workspace(
    tmp_path: Path,
    *,
    final: bool = False,
    representatives_only: bool = False,
) -> tuple[StateStore, Path]:
    workspace = tmp_path / "workspace"
    root = workspace / "books" / BOOK_ID / "episodes" / EPISODE_ID
    root.mkdir(parents=True)
    catalog = load_style_catalog(CATALOG)
    decision = manual_style_decision(
        catalog,
        style_id="warm-flat-storybook",
        source_script_sha256="5" * 64,
    )
    character = CharacterLock(
        role="普通读者",
        age_range="30至39岁",
        face="椭圆脸",
        hair="黑色短发",
        body="中等身材",
        base_clothing="深蓝衬衫",
        allowed_variations=(),
        color_markers=("深蓝",),
        personal_objects=("笔记本",),
        forbidden_changes=("发型",),
        source_script_sha256=decision.source_script_sha256,
        style_fingerprint=decision.style_fingerprint,
    )
    bible = CharacterBible(
        source_script_sha256=decision.source_script_sha256,
        style_fingerprint=decision.style_fingerprint,
        characters=(character,),
    )
    storyboard = _storyboard()
    storyboard = storyboard.model_copy(
        update={
            "style_decision_sha256": canonical_model_sha256(decision),
            "character_lock_sha256": canonical_model_sha256(bible),
            "character_bible_sha256": canonical_model_sha256(bible),
            "scenes": tuple(
                scene.model_copy(update={"character_refs": ("fixture-unbound",)})
                for scene in storyboard.scenes
            ),
        }
    )
    style = next(item for item in catalog if item.style_id == decision.selected_style)
    planned = prepare_image_jobs(storyboard, bible, style, decision, root)
    generated: dict[str, ImageJob] = {}
    for planned_job in planned.jobs:
        asset_id = planned_job.asset_id or ""
        if representatives_only and not planned_job.representative:
            generated[asset_id] = planned_job
            continue
        current = planned.model_copy(
            update={
                "jobs": tuple(
                    generated.get(item.asset_id or "", item) for item in planned.jobs
                )
            }
        )
        job = (
            bind_job_references(planned_job, current)
            if planned_job.phase == "continuation"
            else planned_job
        )
        payload = f"old-{asset_id}".encode()
        job.output_master.parent.mkdir(parents=True, exist_ok=True)
        job.output_master.write_bytes(payload)
        generated[asset_id] = job.model_copy(
            update={
                "master_sha256": _sha_bytes(payload),
                "attempts": 1,
                "status": "generated",
            }
        )
    manifest = planned.model_copy(
        update={
            "jobs": tuple(generated[job.asset_id or ""] for job in planned.jobs)
        }
    )
    illustration = root / "media" / "illustration"
    (illustration / "style_decision.json").write_text(
        decision.model_dump_json(), encoding="utf-8"
    )
    (illustration / "character_bible.json").write_text(
        bible.model_dump_json(), encoding="utf-8"
    )
    (illustration / "illustration_storyboard.json").write_text(
        storyboard.model_dump_json(), encoding="utf-8"
    )
    (illustration / "illustration_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    review = build_representative_review(storyboard, manifest)
    approval = approve_representatives(
        review, approved_image_sha256s=review.image_sha256s, reviewer="user"
    )
    (illustration / "representative_review.json").write_text(
        review.model_dump_json(), encoding="utf-8"
    )
    (illustration / "representative_approval.json").write_text(
        approval.model_dump_json(), encoding="utf-8"
    )

    outputs: dict[str, Path] = {}
    for stage, relative in (
        ("visual_render", "media/visual/silent.mp4"),
        ("render", "media/final/final.mp4"),
        ("qc", "media/final/final.qc.json"),
        ("delivery", "deliveries/2026-09-02/candidate.json"),
        ("final_approval", "deliveries/2026-09-02/final-approval.json"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{stage}-evidence".encode())
        outputs[stage] = path
    audio = root / "media" / "audio" / "final.wav"
    subtitles = root / "media" / "subtitles" / "final.ass"
    for path, payload in ((audio, b"audio"), (subtitles, b"subtitles")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    manifests = {
        stage: StageManifest(
            stage=stage,
            status="completed",
            inputs={},
            outputs={stage: _artifact(path)},
            config_sha256="4" * 64,
        )
        for stage, path in outputs.items()
    }
    manifests.update(
        {
            "select_style": StageManifest(
                stage="select_style",
                status="completed",
                inputs={},
                outputs={
                    "style_decision": _artifact(
                        illustration / "style_decision.json"
                    )
                },
                config_sha256="4" * 64,
            ),
            "plan_illustrations": StageManifest(
                stage="plan_illustrations",
                status="completed",
                inputs={},
                outputs={
                    "character_bible": _artifact(
                        illustration / "character_bible.json"
                    ),
                    "illustration_storyboard": _artifact(
                        illustration / "illustration_storyboard.json"
                    ),
                },
                config_sha256="4" * 64,
            ),
            "prepare_representatives": StageManifest(
                stage="prepare_representatives",
                status="completed",
                inputs={
                    "storyboard_sha256": manifest.storyboard_sha256,
                    "style_fingerprint": manifest.style_fingerprint,
                    "character_lock_sha256": manifest.character_lock_sha256,
                },
                outputs={
                    "illustration_manifest": _artifact(
                        illustration / "illustration_manifest.json"
                    )
                },
                config_sha256="4" * 64,
            ),
            "tts": StageManifest(
                stage="tts", status="completed", inputs={},
                outputs={"audio": _artifact(audio)}, config_sha256="4" * 64,
            ),
            "subtitles": StageManifest(
                stage="subtitles", status="completed", inputs={},
                outputs={"subtitles": _artifact(subtitles)}, config_sha256="4" * 64,
            ),
            "representative_review": StageManifest(
                stage="representative_review", status="completed", inputs={},
                outputs={
                    "review": _artifact(illustration / "representative_review.json"),
                    "approval": _artifact(illustration / "representative_approval.json"),
                },
                config_sha256="4" * 64,
            ),
        }
    )
    completed = [
        "tts", "subtitles", "select_style", "plan_illustrations",
        "prepare_representatives", "illustration_anchor", "illustration_continuation",
        "representative_review", "illustration_images", "visual_render", "render",
        "qc", "delivery", "final_approval",
    ]
    state = EpisodeState(
        book_id=BOOK_ID,
        episode_id=EPISODE_ID,
        status=(
            "final_approved"
            if final
            else "representatives_approved"
            if representatives_only
            else "awaiting_final_review"
        ),
        stage_manifests=manifests,
        completed_stages=completed,
    )
    store = StateStore(workspace)
    store.save_episode(state)
    return store, root


def _manifest(root: Path) -> IllustrationManifest:
    return IllustrationManifest.model_validate_json(
        (root / "media" / "illustration" / "illustration_manifest.json").read_text(
            encoding="utf-8"
        )
    )


def _tree(root: Path) -> tuple[tuple[str, ...], dict[str, bytes]]:
    directories = tuple(
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir())
    )
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    return directories, files


def _refresh_receipt_and_authorization_artifacts(
    store: StateStore,
    root: Path,
    receipt_path: Path,
    receipt: dict[str, object],
) -> None:
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_sha256 = sha256_file(receipt_path)
    state = store.load_episode(BOOK_ID, EPISODE_ID)
    stage = state.stage_manifests["illustration_replacement"]
    stage.inputs["receipt_sha256"] = receipt_sha256
    stage.outputs["replacement_receipt"] = _artifact(receipt_path)
    for label, artifact in tuple(stage.outputs.items()):
        if not label.startswith("authorization_"):
            continue
        path = Path(artifact.path)
        authorization = json.loads(path.read_text(encoding="utf-8"))
        authorization["receipt_sha256"] = receipt_sha256
        path.write_text(json.dumps(authorization), encoding="utf-8")
        stage.outputs[label] = _artifact(path)
    store.save_episode(state)


def _review_sha(root: Path) -> str:
    return sha256_file(root / "media" / "illustration" / "representative_review.json")


def _fake_runner(argv: list[str], **_: object) -> CommandResult:
    source = Path(argv[argv.index("-i") + 1])
    target = Path(argv[-1])
    target.write_bytes(source.read_bytes())
    return CommandResult(argv=tuple(argv), returncode=0, stdout="", stderr="")


def _inspect(_path: Path) -> ImageFacts:
    return ImageFacts(width=1080, height=1920, format="png")


@pytest.fixture(autouse=True)
def _stub_restyle_image_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(restyle_module, "inspect_image", _inspect)


def test_rejecting_b_preserves_a_and_archives_only_affected_evidence(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    before = _manifest(root)
    anchor = before.jobs[0]
    continuation = before.jobs[1]
    other_images = {
        job.asset_id: job.output_master.read_bytes() for job in before.jobs[2:]
    }

    result = prepare_illustration_replacement(
        store=store,
        book_id=BOOK_ID,
        episode_id=EPISODE_ID,
        asset_id="S01-B",
        expected_asset_sha256=continuation.master_sha256,
        expected_representative_review_sha256=_review_sha(root),
        reason="人物服装与锚点图不连续",
    )

    current = _manifest(root)
    current_a, current_b = current.jobs[:2]
    assert current_a == anchor
    assert current_a.output_master.read_bytes() == b"old-S01-A"
    assert current_b.status == "planned"
    assert current_b.master_sha256 is None
    assert current_b.anchor_sha256 == anchor.master_sha256
    assert current_b.reference_image_sha256s[0] == anchor.master_sha256
    assert not continuation.output_master.exists()
    assert result.reset_asset_ids == ("S01-B",)
    archived_b = next(result.history_dir.glob("files/*-S01_continuation.png"))
    assert archived_b.read_bytes() == b"old-S01-B"
    assert (result.history_dir / "illustration_manifest.json").is_file()
    assert (result.history_dir / "episode.json").is_file()
    assert (result.history_dir / "replacement_receipt.json").is_file()
    assert not (root / "media/illustration/representative_review.json").exists()
    assert not (root / "media/illustration/representative_approval.json").exists()
    assert not (root / "media/final/final.mp4").exists()
    assert not (root / "deliveries/2026-09-02/candidate.json").exists()
    assert (root / "media/audio/final.wav").read_bytes() == b"audio"
    assert (root / "media/subtitles/final.ass").read_bytes() == b"subtitles"
    assert {
        job.asset_id: job.output_master.read_bytes() for job in current.jobs[2:]
    } == other_images
    state = store.load_episode(BOOK_ID, EPISODE_ID)
    assert state.status == "representative_generation_running"
    assert "tts" in state.completed_stages and "subtitles" in state.completed_stages
    assert state.stage_manifests["final_approval"].status == "stale"
    authorization = ReplacementAuthorization.model_validate_json(
        result.authorization_paths[0].read_text(encoding="utf-8")
    )
    assert authorization.status == "authorized"
    assert authorization.reason == "人物服装与锚点图不连续"


def test_rejecting_a_resets_pair_and_new_b_must_bind_new_a(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    before = _manifest(root)
    old_a, old_b = before.jobs[:2]
    result = prepare_illustration_replacement(
        store=store,
        book_id=BOOK_ID,
        episode_id=EPISODE_ID,
        asset_id="S01-A",
        expected_asset_sha256=old_a.master_sha256,
        expected_representative_review_sha256=_review_sha(root),
        reason="锚点人物脸型不符合批准方向",
    )
    reset = _manifest(root)
    assert result.reset_asset_ids == ("S01-A", "S01-B")
    assert [job.status for job in reset.jobs[:2]] == ["planned", "planned"]
    assert reset.jobs[0].master_sha256 is None
    assert reset.jobs[1].master_sha256 is None
    assert reset.jobs[1].anchor_sha256 is None
    assert reset.jobs[1].reference_image_sha256s == ()
    assert len(result.authorization_paths) == 2

    b_source = tmp_path / "new-b.png"
    b_source.write_bytes(b"brand-new-b")
    with pytest.raises(IllustrationReplacementError, match="replacement_import_order_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=result.nonce, source_path=b_source, ffmpeg_command="ffmpeg",
            inspector=_inspect, runner=_fake_runner,
        )

    a_source = tmp_path / "new-a.png"
    a_source.write_bytes(b"brand-new-a")
    imported_a = import_authorized_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-A", nonce=result.nonce, source_path=a_source, ffmpeg_command="ffmpeg",
        inspector=_inspect, runner=_fake_runner,
    )
    rebound = _manifest(root)
    new_a, pending_b = rebound.jobs[:2]
    assert imported_a.imported.master_sha256 == _sha_bytes(b"brand-new-a")
    assert pending_b.anchor_sha256 == new_a.master_sha256
    assert pending_b.reference_image_sha256s[0] == new_a.master_sha256
    assert pending_b.anchor_sha256 != old_a.master_sha256

    old_b_source = tmp_path / "old-b.png"
    old_b_source.write_bytes(b"old-S01-B")
    with pytest.raises(IllustrationReplacementError, match="replacement_same_sha256"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=result.nonce, source_path=old_b_source, ffmpeg_command="ffmpeg",
            inspector=_inspect, runner=_fake_runner,
        )
    pending_authorization = ReplacementAuthorization.model_validate_json(
        result.authorization_paths[1].read_text(encoding="utf-8")
    )
    assert pending_authorization.status == "authorized"

    imported_b = import_authorized_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", nonce=result.nonce, source_path=b_source, ffmpeg_command="ffmpeg",
        inspector=_inspect, runner=_fake_runner,
    )
    completed = _manifest(root)
    assert completed.jobs[1].master_sha256 == imported_b.imported.master_sha256
    assert completed.jobs[1].anchor_sha256 == completed.jobs[0].master_sha256
    assert store.load_episode(BOOK_ID, EPISODE_ID).status == "awaiting_representative_review"
    assert ReplacementAuthorization.model_validate_json(
        result.authorization_paths[0].read_text(encoding="utf-8")
    ).status == "consumed"
    assert ReplacementAuthorization.model_validate_json(
        result.authorization_paths[1].read_text(encoding="utf-8")
    ).status == "consumed"


@pytest.mark.parametrize("asset_id", ["S01-A", "S01-B"])
def test_hash_or_review_mismatch_fails_closed_without_tree_change(
    tmp_path: Path, asset_id: str
) -> None:
    store, root = _build_workspace(tmp_path)
    before = _tree(root)
    with pytest.raises(IllustrationReplacementError, match="replacement_evidence_mismatch"):
        prepare_illustration_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id=asset_id, expected_asset_sha256="f" * 64,
            expected_representative_review_sha256=_review_sha(root), reason="不连续",
        )
    assert _tree(root) == before

    with pytest.raises(IllustrationReplacementError, match="replacement_review_mismatch"):
        prepare_illustration_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id=asset_id,
            expected_asset_sha256=next(
                job.master_sha256 for job in _manifest(root).jobs if job.asset_id == asset_id
            ),
            expected_representative_review_sha256="e" * 64,
            reason="不连续",
        )
    assert _tree(root) == before


def test_blank_reason_legacy_and_final_approved_are_rejected(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    current = _manifest(root).jobs[0]
    with pytest.raises(IllustrationReplacementError, match="replacement_reason_required"):
        prepare_illustration_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-A", expected_asset_sha256=current.master_sha256,
            expected_representative_review_sha256=_review_sha(root), reason="  ",
        )
    state = store.load_episode(BOOK_ID, EPISODE_ID)
    state.status = "final_approved"
    store.save_episode(state)
    with pytest.raises(IllustrationReplacementError, match="replacement_final_approved"):
        prepare_illustration_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-A", expected_asset_sha256=current.master_sha256,
            expected_representative_review_sha256=_review_sha(root), reason="不连续",
        )

    store2, root2 = _build_workspace(tmp_path / "legacy")
    manifest = _manifest(root2)
    legacy = manifest.jobs[0].model_copy(
        update={"phase": "legacy", "asset_id": None, "output_bw": root2 / "media/illustration/images/S01_bw.png"}
    )
    (legacy.output_bw).write_bytes(b"bw")
    legacy = legacy.model_copy(update={"bw_sha256": sha256_file(legacy.output_bw)})
    manifest = manifest.model_copy(update={"jobs": (legacy, *manifest.jobs[1:])})
    (root2 / "media/illustration/illustration_manifest.json").write_text(
        manifest.model_dump_json(), encoding="utf-8"
    )
    with pytest.raises(IllustrationReplacementError, match="replacement_pair_only"):
        prepare_illustration_replacement(
            store=store2, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-A", expected_asset_sha256=legacy.master_sha256,
            expected_representative_review_sha256=_review_sha(root2), reason="不连续",
        )


def test_tampered_authorization_and_failed_import_do_not_consume_it(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    result = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不连续",
    )
    authorization_path = result.authorization_paths[0]
    original = authorization_path.read_bytes()
    payload = json.loads(original)
    payload["reason"] = "tampered"
    authorization_path.write_text(json.dumps(payload), encoding="utf-8")
    source = tmp_path / "replacement.png"
    source.write_bytes(b"replacement")
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=result.nonce, source_path=source, ffmpeg_command="ffmpeg",
            inspector=_inspect, runner=_fake_runner,
        )
    assert not job.output_master.exists()
    authorization_path.write_bytes(original)
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"bad")
    with pytest.raises(IllustrationReplacementError, match="replacement_import_failed"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=result.nonce, source_path=bad, ffmpeg_command="ffmpeg",
            inspector=_inspect, runner=_fake_runner,
        )
    assert ReplacementAuthorization.model_validate_json(
        authorization_path.read_text(encoding="utf-8")
    ).status == "authorized"


class _FailingStore(StateStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.fail = False

    def save_episode(self, state: EpisodeState) -> None:
        if self.fail:
            raise OSError("injected save failure")
        super().save_episode(state)


def test_prepare_save_failure_restores_tree_byte_exact(tmp_path: Path) -> None:
    initial, root = _build_workspace(tmp_path)
    store = _FailingStore(initial.root)
    before = _tree(root)
    job = _manifest(root).jobs[0]
    store.fail = True
    with pytest.raises(IllustrationReplacementError, match="replacement_transaction_failed"):
        prepare_illustration_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-A", expected_asset_sha256=job.master_sha256,
            expected_representative_review_sha256=_review_sha(root), reason="脸型不一致",
        )
    assert _tree(root) == before


def test_import_save_failure_restores_authorization_manifest_and_output(tmp_path: Path) -> None:
    initial, root = _build_workspace(tmp_path)
    store = _FailingStore(initial.root)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"replacement-new")
    store.fail = True
    with pytest.raises(IllustrationReplacementError, match="replacement_transaction_failed"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=prepared.nonce, source_path=source, ffmpeg_command="ffmpeg",
            inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before
    assert ReplacementAuthorization.model_validate_json(
        prepared.authorization_paths[0].read_text(encoding="utf-8")
    ).status == "authorized"


def test_anchor_import_save_failure_restores_both_authorizations(tmp_path: Path) -> None:
    initial, root = _build_workspace(tmp_path)
    store = _FailingStore(initial.root)
    job = _manifest(root).jobs[0]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-A", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="脸型不一致",
    )
    before = _tree(root)
    source = tmp_path / "replacement-a.png"
    source.write_bytes(b"replacement-anchor")
    store.fail = True
    with pytest.raises(IllustrationReplacementError, match="replacement_transaction_failed"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-A", nonce=prepared.nonce, source_path=source,
            ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before
    assert all(
        ReplacementAuthorization.model_validate_json(path.read_text(encoding="utf-8")).status
        == "authorized"
        for path in prepared.authorization_paths
    )


def test_existing_history_reparse_or_collision_fails_closed(tmp_path: Path, monkeypatch) -> None:
    store, root = _build_workspace(tmp_path)
    history = root / ".private" / "history"
    history.mkdir(parents=True)
    collision = history / "fixed-replace-S01-B-00000000000000000000000000000000"
    collision.mkdir()
    monkeypatch.setattr(
        replacement_module,
        "_new_history_name",
        lambda _asset_id: collision.name,
    )
    before = _tree(root)
    job = _manifest(root).jobs[1]
    with pytest.raises(IllustrationReplacementError, match="replacement_history_collision"):
        prepare_illustration_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", expected_asset_sha256=job.master_sha256,
            expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
        )
    assert _tree(root) == before


def test_move_or_post_move_hash_failure_restores_tree_byte_exact(
    tmp_path: Path, monkeypatch
) -> None:
    for failure in ("move", "hash"):
        if failure == "hash":
            monkeypatch.undo()
            monkeypatch.setattr(restyle_module, "inspect_image", _inspect)
        store, root = _build_workspace(tmp_path / failure)
        before = _tree(root)
        job = _manifest(root).jobs[1]
        if failure == "move":
            real_replace = replacement_module.os.replace
            failed = False

            def fail_once(source, destination):
                nonlocal failed
                if not failed and Path(source).name == "S01_continuation.png":
                    failed = True
                    raise OSError("injected move failure")
                return real_replace(source, destination)

            monkeypatch.setattr(replacement_module.os, "replace", fail_once)
        else:
            real_hash = replacement_module.sha256_file
            failed = False

            def fail_hash_once(path: Path) -> str:
                nonlocal failed
                target = Path(path)
                if (
                    not failed
                    and target.name.endswith("S01_continuation.png")
                    and ".private" in target.parts
                ):
                    failed = True
                    raise OSError("injected hash failure")
                return real_hash(target)

            monkeypatch.setattr(replacement_module, "sha256_file", fail_hash_once)
        with pytest.raises(IllustrationReplacementError, match="replacement_transaction_failed"):
            prepare_illustration_replacement(
                store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
                asset_id="S01-B", expected_asset_sha256=job.master_sha256,
                expected_representative_review_sha256=_review_sha(root),
                reason="动作不一致",
            )
        assert _tree(root) == before
        monkeypatch.undo()


def test_state_bound_authorization_path_tamper_is_rejected(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    state = store.load_episode(BOOK_ID, EPISODE_ID)
    stage = state.stage_manifests["illustration_replacement"]
    output = stage.outputs["authorization_S01-B"]
    stage.outputs["authorization_S01-B"] = output.model_copy(
        update={"path": str(tmp_path / "outside.json")}
    )
    store.save_episode(state)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"new-b")
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=prepared.nonce, source_path=source, ffmpeg_command="ffmpeg",
            inspector=_inspect, runner=_fake_runner,
        )
    assert not _manifest(root).jobs[1].output_master.exists()
    assert ReplacementAuthorization.model_validate_json(
        prepared.authorization_paths[0].read_text(encoding="utf-8")
    ).status == "authorized"


def test_media_service_exposes_hash_bound_rejection_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bv.workflow.media_runtime as media_runtime_module
    from bv.workflow.media_runtime import (
        LocalIllustrationGateway,
        MediaProductionService,
        MediaWorkflowError,
    )

    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    service = MediaProductionService(
        store=store,
        stages={},
        images=LocalIllustrationGateway(ffmpeg_command="ffmpeg"),
        restyle_catalog_path=CATALOG,
    )
    result = service.reject_illustration(
        BOOK_ID,
        EPISODE_ID,
        asset_id="S01-B",
        expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root),
        reason="动作不一致",
    )
    assert result.reset_asset_ids == ("S01-B",)
    assert result.authorization_paths[0].is_file()
    source = tmp_path / "ordinary-import.png"
    source.write_bytes(b"ordinary-import-must-not-bypass-receipt")
    with pytest.raises(
        MediaWorkflowError, match="replacement_import_requires_authorization"
    ):
        service.import_image(BOOK_ID, EPISODE_ID, "S01-B", source)
    assert ReplacementAuthorization.model_validate_json(
        result.authorization_paths[0].read_text(encoding="utf-8")
    ).status == "authorized"
    real_import = import_authorized_replacement

    def import_with_fake_media(**kwargs: object) -> object:
        return real_import(**kwargs, inspector=_inspect, runner=_fake_runner)

    monkeypatch.setattr(
        media_runtime_module, "import_authorized_replacement", import_with_fake_media
    )
    before = _tree(root)
    with pytest.raises(MediaWorkflowError, match="replacement_authorization_invalid"):
        service.import_replacement(
            BOOK_ID,
            EPISODE_ID,
            asset_id="S01-B",
            nonce="f" * 32,
            source=source,
        )
    assert _tree(root) == before
    imported = service.import_replacement(
        BOOK_ID,
        EPISODE_ID,
        asset_id="S01-B",
        nonce=result.nonce,
        source=source,
    )
    assert imported.transaction_id == result.transaction_id


def test_four_scene_restart_flow_reapproves_pairs_and_returns_render_ready(
    tmp_path: Path,
) -> None:
    from bv.workflow.media_runtime import LocalIllustrationGateway, MediaProductionService

    store, root = _build_workspace(tmp_path, representatives_only=True)
    storyboard = IllustrationStoryboard.model_validate_json(
        (root / "media/illustration/illustration_storyboard.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = _manifest(root)
    approval = RepresentativeApproval.model_validate_json(
        (root / "media/illustration/representative_approval.json").read_text(
            encoding="utf-8"
        )
    )
    assert [job.asset_id for job in manifest.jobs if job.status == "generated"] == [
        "S01-A", "S01-B", "S03-A", "S03-B", "S04-A", "S04-B"
    ]
    assert [job.asset_id for job in unlock_batch_jobs(storyboard, manifest, approval)] == [
        "S02-A", "S02-B"
    ]

    for asset_id, payload in (("S02-A", b"fresh-S02-A"), ("S02-B", b"fresh-S02-B")):
        manifest = _manifest(root)
        job = next(item for item in manifest.jobs if item.asset_id == asset_id)
        source = tmp_path / f"{asset_id}.png"
        source.write_bytes(payload)
        imported = import_image_asset(
            job, source, manifest=manifest, inspector=_inspect,
            ffmpeg_command="ffmpeg", runner=_fake_runner,
        )
        manifest = record_imported_image(manifest, imported)
        (root / "media/illustration/illustration_manifest.json").write_text(
            manifest.model_dump_json(), encoding="utf-8"
        )
    state = store.load_episode(BOOK_ID, EPISODE_ID)
    state.status = "illustrations_ready"
    store.save_episode(state)

    current_b = next(job for job in _manifest(root).jobs if job.asset_id == "S01-B")
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=current_b.master_sha256,
        expected_representative_review_sha256=_review_sha(root),
        reason="代表对后半动作与锚点不连续",
    )
    replacement = tmp_path / "S01-B-replacement.png"
    replacement.write_bytes(b"new-S01-B")
    import_authorized_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", nonce=prepared.nonce, source_path=replacement, ffmpeg_command="ffmpeg",
        inspector=_inspect, runner=_fake_runner,
    )

    restarted_store = StateStore(store.root)
    service = MediaProductionService(
        store=restarted_store,
        stages={},
        images=LocalIllustrationGateway(ffmpeg_command="ffmpeg"),
        restyle_catalog_path=CATALOG,
    )
    restarted = service.status(BOOK_ID, EPISODE_ID)
    assert restarted.status == "awaiting_representative_review"
    assert restarted.present_scene_ids == (
        "S01-A", "S01-B", "S03-A", "S03-B", "S04-A", "S04-B"
    )
    service.approve_representatives(BOOK_ID, EPISODE_ID)
    ready = service.prepare_batch(BOOK_ID, EPISODE_ID)
    assert ready.status == "illustrations_ready"
    assert ready.missing_scene_ids == ()
    assert ready.batch_unlocked is True


def test_nonrepresentative_b_restarts_at_review_then_returns_to_batch(tmp_path: Path) -> None:
    from bv.workflow.media_runtime import LocalIllustrationGateway, MediaProductionService

    store, root = _build_workspace(tmp_path)
    manifest = _manifest(root)
    old_a = next(job for job in manifest.jobs if job.asset_id == "S02-A")
    old_b = next(job for job in manifest.jobs if job.asset_id == "S02-B")
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S02-B", expected_asset_sha256=old_b.master_sha256,
        expected_representative_review_sha256=_review_sha(root),
        reason="非代表场延续动作偏离旁白",
    )
    reset = _manifest(root)
    assert next(job for job in reset.jobs if job.asset_id == "S02-A") == old_a
    pending_b = next(job for job in reset.jobs if job.asset_id == "S02-B")
    assert pending_b.anchor_sha256 == old_a.master_sha256
    assert prepared.status == "awaiting_representative_review"

    service = MediaProductionService(
        store=StateStore(store.root), stages={},
        images=LocalIllustrationGateway(ffmpeg_command="ffmpeg"),
        restyle_catalog_path=CATALOG,
    )
    service.approve_representatives(BOOK_ID, EPISODE_ID)
    batch = service.prepare_batch(BOOK_ID, EPISODE_ID)
    assert batch.status == "illustration_batch_running"
    assert batch.missing_scene_ids == ("S02-B",)
    source = tmp_path / "S02-B-replacement.png"
    source.write_bytes(b"new-S02-B")
    result = import_authorized_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S02-B", nonce=prepared.nonce, source_path=source, ffmpeg_command="ffmpeg",
        inspector=_inspect, runner=_fake_runner,
    )
    assert result.status == "illustrations_ready"


def test_consumed_authorization_is_archived_before_a_second_rejection(tmp_path: Path) -> None:
    from bv.workflow.media_runtime import LocalIllustrationGateway, MediaProductionService

    store, root = _build_workspace(tmp_path)
    first = next(job for job in _manifest(root).jobs if job.asset_id == "S01-B")
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=first.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="第一次拒绝",
    )
    source = tmp_path / "first-replacement.png"
    source.write_bytes(b"first-replacement")
    import_authorized_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", nonce=prepared.nonce, source_path=source, ffmpeg_command="ffmpeg",
        inspector=_inspect, runner=_fake_runner,
    )
    service = MediaProductionService(
        store=store,
        stages={},
        images=LocalIllustrationGateway(ffmpeg_command="ffmpeg"),
        restyle_catalog_path=CATALOG,
    )
    service.approve_representatives(BOOK_ID, EPISODE_ID)
    current = next(job for job in _manifest(root).jobs if job.asset_id == "S01-B")
    second = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=current.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="第二次拒绝",
    )
    assert second.authorization_paths[0] == prepared.authorization_paths[0]
    assert any(
        path.name.endswith("S01-B.json")
        for path in second.history_dir.glob("files/*")
    )
    assert ReplacementAuthorization.model_validate_json(
        second.authorization_paths[0].read_text(encoding="utf-8")
    ).reason == "第二次拒绝"


@pytest.mark.parametrize(("asset_id", "cycles"), (("S01-B", 3), ("S01-A", 2)))
def test_reapproved_asset_can_be_rejected_and_imported_repeatedly(
    tmp_path: Path, asset_id: str, cycles: int
) -> None:
    from bv.workflow.media_runtime import LocalIllustrationGateway, MediaProductionService

    store, root = _build_workspace(tmp_path)
    service = MediaProductionService(
        store=store,
        stages={},
        images=LocalIllustrationGateway(ffmpeg_command="ffmpeg"),
        restyle_catalog_path=CATALOG,
    )
    for cycle in range(1, cycles + 1):
        target = next(job for job in _manifest(root).jobs if job.asset_id == asset_id)
        prepared = prepare_illustration_replacement(
            store=store,
            book_id=BOOK_ID,
            episode_id=EPISODE_ID,
            asset_id=asset_id,
            expected_asset_sha256=target.master_sha256 or "",
            expected_representative_review_sha256=_review_sha(root),
            reason=f"第{cycle}次拒绝",
        )
        for replacement_asset_id in prepared.reset_asset_ids:
            source = tmp_path / f"{asset_id}-{cycle}-{replacement_asset_id}.png"
            source.write_bytes(
                f"replacement-{asset_id}-{cycle}-{replacement_asset_id}".encode()
            )
            result = import_authorized_replacement(
                store=store,
                book_id=BOOK_ID,
                episode_id=EPISODE_ID,
                asset_id=replacement_asset_id,
                nonce=prepared.nonce,
                source_path=source,
                ffmpeg_command="ffmpeg",
                inspector=_inspect,
                runner=_fake_runner,
            )
        assert result.status == "awaiting_representative_review"
        approved = service.approve_representatives(BOOK_ID, EPISODE_ID)
        assert approved.status == "representatives_approved"


def test_prepare_rejects_an_isolated_representative_approval(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    (root / "media/illustration/representative_review.json").unlink()
    target = next(job for job in _manifest(root).jobs if job.asset_id == "S01-B")
    before = _tree(root)

    with pytest.raises(IllustrationReplacementError, match="illustration_provenance_invalid"):
        prepare_illustration_replacement(
            store=store,
            book_id=BOOK_ID,
            episode_id=EPISODE_ID,
            asset_id="S01-B",
            expected_asset_sha256=target.master_sha256 or "",
            reason="拒绝孤立批准证据",
        )
    assert _tree(root) == before


def test_approval_projection_tamper_is_rejected_after_hash_refresh(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    target = next(job for job in _manifest(root).jobs if job.asset_id == "S01-B")
    prepared = prepare_illustration_replacement(
        store=store,
        book_id=BOOK_ID,
        episode_id=EPISODE_ID,
        asset_id="S01-B",
        expected_asset_sha256=target.master_sha256 or "",
        expected_representative_review_sha256=_review_sha(root),
        reason="动作不一致",
    )
    receipt_path = prepared.history_dir / "replacement_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["representative_approval_sha256"] = "f" * 64
    _refresh_receipt_and_authorization_artifacts(store, root, receipt_path, receipt)
    state = store.load_episode(BOOK_ID, EPISODE_ID)
    state.stage_manifests["illustration_replacement"].inputs[
        "representative_approval_sha256"
    ] = "f" * 64
    store.save_episode(state)
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")

    with pytest.raises(
        IllustrationReplacementError, match="replacement_authorization_invalid"
    ):
        import_authorized_replacement(
            store=store,
            book_id=BOOK_ID,
            episode_id=EPISODE_ID,
            asset_id="S01-B",
            nonce=prepared.nonce,
            source_path=source,
            ffmpeg_command="ffmpeg",
            inspector=_inspect,
            runner=_fake_runner,
        )
    assert _tree(root) == before


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink unavailable")
def test_history_descendant_symlink_and_outside_sentinel_are_untouched(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"sentinel")
    history = root / ".private" / "history"
    history.mkdir(parents=True)
    link = history / "unsafe-link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("current Windows account cannot create symlinks")
    job = _manifest(root).jobs[1]
    with pytest.raises(IllustrationReplacementError, match="replacement_unsafe_path"):
        prepare_illustration_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", expected_asset_sha256=job.master_sha256,
            expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
        )
    assert outside.read_bytes() == b"sentinel"


@pytest.mark.parametrize(
    ("surface", "field", "value"),
    (
        ("receipt", "transaction_id", "f" * 32),
        ("receipt", "state_snapshot_sha256", "f" * 64),
        ("receipt_archive", "archive", "files/tampered.png"),
        ("receipt_archive", "source", "../outside.png"),
        ("receipt_archive", "sha256", "f" * 64),
        ("receipt_archive", "size_bytes", 0),
        ("receipt_projection", "S01-B", "f" * 64),
        ("stage_input", "transaction_id", "f" * 32),
        ("stage_input", "nonce", "f" * 32),
        ("stage_input", "rejected_asset_id", "S02-B"),
        ("stage_input", "reason", "tampered reason"),
        ("stage_input", "history_relative_path", ".private/history/tampered"),
        ("stage_input", "representative_review_sha256", "f" * 64),
        ("stage_input", "state_snapshot_sha256", "f" * 64),
        ("stage_input", "manifest_snapshot_sha256", "f" * 64),
        ("stage_input", "affected_stage_manifests_sha256", "f" * 64),
        ("stage_input", "receipt_sha256", "f" * 64),
        ("stage_input", "reset_job_projection_sha256", "f" * 64),
        ("stage", "stage", "render"),
        ("stage", "status", "stale"),
        ("stage", "config_sha256", "f" * 64),
    ),
)
def test_reviewed_transaction_binding_tamper_is_byte_exact_rejected(
    tmp_path: Path, surface: str, field: str, value: object
) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store,
        book_id=BOOK_ID,
        episode_id=EPISODE_ID,
        asset_id="S01-B",
        expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root),
        reason="动作不一致",
    )
    if surface.startswith("receipt"):
        path = prepared.history_dir / "replacement_receipt.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if surface == "receipt_archive":
            payload["archived_files"][0][field] = value
        elif surface == "receipt_projection":
            payload["reset_job_projection_sha256s"][field] = value
        else:
            payload[field] = value
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        state = store.load_episode(BOOK_ID, EPISODE_ID)
        stage = state.stage_manifests["illustration_replacement"]
        if surface == "stage_input":
            stage.inputs[field] = value
        else:
            setattr(stage, field, value)
        store.save_episode(state)
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")
    with pytest.raises(
        IllustrationReplacementError, match="replacement_authorization_invalid"
    ):
        import_authorized_replacement(
            store=store,
            book_id=BOOK_ID,
            episode_id=EPISODE_ID,
            asset_id="S01-B",
            nonce=prepared.nonce,
            source_path=source,
            ffmpeg_command="ffmpeg",
            inspector=_inspect,
            runner=_fake_runner,
        )
    assert _tree(root) == before


@pytest.mark.parametrize(
    ("label", "field", "value"),
    (
        ("replacement_receipt", "path", "C:/outside/receipt.json"),
        ("replacement_receipt", "sha256", "f" * 64),
        ("replacement_receipt", "size_bytes", 0),
        ("authorization_S01-B", "path", "C:/outside/auth.json"),
        ("authorization_S01-B", "sha256", "f" * 64),
        ("authorization_S01-B", "size_bytes", 0),
    ),
)
def test_each_stage_output_reference_is_exact(
    tmp_path: Path, label: str, field: str, value: object
) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    state = store.load_episode(BOOK_ID, EPISODE_ID)
    stage = state.stage_manifests["illustration_replacement"]
    stage.outputs[label] = stage.outputs[label].model_copy(update={field: value})
    store.save_episode(state)
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=prepared.nonce, source_path=source,
            ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before


def test_wrong_replacement_nonce_is_byte_exact_rejected(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce="f" * 32, source_path=source,
            ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before
    assert ReplacementAuthorization.model_validate_json(
        prepared.authorization_paths[0].read_text(encoding="utf-8")
    ).status == "authorized"


def test_refreshed_receipt_artifact_still_requires_authorization_binding(
    tmp_path: Path,
) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    receipt_path = prepared.history_dir / "replacement_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["reason"] = "tampered reason"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    state = store.load_episode(BOOK_ID, EPISODE_ID)
    stage = state.stage_manifests["illustration_replacement"]
    stage.inputs["reason"] = "tampered reason"
    stage.inputs["receipt_sha256"] = sha256_file(receipt_path)
    stage.outputs["replacement_receipt"] = _artifact(receipt_path)
    store.save_episode(state)
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=prepared.nonce, source_path=source,
            ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before


def test_attempts_tamper_blocks_ordinary_and_authorized_import_byte_exact(
    tmp_path: Path,
) -> None:
    from bv.workflow.media_runtime import (
        LocalIllustrationGateway,
        MediaProductionService,
        MediaWorkflowError,
    )

    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store,
        book_id=BOOK_ID,
        episode_id=EPISODE_ID,
        asset_id="S01-B",
        expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root),
        reason="动作不一致",
    )
    manifest = _manifest(root)
    manifest = manifest.model_copy(
        update={
            "jobs": (
                manifest.jobs[0],
                manifest.jobs[1].model_copy(update={"attempts": 0}),
                *manifest.jobs[2:],
            )
        }
    )
    (root / "media/illustration/illustration_manifest.json").write_text(
        manifest.model_dump_json(), encoding="utf-8"
    )
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")
    service = MediaProductionService(
        store=store,
        stages={},
        images=LocalIllustrationGateway(ffmpeg_command="ffmpeg"),
        restyle_catalog_path=CATALOG,
    )
    with pytest.raises(MediaWorkflowError, match="replacement_authorization_invalid"):
        service.import_image(BOOK_ID, EPISODE_ID, "S01-B", source)
    assert _tree(root) == before
    with pytest.raises(
        IllustrationReplacementError, match="replacement_authorization_invalid"
    ):
        import_authorized_replacement(
            store=store,
            book_id=BOOK_ID,
            episode_id=EPISODE_ID,
            asset_id="S01-B",
            nonce=prepared.nonce,
            source_path=source,
            ffmpeg_command="ffmpeg",
            inspector=_inspect,
            runner=_fake_runner,
        )
    assert _tree(root) == before
    assert ReplacementAuthorization.model_validate_json(
        prepared.authorization_paths[0].read_text(encoding="utf-8")
    ).status == "authorized"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 1),
        ("nonce", "f" * 32),
        ("book_id", "other-book"),
        ("episode_id", "other-episode"),
        ("rejected_asset_id", "S02-B"),
        ("reason", "tampered reason"),
        ("expected_asset_sha256", "f" * 64),
        ("representative_review_sha256", None),
        ("representative_approval_sha256", None),
        ("history_relative_path", ".private/history/tampered"),
        ("reset_asset_ids", ["S02-B"]),
        ("authorization_asset_ids", ["S02-B"]),
        ("state_snapshot_sha256", "f" * 64),
        ("manifest_snapshot_sha256", "f" * 64),
        ("affected_stage_manifests_sha256", "f" * 64),
        ("prepared_at_ns", 1),
    ),
)
def test_each_receipt_identity_field_is_bound(
    tmp_path: Path, field: str, value: object
) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    receipt_path = prepared.history_dir / "replacement_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload[field] = value
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=prepared.nonce, source_path=source,
            ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 1),
        ("transaction_id", "f" * 32),
        ("nonce", "f" * 32),
        ("book_id", "other-book"),
        ("episode_id", "other-episode"),
        ("asset_id", "S02-B"),
        ("phase", "anchor"),
        ("reason", "tampered reason"),
        ("rejected_sha256", "f" * 64),
        ("representative_review_sha256", None),
        ("storyboard_sha256", "f" * 64),
        ("style_fingerprint", "f" * 64),
        ("character_lock_sha256", "f" * 64),
        ("prepared_attempt", 2),
        ("archive_relative_path", ".private/history/tampered"),
        ("receipt_sha256", "f" * 64),
        ("state_snapshot_sha256", "f" * 64),
        ("manifest_snapshot_sha256", "f" * 64),
        ("job_projection_sha256", "f" * 64),
        ("prepared_at_ns", 1),
        ("status", "consumed"),
        ("replacement_sha256", "f" * 64),
        ("replacement_size_bytes", 1),
    ),
)
def test_each_authorization_claim_is_cross_bound_even_if_artifact_ref_is_refreshed(
    tmp_path: Path, field: str, value: object
) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    authorization_path = prepared.authorization_paths[0]
    payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    payload[field] = value
    authorization_path.write_text(json.dumps(payload), encoding="utf-8")
    state = store.load_episode(BOOK_ID, EPISODE_ID)
    stage = state.stage_manifests["illustration_replacement"]
    stage.outputs["authorization_S01-B"] = _artifact(authorization_path)
    store.save_episode(state)
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=prepared.nonce, source_path=source,
            ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("attempts", 0),
        ("attempts", 99),
        ("status", "generated"),
        ("output_master", "C:/outside/replacement.png"),
        ("master_sha256", "f" * 64),
        ("anchor_sha256", "f" * 64),
        ("reference_image_sha256s", ["f" * 64]),
    ),
)
def test_active_replacement_job_drift_blocks_both_import_paths(
    tmp_path: Path, field: str, value: object
) -> None:
    from bv.workflow.media_runtime import (
        LocalIllustrationGateway,
        MediaProductionService,
        MediaWorkflowError,
    )

    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    manifest_path = root / "media/illustration/illustration_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["jobs"][1][field] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")
    service = MediaProductionService(
        store=store, stages={},
        images=LocalIllustrationGateway(ffmpeg_command="ffmpeg"),
        restyle_catalog_path=CATALOG,
    )
    with pytest.raises(MediaWorkflowError, match="replacement_authorization_invalid"):
        service.import_image(BOOK_ID, EPISODE_ID, "S01-B", source)
    assert _tree(root) == before
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=prepared.nonce, source_path=source,
            ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before


def test_concurrent_replacement_import_has_one_committed_winner(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    sources = (tmp_path / "replacement-one.png", tmp_path / "replacement-two.png")
    sources[0].write_bytes(b"fresh-replacement-one")
    sources[1].write_bytes(b"fresh-replacement-two")

    def attempt(source: Path) -> ReplacementImportResult | IllustrationReplacementError:
        try:
            return import_authorized_replacement(
                store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
                asset_id="S01-B", nonce=prepared.nonce, source_path=source,
                ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
            )
        except IllustrationReplacementError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(attempt, sources))
    winners = tuple(item for item in outcomes if isinstance(item, ReplacementImportResult))
    losers = tuple(item for item in outcomes if isinstance(item, IllustrationReplacementError))
    assert len(winners) == 1
    assert len(losers) == 1
    assert losers[0].error_code == "replacement_authorization_invalid"
    authorization = ReplacementAuthorization.model_validate_json(
        prepared.authorization_paths[0].read_text(encoding="utf-8")
    )
    current = _manifest(root).jobs[1]
    assert authorization.status == "consumed"
    assert authorization.replacement_sha256 == current.master_sha256
    assert current.output_master.read_bytes() in {
        b"fresh-replacement-one", b"fresh-replacement-two"
    }


@pytest.mark.parametrize("entrypoint", ("authorized", "ordinary"))
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("attempts", 99),
        ("status", "approved"),
        ("prompt", "tampered prompt"),
        ("style_fingerprint", "f" * 64),
        ("output_master", "C:/outside/tampered-a.png"),
        ("master_sha256", "f" * 64),
        ("__file_bytes__", "tampered-a-file"),
    ),
)
def test_consumed_a_drift_blocks_b_import_without_laundering_projection(
    tmp_path: Path, entrypoint: str, field: str, value: object
) -> None:
    from bv.workflow.media_runtime import (
        LocalIllustrationGateway,
        MediaProductionService,
        MediaWorkflowError,
    )

    store, root = _build_workspace(tmp_path)
    old_a = _manifest(root).jobs[0]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-A", expected_asset_sha256=old_a.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="脸型不一致",
    )
    source_a = tmp_path / "replacement-a.png"
    source_a.write_bytes(b"fresh-replacement-a")
    import_authorized_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-A", nonce=prepared.nonce, source_path=source_a,
        ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
    )
    manifest_path = root / "media/illustration/illustration_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "__file_bytes__":
        Path(manifest_payload["jobs"][0]["output_master"]).write_bytes(
            str(value).encode("utf-8")
        )
    else:
        manifest_payload["jobs"][0][field] = value
        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    before = _tree(root)
    source_b = tmp_path / "replacement-b.png"
    source_b.write_bytes(b"fresh-replacement-b")
    if entrypoint == "authorized":
        with pytest.raises(
            IllustrationReplacementError, match="replacement_authorization_invalid"
        ):
            import_authorized_replacement(
                store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
                asset_id="S01-B", nonce=prepared.nonce, source_path=source_b,
                ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
            )
    else:
        service = MediaProductionService(
            store=store, stages={},
            images=LocalIllustrationGateway(ffmpeg_command="ffmpeg"),
            restyle_catalog_path=CATALOG,
        )
        with pytest.raises(MediaWorkflowError, match="replacement_authorization_invalid"):
            service.import_image(BOOK_ID, EPISODE_ID, "S01-B", source_b)
    assert _tree(root) == before


def test_b_import_does_not_rewrite_consumed_a_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, root = _build_workspace(tmp_path)
    old_a = _manifest(root).jobs[0]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-A", expected_asset_sha256=old_a.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="脸型不一致",
    )
    source_a = tmp_path / "replacement-a.png"
    source_a.write_bytes(b"fresh-replacement-a")
    import_authorized_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-A", nonce=prepared.nonce, source_path=source_a,
        ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
    )
    consumed_a_path = prepared.authorization_paths[0]
    real_atomic_write_json = replacement_module._atomic_write_json

    def reject_consumed_a_rewrite(path: Path, payload: object) -> None:
        if Path(path) == consumed_a_path:
            raise AssertionError("consumed A authorization must remain immutable")
        real_atomic_write_json(path, payload)

    monkeypatch.setattr(
        replacement_module, "_atomic_write_json", reject_consumed_a_rewrite
    )
    source_b = tmp_path / "replacement-b.png"
    source_b.write_bytes(b"fresh-replacement-b")
    result = import_authorized_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", nonce=prepared.nonce, source_path=source_b,
        ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
    )
    assert result.imported.master_sha256 == _sha_bytes(b"fresh-replacement-b")


@pytest.mark.parametrize(
    "source_suffix",
    (
        "episode.json",
        "media/illustration/illustration_manifest.json",
        ".transaction/affected_stage_manifests.json",
        "media/illustration/images/S01_continuation.png",
        "media/illustration/representative_review.json",
        "media/illustration/representative_approval.json",
        "media/visual/silent.mp4",
        "media/final/final.mp4",
        "media/final/final.qc.json",
        "deliveries/2026-09-02/candidate.json",
        "deliveries/2026-09-02/final-approval.json",
    ),
)
def test_snapshot_required_archive_entry_cannot_be_omitted_with_refreshed_hashes(
    tmp_path: Path, source_suffix: str
) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    receipt_path = prepared.history_dir / "replacement_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    matches = [
        entry for entry in receipt["archived_files"]
        if str(entry["source"]).endswith(source_suffix)
    ]
    assert len(matches) == 1
    removed = matches[0]
    (prepared.history_dir / str(removed["archive"])).unlink()
    receipt["archived_files"].remove(removed)
    _refresh_receipt_and_authorization_artifacts(store, root, receipt_path, receipt)
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=prepared.nonce, source_path=source,
            ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before


def test_archive_extra_entry_is_rejected_even_with_refreshed_hashes(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    receipt_path = prepared.history_dir / "replacement_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    extra = prepared.history_dir / "files/999-extra.bin"
    extra.write_bytes(b"extra-history-evidence")
    receipt["archived_files"].append(
        {
            "source": "media/extra.bin",
            "archive": "files/999-extra.bin",
            "sha256": sha256_file(extra),
            "size_bytes": extra.stat().st_size,
        }
    )
    _refresh_receipt_and_authorization_artifacts(store, root, receipt_path, receipt)
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=prepared.nonce, source_path=source,
            ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before


def test_affected_stage_manifest_archive_is_required(tmp_path: Path) -> None:
    store, root = _build_workspace(tmp_path)
    job = _manifest(root).jobs[1]
    prepared = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="动作不一致",
    )
    (prepared.history_dir / "affected_stage_manifests.json").unlink()
    before = _tree(root)
    source = tmp_path / "replacement.png"
    source.write_bytes(b"fresh-replacement")
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=prepared.nonce, source_path=source,
            ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before


def test_previous_replacement_authorization_archive_cannot_be_omitted(
    tmp_path: Path,
) -> None:
    from bv.workflow.media_runtime import LocalIllustrationGateway, MediaProductionService

    store, root = _build_workspace(tmp_path)
    first_job = _manifest(root).jobs[1]
    first = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=first_job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="第一次拒绝",
    )
    first_source = tmp_path / "first-replacement.png"
    first_source.write_bytes(b"first-replacement")
    import_authorized_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", nonce=first.nonce, source_path=first_source,
        ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
    )
    service = MediaProductionService(
        store=store,
        stages={},
        images=LocalIllustrationGateway(ffmpeg_command="ffmpeg"),
        restyle_catalog_path=CATALOG,
    )
    service.approve_representatives(BOOK_ID, EPISODE_ID)
    current_job = _manifest(root).jobs[1]
    second = prepare_illustration_replacement(
        store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
        asset_id="S01-B", expected_asset_sha256=current_job.master_sha256,
        expected_representative_review_sha256=_review_sha(root), reason="第二次拒绝",
    )
    receipt_path = second.history_dir / "replacement_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    matches = [
        entry for entry in receipt["archived_files"]
        if entry["source"] == ".private/replacements/S01-B.json"
    ]
    assert len(matches) == 1
    removed = matches[0]
    (second.history_dir / str(removed["archive"])).unlink()
    receipt["archived_files"].remove(removed)
    _refresh_receipt_and_authorization_artifacts(store, root, receipt_path, receipt)
    before = _tree(root)
    second_source = tmp_path / "second-replacement.png"
    second_source.write_bytes(b"second-replacement")
    with pytest.raises(IllustrationReplacementError, match="replacement_authorization_invalid"):
        import_authorized_replacement(
            store=store, book_id=BOOK_ID, episode_id=EPISODE_ID,
            asset_id="S01-B", nonce=second.nonce, source_path=second_source,
            ffmpeg_command="ffmpeg", inspector=_inspect, runner=_fake_runner,
        )
    assert _tree(root) == before
