from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bv.state.models import EpisodeState
from bv.state.store import StateStore
from bv.config import AppConfig
from bv.production.profile import ProductionProfile
from bv.delivery.exporter import ApprovalRecord, DeliveryBundle
from bv.core.hashing import sha256_file
from bv.workflow.media_runtime import (
    LocalSocialCoverGateway,
    LocalDeliveryGateway,
    MediaProductionService,
    MediaWorkflowError,
)
from bv.workflow.runtime import RuntimeAuthorization, build_media_runtime_bindings
from bv.workflow.stages import HANDDRAWN_MEDIA_PREPARE_ORDER, StageOutcome


class ArtifactStage:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.statuses: list[str] = []

    def run(self, context) -> StageOutcome:
        self.calls += 1
        self.statuses.append(context.episode_state.status)
        path = context.episode_root / "media" / "stage-fixtures" / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"stage":"{self.name}"}}', encoding="utf-8")
        return StageOutcome(outputs={self.name: path}, inputs={"script_sha256": "a" * 64})


class FakeIllustrationGateway:
    representative_ids = ("S01", "S05", "S10")
    all_ids = ("S01", "S02", "S03", "S05", "S10")

    def __init__(self) -> None:
        self.present: set[str] = set()
        self.approved = False
        self.current = True

    def status(self, _context):
        present = tuple(item for item in self.representative_ids if item in self.present)
        missing = tuple(item for item in self.representative_ids if item not in self.present)
        return present, missing

    def batch_status(self, _context):
        present = tuple(item for item in self.all_ids if item in self.present)
        missing = tuple(item for item in self.all_ids if item not in self.present)
        return present, missing

    def import_master(
        self, _context, scene_id: str, source: Path, *, representative_only: bool
    ) -> None:
        assert source.is_file()
        allowed = self.representative_ids if representative_only else self.all_ids
        if scene_id not in allowed:
            raise MediaWorkflowError("scene_not_in_representatives")
        self.present.add(scene_id)

    def approve(self, _context) -> None:
        if set(self.representative_ids) != self.present:
            raise MediaWorkflowError("representatives_incomplete")
        self.approved = True

    def unlock(self, _context):
        if not self.approved:
            raise MediaWorkflowError("representatives_not_approved")
        if not self.current:
            raise MediaWorkflowError("representative_approval_stale")
        return ("S02", "S03")


class FakeCoverGateway:
    def __init__(self) -> None:
        self.imported = False

    def ready(self, _context) -> bool:
        return self.imported

    def import_cover(
        self, _context, source: Path, *, source_type: str,
        matches_product_version: bool,
    ) -> None:
        assert source.is_file()
        assert source_type == "user_provided"
        assert matches_product_version is True
        self.imported = True


class FakeSocialCoverGateway:
    def __init__(self) -> None:
        self.imported = False

    def ready(self, _context) -> bool:
        return self.imported

    def import_current_run(
        self,
        context,
        *,
        cover_brief_path: Path,
        source: Path,
        source_sha256: str,
    ) -> StageOutcome:
        assert cover_brief_path.is_file()
        assert source.is_file()
        assert len(source_sha256) == 64
        output = context.episode_root / "media" / "final" / "social_cover.png"
        manifest = output.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())
        manifest.write_text('{"kind":"social-cover"}', encoding="utf-8")
        self.imported = True
        return StageOutcome(
            outputs={"social_cover": output, "social_cover_manifest": manifest},
            inputs={"source_image_sha256": source_sha256},
        )


class FakeDeliveryGateway:
    def export_candidate(self, context) -> DeliveryBundle:
        root = context.episode_root / "deliveries" / "2026-08-23"
        root.mkdir(parents=True, exist_ok=True)
        video = root / "2026-08-23-活着-成片.mp4"
        cover = root / "2026-08-23-活着-作品封面.png"
        manifest = root / "2026-08-23-活着-交付清单.json"
        video.write_bytes(b"video")
        cover.write_bytes(b"cover")
        manifest.write_text('{"state":"awaiting_final_review"}', encoding="utf-8")
        return DeliveryBundle(
            root=root,
            version=1,
            manifest_path=manifest,
            manifest_sha256=sha256_file(manifest),
            files=(video, cover, manifest),
            video_path=video,
            cover_path=cover,
        )

    def record_final_approval(
        self,
        context,
        *,
        manifest_path: Path,
        version: int,
    ) -> ApprovalRecord:
        assert context.episode_state.status == "awaiting_final_review"
        assert version == 1
        root = manifest_path.parent
        video = root / "2026-08-23-活着-成片.mp4"
        cover = root / "2026-08-23-活着-作品封面.png"
        approval = root / "2026-08-23-活着-最终批准.json"
        approval.write_text('{"status":"final_approved"}', encoding="utf-8")
        return ApprovalRecord(
            delivery_manifest_sha256=sha256_file(manifest_path),
            video_sha256=sha256_file(video),
            cover_sha256=sha256_file(cover),
            approved_at=datetime.now(UTC),
            path=approval,
        )


def _service(tmp_path: Path):
    workspace = tmp_path / "workspace"
    root = workspace / "books" / "book-demo" / "episodes" / "E001"
    root.mkdir(parents=True)
    store = StateStore(workspace)
    store.save_episode(
        EpisodeState(
            book_id="book-demo", episode_id="E001", status="script_approved",
            script_hash="a" * 64,
        )
    )
    stages = {
        name: ArtifactStage(name)
        for name in (
            "tts", "asr", "subtitles", "select_style", "plan_illustrations",
            "prepare_representatives", "visual_render", "render", "qc",
        )
    }
    gateway = FakeIllustrationGateway()
    social_cover = FakeSocialCoverGateway()
    return MediaProductionService(
        store=store,
        stages=stages,
        images=gateway,
        cover=FakeCoverGateway(),
        social_cover=social_cover,
        delivery=FakeDeliveryGateway(),
    ), gateway, stages


def test_prepare_stops_at_three_representatives(tmp_path: Path) -> None:
    service, gateway, stages = _service(tmp_path)

    view = service.prepare("book-demo", "E001", mode="full")

    assert view.status == "representative_generation_running"
    assert view.missing_scene_ids == ("S01", "S05", "S10")
    fixture = tmp_path / "generated.png"
    fixture.write_bytes(b"fixture")
    for scene_id in view.missing_scene_ids:
        view = service.import_image("book-demo", "E001", scene_id, fixture)
    assert view.status == "awaiting_representative_review"
    assert view.present_scene_ids == ("S01", "S05", "S10")
    assert view.batch_unlocked is False
    assert all(stages[name].calls == 1 for name in (
        "tts", "asr", "subtitles", "select_style", "plan_illustrations",
        "prepare_representatives",
    ))
    assert all(
        stages[name].statuses == ["script_approved"]
        for name in HANDDRAWN_MEDIA_PREPARE_ORDER
    )


def test_doubao_full_tts_is_blocked_before_voice_approval(tmp_path: Path) -> None:
    service, _, stages = _service(tmp_path)
    root = (
        tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    )
    (root / "production_profile.json").write_text(
        ProductionProfile.living_default().model_dump_json(indent=2),
        encoding="utf-8",
    )

    with pytest.raises(MediaWorkflowError, match="voice_profile_not_approved"):
        service.prepare("book-demo", "E001", mode="full")

    assert stages["tts"].calls == 0


def test_approval_unlocks_only_current_manifest(tmp_path: Path) -> None:
    service, gateway, _ = _service(tmp_path)
    view = service.prepare("book-demo", "E001", mode="full")
    fixture = tmp_path / "generated.png"
    fixture.write_bytes(b"fixture")
    for scene_id in view.missing_scene_ids:
        service.import_image("book-demo", "E001", scene_id, fixture)

    approval = service.approve_representatives("book-demo", "E001")
    assert approval.status == "representatives_approved"
    gateway.current = False

    with pytest.raises(MediaWorkflowError, match="representative_approval_stale"):
        service.prepare_batch("book-demo", "E001")


def test_prepare_is_idempotent_for_current_stage_manifests(tmp_path: Path) -> None:
    service, _, stages = _service(tmp_path)

    service.prepare("book-demo", "E001", mode="technical_sample")
    service.prepare("book-demo", "E001", mode="technical_sample")

    assert all(stage.calls == 1 for name, stage in stages.items() if name in {
        "tts", "asr", "subtitles", "select_style", "plan_illustrations",
        "prepare_representatives",
    })


def test_runtime_builder_uses_codex_handdrawn_stages_without_grok(tmp_path: Path) -> None:
    config = AppConfig(
        workspace_dir=tmp_path / "workspace",
        voice_path=tmp_path / "voice.wav",
        subtitle_font_path=tmp_path / "font.ttf",
        handdrawn={"vendor_dir": tmp_path / "vendor"},
    )

    bindings = build_media_runtime_bindings(
        config,
        RuntimeAuthorization(allow_external=True),
        mode="technical_sample",
        sample_span=(0, 10),
    )

    assert set(bindings.stages) == {
        "tts", "asr", "subtitles", "select_style", "plan_illustrations",
        "prepare_representatives", "visual_render", "render", "qc",
    }
    assert bindings.service.stages == dict(bindings.stages)
    assert isinstance(bindings.service.social_cover, LocalSocialCoverGateway)
    assert isinstance(bindings.service.delivery, LocalDeliveryGateway)
    assert bindings.gate_approvers == {}
    assert bindings.stages["tts"].mode == "technical_sample"
    with pytest.raises(MediaWorkflowError, match="media_runtime_mode_mismatch"):
        bindings.service.prepare("book-demo", "E001", mode="full")


def test_batch_import_reaches_illustrations_ready_only_when_all_images_exist(
    tmp_path: Path,
) -> None:
    service, _, _ = _service(tmp_path)
    view = service.prepare("book-demo", "E001", mode="full")
    fixture = tmp_path / "generated.png"
    fixture.write_bytes(b"fixture")
    for scene_id in view.missing_scene_ids:
        service.import_image("book-demo", "E001", scene_id, fixture)
    service.approve_representatives("book-demo", "E001")

    batch = service.prepare_batch("book-demo", "E001")
    assert batch.status == "illustration_batch_running"
    assert batch.missing_scene_ids == ("S02", "S03")
    for scene_id in batch.missing_scene_ids:
        batch = service.import_image("book-demo", "E001", scene_id, fixture)

    assert batch.status == "illustrations_ready"
    assert batch.missing_scene_ids == ()


def test_mode_switch_archives_sample_media_before_full_rebuild(tmp_path: Path) -> None:
    service, _, stages = _service(tmp_path)
    service.prepare("book-demo", "E001", mode="technical_sample")

    view = service.prepare("book-demo", "E001", mode="full")

    assert view.status == "representative_generation_running"
    assert all(stages[name].calls == 2 for name in HANDDRAWN_MEDIA_PREPARE_ORDER)
    recovery = (
        tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
        / ".recovery" / "media-mode-switch"
    )
    assert len(tuple(recovery.glob("technical_sample-to-full-*"))) == 1


def test_final_render_allows_no_product_cover_and_stops_at_final_review(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    image = tmp_path / "generated.png"
    image.write_bytes(b"fixture")
    view = service.prepare("book-demo", "E001", mode="full")
    for scene_id in view.missing_scene_ids:
        service.import_image("book-demo", "E001", scene_id, image)
    service.approve_representatives("book-demo", "E001")
    view = service.prepare_batch("book-demo", "E001")
    for scene_id in view.missing_scene_ids:
        view = service.import_image("book-demo", "E001", scene_id, image)
    assert view.status == "illustrations_ready"

    with pytest.raises(MediaWorkflowError, match="social_cover_not_ready"):
        service.render("book-demo", "E001")

    brief = tmp_path / "cover-brief.json"
    brief.write_text("{}", encoding="utf-8")
    social_source = tmp_path / "social-cover.png"
    social_source.write_bytes(b"social cover fixture")
    imported = service.import_social_cover(
        "book-demo",
        "E001",
        cover_brief_path=brief,
        source=social_source,
        source_sha256="b" * 64,
    )
    assert imported.social_cover_ready is True
    assert "social_cover" in service.store.load_episode(
        "book-demo", "E001"
    ).stage_manifests

    rendered = service.render("book-demo", "E001")
    assert rendered.status == "awaiting_final_review"
    assert rendered.delivery_version == 1
    assert rendered.delivery_root is not None
    approved = service.approve_final("book-demo", "E001")
    assert approved.status == "final_approved"
    assert approved.approval_record_path is not None
    assert approved.approval_record_path.is_file()
