from __future__ import annotations

import hashlib
import json
import shutil
import struct
import wave
import zlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bv.delivery.exporter import DeliveryExporter
from bv.production.profile import ProductionProfile, production_profile_sha256
from bv.state.models import BookState, EpisodeState
from bv.state.store import StateStore
from bv.video.cover_brief import CoverBriefBuilder, CoverDerivedFields
from bv.voice.audition import VoiceAuditionService
from bv.voice.providers import NarrationArtifact, NarrationRequest, ProviderReceipt
from bv.workflow.media_runtime import (
    LocalDeliveryGateway,
    LocalSocialCoverGateway,
    MediaProductionService,
    MediaWorkflowError,
)
from bv.workflow.runtime import RuntimeAuthorization
from bv.workflow.stages import HANDDRAWN_MEDIA_PREPARE_ORDER, StageOutcome


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, *, seconds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x01\x00" * 48_000 * seconds)


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


def _write_png(path: Path, *, width: int = 540, height: int = 720) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = b"\x00" + (b"\xdc\xcf\xb5" * width)
    raw = row * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=1))
        + chunk(b"IEND", b"")
    )


class _FakeDoubaoAuditionSynthesizer:
    provider = "doubao"

    def synthesize(
        self,
        request: NarrationRequest,
        authorization: RuntimeAuthorization,
    ) -> NarrationArtifact:
        authorization.assert_tts_submission(
            book_id=request.book_id,
            episode_id=request.episode_id,
            script_sha256=request.approved_script_sha256,
            provider_voice_id=request.provider_voice_id,
            kind="audition",
        )
        raw = request.output_dir / "voice_raw.wav"
        receipt_path = request.output_dir / "doubao-audition-receipt.json"
        _write_wav(raw, seconds=20)
        task_id = f"task-{request.provider_voice_id}"
        receipt = ProviderReceipt(
            provider="doubao",
            request_sha256="a" * 64,
            task_id=task_id,
            task_id_sha256=hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
            response_sha256="b" * 64,
            raw_audio_sha256=_sha256(raw),
            status="success",
            error_code=None,
        )
        receipt_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        return NarrationArtifact(
            raw_audio_path=raw,
            raw_audio_sha256=_sha256(raw),
            receipt_path=receipt_path,
            receipt_sha256=_sha256(receipt_path),
        )


class _LocalFixtureStage:
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, context) -> StageOutcome:
        episode = context.episode_root
        if self.name == "tts":
            wav = episode / "media" / "voice" / "voice_master.wav"
            manifest = wav.with_suffix(".json")
            _write_wav(wav, seconds=135)
            manifest.write_text(
                json.dumps(
                    {
                        "provider": "doubao",
                        "provider_voice_id": "reader-calm",
                        "duration_seconds": 135.0,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return StageOutcome(
                outputs={"voice_master": wav, "voice_manifest": manifest}
            )
        if self.name == "subtitles":
            ass = episode / "media" / "subtitles" / "subtitles.ass"
            ass.parent.mkdir(parents=True, exist_ok=True)
            ass.write_text(
                "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n"
                "[Events]\nDialogue: 0,0:00:00.00,0:02:15.00,Default,,0,0,0,,活着不是向苦难低头\n",
                encoding="utf-8",
            )
            return StageOutcome(outputs={"subtitles_ass": ass})
        if self.name == "render":
            video = episode / "media" / "final" / "final.mp4"
            manifest = video.with_suffix(".render.json")
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"deterministic-local-final-video")
            manifest.write_text('{"duration_seconds":135.0}', encoding="utf-8")
            return StageOutcome(outputs={"final_video": video, "render_manifest": manifest})
        if self.name == "qc":
            manifest = episode / "media" / "final" / "final.qc.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text('{"technical_status":"passed"}', encoding="utf-8")
            return StageOutcome(outputs={"qc_manifest": manifest})

        fixture = episode / "media" / "fixtures" / f"{self.name}.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps({"stage": self.name}), encoding="utf-8")
        return StageOutcome(outputs={self.name: fixture})


class _ConversationIllustrationGateway:
    scene_ids = tuple(f"S{index:02d}" for index in range(1, 16))
    representative_ids = ("S01", "S08", "S15")

    def __init__(self) -> None:
        self.present: set[str] = set()
        self.batch_unlocked = False

    def status(self, context) -> tuple[tuple[str, ...], tuple[str, ...]]:
        present = tuple(item for item in self.representative_ids if item in self.present)
        missing = tuple(item for item in self.representative_ids if item not in self.present)
        return present, missing

    def batch_status(self, context) -> tuple[tuple[str, ...], tuple[str, ...]]:
        present = tuple(item for item in self.scene_ids if item in self.present)
        missing = tuple(item for item in self.scene_ids if item not in self.present)
        return present, missing

    def import_master(
        self,
        context,
        scene_id: str,
        source: Path,
        *,
        representative_only: bool,
    ) -> None:
        allowed = self.representative_ids if representative_only else self.scene_ids
        if scene_id not in allowed or not source.is_file():
            raise MediaWorkflowError("illustration_import_failed")
        self.present.add(scene_id)

    def approve(self, context) -> None:
        if any(item not in self.present for item in self.representative_ids):
            raise MediaWorkflowError("representatives_incomplete")

    def unlock(self, context) -> None:
        self.batch_unlocked = True


def _build_cover_brief(episode_root: Path) -> Path:
    script = episode_root / "script" / "approved.txt"
    semantic_lock = episode_root / "script" / "semantic-lock.json"
    semantic_lock.write_text(
        '{"value_thesis":"看清生活以后仍然珍惜具体的人和日常"}',
        encoding="utf-8",
    )
    style_root = episode_root / ".private" / "cover-style"
    style_root.mkdir(parents=True)
    meta = style_root / "META.md"
    atom = style_root / "STYLE.md"
    blueprint = style_root / "cover-prompt-blueprint.md"
    meta.write_text(
        "```yaml\nid: french-minimal-ink-poster\noutputs: [cover, poster]\n```",
        encoding="utf-8",
    )
    atom.write_text("warm ivory paper and restrained black ink", encoding="utf-8")
    blueprint.write_text("single integrated vertical book cover", encoding="utf-8")
    brief = CoverBriefBuilder(blueprint_path=blueprint).build(
        episode_root=episode_root,
        slug="huo-zhe-e001",
        fields=CoverDerivedFields(
            title="活着",
            summary="看清失去以后仍然珍惜具体生活",
            visual_subject="老人和老牛",
            audience="正在承受现实压力的成年人",
            mood="克制沉静",
            visual_metaphor="老人和老牛在黄昏田埂并肩前行",
        ),
        compiled_prompt=(
            "Create one 3:4 cover in french-minimal-ink-poster style. "
            "Place the exact title 活着 large and centered. "
            "Show an old man and an old ox walking together at dusk."
        ),
        approved_script_path=script,
        approved_script_sha256=_sha256(script),
        semantic_lock_path=semantic_lock,
        semantic_lock_sha256=_sha256(semantic_lock),
        production_profile_sha256=production_profile_sha256(episode_root),
        style_id="french-minimal-ink-poster",
        style_meta_path=meta,
        style_atom_path=atom,
    )
    return brief.manifest_path


def test_approved_longform_episode_reaches_immutable_dated_delivery(
    tmp_path: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("local FFmpeg is required")

    workspace = tmp_path / "workspace"
    episode_root = workspace / "books" / "book-living" / "episodes" / "E001"
    script = episode_root / "script" / "approved.txt"
    script.parent.mkdir(parents=True)
    script.write_text(
        "有些人读活着 是因为现实已经压得人喘不过气 "
        "余华没有把苦难写成奖章 也没有保证熬过去就会得到补偿 "
        "他只是让我们看见 当命运一次次拿走熟悉的人和事 "
        "一个普通人仍然可以把今天交给明天 "
        "读完以后真正留下来的 不是应该忍受一切 "
        "而是别等失去以后 才看见眼前的人 一顿饭和一个平常晚上有多珍贵",
        encoding="utf-8",
    )
    episode_root.joinpath("production_profile.json").write_text(
        ProductionProfile.living_default().model_dump_json(indent=2),
        encoding="utf-8",
    )

    store = StateStore(workspace)
    store.save_book(
        BookState(
            book_id="book-living",
            title="活着",
            authors=["余华"],
            source_mode="title_author",
        )
    )
    store.save_episode(
        EpisodeState(
            book_id="book-living",
            episode_id="E001",
            status="script_approved",
            script_hash=_sha256(script),
        )
    )

    illustrations = _ConversationIllustrationGateway()
    audition = VoiceAuditionService(
        synthesizers={"doubao": _FakeDoubaoAuditionSynthesizer()},
        now=lambda: datetime(2026, 8, 24, tzinfo=UTC),
    )
    stages = {
        name: _LocalFixtureStage(name)
        for name in (*HANDDRAWN_MEDIA_PREPARE_ORDER, "visual_render", "render", "qc")
    }
    service = MediaProductionService(
        store=store,
        stages=stages,
        images=illustrations,
        social_cover=LocalSocialCoverGateway(ffmpeg_command=ffmpeg),
        delivery=LocalDeliveryGateway(
            store=store,
            exporter=DeliveryExporter(
                clock=lambda: datetime(2026, 8, 24, 1, tzinfo=UTC)
            ),
        ),
        voice_auditions=audition,
    )

    with pytest.raises(MediaWorkflowError, match="voice_profile_not_approved"):
        service.prepare("book-living", "E001", mode="full")

    voices = ("reader-warm", "reader-calm")
    authorization = RuntimeAuthorization(
        allow_external=True,
        book_id="book-living",
        episode_id="E001",
        script_sha256=_sha256(script),
        audition_voice_ids=voices,
        max_audition_submissions=2,
        allow_fallback=False,
    )
    view = service.prepare_voice_audition(
        "book-living",
        "E001",
        voice_ids=voices,
        supported_voice_ids=voices,
        authorization=authorization,
    )
    assert view.status == "awaiting_voice_audition_review"
    view = service.approve_voice_audition("book-living", "E001", "candidate-02")
    assert view.status == "voice_profile_approved"

    view = service.prepare("book-living", "E001", mode="full")
    assert view.status == "representative_generation_running"
    assert view.missing_scene_ids == ("S01", "S08", "S15")
    assert _wav_seconds(episode_root / "media" / "voice" / "voice_master.wav") == 135.0

    image = tmp_path / "generated-illustration.png"
    _write_png(image)
    for scene_id in view.missing_scene_ids:
        view = service.import_image("book-living", "E001", scene_id, image)
    assert view.status == "awaiting_representative_review"
    service.approve_representatives("book-living", "E001")

    view = service.prepare_batch("book-living", "E001")
    assert len(view.missing_scene_ids) == 12
    for scene_id in view.missing_scene_ids:
        view = service.import_image("book-living", "E001", scene_id, image)
    assert view.status == "illustrations_ready"

    cover_brief = _build_cover_brief(episode_root)
    cover_source = tmp_path / "current-run" / "generated-cover.png"
    _write_png(cover_source)
    view = service.import_social_cover(
        "book-living",
        "E001",
        cover_brief_path=cover_brief,
        source=cover_source,
        source_sha256=_sha256(cover_source),
    )
    assert view.social_cover_ready is True

    view = service.render("book-living", "E001")
    assert view.status == "awaiting_final_review"
    assert view.delivery_version == 1
    assert view.delivery_root is not None
    assert view.delivery_root.name == "2026-08-24"
    assert {path.suffix for path in view.delivery_root.iterdir()} == {
        ".mp4",
        ".png",
        ".txt",
        ".wav",
        ".ass",
        ".json",
    }
    assert len(tuple(view.delivery_root.iterdir())) == 6

    final = service.approve_final("book-living", "E001")
    assert final.status == "final_approved"
    assert final.approval_record_path is not None
    assert final.approval_record_path.is_file()
    assert len(tuple(view.delivery_root.iterdir())) == 7
