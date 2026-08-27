from __future__ import annotations

import hashlib
import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bv.production.profile import ProductionProfile, load_production_profile
from bv.state.models import EpisodeState
from bv.voice.audition import (
    VoiceAuditionError,
    VoiceAuditionService,
    VoiceProfile,
    require_current_voice_profile,
)
from bv.voice.providers import NarrationArtifact, NarrationRequest, ProviderReceipt
from bv.workflow.runtime import RuntimeAuthorization
from bv.workflow.stages import StageContext


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, *, seconds: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x01\x00" * 48_000 * seconds)


class FakeDoubaoSynthesizer:
    provider = "doubao"

    def __init__(self) -> None:
        self.requests: list[NarrationRequest] = []

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
        self.requests.append(request)
        raw = request.output_dir / "voice_raw.wav"
        receipt_path = request.output_dir / "doubao-audition-receipt.json"
        _write_wav(raw)
        receipt = ProviderReceipt(
            provider="doubao",
            request_sha256="a" * 64,
            task_id=f"task-{request.provider_voice_id}",
            task_id_sha256=hashlib.sha256(
                f"task-{request.provider_voice_id}".encode("utf-8")
            ).hexdigest(),
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


def _context(tmp_path: Path) -> StageContext:
    root = tmp_path / "books" / "book-demo" / "episodes" / "E001"
    script = root / "script" / "approved.txt"
    script.parent.mkdir(parents=True)
    script.write_text(
        "有些人读活着，是因为生活正压得人喘不过气。"
        "它没有许诺苦难会自动变成礼物，只让人看见一个普通人怎样把一天交给下一天。"
        "当命运不断拿走熟悉的人和事，活着本身仍然保留着尊严。"
        "这不是歌颂受苦，而是提醒我们珍惜眼前还能握住的关系和日常。",
        encoding="utf-8",
    )
    profile = ProductionProfile.living_default()
    (root / "production_profile.json").write_text(
        profile.model_dump_json(indent=2), encoding="utf-8"
    )
    state = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        status="script_approved",
        script_hash=_sha256(script),
    )
    return StageContext(
        book_id="book-demo",
        episode_id="E001",
        episode_root=root,
        episode_state=state,
    )


def _authorization(context: StageContext, voice_ids: tuple[str, ...]) -> RuntimeAuthorization:
    return RuntimeAuthorization(
        allow_external=True,
        book_id=context.book_id,
        episode_id=context.episode_id,
        script_sha256=context.episode_state.script_hash,
        audition_voice_ids=voice_ids,
        max_audition_submissions=len(voice_ids),
        allow_fallback=False,
    )


def test_audition_uses_same_excerpt_and_at_most_three_candidates(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    synth = FakeDoubaoSynthesizer()
    voice_ids = ("voice-a", "voice-b", "voice-c")
    service = VoiceAuditionService(
        synthesizers={"doubao": synth},
        now=lambda: datetime(2026, 8, 23, tzinfo=UTC),
    )

    manifest = service.prepare_candidates(
        context,
        voice_ids=voice_ids,
        supported_voice_ids=voice_ids,
        authorization=_authorization(context, voice_ids),
    )

    assert len(manifest.candidates) == 3
    assert len({item.excerpt_sha256 for item in manifest.candidates}) == 1
    assert len({item.speed for item in manifest.candidates}) == 1
    assert {item.provider_voice_id for item in manifest.candidates} == set(voice_ids)
    assert len({request.approved_script_sha256 for request in synth.requests}) == 1
    assert manifest.manifest_path.name == "manifest.json"
    assert "tts-audition-20260823" in manifest.manifest_path.as_posix()

    with pytest.raises(VoiceAuditionError, match="too_many_voice_candidates"):
        service.prepare_candidates(
            context,
            voice_ids=("a", "b", "c", "d"),
            supported_voice_ids=("a", "b", "c", "d"),
            authorization=_authorization(context, ("a", "b", "c", "d")),
        )


def test_approval_updates_profile_and_creates_current_voice_evidence(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    synth = FakeDoubaoSynthesizer()
    voice_ids = ("voice-a", "voice-b")
    service = VoiceAuditionService(
        synthesizers={"doubao": synth},
        now=lambda: datetime(2026, 8, 23, tzinfo=UTC),
    )
    manifest = service.prepare_candidates(
        context,
        voice_ids=voice_ids,
        supported_voice_ids=voice_ids,
        authorization=_authorization(context, voice_ids),
    )

    profile = service.approve_candidate(context, "candidate-02")

    assert profile.provider == "doubao"
    assert profile.provider_voice_id == "voice-b"
    assert profile.approved_audio_sha256 == manifest.candidates[1].audio_sha256
    assert load_production_profile(context.episode_root).tts.voice_type == "voice-b"
    persisted = VoiceProfile.model_validate_json(
        (context.episode_root / "voice_profile.json").read_text(encoding="utf-8")
    )
    assert persisted == profile
    assert require_current_voice_profile(
        context.episode_root,
        load_production_profile(context.episode_root),
        script_sha256=context.episode_state.script_hash,
    ) == profile


def test_tampered_approved_candidate_blocks_formal_profile(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    synth = FakeDoubaoSynthesizer()
    voice_ids = ("voice-a",)
    service = VoiceAuditionService(
        synthesizers={"doubao": synth},
        now=lambda: datetime(2026, 8, 23, tzinfo=UTC),
    )
    service.prepare_candidates(
        context,
        voice_ids=voice_ids,
        supported_voice_ids=voice_ids,
        authorization=_authorization(context, voice_ids),
    )
    profile = service.approve_candidate(context, "candidate-01")
    approved_audio = context.episode_root / profile.approved_audio_path
    approved_audio.write_bytes(b"tampered")

    with pytest.raises(VoiceAuditionError, match="voice_profile_stale"):
        require_current_voice_profile(
            context.episode_root,
            load_production_profile(context.episode_root),
            script_sha256=context.episode_state.script_hash,
        )


def test_manifest_contains_no_source_script_body(tmp_path: Path) -> None:
    context = _context(tmp_path)
    synth = FakeDoubaoSynthesizer()
    voice_ids = ("voice-a",)
    service = VoiceAuditionService(
        synthesizers={"doubao": synth},
        now=lambda: datetime(2026, 8, 23, tzinfo=UTC),
    )
    manifest = service.prepare_candidates(
        context,
        voice_ids=voice_ids,
        supported_voice_ids=voice_ids,
        authorization=_authorization(context, voice_ids),
    )

    payload = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    assert "有些人读活着" not in json.dumps(payload, ensure_ascii=False)


def test_completed_audition_is_idempotent_without_new_submissions(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    synth = FakeDoubaoSynthesizer()
    voice_ids = ("voice-a",)
    service = VoiceAuditionService(
        synthesizers={"doubao": synth},
        now=lambda: datetime(2026, 8, 23, tzinfo=UTC),
    )
    first = service.prepare_candidates(
        context,
        voice_ids=voice_ids,
        supported_voice_ids=voice_ids,
        authorization=_authorization(context, voice_ids),
    )

    second = service.prepare_candidates(
        context,
        voice_ids=voice_ids,
        supported_voice_ids=voice_ids,
        authorization=RuntimeAuthorization(),
    )

    assert second == first
    assert len(synth.requests) == 1
