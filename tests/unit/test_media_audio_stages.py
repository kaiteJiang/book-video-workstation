from __future__ import annotations

import hashlib
import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bv.asr.volcengine import VolcCredentials
from bv.config import IndexTTS2Config
from bv.content.scripts import SemanticLock
from bv.core.hashing import sha256_file
from bv.production.profile import ProductionProfile, production_profile_sha256
from bv.state.models import EpisodeState
from bv.voice.indextts2 import SynthesizedVoice
from bv.voice.audition import (
    VoiceAuditionCandidate,
    VoiceAuditionManifest,
    VoiceProfile,
)
from bv.voice.providers import NarrationArtifact, NarrationRequest, ProviderReceipt
from bv.voice.processing import (
    FULL_EPISODE_POLICY,
    TECHNICAL_SAMPLE_POLICY,
    NarrationDuration,
    VoiceManifest,
    classify_narration_duration,
    duration_policy_from,
)
from bv.workflow.media_stages import (
    AsrStage,
    MediaStageError,
    TtsStage,
    _visual_sources,
    approved_narration_input,
)
from bv.workflow.runtime import ExternalAuthorizationError, RuntimeAuthorization
from bv.workflow.stages import StageContext


TEXT = "第一段保留原文。第二段用于技术样片。第三段仍属于完整批准文稿。"


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_wav(path: Path, seconds: float = 15.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x01\x00" * round(48_000 * seconds))


def _context(tmp_path: Path) -> StageContext:
    root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    script = root / "script" / "approved.txt"
    script.parent.mkdir(parents=True)
    script.write_text(TEXT, encoding="utf-8")
    digest = _sha_bytes(script.read_bytes())
    return StageContext(
        book_id="book-demo",
        episode_id="E001",
        episode_root=root,
        episode_state=EpisodeState(
            book_id="book-demo", episode_id="E001", status="script_approved",
            script_hash=digest,
        ),
    )


def _write_standalone_visual_sources(context: StageContext) -> SemanticLock:
    voice_root = context.episode_root / "media" / "voice"
    voice_root.mkdir(parents=True)
    script_sha256 = context.episode_state.script_hash
    assert script_sha256 is not None
    voice = VoiceManifest(
        voice_id="黑金3",
        script_sha256=script_sha256,
        reference_sha256="1" * 64,
        raw_sha256="2" * 64,
        master_sha256="3" * 64,
        processing_sha256="4" * 64,
        sample_rate=48_000,
        channels=1,
        sample_width=2,
        raw_duration_ms=50_000,
        master_duration_ms=50_000,
        qualifying_start_trim_ms=0,
        qualifying_end_trim_ms=0,
        duration=NarrationDuration(seconds=50.0, level="pass", band="ideal"),
        created_at=datetime.now(UTC),
    )
    (voice_root / "voice_master.json").write_text(
        voice.model_dump_json(), encoding="utf-8"
    )
    lock_fields = {
        "episode_id": "E001",
        "topic_identity_sha256": "5" * 64,
        "value_thesis": "继续过具体的生活",
        "target_reader": "承受生活压力的成年人",
        "reader_before": "觉得没有盼头",
        "reader_after": "重新看见眼前的人",
        "central_life_tension": "生活不给解释时怎样继续",
        "life_connection": "工作家庭和饭桌",
        "required_cluster_ids": ["cluster-a"],
        "allowed_claim_ids_by_cluster": {"cluster-a": ["claim-a"]},
        "chapter_regions_by_cluster": {"cluster-a": ["whole-book"]},
        "cluster_coverage_terms": {"cluster-a": "继续生活"},
        "practical_boundary": "不承诺解决困境",
        "reading_reason": "看见具体生活",
        "allowed_numbers": [],
        "allowed_negations": ["不"],
    }
    lock_sha256 = hashlib.sha256(
        json.dumps(
            lock_fields,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    lock = SemanticLock.model_validate({**lock_fields, "sha256": lock_sha256})
    script_root = context.episode_root / "script"
    (script_root / "semantic-lock.json").write_text(
        lock.model_dump_json(), encoding="utf-8"
    )
    (script_root / "script_manifest.json").write_text(
        json.dumps(
            {
                "script_sha256": script_sha256,
                "semantic_lock_sha256": lock_sha256,
            }
        ),
        encoding="utf-8",
    )
    return lock


def test_technical_sample_policy_accepts_fifteen_seconds_without_weakening_full_mode() -> None:
    assert classify_narration_duration(15.0, policy=TECHNICAL_SAMPLE_POLICY).level == "pass"
    assert classify_narration_duration(15.0, policy=FULL_EPISODE_POLICY).level == "fail"


@pytest.mark.parametrize(
    ("seconds", "level"),
    [
        (119.9, "fail"),
        (120.0, "pass"),
        (135.0, "pass"),
        (150.0, "pass"),
        (150.1, "fail"),
    ],
)
def test_living_duration_boundaries(seconds: float, level: str) -> None:
    policy = duration_policy_from(ProductionProfile.living_default().duration)

    assert classify_narration_duration(seconds, policy=policy).level == level


def test_story_duration_over_soft_max_is_review_warning_not_failure() -> None:
    policy = duration_policy_from(ProductionProfile.longform_story_default().duration)

    result = classify_narration_duration(720.0, policy=policy)

    assert result.level == "pass"
    assert result.band == "edge_long"
    assert result.warnings == ["duration_above_soft_max"]


def test_sample_projection_is_an_exact_contiguous_approved_span(tmp_path: Path) -> None:
    context = _context(tmp_path)
    start = TEXT.index("第二段")
    end = TEXT.index("第三段")

    projected = approved_narration_input(
        context,
        mode="technical_sample",
        sample_span=(start, end),
    )

    assert projected.text == TEXT[start:end]
    assert projected.source_span == (start, end)
    assert projected.path.read_text(encoding="utf-8") == TEXT[start:end]
    assert projected.projection_path is not None and projected.projection_path.is_file()


def test_visual_sources_accept_approved_standalone_semantic_lock(tmp_path: Path) -> None:
    context = _context(tmp_path)
    lock = _write_standalone_visual_sources(context)
    script_sha256 = context.episode_state.script_hash
    assert script_sha256 is not None

    narration, loaded_lock = _visual_sources(context)

    assert narration.sha256 == script_sha256
    assert loaded_lock == lock


def test_visual_sources_do_not_bypass_existing_invalid_script_package(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_standalone_visual_sources(context)
    (context.episode_root / "script" / "script_package.json").mkdir()

    with pytest.raises(MediaStageError, match="visual_source_invalid"):
        _visual_sources(context)


def test_tts_stage_publishes_only_verified_master_and_manifest(tmp_path: Path) -> None:
    context = _context(tmp_path)
    reference = tmp_path / "voice.wav"
    _write_wav(reference, 1.0)

    def synthesize(**kwargs: object) -> SynthesizedVoice:
        raw = Path(kwargs["raw_output_path"])
        _write_wav(raw)
        return SynthesizedVoice(raw_path=raw, raw_sha256=_sha_bytes(raw.read_bytes()))

    def master(**kwargs: object) -> VoiceManifest:
        output = Path(kwargs["master_path"])
        _write_wav(output)
        script = Path(kwargs["script_path"])
        manifest = VoiceManifest(
            voice_id="黑金3", script_sha256=_sha_bytes(script.read_bytes()),
            reference_sha256=_sha_bytes(reference.read_bytes()),
            raw_sha256="1" * 64, master_sha256=_sha_bytes(output.read_bytes()),
            processing_sha256="2" * 64, sample_rate=48_000, channels=1,
            sample_width=2, raw_duration_ms=15_000, master_duration_ms=15_000,
            qualifying_start_trim_ms=0, qualifying_end_trim_ms=0,
            duration=NarrationDuration(seconds=15.0, level="pass", band="ideal"),
            created_at=datetime.now(UTC),
        )
        output.with_suffix(".json").write_text(manifest.model_dump_json(), encoding="utf-8")
        return manifest

    stage = TtsStage(
        config=IndexTTS2Config(install_dir=tmp_path, model_dir=tmp_path),
        reference_voice_path=reference,
        ffmpeg_command="ffmpeg",
        mode="technical_sample",
        sample_span=(0, len(TEXT)),
        synthesizer=synthesize,
        master_builder=master,
    )

    outcome = stage.run(context)

    assert set(outcome.outputs) == {"voice_master", "voice_manifest"}
    assert outcome.inputs["script_sha256"] == _sha_bytes(TEXT.encode("utf-8"))
    assert all(path.is_file() for path in outcome.outputs.values())


def test_asr_stage_requires_external_authorization_before_inputs(tmp_path: Path) -> None:
    stage = AsrStage(
        credentials=VolcCredentials(api_key="placeholder"),
        endpoint="https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash",
        authorization=RuntimeAuthorization(),
    )

    with pytest.raises(ExternalAuthorizationError, match="external_not_authorized"):
        stage.run(_context(tmp_path))


def test_tts_stage_routes_doubao_from_approved_episode_profile(tmp_path: Path) -> None:
    context = _context(tmp_path)
    script = context.episode_root / "script" / "approved.txt"
    script.write_text(TEXT * 4, encoding="utf-8")
    context.episode_state.script_hash = _sha_bytes(script.read_bytes())
    base = ProductionProfile.living_default()
    approved = base.model_copy(
        update={"tts": base.tts.model_copy(update={"voice_type": "voice-a"})}
    )
    profile_path = context.episode_root / "production_profile.json"
    profile_path.write_text(approved.model_dump_json(indent=2), encoding="utf-8")

    audition_root = (
        context.episode_root / ".private" / "candidates" / "tts-audition-20260823"
    )
    audio = audition_root / "candidate-01" / "voice_raw.wav"
    receipt = audition_root / "candidate-01" / "doubao-audition-receipt.json"
    _write_wav(audio, 20.0)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text("{}", encoding="utf-8")
    excerpt = audition_root / "excerpt.txt"
    excerpt.write_text(TEXT, encoding="utf-8")
    excerpt_sha = sha256_file(excerpt)
    manifest_path = audition_root / "manifest.json"
    manifest = VoiceAuditionManifest(
        book_id=context.book_id,
        episode_id=context.episode_id,
        provider="doubao",
        script_sha256=context.episode_state.script_hash,
        profile_sha256_before="a" * 64,
        excerpt_path=excerpt.relative_to(context.episode_root).as_posix(),
        excerpt_sha256=excerpt_sha,
        excerpt_span=(0, len(TEXT)),
        speed=1.0,
        candidates=(
            VoiceAuditionCandidate(
                candidate_id="candidate-01",
                provider_voice_id="voice-a",
                excerpt_sha256=excerpt_sha,
                speed=1.0,
                audio_path=audio.relative_to(context.episode_root).as_posix(),
                audio_sha256=sha256_file(audio),
                receipt_path=receipt.relative_to(context.episode_root).as_posix(),
                receipt_sha256=sha256_file(receipt),
                duration_seconds=20.0,
            ),
        ),
        created_at=datetime.now(UTC),
        manifest_path=manifest_path,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    voice_profile = VoiceProfile(
        provider="doubao",
        provider_voice_id="voice-a",
        speed=1.0,
        script_sha256=context.episode_state.script_hash,
        excerpt_sha256=excerpt_sha,
        approved_audio_sha256=sha256_file(audio),
        audition_manifest_sha256=sha256_file(manifest_path),
        production_profile_sha256=production_profile_sha256(context.episode_root),
        candidate_id="candidate-01",
        audition_manifest_path=manifest_path.relative_to(context.episode_root).as_posix(),
        approved_audio_path=audio.relative_to(context.episode_root).as_posix(),
        approved_at=datetime.now(UTC),
    )
    (context.episode_root / "voice_profile.json").write_text(
        voice_profile.model_dump_json(indent=2), encoding="utf-8"
    )
    requests: list[NarrationRequest] = []

    class FormalDoubao:
        provider = "doubao"

        def synthesize(self, request, authorization):
            authorization.assert_tts_submission(
                book_id=request.book_id,
                episode_id=request.episode_id,
                script_sha256=request.approved_script_sha256,
                provider_voice_id=request.provider_voice_id,
                kind="formal",
            )
            requests.append(request)
            raw = request.output_dir / "voice_raw.wav"
            provider_receipt = request.output_dir / "doubao-formal-receipt.json"
            _write_wav(raw, 50.0)
            provider_receipt.write_text("{}", encoding="utf-8")
            return NarrationArtifact(
                raw_audio_path=raw,
                raw_audio_sha256=sha256_file(raw),
                receipt_path=provider_receipt,
                receipt_sha256=sha256_file(provider_receipt),
            )

    def master(**kwargs: object) -> VoiceManifest:
        output = Path(kwargs["master_path"])
        _write_wav(output, 50.0)
        result = VoiceManifest(
            provider="doubao",
            provider_voice_id="voice-a",
            script_sha256=context.episode_state.script_hash,
            reference_sha256=None,
            provider_receipt_sha256=sha256_file(Path(kwargs["provider_receipt_path"])),
            raw_sha256="1" * 64,
            master_sha256=sha256_file(output),
            processing_sha256="2" * 64,
            sample_rate=48_000,
            channels=1,
            sample_width=2,
            raw_duration_ms=50_000,
            master_duration_ms=50_000,
            qualifying_start_trim_ms=0,
            qualifying_end_trim_ms=0,
            duration=NarrationDuration(seconds=50.0, level="pass", band="ideal"),
            created_at=datetime.now(UTC),
        )
        output.with_suffix(".json").write_text(
            result.model_dump_json(), encoding="utf-8"
        )
        return result

    stage = TtsStage(
        config=IndexTTS2Config(install_dir=tmp_path, model_dir=tmp_path),
        reference_voice_path=tmp_path / "unused.wav",
        ffmpeg_command="ffmpeg",
        mode="full",
        synthesizers={"doubao": FormalDoubao()},
        authorization=RuntimeAuthorization(
            allow_external=True,
            book_id=context.book_id,
            episode_id=context.episode_id,
            script_sha256=context.episode_state.script_hash,
            provider_voice_id="voice-a",
            max_formal_submissions=1,
        ),
        master_builder=master,
    )

    outcome = stage.run(context)

    assert len(requests) == 1
    assert requests[0].provider_voice_id == "voice-a"
    assert outcome.inputs["provider"] == "doubao"
