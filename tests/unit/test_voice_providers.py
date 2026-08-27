from __future__ import annotations

import hashlib
import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from bv.config import IndexTTS2Config
from bv.voice.indextts2 import SynthesizedVoice
from bv.voice.processing import NarrationDuration, VoiceManifest
from bv.voice.providers import (
    IndexTTS2NarrationSynthesizer,
    NarrationRequest,
    ProviderReceipt,
)
from bv.workflow.runtime import ExternalAuthorizationError, RuntimeAuthorization


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x01\x00" * 48_000)


def _legacy_manifest_payload() -> dict[str, object]:
    return {
        "voice_id": "黑金3",
        "script_sha256": "a" * 64,
        "reference_sha256": "b" * 64,
        "raw_sha256": "c" * 64,
        "master_sha256": "d" * 64,
        "processing_sha256": "e" * 64,
        "sample_rate": 48_000,
        "channels": 1,
        "sample_width": 2,
        "raw_duration_ms": 15_000,
        "master_duration_ms": 15_000,
        "qualifying_start_trim_ms": 0,
        "qualifying_end_trim_ms": 0,
        "duration": {
            "seconds": 15.0,
            "level": "pass",
            "band": "ideal",
            "warnings": [],
        },
        "created_at": datetime.now(UTC).isoformat(),
    }


def test_legacy_voice_manifest_migrates_to_indextts2() -> None:
    manifest = VoiceManifest.model_validate(_legacy_manifest_payload())

    assert manifest.provider == "indextts2"
    assert manifest.provider_voice_id == "黑金3"
    assert manifest.voice_id == "黑金3"
    assert manifest.reference_sha256 == "b" * 64
    assert manifest.provider_receipt_sha256 is None
    dumped = manifest.model_dump(mode="json")
    assert dumped["provider_voice_id"] == "黑金3"
    assert "voice_id" not in dumped


@pytest.mark.parametrize(
    ("provider", "reference_sha256", "receipt_sha256", "error"),
    [
        ("indextts2", None, None, "indextts2_reference_required"),
        ("doubao", "b" * 64, "f" * 64, "doubao_reference_forbidden"),
        ("doubao", None, None, "doubao_receipt_required"),
    ],
)
def test_manifest_enforces_provider_specific_fields(
    provider: str,
    reference_sha256: str | None,
    receipt_sha256: str | None,
    error: str,
) -> None:
    payload = _legacy_manifest_payload()
    payload.pop("voice_id")
    payload.update(
        {
            "provider": provider,
            "provider_voice_id": "黑金3" if provider == "indextts2" else "reader-a",
            "reference_sha256": reference_sha256,
            "provider_receipt_sha256": receipt_sha256,
        }
    )

    with pytest.raises(ValidationError, match=error):
        VoiceManifest.model_validate(payload)


def test_runtime_authorization_matches_exact_formal_scope() -> None:
    authorization = RuntimeAuthorization(
        allow_external=True,
        book_id="book-demo",
        episode_id="E001",
        script_sha256="a" * 64,
        provider_voice_id="reader-a",
        max_formal_submissions=1,
        allow_fallback=False,
    )

    authorization.assert_tts_submission(
        book_id="book-demo",
        episode_id="E001",
        script_sha256="a" * 64,
        provider_voice_id="reader-a",
        kind="formal",
    )

    with pytest.raises(ExternalAuthorizationError, match="external_scope_mismatch"):
        authorization.assert_tts_submission(
            book_id="book-demo",
            episode_id="E001",
            script_sha256="a" * 64,
            provider_voice_id="reader-b",
            kind="formal",
        )


def test_provider_receipt_binds_task_hash_and_success_audio_hash() -> None:
    with pytest.raises(ValidationError, match="provider_task_hash_mismatch"):
        ProviderReceipt(
            provider="doubao",
            request_sha256="a" * 64,
            task_id="task-a",
            task_id_sha256="b" * 64,
            response_sha256="c" * 64,
            status="working",
            error_code=None,
        )

    with pytest.raises(ValidationError, match="provider_error_code_invalid"):
        ProviderReceipt(
            provider="doubao",
            request_sha256="a" * 64,
            task_id="task-a",
            task_id_sha256=hashlib.sha256(b"task-a").hexdigest(),
            response_sha256="c" * 64,
            status="failed",
            error_code="PRIVATE_PROVIDER_BODY token=secret",
        )

    with pytest.raises(ValidationError, match="provider_success_audio_required"):
        ProviderReceipt(
            provider="doubao",
            request_sha256="a" * 64,
            task_id="task-a",
            task_id_sha256=hashlib.sha256(b"task-a").hexdigest(),
            response_sha256="c" * 64,
            status="success",
            error_code=None,
        )


def test_local_index_adapter_writes_minimal_receipt_and_preserves_raw(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    script = episode / "script" / "approved.txt"
    reference = tmp_path / "reference.wav"
    script.parent.mkdir(parents=True)
    script.write_text("批准文稿", encoding="utf-8")
    _write_wav(reference)
    calls: list[dict[str, object]] = []

    def fake_synthesize(**kwargs: object) -> SynthesizedVoice:
        calls.append(kwargs)
        raw = Path(kwargs["raw_output_path"])
        _write_wav(raw)
        return SynthesizedVoice(raw_path=raw, raw_sha256=_sha256(raw))

    request = NarrationRequest(
        book_id="book-demo",
        episode_id="E001",
        approved_script_path=script,
        approved_script_sha256=_sha256(script),
        provider_resource_id="local.indextts2",
        provider_voice_id="黑金3",
        speed=1.0,
        output_dir=episode / ".private" / "voice",
        kind="formal",
    )
    adapter = IndexTTS2NarrationSynthesizer(
        config=IndexTTS2Config(install_dir=tmp_path, model_dir=tmp_path),
        reference_voice_path=reference,
        synthesize=fake_synthesize,
    )

    artifact = adapter.synthesize(request, RuntimeAuthorization())

    assert len(calls) == 1
    assert artifact.raw_audio_path.is_file()
    assert artifact.raw_audio_sha256 == _sha256(artifact.raw_audio_path)
    receipt = ProviderReceipt.model_validate_json(
        artifact.receipt_path.read_text(encoding="utf-8")
    )
    assert receipt.provider == "indextts2"
    assert receipt.status == "local_complete"
    assert receipt.task_id_sha256 is None
    assert receipt.response_sha256 is None
    serialized = json.loads(artifact.receipt_path.read_text(encoding="utf-8"))
    assert set(serialized) == {
        "provider",
        "request_sha256",
        "task_id",
        "task_id_sha256",
        "response_sha256",
        "raw_audio_sha256",
        "status",
        "error_code",
    }
