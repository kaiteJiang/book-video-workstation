from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from bv.config import IndexTTS2Config
from bv.voice.indextts2 import SynthesizedVoice, synthesize_voice

if TYPE_CHECKING:
    from bv.workflow.runtime import RuntimeAuthorization


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class NarrationProviderError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _ProviderModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderReceipt(_ProviderModel):
    provider: Literal["indextts2", "doubao"]
    request_sha256: str
    task_id: str | None = None
    task_id_sha256: str | None
    response_sha256: str | None
    raw_audio_sha256: str | None = None
    status: Literal[
        "local_complete",
        "submitted",
        "working",
        "success",
        "failed",
    ]
    error_code: str | None = None

    @field_validator(
        "request_sha256",
        "task_id_sha256",
        "response_sha256",
        "raw_audio_sha256",
    )
    @classmethod
    def _validate_hash(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid_provider_receipt_hash")
        return value

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or len(value) > 256):
            raise ValueError("invalid_provider_task_id")
        return value

    @field_validator("error_code")
    @classmethod
    def _validate_error_code(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) is None:
            raise ValueError("provider_error_code_invalid")
        return value

    @model_validator(mode="after")
    def _validate_provider_evidence(self) -> ProviderReceipt:
        if self.provider == "indextts2":
            if (
                self.status != "local_complete"
                or self.task_id is not None
                or self.task_id_sha256 is not None
                or self.response_sha256 is not None
                or self.raw_audio_sha256 is None
            ):
                raise ValueError("indextts2_receipt_invalid")
            return self
        if self.task_id is None or self.task_id_sha256 is None:
            raise ValueError("provider_task_id_required")
        if _sha256_text(self.task_id) != self.task_id_sha256:
            raise ValueError("provider_task_hash_mismatch")
        if self.response_sha256 is None:
            raise ValueError("provider_response_hash_required")
        if self.status == "success" and self.raw_audio_sha256 is None:
            raise ValueError("provider_success_audio_required")
        if self.status != "success" and self.raw_audio_sha256 is not None:
            raise ValueError("provider_non_success_audio_forbidden")
        return self


class NarrationRequest(_ProviderModel):
    book_id: str
    episode_id: str
    approved_script_path: Path
    approved_script_sha256: str
    provider_resource_id: str
    provider_voice_id: str
    speed: float
    output_dir: Path
    kind: Literal["audition", "formal"]

    @field_validator("book_id", "episode_id")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        if _IDENTIFIER_PATTERN.fullmatch(value) is None:
            raise ValueError("unsafe_narration_identifier")
        return value

    @field_validator("approved_script_sha256")
    @classmethod
    def _validate_script_hash(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid_approved_script_sha256")
        return value

    @field_validator("provider_voice_id")
    @classmethod
    def _validate_voice_id(cls, value: str) -> str:
        if not value.strip() or len(value) > 256:
            raise ValueError("invalid_provider_voice_id")
        return value

    @field_validator("provider_resource_id")
    @classmethod
    def _validate_resource_id(cls, value: str) -> str:
        if not value.strip() or len(value) > 256:
            raise ValueError("invalid_provider_resource_id")
        return value

    @field_validator("speed")
    @classmethod
    def _validate_speed(cls, value: float) -> float:
        if not 0.5 < value <= 2.0:
            raise ValueError("invalid_narration_speed")
        return value


class NarrationArtifact(_ProviderModel):
    raw_audio_path: Path
    raw_audio_sha256: str
    receipt_path: Path
    receipt_sha256: str

    @field_validator("raw_audio_sha256", "receipt_sha256")
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid_narration_artifact_hash")
        return value


class NarrationSynthesizer(Protocol):
    provider: Literal["indextts2", "doubao"]

    def synthesize(
        self,
        request: NarrationRequest,
        authorization: RuntimeAuthorization,
    ) -> NarrationArtifact:
        raise NotImplementedError


class IndexTTS2NarrationSynthesizer:
    provider: Literal["indextts2"] = "indextts2"

    def __init__(
        self,
        *,
        config: IndexTTS2Config,
        reference_voice_path: Path,
        synthesize: object = synthesize_voice,
    ) -> None:
        if not callable(synthesize):
            raise TypeError("synthesize must be callable")
        self.config = config
        self.reference_voice_path = Path(reference_voice_path)
        self._synthesize = synthesize

    def synthesize(
        self,
        request: NarrationRequest,
        authorization: RuntimeAuthorization,
    ) -> NarrationArtifact:
        del authorization
        if request.provider_voice_id != self.config.voice_id:
            raise NarrationProviderError("voice_profile_mismatch")
        if request.speed != 1.0:
            raise NarrationProviderError("indextts2_speed_unsupported")
        _require_episode_output(request)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = request.output_dir / "voice_raw.wav"
        result = self._synthesize(
            episode_root=_episode_root_from_request(request),
            install_dir=self.config.install_dir,
            model_dir=self.config.model_dir,
            script_path=request.approved_script_path,
            approved_script_sha256=request.approved_script_sha256,
            reference_voice_path=self.reference_voice_path,
            raw_output_path=raw_path,
            uv_command=self.config.uv_command,
            device=self.config.device,
        )
        if not isinstance(result, SynthesizedVoice):
            raise NarrationProviderError("indextts2_artifact_invalid")
        if not result.raw_path.is_file() or _sha256_file(result.raw_path) != result.raw_sha256:
            raise NarrationProviderError("indextts2_artifact_invalid")
        receipt = ProviderReceipt(
            provider="indextts2",
            request_sha256=_request_sha256(request),
            task_id=None,
            task_id_sha256=None,
            response_sha256=None,
            raw_audio_sha256=result.raw_sha256,
            status="local_complete",
            error_code=None,
        )
        receipt_path = request.output_dir / f"indextts2-{request.kind}-receipt.json"
        _atomic_write_receipt(receipt_path, receipt)
        return NarrationArtifact(
            raw_audio_path=result.raw_path,
            raw_audio_sha256=result.raw_sha256,
            receipt_path=receipt_path,
            receipt_sha256=_sha256_file(receipt_path),
        )


def _request_sha256(request: NarrationRequest) -> str:
    payload = {
        "book_id": request.book_id,
        "episode_id": request.episode_id,
        "approved_script_sha256": request.approved_script_sha256,
        "provider_resource_id": request.provider_resource_id,
        "provider_voice_id": request.provider_voice_id,
        "speed": request.speed,
        "kind": request.kind,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def narration_request_sha256(request: NarrationRequest) -> str:
    return _request_sha256(request)


def _episode_root_from_request(request: NarrationRequest) -> Path:
    try:
        return request.approved_script_path.parents[1]
    except IndexError:
        raise NarrationProviderError("unsafe_narration_path") from None


def _require_episode_output(request: NarrationRequest) -> None:
    episode_root = _episode_root_from_request(request).absolute()
    script = request.approved_script_path.absolute()
    output = request.output_dir.absolute()
    if not script.is_relative_to(episode_root) or not output.is_relative_to(episode_root):
        raise NarrationProviderError("unsafe_narration_path")
    if request.output_dir.exists() and request.output_dir.is_symlink():
        raise NarrationProviderError("unsafe_narration_path")


def _atomic_write_receipt(path: Path, receipt: ProviderReceipt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(receipt.model_dump_json(indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
