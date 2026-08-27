from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import wave
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from bv.core.atomic import atomic_write_json
from bv.core.hashing import sha256_file
from bv.production.profile import (
    ProductionProfile,
    load_production_profile,
    production_profile_sha256,
)
from bv.workflow.runtime import RuntimeAuthorization
from bv.workflow.stages import StageContext

from .providers import NarrationRequest, NarrationSynthesizer


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_VOICE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")


class VoiceAuditionError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _AuditionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VoiceAuditionCandidate(_AuditionModel):
    candidate_id: str
    provider_voice_id: str
    excerpt_sha256: str
    speed: float
    audio_path: str
    audio_sha256: str
    receipt_path: str
    receipt_sha256: str
    duration_seconds: float

    @field_validator(
        "excerpt_sha256", "audio_sha256", "receipt_sha256"
    )
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid_voice_audition_hash")
        return value


class VoiceAuditionManifest(_AuditionModel):
    schema_version: Literal[1] = 1
    book_id: str
    episode_id: str
    provider: Literal["doubao"]
    script_sha256: str
    profile_sha256_before: str
    excerpt_path: str
    excerpt_sha256: str
    excerpt_span: tuple[int, int]
    speed: float
    candidates: tuple[VoiceAuditionCandidate, ...]
    created_at: datetime
    manifest_path: Path

    @field_validator("script_sha256", "profile_sha256_before", "excerpt_sha256")
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid_voice_audition_hash")
        return value


class VoiceProfile(_AuditionModel):
    schema_version: Literal[1] = 1
    provider: Literal["indextts2", "doubao"]
    provider_voice_id: str
    speed: float
    script_sha256: str
    excerpt_sha256: str
    approved_audio_sha256: str
    audition_manifest_sha256: str
    production_profile_sha256: str
    candidate_id: str
    audition_manifest_path: str
    approved_audio_path: str
    approved_at: datetime

    @field_validator(
        "script_sha256",
        "excerpt_sha256",
        "approved_audio_sha256",
        "audition_manifest_sha256",
        "production_profile_sha256",
    )
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid_voice_profile_hash")
        return value


class VoiceAuditionService:
    def __init__(
        self,
        *,
        synthesizers: Mapping[str, NarrationSynthesizer],
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.synthesizers = dict(synthesizers)
        self.now = now

    def prepare_candidates(
        self,
        context: StageContext,
        *,
        voice_ids: tuple[str, ...],
        supported_voice_ids: tuple[str, ...],
        authorization: RuntimeAuthorization,
    ) -> VoiceAuditionManifest:
        profile = load_production_profile(context.episode_root)
        _, script_text, script_sha256 = _approved_script(context)
        _validate_voice_ids(voice_ids, supported_voice_ids)
        if profile.tts.provider != "doubao" or profile.tts.voice_type is not None:
            raise VoiceAuditionError("voice_audition_not_required")
        excerpt, span = _select_excerpt(script_text, speed=profile.tts.speed)
        excerpt_sha256 = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        day = self.now().astimezone(
            timezone(timedelta(hours=8), name="Asia/Shanghai")
        ).strftime("%Y%m%d")
        root = (
            context.episode_root
            / ".private"
            / "candidates"
            / f"tts-audition-{day}"
        )
        _require_episode_child(context.episode_root, root)
        manifest_path = root / "manifest.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            existing = _load_manifest(context, manifest_path)
            if (
                existing.script_sha256 != script_sha256
                or existing.profile_sha256_before
                != production_profile_sha256(context.episode_root)
                or existing.excerpt_sha256 != excerpt_sha256
                or existing.excerpt_span != span
                or tuple(
                    candidate.provider_voice_id
                    for candidate in existing.candidates
                )
                != voice_ids
            ):
                raise VoiceAuditionError("voice_audition_already_exists")
            for candidate in existing.candidates:
                _validate_artifact(
                    context.episode_root / Path(candidate.audio_path).parent,
                    context.episode_root / candidate.audio_path,
                    candidate.audio_sha256,
                )
                _validate_artifact(
                    context.episode_root / Path(candidate.receipt_path).parent,
                    context.episode_root / candidate.receipt_path,
                    candidate.receipt_sha256,
                )
            return existing
        if "doubao" not in self.synthesizers:
            raise VoiceAuditionError("voice_audition_provider_unavailable")
        authorization.assert_audition_batch(
            book_id=context.book_id,
            episode_id=context.episode_id,
            script_sha256=script_sha256,
            provider_voice_ids=voice_ids,
        )

        request_plan = {
            "schema_version": 1,
            "book_id": context.book_id,
            "episode_id": context.episode_id,
            "provider": "doubao",
            "script_sha256": script_sha256,
            "profile_sha256_before": production_profile_sha256(
                context.episode_root
            ),
            "excerpt_sha256": excerpt_sha256,
            "excerpt_span": list(span),
            "speed": profile.tts.speed,
            "voice_ids": list(voice_ids),
        }
        plan_path = root / "request.json"
        excerpt_path = root / "excerpt.txt"
        if root.exists():
            try:
                _require_safe_regular(context.episode_root, plan_path)
                _require_safe_regular(context.episode_root, excerpt_path)
                persisted = json.loads(plan_path.read_text(encoding="utf-8"))
            except Exception:
                raise VoiceAuditionError("voice_audition_resume_invalid") from None
            if (
                persisted != request_plan
                or sha256_file(excerpt_path) != excerpt_sha256
                or excerpt_path.read_text(encoding="utf-8") != excerpt
            ):
                raise VoiceAuditionError("voice_audition_resume_mismatch")
        else:
            root.mkdir(parents=True, exist_ok=False)
            _atomic_write_text(excerpt_path, excerpt)
            atomic_write_json(plan_path, request_plan)

        candidates: list[VoiceAuditionCandidate] = []
        try:
            for index, voice_id in enumerate(voice_ids, start=1):
                candidate_id = f"candidate-{index:02d}"
                output_dir = root / candidate_id
                request = NarrationRequest(
                    book_id=context.book_id,
                    episode_id=context.episode_id,
                    approved_script_path=excerpt_path,
                    approved_script_sha256=excerpt_sha256,
                    provider_resource_id=profile.tts.resource_id,
                    provider_voice_id=voice_id,
                    speed=profile.tts.speed,
                    output_dir=output_dir,
                    kind="audition",
                )
                scoped = RuntimeAuthorization(
                    allow_external=True,
                    book_id=context.book_id,
                    episode_id=context.episode_id,
                    script_sha256=excerpt_sha256,
                    provider_voice_id=voice_id,
                    max_audition_submissions=1,
                    allow_fallback=False,
                )
                artifact = self.synthesizers["doubao"].synthesize(request, scoped)
                _validate_artifact(output_dir, artifact.raw_audio_path, artifact.raw_audio_sha256)
                _validate_artifact(output_dir, artifact.receipt_path, artifact.receipt_sha256)
                duration = _wav_duration_seconds(artifact.raw_audio_path)
                if not 15.0 <= duration <= 25.0:
                    raise VoiceAuditionError("voice_audition_duration_invalid")
                candidates.append(
                    VoiceAuditionCandidate(
                        candidate_id=candidate_id,
                        provider_voice_id=voice_id,
                        excerpt_sha256=excerpt_sha256,
                        speed=profile.tts.speed,
                        audio_path=_relative(context.episode_root, artifact.raw_audio_path),
                        audio_sha256=artifact.raw_audio_sha256,
                        receipt_path=_relative(context.episode_root, artifact.receipt_path),
                        receipt_sha256=artifact.receipt_sha256,
                        duration_seconds=duration,
                    )
                )
            manifest = VoiceAuditionManifest(
                book_id=context.book_id,
                episode_id=context.episode_id,
                provider="doubao",
                script_sha256=script_sha256,
                profile_sha256_before=production_profile_sha256(context.episode_root),
                excerpt_path=_relative(context.episode_root, excerpt_path),
                excerpt_sha256=excerpt_sha256,
                excerpt_span=span,
                speed=profile.tts.speed,
                candidates=tuple(candidates),
                created_at=self.now(),
                manifest_path=manifest_path,
            )
            atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
            return manifest
        except Exception:
            raise

    def approve_candidate(
        self,
        context: StageContext,
        candidate_id: str,
    ) -> VoiceProfile:
        manifest_path = _latest_manifest_path(context.episode_root)
        manifest = _load_manifest(context, manifest_path)
        matches = tuple(
            candidate for candidate in manifest.candidates
            if candidate.candidate_id == candidate_id
        )
        if len(matches) != 1:
            raise VoiceAuditionError("voice_candidate_not_found")
        candidate = matches[0]
        audio_path = context.episode_root / candidate.audio_path
        _validate_artifact(
            audio_path.parent,
            audio_path,
            candidate.audio_sha256,
        )
        current = load_production_profile(context.episode_root)
        if (
            current.tts.provider != manifest.provider
            or current.tts.speed != manifest.speed
            or production_profile_sha256(context.episode_root)
            != manifest.profile_sha256_before
        ):
            raise VoiceAuditionError("voice_audition_stale")
        updated = current.model_copy(
            update={
                "tts": current.tts.model_copy(
                    update={"voice_type": candidate.provider_voice_id}
                )
            }
        )
        profile_path = context.episode_root / "production_profile.json"
        if not profile_path.is_file() or profile_path.is_symlink():
            raise VoiceAuditionError("production_profile_missing")
        atomic_write_json(profile_path, updated.model_dump(mode="json"))
        updated_hash = production_profile_sha256(context.episode_root)
        voice_profile = VoiceProfile(
            provider=manifest.provider,
            provider_voice_id=candidate.provider_voice_id,
            speed=manifest.speed,
            script_sha256=manifest.script_sha256,
            excerpt_sha256=manifest.excerpt_sha256,
            approved_audio_sha256=candidate.audio_sha256,
            audition_manifest_sha256=sha256_file(manifest_path),
            production_profile_sha256=updated_hash,
            candidate_id=candidate.candidate_id,
            audition_manifest_path=_relative(context.episode_root, manifest_path),
            approved_audio_path=candidate.audio_path,
            approved_at=self.now(),
        )
        atomic_write_json(
            context.episode_root / "voice_profile.json",
            voice_profile.model_dump(mode="json"),
        )
        return voice_profile


def require_current_voice_profile(
    episode_root: Path,
    profile: ProductionProfile,
    *,
    script_sha256: str | None,
) -> VoiceProfile:
    if profile.tts.provider == "indextts2":
        raise VoiceAuditionError("voice_profile_not_required")
    path = Path(episode_root) / "voice_profile.json"
    try:
        _require_safe_regular(Path(episode_root), path)
        voice = VoiceProfile.model_validate_json(path.read_text(encoding="utf-8"))
        manifest_path = Path(episode_root) / voice.audition_manifest_path
        audio_path = Path(episode_root) / voice.approved_audio_path
        _require_safe_regular(Path(episode_root), manifest_path)
        _require_safe_regular(Path(episode_root), audio_path)
        if (
            script_sha256 is None
            or voice.script_sha256 != script_sha256
            or voice.provider != profile.tts.provider
            or voice.provider_voice_id != profile.tts.voice_type
            or voice.speed != profile.tts.speed
            or voice.production_profile_sha256
            != production_profile_sha256(Path(episode_root))
            or voice.audition_manifest_sha256 != sha256_file(manifest_path)
            or voice.approved_audio_sha256 != sha256_file(audio_path)
        ):
            raise VoiceAuditionError("voice_profile_stale")
        return voice
    except VoiceAuditionError:
        raise
    except Exception:
        raise VoiceAuditionError("voice_profile_not_approved") from None


def _approved_script(context: StageContext) -> tuple[Path, str, str]:
    path = context.episode_root / "script" / "approved.txt"
    try:
        _require_safe_regular(context.episode_root, path)
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except Exception:
        raise VoiceAuditionError("approved_script_invalid") from None
    digest = hashlib.sha256(raw).hexdigest()
    if (
        context.episode_state is None
        or context.episode_state.script_hash != digest
        or not text.strip()
    ):
        raise VoiceAuditionError("approved_script_invalid")
    return path, text, digest


def _validate_voice_ids(
    voice_ids: tuple[str, ...],
    supported_voice_ids: tuple[str, ...],
) -> None:
    if not voice_ids:
        raise VoiceAuditionError("voice_candidates_required")
    if len(voice_ids) > 3:
        raise VoiceAuditionError("too_many_voice_candidates")
    if len(set(voice_ids)) != len(voice_ids):
        raise VoiceAuditionError("duplicate_voice_candidates")
    supported = set(supported_voice_ids)
    for voice_id in voice_ids:
        if _VOICE_ID_PATTERN.fullmatch(voice_id) is None:
            raise VoiceAuditionError("invalid_voice_candidate")
        if voice_id not in supported:
            raise VoiceAuditionError("voice_candidate_not_supported")


def _select_excerpt(text: str, *, speed: float) -> tuple[str, tuple[int, int]]:
    target_units = round(64 * speed)
    minimum_units = round(48 * speed)
    units = 0
    end = 0
    for end, character in enumerate(text, start=1):
        if character.strip() and character not in "，。！？；：、,.!?;:\n\r\t“”‘’（）()《》":
            units += 1
        if units >= target_units:
            break
    if units < minimum_units:
        raise VoiceAuditionError("voice_audition_excerpt_too_short")
    excerpt = text[:end].strip()
    estimated = units / (3.2 * speed)
    if not 15.0 <= estimated <= 25.0:
        raise VoiceAuditionError("voice_audition_excerpt_invalid")
    start = text.index(excerpt)
    return excerpt, (start, start + len(excerpt))


def _latest_manifest_path(episode_root: Path) -> Path:
    root = Path(episode_root) / ".private" / "candidates"
    try:
        matches = tuple(sorted(root.glob("tts-audition-*/manifest.json")))
    except OSError:
        matches = ()
    if len(matches) != 1:
        raise VoiceAuditionError("voice_audition_manifest_unavailable")
    return matches[0]


def _load_manifest(
    context: StageContext,
    path: Path,
) -> VoiceAuditionManifest:
    try:
        _require_safe_regular(context.episode_root, path)
        manifest = VoiceAuditionManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        raise VoiceAuditionError("voice_audition_manifest_invalid") from None
    if (
        manifest.book_id != context.book_id
        or manifest.episode_id != context.episode_id
        or manifest.manifest_path.absolute() != path.absolute()
    ):
        raise VoiceAuditionError("voice_audition_manifest_invalid")
    return manifest


def _validate_artifact(root: Path, path: Path, expected_sha256: str) -> None:
    try:
        _require_safe_regular(root, path)
        if sha256_file(path) != expected_sha256:
            raise VoiceAuditionError("voice_audition_artifact_invalid")
    except VoiceAuditionError:
        raise
    except Exception:
        raise VoiceAuditionError("voice_audition_artifact_invalid") from None


def _wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as stream:
            return stream.getnframes() / stream.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        raise VoiceAuditionError("voice_audition_audio_invalid") from None


def _relative(root: Path, path: Path) -> str:
    try:
        return path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError:
        raise VoiceAuditionError("unsafe_voice_audition_path") from None


def _require_episode_child(root: Path, path: Path) -> None:
    if not path.absolute().is_relative_to(root.absolute()):
        raise VoiceAuditionError("unsafe_voice_audition_path")
    if _redirect_in_existing_chain(path):
        raise VoiceAuditionError("unsafe_voice_audition_path")


def _require_safe_regular(root: Path, path: Path) -> None:
    _require_episode_child(root, path)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise VoiceAuditionError("unsafe_voice_audition_path")


def _redirect_in_existing_chain(path: Path) -> bool:
    candidate = Path(path)
    while True:
        if candidate.exists() or candidate.is_symlink():
            if candidate.is_symlink() or _is_windows_reparse(candidate):
                return True
        parent = candidate.parent
        if parent == candidate:
            return False
        candidate = parent


def _is_windows_reparse(path: Path) -> bool:
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & 0x400)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
