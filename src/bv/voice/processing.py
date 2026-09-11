"""Deterministic, local processing for one IndexTTS2 narration WAV."""

from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import uuid
import wave
from collections.abc import Mapping
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bv.core.process import CommandResult, run_command
from bv.production.profile import DurationProfile


class VoiceProcessingError(RuntimeError):
    """A stable, privacy-safe processing failure."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class NarrationDuration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seconds: float
    level: Literal["pass", "fail"]
    band: Literal["too_short", "edge_short", "ideal", "edge_long", "too_long"]
    warnings: list[str] = Field(default_factory=list)


class PcmWavInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int
    duration_seconds: float


class EdgeSilence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_trim_ms: int
    end_trim_ms: int
    all_silence: bool


class VoiceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["indextts2", "doubao"]
    provider_voice_id: str
    script_sha256: str
    reference_sha256: str | None
    provider_receipt_sha256: str | None = None
    raw_sha256: str
    master_sha256: str
    processing_sha256: str
    sample_rate: int
    channels: int
    sample_width: int
    raw_duration_ms: int
    master_duration_ms: int
    qualifying_start_trim_ms: int
    qualifying_end_trim_ms: int
    measured_i: float | None = None
    measured_tp: float | None = None
    measured_lra: float | None = None
    measured_thresh: float | None = None
    target_offset: float | None = None
    tempo_factor: float = 1.0
    duration: NarrationDuration
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_manifest(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        migrated = dict(value)
        legacy_voice = migrated.pop("voice_id", None)
        current_voice = migrated.get("provider_voice_id")
        if (
            legacy_voice is not None
            and current_voice is not None
            and legacy_voice != current_voice
        ):
            raise ValueError("conflicting_voice_id")
        if current_voice is None and legacy_voice is not None:
            migrated["provider_voice_id"] = legacy_voice
        migrated.setdefault("provider", "indextts2")
        migrated.setdefault("provider_receipt_sha256", None)
        return migrated

    @model_validator(mode="after")
    def _validate_provider_evidence(self) -> VoiceManifest:
        if not self.provider_voice_id.strip():
            raise ValueError("provider_voice_id_required")
        if self.provider == "indextts2":
            if self.reference_sha256 is None:
                raise ValueError("indextts2_reference_required")
            if self.provider_receipt_sha256 is not None:
                raise ValueError("indextts2_receipt_forbidden")
        else:
            if self.reference_sha256 is not None:
                raise ValueError("doubao_reference_forbidden")
            if self.provider_receipt_sha256 is None:
                raise ValueError("doubao_receipt_required")
        return self

    @property
    def voice_id(self) -> str:
        """One-release read alias for legacy callers."""

        return self.provider_voice_id


Runner = Callable[..., CommandResult]


@dataclass(frozen=True, slots=True)
class NarrationDurationPolicy:
    hard_min_seconds: float
    ideal_min_seconds: float
    ideal_max_seconds: float
    hard_max_seconds: float
    advisory_only: bool = False


TECHNICAL_SAMPLE_POLICY = NarrationDurationPolicy(10.0, 13.0, 17.0, 20.0)
FULL_EPISODE_POLICY = NarrationDurationPolicy(45.0, 48.0, 56.0, 60.0)


def duration_policy_from(profile: DurationProfile) -> NarrationDurationPolicy:
    return NarrationDurationPolicy(
        hard_min_seconds=profile.hard_min_seconds,
        ideal_min_seconds=profile.ideal_min_seconds,
        ideal_max_seconds=profile.ideal_max_seconds,
        hard_max_seconds=profile.hard_max_seconds,
        advisory_only=profile.advisory_only,
    )


def classify_narration_duration(
    seconds: float,
    *,
    policy: NarrationDurationPolicy = FULL_EPISODE_POLICY,
) -> NarrationDuration:
    if not (
        0 < policy.hard_min_seconds <= policy.ideal_min_seconds
        <= policy.ideal_max_seconds <= policy.hard_max_seconds
    ):
        raise VoiceProcessingError("invalid_duration_policy")
    if seconds < policy.hard_min_seconds and not policy.advisory_only:
        return NarrationDuration(seconds=seconds, level="fail", band="too_short")
    if seconds < policy.ideal_min_seconds:
        return NarrationDuration(seconds=seconds, level="pass", band="edge_short", warnings=["edge_short_duration"])
    if seconds <= policy.ideal_max_seconds:
        return NarrationDuration(seconds=seconds, level="pass", band="ideal")
    if seconds <= policy.hard_max_seconds:
        return NarrationDuration(seconds=seconds, level="pass", band="edge_long", warnings=["edge_long_duration"])
    if policy.advisory_only:
        return NarrationDuration(seconds=seconds, level="pass", band="edge_long", warnings=["duration_above_soft_max"])
    return NarrationDuration(seconds=seconds, level="fail", band="too_long")


def inspect_pcm_wav(path: Path) -> PcmWavInfo:
    try:
        with wave.open(str(path), "rb") as stream:
            if stream.getcomptype() != "NONE":
                raise VoiceProcessingError("invalid_pcm_wav")
            frames = stream.getnframes()
            rate = stream.getframerate()
            channels = stream.getnchannels()
            width = stream.getsampwidth()
    except VoiceProcessingError:
        raise
    except (OSError, EOFError, wave.Error):
        raise VoiceProcessingError("invalid_pcm_wav") from None
    if frames <= 0 or rate <= 0 or channels <= 0 or width not in {1, 2, 3, 4}:
        raise VoiceProcessingError("invalid_pcm_wav")
    return PcmWavInfo(
        sample_rate=rate, channels=channels, sample_width=width, frame_count=frames,
        duration_seconds=frames / rate,
    )


def detect_qualifying_edge_silence(
    path: Path, *, silence_threshold_db: float = -45.0, removable_edge_silence_ms: int = 300,
) -> EdgeSilence:
    info = inspect_pcm_wav(path)
    if removable_edge_silence_ms < 0 or silence_threshold_db >= 0:
        raise VoiceProcessingError("invalid_processing_configuration")
    try:
        with wave.open(str(path), "rb") as stream:
            data = stream.readframes(info.frame_count)
    except (OSError, EOFError, wave.Error):
        raise VoiceProcessingError("invalid_pcm_wav") from None
    frame_silent = _silent_frames(data, info, silence_threshold_db)
    if all(frame_silent):
        return EdgeSilence(start_trim_ms=0, end_trim_ms=0, all_silence=True)
    leading = _edge_run(frame_silent, from_start=True)
    trailing = _edge_run(frame_silent, from_start=False)
    minimum_frames = math.ceil(removable_edge_silence_ms * info.sample_rate / 1_000)
    return EdgeSilence(
        start_trim_ms=round(leading * 1_000 / info.sample_rate) if leading >= minimum_frames else 0,
        end_trim_ms=round(trailing * 1_000 / info.sample_rate) if trailing >= minimum_frames else 0,
        all_silence=False,
    )


def build_voice_master(
    *,
    episode_root: Path,
    raw_path: Path,
    master_path: Path,
    script_path: Path,
    approved_script_sha256: str,
    ffmpeg_command: Path | str,
    reference_voice_path: Path | None = None,
    voice_id: str | None = None,
    provider: Literal["indextts2", "doubao"] = "indextts2",
    provider_voice_id: str | None = None,
    provider_receipt_path: Path | None = None,
    expected_reference_sha256: str | None = None,
    silence_threshold_db: float = -45.0,
    removable_edge_silence_ms: int = 300,
    target_lufs: float = -18.0,
    true_peak_db: float = -1.5,
    runner: Runner = run_command,
    timeout: float = 300.0,
    duration_policy: NarrationDurationPolicy = FULL_EPISODE_POLICY,
) -> VoiceManifest:
    """Decode, edge-trim, loudness-normalize, and atomically publish a master WAV."""

    root, raw, master, script = map(
        Path,
        (episode_root, raw_path, master_path, script_path),
    )
    effective_voice_id = provider_voice_id or voice_id
    if (
        effective_voice_id is None
        or not effective_voice_id.strip()
        or (
            provider_voice_id is not None
            and voice_id is not None
            and provider_voice_id != voice_id
        )
    ):
        raise VoiceProcessingError("provider_voice_id_invalid")
    manifest_path = master.with_suffix(".json")
    _require_safe_existing_directory(root)
    for path in (raw, script):
        _require_safe_existing_file(path)
        _require_child(root, path)
    reference: Path | None = (
        None if reference_voice_path is None else Path(reference_voice_path)
    )
    receipt: Path | None = (
        None if provider_receipt_path is None else Path(provider_receipt_path)
    )
    if provider == "indextts2":
        if reference is None or receipt is not None:
            raise VoiceProcessingError("provider_evidence_invalid")
        _require_safe_existing_file(reference)
    elif provider == "doubao":
        if reference is not None or receipt is None:
            raise VoiceProcessingError("provider_evidence_invalid")
        _require_safe_existing_file(receipt)
        _require_child(root, receipt)
    else:
        raise VoiceProcessingError("provider_invalid")
    for path in (master, manifest_path):
        _require_child(root, path)
        if _redirect_in_existing_chain(path):
            raise VoiceProcessingError("unsafe_voice_path")
    _ensure_safe_directory(master.parent)
    temporary_root = root / ".private" / "voice-processing"
    _ensure_safe_directory(temporary_root)

    script_hash = _sha256(script)
    if not re.fullmatch(r"[0-9a-f]{64}", approved_script_sha256) or script_hash != approved_script_sha256:
        raise VoiceProcessingError("approved_script_hash_mismatch")
    reference_hash = None if reference is None else _sha256(reference)
    receipt_hash = None if receipt is None else _sha256(receipt)
    if (
        expected_reference_sha256 is not None
        and reference_hash != expected_reference_sha256
    ):
        raise VoiceProcessingError("reference_hash_mismatch")
    raw_hash = _sha256(raw)
    raw_info = inspect_pcm_wav(raw)
    settings = {
        "format": "pcm_s16le/48000/mono", "silence_threshold_db": silence_threshold_db,
        "removable_edge_silence_ms": removable_edge_silence_ms, "target_lufs": target_lufs,
        "true_peak_db": true_peak_db,
        "max_tempo_factor": 1.08,
        "duration_target_margin_seconds": 0.25,
        "provider": provider,
        "provider_voice_id": effective_voice_id,
        "provider_receipt_sha256": receipt_hash,
        "duration_policy": {
            "hard_min_seconds": duration_policy.hard_min_seconds,
            "ideal_min_seconds": duration_policy.ideal_min_seconds,
            "ideal_max_seconds": duration_policy.ideal_max_seconds,
            "hard_max_seconds": duration_policy.hard_max_seconds,
        },
    }
    processing_hash = _hash_json(settings)
    existing = _existing_master_if_identical(
        master=master, manifest_path=manifest_path, raw_hash=raw_hash,
        script_hash=script_hash, reference_hash=reference_hash,
        provider_receipt_hash=receipt_hash,
        processing_hash=processing_hash,
        provider=provider,
        provider_voice_id=effective_voice_id,
    )
    if existing is not None:
        return existing
    decoded = _temporary_wav(temporary_root, "decoded")
    normalized = _temporary_wav(temporary_root, "normalized")
    master_published = False
    try:
        _run_ffmpeg(
            [str(ffmpeg_command), "-hide_banner", "-nostdin", "-y", "-i", str(raw), "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(decoded)],
            cwd=root, runner=runner, timeout=timeout, error_code="ffmpeg_decode_failed",
        )
        decoded_info = inspect_pcm_wav(decoded)
        if (decoded_info.sample_rate, decoded_info.channels, decoded_info.sample_width) != (48_000, 1, 2):
            raise VoiceProcessingError("decoded_format_invalid")
        edges = detect_qualifying_edge_silence(decoded, silence_threshold_db=silence_threshold_db, removable_edge_silence_ms=removable_edge_silence_ms)
        if edges.all_silence:
            raise VoiceProcessingError("audio_all_silence")
        start_samples = round(edges.start_trim_ms * decoded_info.sample_rate / 1_000)
        end_samples = decoded_info.frame_count - round(edges.end_trim_ms * decoded_info.sample_rate / 1_000)
        if end_samples <= start_samples:
            raise VoiceProcessingError("audio_all_silence")
        trimmed_seconds = (end_samples - start_samples) / decoded_info.sample_rate
        tempo_factor = _duration_tempo_factor(trimmed_seconds, duration_policy)
        measurement = _measure_loudness(
            decoded, ffmpeg_command, root, runner, timeout, target_lufs,
            true_peak_db, start_samples, end_samples, tempo_factor,
        )
        filter_text = _loudnorm_filter(
            measurement,
            target_lufs,
            true_peak_db,
            start_samples,
            end_samples,
            tempo_factor,
        )
        _run_ffmpeg(
            [str(ffmpeg_command), "-hide_banner", "-nostdin", "-y", "-i", str(decoded), "-af", filter_text, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(normalized)],
            cwd=root, runner=runner, timeout=timeout, error_code="ffmpeg_normalize_failed",
        )
        master_info = inspect_pcm_wav(normalized)
        if (master_info.sample_rate, master_info.channels, master_info.sample_width) != (48_000, 1, 2):
            raise VoiceProcessingError("normalized_format_invalid")
        if (
            _sha256(raw) != raw_hash
            or _sha256(script) != script_hash
            or (reference is not None and _sha256(reference) != reference_hash)
            or (receipt is not None and _sha256(receipt) != receipt_hash)
        ):
            raise VoiceProcessingError("input_hash_changed")
        if _redirect_in_existing_chain(master) or _redirect_in_existing_chain(manifest_path):
            raise VoiceProcessingError("unsafe_voice_path")
        os.replace(normalized, master)
        master_published = True
        master_info = inspect_pcm_wav(master)
        manifest = VoiceManifest(
            provider=provider,
            provider_voice_id=effective_voice_id,
            script_sha256=script_hash,
            reference_sha256=reference_hash,
            provider_receipt_sha256=receipt_hash,
            raw_sha256=raw_hash, master_sha256=_sha256(master), processing_sha256=processing_hash,
            sample_rate=master_info.sample_rate, channels=master_info.channels, sample_width=master_info.sample_width,
            raw_duration_ms=round(raw_info.duration_seconds * 1_000), master_duration_ms=round(master_info.duration_seconds * 1_000),
            qualifying_start_trim_ms=edges.start_trim_ms, qualifying_end_trim_ms=edges.end_trim_ms,
            measured_i=measurement["input_i"], measured_tp=measurement["input_tp"], measured_lra=measurement["input_lra"],
            measured_thresh=measurement["input_thresh"], target_offset=measurement["target_offset"],
            tempo_factor=tempo_factor,
            duration=classify_narration_duration(
                master_info.duration_seconds,
                policy=duration_policy,
            ), created_at=datetime.now(UTC),
        )
        _atomic_write_manifest(manifest_path, manifest, root)
        return manifest
    except VoiceProcessingError:
        if master_published:
            _remove_task_owned_master(master)
        raise
    except OSError:
        if master_published:
            _remove_task_owned_master(master)
        raise VoiceProcessingError("master_publish_failed") from None
    finally:
        for temporary in (decoded, normalized):
            try:
                if temporary.is_file() and not temporary.is_symlink():
                    temporary.unlink()
            except OSError:
                pass


def _measure_loudness(
    path: Path, ffmpeg_command: Path | str, cwd: Path, runner: Runner,
    timeout: float, target_lufs: float, true_peak_db: float,
    start_samples: int, end_samples: int, tempo_factor: float = 1.0,
) -> dict[str, float]:
    filter_text = (
        f"{_trim_filter(start_samples, end_samples, tempo_factor)}"
        f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA=11:print_format=json"
    )
    result = _run_ffmpeg(
        [str(ffmpeg_command), "-hide_banner", "-nostdin", "-i", str(path), "-af", filter_text, "-f", "null", "NUL"],
        cwd=cwd, runner=runner, timeout=timeout, error_code="ffmpeg_measure_failed",
    )
    for text in (result.stderr, result.stdout):
        for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
            try:
                payload = json.loads(match.group(0))
                measured = {name: float(payload[name]) for name in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if all(math.isfinite(value) for value in measured.values()):
                return measured
    raise VoiceProcessingError("loudnorm_measurement_invalid")


def _loudnorm_filter(
    measured: dict[str, float],
    target_lufs: float,
    true_peak_db: float,
    start_samples: int,
    end_samples: int,
    tempo_factor: float = 1.0,
) -> str:
    return (
        f"{_trim_filter(start_samples, end_samples, tempo_factor)}"
        f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA=11:measured_I={measured['input_i']}:"
        f"measured_LRA={measured['input_lra']}:measured_TP={measured['input_tp']}:"
        f"measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true:print_format=json"
    )


def _trim_filter(start_samples: int, end_samples: int, tempo_factor: float = 1.0) -> str:
    tempo = "" if tempo_factor == 1.0 else f"atempo={tempo_factor:.6f},"
    return (
        f"atrim=start_sample={start_samples}:end_sample={end_samples},"
        f"asetpts=N/SR/TB,{tempo}"
    )


def _duration_tempo_factor(
    seconds: float,
    policy: NarrationDurationPolicy,
    *,
    max_tempo_factor: float = 1.08,
    target_margin_seconds: float = 0.25,
) -> float:
    if policy.advisory_only:
        return 1.0
    if seconds <= policy.hard_max_seconds:
        return 1.0
    target = policy.hard_max_seconds - target_margin_seconds
    if target <= 0:
        raise VoiceProcessingError("invalid_duration_policy")
    factor = seconds / target
    if factor > max_tempo_factor:
        raise VoiceProcessingError("duration_too_long_for_bounded_tempo")
    return factor


def _run_ffmpeg(argv: list[str], *, cwd: Path, runner: Runner, timeout: float, error_code: str) -> CommandResult:
    try:
        result = runner(argv, cwd=cwd, timeout=timeout, secrets=())
    except (OSError, TypeError, ValueError):
        raise VoiceProcessingError(error_code) from None
    if result.returncode != 0:
        raise VoiceProcessingError(error_code)
    return result


def _silent_frames(data: bytes, info: PcmWavInfo, threshold_db: float) -> list[bool]:
    frame_width = info.channels * info.sample_width
    maximum = (1 << (info.sample_width * 8 - 1)) - 1
    threshold = maximum * 10 ** (threshold_db / 20)
    frames: list[bool] = []
    for offset in range(0, len(data), frame_width):
        values = [_sample_value(data[offset + channel * info.sample_width:offset + (channel + 1) * info.sample_width], info.sample_width) for channel in range(info.channels)]
        frames.append(max(abs(value) for value in values) <= threshold)
    return frames


def _sample_value(value: bytes, width: int) -> int:
    if width == 1:
        return value[0] - 128
    return int.from_bytes(value, byteorder="little", signed=True)


def _edge_run(values: list[bool], *, from_start: bool) -> int:
    source = values if from_start else list(reversed(values))
    for index, value in enumerate(source):
        if not value:
            return index
    return len(values)


def _temporary_wav(root: Path, label: str) -> Path:
    return root / f"{label}-{uuid.uuid4().hex}.wav"


def _hash_json(value: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_manifest(path: Path, manifest: VoiceManifest, root: Path) -> None:
    if _redirect_in_existing_chain(path):
        raise VoiceProcessingError("unsafe_voice_path")
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f"{path.name}-", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(manifest.model_dump_json(indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        if _redirect_in_existing_chain(path) or _redirect_in_existing_chain(root):
            raise VoiceProcessingError("unsafe_voice_path")
        os.replace(temporary, path)
    except VoiceProcessingError:
        raise
    except OSError:
        raise VoiceProcessingError("manifest_write_failed") from None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _existing_master_if_identical(
    *, master: Path, manifest_path: Path, raw_hash: str, script_hash: str,
    reference_hash: str | None,
    provider_receipt_hash: str | None,
    processing_hash: str,
    provider: Literal["indextts2", "doubao"],
    provider_voice_id: str,
) -> VoiceManifest | None:
    if not master.exists() and not manifest_path.exists():
        return None
    if not master.is_file() or not manifest_path.is_file():
        raise VoiceProcessingError("existing_master_integrity_invalid")
    try:
        manifest = VoiceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        info = inspect_pcm_wav(master)
    except (OSError, ValueError, VoiceProcessingError):
        raise VoiceProcessingError("existing_master_integrity_invalid") from None
    if (
        manifest.master_sha256 != _sha256(master)
        or (manifest.sample_rate, manifest.channels, manifest.sample_width) != (info.sample_rate, info.channels, info.sample_width)
        or manifest.master_duration_ms != round(info.duration_seconds * 1_000)
    ):
        raise VoiceProcessingError("existing_master_integrity_invalid")
    if (
        manifest.provider == provider
        and manifest.provider_voice_id == provider_voice_id
        and manifest.raw_sha256 == raw_hash
        and manifest.script_sha256 == script_hash and manifest.reference_sha256 == reference_hash
        and manifest.provider_receipt_sha256 == provider_receipt_hash
        and manifest.processing_sha256 == processing_hash
    ):
        return manifest
    raise VoiceProcessingError("existing_master_mismatch")


def _remove_task_owned_master(path: Path) -> None:
    try:
        if path.is_file() and not path.is_symlink():
            path.unlink()
    except OSError:
        pass


def _require_child(root: Path, path: Path) -> None:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        raise VoiceProcessingError("unsafe_voice_path") from None


def _require_safe_existing_directory(path: Path) -> None:
    if not path.is_dir() or path.is_symlink() or _redirect_in_existing_chain(path):
        raise VoiceProcessingError("unsafe_voice_path")


def _require_safe_existing_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or _redirect_in_existing_chain(path):
        raise VoiceProcessingError("unsafe_voice_path")


def _ensure_safe_directory(path: Path) -> None:
    if _redirect_in_existing_chain(path):
        raise VoiceProcessingError("unsafe_voice_path")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise VoiceProcessingError("unsafe_voice_path") from None
    _require_safe_existing_directory(path)


def _redirect_in_existing_chain(path: Path) -> bool:
    candidate = Path(path)
    while True:
        if candidate.exists() or candidate.is_symlink():
            if _is_redirected(candidate):
                return True
        parent = candidate.parent
        if parent == candidate:
            return False
        candidate = parent


def _is_redirected(path: Path) -> bool:
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        attributes = 0
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
