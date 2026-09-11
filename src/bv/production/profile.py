from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_validator, model_serializer, model_validator


VisualSequenceMode = Literal[
    "legacy-monochrome-reveal",
    "color-story-pair",
]


class ProductionProfileError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _ProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DurationProfile(_ProfileModel):
    hard_min_seconds: float = Field(gt=0)
    ideal_min_seconds: float = Field(gt=0)
    ideal_max_seconds: float = Field(gt=0)
    hard_max_seconds: float = Field(gt=0)
    soft_max_seconds: float | None = Field(default=None, gt=0)
    advisory_only: bool = False

    @model_serializer(mode="wrap")
    def _serialize_legacy_compatibly(self, handler: SerializerFunctionWrapHandler) -> object:
        payload = handler(self)
        if isinstance(payload, dict) and self.soft_max_seconds is None:
            payload.pop("soft_max_seconds", None)
        if isinstance(payload, dict) and not self.advisory_only:
            payload.pop("advisory_only", None)
        return payload

    @model_validator(mode="after")
    def _validate_order(self) -> DurationProfile:
        if not (
            self.hard_min_seconds
            <= self.ideal_min_seconds
            <= self.ideal_max_seconds
            <= self.hard_max_seconds
        ):
            raise ValueError("invalid_duration_profile")
        if self.soft_max_seconds is not None and self.soft_max_seconds < self.ideal_max_seconds:
            raise ValueError("invalid_duration_profile")
        return self


class TtsProfile(_ProfileModel):
    provider: Literal["indextts2", "doubao"]
    resource_id: str
    voice_type: str | None
    speed: float = Field(gt=0.5, le=2.0)

    @field_validator("resource_id")
    @classmethod
    def _require_resource_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("blank_tts_resource_id")
        return value

    @field_validator("voice_type")
    @classmethod
    def _validate_voice_type(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("blank_voice_type")
        return value

    @model_validator(mode="after")
    def _validate_provider_fields(self) -> TtsProfile:
        if self.provider == "indextts2" and self.voice_type != "黑金3":
            raise ValueError("legacy_voice_type_invalid")
        if (
            self.provider == "doubao"
            and self.resource_id != "seed-tts-2.0"
        ):
            raise ValueError("doubao_resource_id_invalid")
        return self


class VisualProfile(_ProfileModel):
    seconds_per_scene_min: float = Field(gt=0)
    seconds_per_scene_max: float = Field(gt=0)
    representative_count: Literal[3]
    transition: Literal["cross-dissolve"]
    sequence_mode: VisualSequenceMode = "legacy-monochrome-reveal"
    scene_count: int | None = Field(default=None, ge=3, le=48)
    style_id: str | None = None

    @field_validator("style_id")
    @classmethod
    def _validate_style_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("blank_style_id")
        return value

    @model_validator(mode="after")
    def _validate_scene_duration(self) -> VisualProfile:
        if self.seconds_per_scene_min > self.seconds_per_scene_max:
            raise ValueError("invalid_scene_duration_profile")
        return self


DeliveryArtifact = Literal[
    "video",
    "cover",
    "script",
    "voice",
    "subtitles",
    "manifest",
]
_DELIVERY_ARTIFACTS: tuple[DeliveryArtifact, ...] = (
    "video",
    "cover",
    "script",
    "voice",
    "subtitles",
    "manifest",
)


class DeliveryProfile(_ProfileModel):
    timezone: Literal["Asia/Shanghai"]
    include: tuple[DeliveryArtifact, ...]

    @field_validator("include")
    @classmethod
    def _require_exact_artifacts(
        cls,
        value: tuple[DeliveryArtifact, ...],
    ) -> tuple[DeliveryArtifact, ...]:
        if value != _DELIVERY_ARTIFACTS:
            raise ValueError("invalid_delivery_artifacts")
        return value


class ProductionProfile(_ProfileModel):
    schema_version: Literal[1]
    narrative_mode: Literal["value", "story"] = "value"
    duration: DurationProfile
    tts: TtsProfile
    visual: VisualProfile
    delivery: DeliveryProfile

    @model_serializer(mode="wrap")
    def _serialize_legacy_compatibly(self, handler: SerializerFunctionWrapHandler) -> object:
        payload = handler(self)
        if isinstance(payload, dict) and self.narrative_mode == "value":
            payload.pop("narrative_mode", None)
        return payload

    @classmethod
    def short_book_default(cls) -> ProductionProfile:
        return cls(
            schema_version=1,
            duration=DurationProfile(
                hard_min_seconds=30.0,
                ideal_min_seconds=35.0,
                ideal_max_seconds=40.0,
                hard_max_seconds=45.0,
            ),
            tts=TtsProfile(
                provider="doubao",
                resource_id="seed-tts-2.0",
                voice_type=None,
                speed=1.0,
            ),
            visual=VisualProfile(
                seconds_per_scene_min=7.5,
                seconds_per_scene_max=11.25,
                representative_count=3,
                transition="cross-dissolve",
                sequence_mode="color-story-pair",
                scene_count=4,
                style_id="retro-gouache-concept",
            ),
            delivery=DeliveryProfile(
                timezone="Asia/Shanghai",
                include=_DELIVERY_ARTIFACTS,
            ),
        )

    @classmethod
    def living_default(cls) -> ProductionProfile:
        return cls(
            schema_version=1,
            duration=DurationProfile(
                hard_min_seconds=120.0,
                ideal_min_seconds=128.0,
                ideal_max_seconds=142.0,
                hard_max_seconds=150.0,
            ),
            tts=TtsProfile(
                provider="doubao",
                resource_id="seed-tts-2.0",
                voice_type=None,
                speed=1.0,
            ),
            visual=VisualProfile(
                seconds_per_scene_min=7.0,
                seconds_per_scene_max=11.0,
                representative_count=3,
                transition="cross-dissolve",
                sequence_mode="color-story-pair",
                scene_count=4,
            ),
            delivery=DeliveryProfile(
                timezone="Asia/Shanghai",
                include=_DELIVERY_ARTIFACTS,
            ),
        )

    @classmethod
    def longform_story_default(cls) -> ProductionProfile:
        """Long-form story guidance: 6-8 minutes, with a review-only 10 minute cap."""
        return cls(
            schema_version=1,
            narrative_mode="story",
            duration=DurationProfile(
                hard_min_seconds=1.0,
                ideal_min_seconds=360.0,
                ideal_max_seconds=480.0,
                hard_max_seconds=600.0,
                soft_max_seconds=600.0,
                advisory_only=True,
            ),
            tts=TtsProfile(provider="doubao", resource_id="seed-tts-2.0", voice_type=None, speed=1.12),
            visual=VisualProfile(
                seconds_per_scene_min=12.0,
                seconds_per_scene_max=30.0,
                representative_count=3,
                transition="cross-dissolve",
                sequence_mode="color-story-pair",
                scene_count=None,
            ),
            delivery=DeliveryProfile(timezone="Asia/Shanghai", include=_DELIVERY_ARTIFACTS),
        )

    def duration_advisories(self, actual_seconds: float) -> tuple[str, ...]:
        """Return review hints without rejecting a valid story duration."""
        if self.narrative_mode != "story":
            return ()
        advisories: list[str] = []
        if not self.duration.ideal_min_seconds <= actual_seconds <= self.duration.ideal_max_seconds:
            advisories.append("outside_recommended_story_duration")
        if self.duration.soft_max_seconds is not None and actual_seconds > self.duration.soft_max_seconds:
            advisories.append("story_duration_above_soft_max")
        return tuple(advisories)


LEGACY_PRODUCTION_PROFILE = ProductionProfile(
    schema_version=1,
    duration=DurationProfile(
        hard_min_seconds=45.0,
        ideal_min_seconds=48.0,
        ideal_max_seconds=56.0,
        hard_max_seconds=60.0,
    ),
    tts=TtsProfile(
        provider="indextts2",
        resource_id="local.indextts2",
        voice_type="黑金3",
        speed=1.0,
    ),
    visual=VisualProfile(
        seconds_per_scene_min=4.0,
        seconds_per_scene_max=7.0,
        representative_count=3,
        transition="cross-dissolve",
        sequence_mode="legacy-monochrome-reveal",
    ),
    delivery=DeliveryProfile(
        timezone="Asia/Shanghai",
        include=_DELIVERY_ARTIFACTS,
    ),
)


def load_production_profile(episode_root: Path) -> ProductionProfile:
    root = Path(episode_root)
    path = root / "production_profile.json"
    if not path.exists() and not path.is_symlink():
        return LEGACY_PRODUCTION_PROFILE
    _require_safe_file(root, path)
    try:
        payload = path.read_text(encoding="utf-8")
        if len(payload.encode("utf-8")) > 128 * 1024:
            raise ProductionProfileError("production_profile_too_large")
        return ProductionProfile.model_validate_json(payload)
    except ProductionProfileError:
        raise
    except (OSError, UnicodeDecodeError):
        raise ProductionProfileError("production_profile_read_failed") from None


def production_profile_sha256(episode_root: Path) -> str:
    root = Path(episode_root)
    path = root / "production_profile.json"
    profile = load_production_profile(root)
    if path.exists():
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            raise ProductionProfileError("production_profile_read_failed") from None
    payload = profile.model_dump(mode="json", exclude_none=True)
    visual = payload.get("visual")
    if isinstance(visual, dict):
        visual.pop("sequence_mode", None)
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def estimated_script_character_band(
    duration: DurationProfile,
) -> tuple[int, int]:
    return (
        round(duration.ideal_min_seconds * 3.0),
        round(duration.ideal_max_seconds * 4.5),
    )


def _require_safe_file(root: Path, path: Path) -> None:
    try:
        canonical_root = Path(os.path.abspath(root))
        canonical_path = Path(os.path.abspath(path))
        if not canonical_path.is_relative_to(canonical_root):
            raise ProductionProfileError("unsafe_production_profile_path")
        for candidate in (canonical_path, canonical_root):
            if candidate.is_symlink() or _is_reparse_point(candidate):
                raise ProductionProfileError("unsafe_production_profile_path")
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ProductionProfileError("unsafe_production_profile_path")
    except ProductionProfileError:
        raise
    except OSError:
        raise ProductionProfileError("unsafe_production_profile_path") from None


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)
