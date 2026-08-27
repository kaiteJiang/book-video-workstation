from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


_SAFE_IDENTIFIER_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def _validate_identifier(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("unsafe_identifier")
    if (
        not value
        or value in {".", ".."}
        or len(value) > 128
        or value[0] not in _SAFE_IDENTIFIER_CHARS - {".", "-", "_"}
        or any(character not in _SAFE_IDENTIFIER_CHARS for character in value)
    ):
        raise ValueError("unsafe_identifier")
    return value


def _validate_sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("invalid_sha256")
    return value


SafeIdentifier = Annotated[str, BeforeValidator(_validate_identifier)]
Sha256 = Annotated[str, BeforeValidator(_validate_sha256)]


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


def _canonical_path_without_redirects(value: object, *, error_code: str) -> Path:
    try:
        raw = Path(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(error_code) from exc
    canonical = Path(os.path.abspath(os.fspath(raw)))
    for candidate in (canonical, *canonical.parents):
        if candidate.is_symlink() or _is_reparse_point(candidate):
            raise ValueError(error_code)
    return canonical


class _IllustrationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
    )


class SafeArea(_IllustrationModel):
    top_reserved: int = 140
    keyline_top: int = 140
    keyline_bottom: int = 340
    illustration_top: int = 340
    illustration_bottom: int = 1460
    subtitle_top: int = 1460
    subtitle_bottom: int = 1660
    bottom_reserved: int = 260
    right_reserved: int = 180

    @model_validator(mode="after")
    def _validate_layout(self) -> SafeArea:
        values = (
            self.top_reserved,
            self.keyline_top,
            self.keyline_bottom,
            self.illustration_top,
            self.illustration_bottom,
            self.subtitle_top,
            self.subtitle_bottom,
            self.bottom_reserved,
            self.right_reserved,
        )
        if any(value < 0 for value in values):
            raise ValueError("invalid_safe_area")
        if (
            self.top_reserved != self.keyline_top
            or self.keyline_top >= self.keyline_bottom
            or self.keyline_bottom != self.illustration_top
            or self.illustration_top >= self.illustration_bottom
            or self.illustration_bottom != self.subtitle_top
            or self.subtitle_top >= self.subtitle_bottom
            or self.subtitle_bottom + self.bottom_reserved != 1920
            or not 0 < self.right_reserved < 1080
        ):
            raise ValueError("invalid_safe_area")
        return self


class StyleDecision(_IllustrationModel):
    library_version: str
    selected_style: str
    candidate_styles: tuple[str, str, str]
    selection_reasons: tuple[str, ...]
    rejected_reasons: dict[str, str]
    confidence: float = Field(ge=0.0, le=1.0)
    manual_override: bool
    source_script_sha256: Sha256
    style_fingerprint: Sha256


class CharacterLock(_IllustrationModel):
    character_id: SafeIdentifier = "reader-01"
    role: str
    age_range: str
    face: str
    hair: str
    body: str
    base_clothing: str
    allowed_variations: tuple[str, ...]
    color_markers: tuple[str, ...]
    personal_objects: tuple[str, ...]
    forbidden_changes: tuple[str, ...]
    source_script_sha256: Sha256
    style_fingerprint: Sha256


class CharacterBible(_IllustrationModel):
    source_script_sha256: Sha256
    style_fingerprint: Sha256
    characters: tuple[CharacterLock, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def _validate_characters(self) -> CharacterBible:
        identifiers = tuple(item.character_id for item in self.characters)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate_character_id")
        if not set(identifiers).issubset(
            {"reader-01", "source-protagonist", "source-family-01"}
        ):
            raise ValueError("unsupported_character_id")
        if any(
            item.source_script_sha256 != self.source_script_sha256
            or item.style_fingerprint != self.style_fingerprint
            for item in self.characters
        ):
            raise ValueError("character_bible_identity_mismatch")
        return self


def load_character_bible(value: object) -> CharacterBible:
    if isinstance(value, CharacterBible):
        return value
    if isinstance(value, CharacterLock):
        character = value
    elif isinstance(value, Mapping) and "characters" in value:
        return CharacterBible.model_validate(value)
    else:
        character = CharacterLock.model_validate(value)
    if character.character_id != "reader-01":
        character = character.model_copy(update={"character_id": "reader-01"})
    return CharacterBible(
        source_script_sha256=character.source_script_sha256,
        style_fingerprint=character.style_fingerprint,
        characters=(character,),
    )


class IllustrationScene(_IllustrationModel):
    scene_id: SafeIdentifier
    start_ms: int
    end_ms: int
    from_frame: int
    to_frame: int
    narration: str
    narration_span: tuple[int, int]
    key_line: str
    visual_purpose: str
    setting: str
    character_action: str
    metaphor: str | None
    composition: str
    character_refs: tuple[SafeIdentifier, ...]
    image_prompt: str
    negative_constraints: tuple[str, ...]
    representative_frame: bool
    asset_status: Literal["planned", "generated", "approved"]

    @field_validator(
        "narration",
        "visual_purpose",
        "setting",
        "character_action",
        "composition",
        "image_prompt",
    )
    @classmethod
    def _require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("blank_scene_text")
        return value

    @field_validator("key_line")
    @classmethod
    def _validate_key_line(cls, value: str) -> str:
        normalized = value.strip()
        if not 6 <= len(normalized) <= 14:
            raise ValueError("invalid_key_line_length")
        return normalized

    @model_validator(mode="after")
    def _validate_ranges(self) -> IllustrationScene:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("invalid_scene_time_range")
        if self.from_frame < 0 or self.to_frame <= self.from_frame:
            raise ValueError("invalid_scene_frame_range")
        span_start, span_end = self.narration_span
        if span_start < 0 or span_end <= span_start:
            raise ValueError("invalid_narration_span")
        return self


class IllustrationStoryboard(_IllustrationModel):
    book_id: SafeIdentifier
    episode_id: SafeIdentifier
    width: Literal[1080]
    height: Literal[1920]
    fps: Literal[30]
    master_duration_ms: int = Field(gt=0)
    total_frames: int = Field(gt=0)
    script_sha256: Sha256
    audio_sha256: Sha256
    subtitle_sha256: Sha256
    style_decision_sha256: Sha256
    character_lock_sha256: Sha256
    character_bible_sha256: Sha256 | None = None
    narrative_deviation_reason: str | None = None
    scenes: tuple[IllustrationScene, ...]

    @model_validator(mode="after")
    def _validate_timeline(self) -> IllustrationStoryboard:
        expected_total = round(self.master_duration_ms * self.fps / 1000)
        if self.total_frames != expected_total:
            raise ValueError("total_frames_mismatch")
        if not self.scenes:
            raise ValueError("storyboard_scenes_empty")
        if self.scenes[0].from_frame != 0:
            raise ValueError("first_scene_frame")
        if self.scenes[-1].to_frame != self.total_frames:
            raise ValueError("final_scene_frame")
        for previous, current in zip(self.scenes, self.scenes[1:]):
            if current.from_frame > previous.to_frame:
                raise ValueError("scene_frame_gap")
            if current.from_frame < previous.to_frame:
                raise ValueError("scene_frame_overlap")
        return self


class SilentRenderRequest(_IllustrationModel):
    episode_root: Path
    storyboard_path: Path
    storyboard_sha256: Sha256
    output_path: Path
    vendor_dir: Path

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_confine_paths(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        normalized = dict(data)
        episode_root = _canonical_path_without_redirects(
            normalized.get("episode_root"),
            error_code="unsafe_episode_path",
        )
        storyboard_path = _canonical_path_without_redirects(
            normalized.get("storyboard_path"),
            error_code="unsafe_episode_path",
        )
        output_path = _canonical_path_without_redirects(
            normalized.get("output_path"),
            error_code="unsafe_episode_path",
        )
        vendor_dir = _canonical_path_without_redirects(
            normalized.get("vendor_dir"),
            error_code="unsafe_vendor_path",
        )
        if not storyboard_path.is_relative_to(episode_root):
            raise ValueError("unsafe_episode_path")
        if not output_path.is_relative_to(episode_root):
            raise ValueError("unsafe_episode_path")
        normalized.update(
            episode_root=episode_root,
            storyboard_path=storyboard_path,
            output_path=output_path,
            vendor_dir=vendor_dir,
        )
        return normalized


class SilentRenderResult(_IllustrationModel):
    output_path: Path
    output_sha256: Sha256
    duration_ms: int = Field(gt=0)
    width: Literal[1080]
    height: Literal[1920]
    frame_rate: Literal[30.0]
    audio_present: Literal[False]
