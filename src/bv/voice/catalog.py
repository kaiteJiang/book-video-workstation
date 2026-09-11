from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class VoiceCatalogError(RuntimeError):
    pass


class _CatalogModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VoiceCompatibility(_CatalogModel):
    model: Literal["doubao-seed-tts-2.0"]
    resource_id: Literal["seed-tts-2.0"]
    async_v3: Literal["verified", "candidate", "unavailable"]


class VoiceCatalogEntry(_CatalogModel):
    voice_id: str
    display_name: str
    gender: Literal["female", "male", "unknown"]
    age_group: Literal["child", "youth", "adult", "unknown"]
    official_category: str
    style_tags: tuple[str, ...]
    story_tags: tuple[str, ...]
    compatibility: VoiceCompatibility
    source_url: str
    source_updated_at: str
    source_verified_at: str
    popularity_evidence: str | None = None
    audition_status: Literal["pending"] = "pending"

    @field_validator("voice_id", "display_name", "official_category")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("voice_catalog_blank_field")
        return value


class VoiceCatalog(_CatalogModel):
    schema_version: Literal[1] = 1
    entries: tuple[VoiceCatalogEntry, ...]


class VoiceSelection(_CatalogModel):
    voice: VoiceCatalogEntry
    score: int
    reasons: tuple[str, ...]


def load_voice_catalog(path: Path | None = None) -> VoiceCatalog:
    catalog_path = path or Path(__file__).with_name("data") / "doubao_voice_catalog.json"
    try:
        catalog = VoiceCatalog.model_validate_json(catalog_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VoiceCatalogError("voice_catalog_invalid") from exc
    ids = tuple(entry.voice_id for entry in catalog.entries)
    if not 8 <= len(ids) <= 12 or len(ids) != len(set(ids)):
        raise VoiceCatalogError("voice_catalog_size_or_ids_invalid")
    return catalog


def supported_voice_ids(
    catalog: VoiceCatalog,
    *,
    resource_id: str = "seed-tts-2.0",
    allow_candidate_compatibility: bool = True,
) -> tuple[str, ...]:
    accepted = {"verified"}
    if allow_candidate_compatibility:
        accepted.add("candidate")
    return tuple(
        entry.voice_id
        for entry in catalog.entries
        if entry.compatibility.resource_id == resource_id
        and entry.compatibility.async_v3 in accepted
    )


def select_story_voices(
    catalog: VoiceCatalog,
    *,
    narrator_gender: Literal["female", "male", "any"] = "any",
    desired_story_tags: tuple[str, ...] = ("natural",),
    resource_id: str = "seed-tts-2.0",
    limit: int = 3,
    allow_candidate_compatibility: bool = True,
) -> tuple[VoiceSelection, ...]:
    if not 1 <= limit <= 3:
        raise VoiceCatalogError("voice_selection_limit_invalid")
    desired = set(desired_story_tags)
    eligible = set(
        supported_voice_ids(
            catalog,
            resource_id=resource_id,
            allow_candidate_compatibility=allow_candidate_compatibility,
        )
    )
    ranked: list[VoiceSelection] = []
    for entry in catalog.entries:
        if entry.voice_id not in eligible:
            continue
        if narrator_gender != "any" and entry.gender != narrator_gender:
            continue
        matched = tuple(sorted(desired.intersection(entry.story_tags)))
        score = len(matched) * 10
        if "natural" in entry.story_tags:
            score += 3
        if "audiobook" in entry.story_tags:
            score += 2
        reasons = [f"匹配故事标签：{', '.join(matched)}" if matched else "无额外故事标签命中"]
        reasons.append(f"旁白性别：{entry.gender}")
        if entry.compatibility.async_v3 == "candidate":
            reasons.append("TTS 2.0 音色已核验；seed-tts-2.0 异步端点仍待实际验证")
        else:
            reasons.append("seed-tts-2.0 异步端点兼容性已核验")
        ranked.append(VoiceSelection(voice=entry, score=score, reasons=tuple(reasons)))
    ranked.sort(key=lambda item: (-item.score, item.voice.voice_id))
    return tuple(ranked[:limit])
