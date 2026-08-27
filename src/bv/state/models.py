from typing import Literal, Mapping

from pydantic import AliasChoices, BaseModel, Field, model_validator


SourceMode = Literal["file", "title_author"]

StageStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "stale",
]

EpisodeStatus = Literal[
    "source_imported",
    "source_invalid",
    "source_parsed",
    "chapter_analysis_running",
    "chapter_analysis_failed",
    "book_analysis_ready",
    "topic_selected",
    "insufficient_distinct_topic",
    "script_drafting",
    "evidence_invalid",
    "awaiting_script_review",
    "script_approved",
    "awaiting_voice_audition_review",
    "voice_profile_approved",
    "voice_reference_required",
    "tts_running",
    "tts_failed",
    "tts_duration_out_of_range",
    "tts_ready",
    "asr_not_configured",
    "asr_running",
    "asr_failed",
    "tts_asr_mismatch",
    "audio_ready",
    "illustration_planning",
    "style_selection_failed",
    "illustration_plan_failed",
    "illustration_plan_ready",
    "representative_generation_running",
    "representative_generation_failed",
    "awaiting_representative_review",
    "representatives_rejected",
    "representatives_approved",
    "illustration_batch_running",
    "illustration_batch_failed",
    "illustrations_ready",
    "visual_render_running",
    "visual_render_failed",
    "visual_render_ready",
    "storyboard_ready",
    "awaiting_video_generation",
    "video_import_invalid",
    "visual_sample_ready",
    "video_partial",
    "video_imported",
    "render_running",
    "render_failed",
    "awaiting_final_review",
    "final_approved",
]


class ArtifactRef(BaseModel):
    path: str
    sha256: str
    size_bytes: int


class StageManifest(BaseModel):
    stage: str
    status: StageStatus
    inputs: dict[str, str]
    outputs: dict[str, ArtifactRef]
    config_sha256: str
    prompt_sha256: str | None = None
    attempt: int = 1
    error_code: str | None = None


class BookState(BaseModel):
    book_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    source_mode: SourceMode = "file"
    source_ids: list[str] = Field(default_factory=list)


_LEGACY_EPISODE_STATUSES = {
    "awaiting_h3_generation": "awaiting_video_generation",
    "h3_import_invalid": "video_import_invalid",
    "h3_partial": "video_partial",
    "h3_imported": "video_imported",
}
_LEGACY_VIDEO_STAGE = "await_h3_segments"
_NEUTRAL_VIDEO_STAGE = "await_video_segments"


def _migrate_stage_name(name: object) -> object:
    return _NEUTRAL_VIDEO_STAGE if name == _LEGACY_VIDEO_STAGE else name


def _migrate_stage_list(values: object) -> object:
    if not isinstance(values, list):
        return values
    migrated: list[object] = []
    for item in values:
        name = _migrate_stage_name(item)
        if any(existing == name for existing in migrated):
            continue
        migrated.append(name)
    return migrated


def _migrate_manifest_payload(value: object) -> object:
    if isinstance(value, StageManifest):
        if value.stage == _LEGACY_VIDEO_STAGE:
            return value.model_copy(update={"stage": _NEUTRAL_VIDEO_STAGE})
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        if payload.get("stage") == _LEGACY_VIDEO_STAGE:
            payload["stage"] = _NEUTRAL_VIDEO_STAGE
        return payload
    return value


def _manifest_identity(value: object) -> object:
    if isinstance(value, StageManifest):
        payload = value.model_dump()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        return value
    if payload.get("stage") == _LEGACY_VIDEO_STAGE:
        payload["stage"] = _NEUTRAL_VIDEO_STAGE
    return payload


def _normalize_manifest_mapping(raw_manifests: Mapping[object, object]) -> dict[object, object]:
    collapsed: dict[object, object] = {}
    for key, value in raw_manifests.items():
        new_key = _migrate_stage_name(key)
        new_value = _migrate_manifest_payload(value)
        if new_key in collapsed:
            if _manifest_identity(collapsed[new_key]) != _manifest_identity(new_value):
                raise ValueError("conflicting_video_stage_manifests")
            continue
        collapsed[new_key] = new_value
    return collapsed


def _maybe_normalize_manifests(value: object) -> object:
    if isinstance(value, Mapping):
        return _normalize_manifest_mapping(value)
    return value


def _normalized_manifests_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if len(left) != len(right):
            return False
        for key, value in left.items():
            if key not in right:
                return False
            if _manifest_identity(value) != _manifest_identity(right[key]):
                return False
        return True
    return left == right


class EpisodeState(BaseModel):
    book_id: str
    episode_id: str
    status: EpisodeStatus = "source_imported"
    stage_manifests: dict[str, StageManifest] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("stage_manifests", "manifests"),
    )
    completed_stages: list[str] = Field(default_factory=list)
    stale_stages: list[str] = Field(default_factory=list)
    script_hash: str | None = Field(
        default=None,
        validation_alias=AliasChoices("script_hash", "script_sha256"),
    )
    voice_id: str | None = None
    topic_id: str | None = None
    failure_summary: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_video_fields(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        migrated = dict(data)
        status = migrated.get("status")
        if isinstance(status, str) and status in _LEGACY_EPISODE_STATUSES:
            migrated["status"] = _LEGACY_EPISODE_STATUSES[status]
        has_stage_manifests = "stage_manifests" in migrated
        has_manifests_alias = "manifests" in migrated
        if has_stage_manifests and has_manifests_alias:
            normalized_stage = _maybe_normalize_manifests(migrated["stage_manifests"])
            normalized_alias = _maybe_normalize_manifests(migrated["manifests"])
            if not _normalized_manifests_equal(normalized_stage, normalized_alias):
                raise ValueError("conflicting_video_stage_manifests")
            migrated["stage_manifests"] = normalized_stage
            migrated.pop("manifests", None)
        elif has_stage_manifests:
            migrated["stage_manifests"] = _maybe_normalize_manifests(
                migrated["stage_manifests"]
            )
        elif has_manifests_alias:
            migrated["stage_manifests"] = _maybe_normalize_manifests(migrated["manifests"])
            migrated.pop("manifests", None)
        if "completed_stages" in migrated:
            migrated["completed_stages"] = _migrate_stage_list(migrated["completed_stages"])
        if "stale_stages" in migrated:
            migrated["stale_stages"] = _migrate_stage_list(migrated["stale_stages"])
        return migrated

    @property
    def manifests(self) -> dict[str, StageManifest]:
        return self.stage_manifests

    @property
    def script_sha256(self) -> str | None:
        return self.script_hash
