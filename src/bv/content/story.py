from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bv.core.atomic import atomic_write_json
from bv.state.invalidation import invalidate_from
from bv.state.models import ArtifactRef, EpisodeState, StageManifest
from bv.state.store import StateStore
from bv.production.profile import ProductionProfile, load_production_profile


_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class StoryCharacter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str = Field(
        pattern=r"source-[A-Za-z0-9][A-Za-z0-9._-]{0,56}"
    )
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("name", "description")
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("story_character_blank")
        return value.strip()


class StoryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    book_title: str = Field(min_length=1)
    authors: list[str]
    point_of_view: Literal["first", "third"]
    direct_writing: Literal[True]
    selected_passage_reason: str = Field(min_length=1)
    narration_chars_per_minute: int = Field(default=250, ge=120, le=500)
    characters: list[StoryCharacter] = Field(default_factory=list, max_length=24)

    @field_validator("title", "book_title", "selected_passage_reason")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("story_metadata_blank")
        return value.strip()

    @field_validator("authors")
    @classmethod
    def _authors_nonblank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if len(cleaned) != len(values):
            raise ValueError("story_authors_blank")
        return cleaned


class StoryEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    source_title: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    source_type: Literal["primary_text", "publisher", "scholarship", "reputable_secondary"]
    support: str = Field(min_length=1)
    url: str | None = None


class StorySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    beat: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    causal_from_section_ids: list[str] = Field(default_factory=list)
    expression: Literal["narration", "direct_quote", "paraphrase"] = "narration"


class StoryReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    verdict: Literal["supported", "bounded_interpretation"]
    evidence_ids: list[str] = Field(min_length=1)
    note: str = Field(min_length=1)


class StoryFactualReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str = Field(min_length=1)
    findings: list[StoryReviewFinding] = Field(min_length=1)


class StoryCandidateManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["candidate"] = "candidate"
    writing_method: Literal["direct"] = "direct"
    narrative_mode: Literal["story"] = "story"
    candidate_sha256: str
    metadata_sha256: str
    evidence_sha256: str
    sections_sha256: str
    factual_review_sha256: str
    estimated_duration_minutes: float
    warnings: list[str]
    metadata: StoryMetadata
    evidence: list[StoryEvidence]
    sections: list[StorySection]
    factual_review: StoryFactualReview


class StoryApprovalManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["approved"] = "approved"
    script_sha256: str
    candidate_manifest_sha256: str
    metadata_sha256: str
    evidence_sha256: str
    sections_sha256: str
    factual_review_sha256: str


class StoryImportResult(BaseModel):
    candidate_path: Path
    manifest_path: Path
    warnings: list[str]


class StoryApprovalResult(BaseModel):
    approved_path: Path
    manifest_path: Path
    episode_state: EpisodeState


def _json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> ArtifactRef:
    return ArtifactRef(path=str(path), sha256=_file_hash(path), size_bytes=path.stat().st_size)


def _is_redirect(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        attributes = os.lstat(path).st_file_attributes
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except (AttributeError, FileNotFoundError, OSError):
        return False


def _path_chain_redirected(path: Path) -> bool:
    current = Path(path)
    while True:
        if current.exists() and _is_redirect(current):
            return True
        if current.parent == current:
            return False
        current = current.parent


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()


def _validate_contract(
    text: str,
    metadata: StoryMetadata,
    evidence: list[StoryEvidence],
    sections: list[StorySection],
    review: StoryFactualReview,
) -> None:
    if not text.strip():
        raise ValueError("story_text_empty")
    evidence_ids = [item.evidence_id for item in evidence]
    if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("story_evidence_invalid")
    character_ids = [character.character_id for character in metadata.characters]
    if len(character_ids) != len(set(character_ids)):
        raise ValueError("story_characters_invalid")
    for character in metadata.characters:
        if not set(character.evidence_ids) <= set(evidence_ids):
            raise ValueError("story_character_evidence_invalid")
    section_ids = [section.section_id for section in sections]
    if not section_ids or len(section_ids) != len(set(section_ids)):
        raise ValueError("story_sections_invalid")
    cursor = 0
    seen_sections: set[str] = set()
    for section in sections:
        if section.start != cursor or section.end <= section.start or section.end > len(text):
            raise ValueError("story_section_spans_invalid")
        if not set(section.evidence_ids) <= set(evidence_ids):
            raise ValueError("story_section_evidence_invalid")
        if not set(section.causal_from_section_ids) <= seen_sections:
            raise ValueError("story_causal_sequence_invalid")
        cursor = section.end
        seen_sections.add(section.section_id)
    if cursor != len(text):
        raise ValueError("story_section_spans_invalid")
    finding_ids: set[str] = set()
    for finding in review.findings:
        if finding.finding_id in finding_ids or finding.end <= finding.start or finding.end > len(text):
            raise ValueError("story_factual_review_invalid")
        if not set(finding.evidence_ids) <= set(evidence_ids):
            raise ValueError("story_factual_review_evidence_invalid")
        finding_ids.add(finding.finding_id)
    for section in sections:
        if not any(finding.start <= section.start and finding.end >= section.end for finding in review.findings):
            raise ValueError("story_factual_review_incomplete")


def _ensure_story_profile(episode_root: Path, episode: EpisodeState) -> None:
    profile_path = episode_root / "production_profile.json"
    current = load_production_profile(episode_root)
    if current.narrative_mode == "story":
        return
    if profile_path.is_file():
        prior_hash = _file_hash(profile_path)
        history_root = episode_root / "script" / "history"
        history_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            profile_path,
            history_root / f"production-profile-{prior_hash[:16]}.json",
        )
    atomic_write_json(
        profile_path,
        ProductionProfile.longform_story_default().model_dump(mode="json"),
    )
    invalidate_from(episode, "production_profile")


def import_story_candidate(
    *,
    store: StateStore,
    book_id: str,
    episode_id: str,
    text: str,
    metadata: StoryMetadata,
    evidence: list[StoryEvidence],
    sections: list[StorySection],
    factual_review: StoryFactualReview,
    candidate_filename: str = "candidate.txt",
) -> StoryImportResult:
    if not _SAFE_SEGMENT.fullmatch(book_id) or not _SAFE_SEGMENT.fullmatch(episode_id):
        raise ValueError("unsafe_story_identity")
    if (
        Path(candidate_filename).name != candidate_filename
        or not candidate_filename.endswith(".txt")
        or not _SAFE_SEGMENT.fullmatch(candidate_filename)
    ):
        raise ValueError("unsafe_story_candidate_filename")
    if candidate_filename.casefold() == "approved.txt":
        raise ValueError("reserved_story_candidate_filename")
    _validate_contract(text, metadata, evidence, sections, factual_review)
    episode = store.load_episode(book_id, episode_id)
    book = store.load_book(book_id)
    if (
        metadata.book_title.strip().casefold() != book.title.strip().casefold()
        or {author.strip().casefold() for author in metadata.authors}
        != {author.strip().casefold() for author in book.authors}
    ):
        raise ValueError("story_book_identity_mismatch")
    if episode.status not in {"source_imported", "awaiting_script_review", "script_approved"}:
        raise ValueError("story_import_not_ready")

    metadata_payload = metadata.model_dump(mode="json")
    evidence_payload = [item.model_dump(mode="json") for item in evidence]
    sections_payload = [item.model_dump(mode="json") for item in sections]
    review_payload = factual_review.model_dump(mode="json")
    character_count = len(re.sub(r"\s+", "", text))
    duration = character_count / metadata.narration_chars_per_minute
    warnings = ["estimated_duration_over_10_minutes"] if duration > 10 else []
    manifest = StoryCandidateManifest(
        candidate_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        metadata_sha256=_json_hash(metadata_payload),
        evidence_sha256=_json_hash(evidence_payload),
        sections_sha256=_json_hash(sections_payload),
        factual_review_sha256=_json_hash(review_payload),
        estimated_duration_minutes=round(duration, 3),
        warnings=warnings,
        metadata=metadata,
        evidence=evidence,
        sections=sections,
        factual_review=factual_review,
    )
    script_root = store.root / "books" / book_id / "episodes" / episode_id / "script"
    candidate_path = script_root / candidate_filename
    manifest_path = script_root / "story_candidate.json"
    episode_root = script_root.parent
    if any(
        _path_chain_redirected(path)
        for path in (
            store.root,
            episode_root,
            script_root,
            script_root / "history",
            script_root / "archive",
            candidate_path,
            manifest_path,
            episode_root / "production_profile.json",
        )
    ):
        raise ValueError("unsafe_story_output_path")
    _ensure_story_profile(script_root.parent, episode)
    approved_path = script_root / "approved.txt"
    approval_manifest_path = script_root / "story_manifest.json"
    if approved_path.is_file():
        approved_hash = _file_hash(approved_path)
        archive_root = script_root / "archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        os.replace(approved_path, archive_root / f"approved-{approved_hash[:16]}.txt")
        if approval_manifest_path.is_file():
            os.replace(
                approval_manifest_path,
                archive_root / f"story-manifest-{approved_hash[:16]}.json",
            )
    if manifest_path.is_file():
        try:
            previous_ref = episode.stage_manifests["draft_script"].outputs[
                "story_candidate"
            ]
            previous_candidate_path = Path(previous_ref.path)
        except KeyError:
            raise ValueError("story_candidate_integrity_failed")
        if (
            previous_candidate_path.parent.resolve() != script_root.resolve()
            or _path_chain_redirected(previous_candidate_path)
            or not previous_candidate_path.is_file()
        ):
            raise ValueError("story_candidate_integrity_failed")
        try:
            previous = StoryCandidateManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            raise ValueError("story_candidate_integrity_failed") from None
        if (
            previous != manifest
            or _file_hash(previous_candidate_path) != manifest.candidate_sha256
            or previous_candidate_path != candidate_path
        ):
            archive_root = script_root / "archive"
            archive_root.mkdir(parents=True, exist_ok=True)
            previous_candidate_hash = _file_hash(previous_candidate_path)
            previous_manifest_hash = _file_hash(manifest_path)
            os.replace(
                previous_candidate_path,
                archive_root / f"candidate-{previous_candidate_hash[:16]}.txt",
            )
            os.replace(
                manifest_path,
                archive_root / f"candidate-manifest-{previous_manifest_hash[:16]}.json",
            )
    elif candidate_path.exists() and _file_hash(candidate_path) != manifest.candidate_sha256:
        raise ValueError("story_candidate_integrity_failed")
    _atomic_write_text(candidate_path, text)
    atomic_write_json(manifest_path, manifest.model_dump(mode="json"))

    invalidate_from(episode, "script")
    episode.status = "awaiting_script_review"
    episode.script_hash = None
    episode.stage_manifests["draft_script"] = StageManifest(
        stage="draft_script",
        status="completed",
        inputs={
            "writing_method": "direct",
            "narrative_mode": "story",
            "metadata": manifest.metadata_sha256,
            "evidence": manifest.evidence_sha256,
            "sections": manifest.sections_sha256,
            "factual_review": manifest.factual_review_sha256,
        },
        outputs={"story_candidate": _artifact(candidate_path), "story_manifest": _artifact(manifest_path)},
        config_sha256="0" * 64,
    )
    if "draft_script" not in episode.completed_stages:
        episode.completed_stages.append("draft_script")
    episode.stale_stages = [stage for stage in episode.stale_stages if stage != "draft_script"]
    store.save_episode(episode)
    return StoryImportResult(candidate_path=candidate_path, manifest_path=manifest_path, warnings=warnings)


def approve_story_candidate(store: StateStore, book_id: str, episode_id: str) -> StoryApprovalResult:
    episode = store.load_episode(book_id, episode_id)
    if episode.status != "awaiting_script_review":
        raise ValueError("story_approval_not_ready")
    script_root = store.root / "books" / book_id / "episodes" / episode_id / "script"
    candidate_manifest_path = script_root / "story_candidate.json"
    approved_path = script_root / "approved.txt"
    approval_manifest_path = script_root / "story_manifest.json"
    if any(
        _path_chain_redirected(path)
        for path in (
            store.root,
            script_root,
            candidate_manifest_path,
            approved_path,
            approval_manifest_path,
            script_root / "archive",
        )
    ):
        raise ValueError("unsafe_story_output_path")
    try:
        stage = episode.stage_manifests["draft_script"]
        candidate_ref = stage.outputs["story_candidate"]
        candidate_path = Path(candidate_ref.path)
        if (
            candidate_path.parent.resolve() != script_root.resolve()
            or _path_chain_redirected(candidate_path)
            or not candidate_path.name.endswith(".txt")
            or not _SAFE_SEGMENT.fullmatch(candidate_path.name)
        ):
            raise ValueError
        manifest = StoryCandidateManifest.model_validate_json(candidate_manifest_path.read_text(encoding="utf-8"))
        text = candidate_path.read_text(encoding="utf-8")
        if (
            stage.status != "completed"
            or stage.outputs["story_candidate"] != _artifact(candidate_path)
            or stage.outputs["story_manifest"] != _artifact(candidate_manifest_path)
            or manifest.candidate_sha256 != hashlib.sha256(text.encode("utf-8")).hexdigest()
            or manifest.metadata_sha256 != _json_hash(manifest.metadata.model_dump(mode="json"))
            or manifest.evidence_sha256 != _json_hash([item.model_dump(mode="json") for item in manifest.evidence])
            or manifest.sections_sha256 != _json_hash([item.model_dump(mode="json") for item in manifest.sections])
            or manifest.factual_review_sha256 != _json_hash(manifest.factual_review.model_dump(mode="json"))
        ):
            raise ValueError
        _validate_contract(
            text,
            manifest.metadata,
            manifest.evidence,
            manifest.sections,
            manifest.factual_review,
        )
    except (KeyError, OSError, ValueError):
        raise ValueError("story_candidate_integrity_failed") from None

    _atomic_write_text(approved_path, text)
    approval_manifest = StoryApprovalManifest(
        script_sha256=manifest.candidate_sha256,
        candidate_manifest_sha256=_file_hash(candidate_manifest_path),
        metadata_sha256=manifest.metadata_sha256,
        evidence_sha256=manifest.evidence_sha256,
        sections_sha256=manifest.sections_sha256,
        factual_review_sha256=manifest.factual_review_sha256,
    )
    atomic_write_json(approval_manifest_path, approval_manifest.model_dump(mode="json"))
    if episode.script_hash != manifest.candidate_sha256:
        invalidate_from(episode, "script")
    episode.script_hash = manifest.candidate_sha256
    episode.status = "script_approved"
    store.save_episode(episode)
    return StoryApprovalResult(
        approved_path=approved_path,
        manifest_path=approval_manifest_path,
        episode_state=episode,
    )
