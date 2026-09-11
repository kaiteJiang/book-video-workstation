"""Read-only, hash-bound projection of an approved story for media stages."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

from bv.content.scripts import SemanticLock
from bv.content.story import (
    StoryApprovalManifest,
    StoryCandidateManifest,
    StoryCharacter,
    StorySection,
    _validate_contract,
)
from bv.production.profile import load_production_profile
from bv.state.models import EpisodeState


class StoryVisualContextError(ValueError):
    pass


@dataclass(frozen=True)
class StoryVisualContext:
    semantic_lock: SemanticLock
    sections: tuple[StorySection, ...]
    important_events: tuple[str, ...]
    point_of_view: str
    source_contract_sha256: str
    characters: tuple[StoryCharacter, ...]


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read(root: Path, path: Path) -> bytes:
    if not path.absolute().is_relative_to(root.absolute()):
        raise StoryVisualContextError("story_visual_source_invalid")
    for part in (path, *path.parents):
        if os.path.lexists(part):
            attrs = getattr(part.stat(follow_symlinks=False), "st_file_attributes", 0)
            if part.is_symlink() or attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise StoryVisualContextError("story_visual_source_invalid")
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise StoryVisualContextError("story_visual_source_invalid")
    return path.read_bytes()


def load_approved_story_visual_context(
    episode_root: Path,
    *,
    book_id: str,
    episode_id: str,
    approved_text: str,
    approved_sha256: str,
) -> StoryVisualContext | None:
    """Return None for legacy profiles only; corrupt story input never falls back."""
    root = Path(episode_root)
    if load_production_profile(root).narrative_mode != "story":
        return None
    try:
        script = root / "script"
        candidate_bytes = _read(root, script / "story_candidate.json")
        approval_bytes = _read(root, script / "story_manifest.json")
        state = EpisodeState.model_validate_json(_read(root, root / "episode.json"))
        approved_bytes = _read(root, script / "approved.txt")
        candidate = StoryCandidateManifest.model_validate_json(candidate_bytes)
        approval = StoryApprovalManifest.model_validate_json(approval_bytes)
        candidate_digest = hashlib.sha256(candidate_bytes).hexdigest()
        stage = state.stage_manifests["draft_script"]
        current_ref = stage.outputs["story_candidate"]
        current_path = Path(current_ref.path)
        manifest_ref = stage.outputs["story_manifest"]
        if (
            state.book_id != book_id or state.episode_id != episode_id
            or state.script_hash != approved_sha256
            or state.status in {"source_imported", "awaiting_script_review"}
            or stage.status != "completed"
            or hashlib.sha256(approved_bytes).hexdigest() != approved_sha256
            or approved_bytes.decode("utf-8") != approved_text
            or candidate.candidate_sha256 != approved_sha256
            or approval.script_sha256 != approved_sha256
            or approval.candidate_manifest_sha256 != candidate_digest
            or manifest_ref.sha256 != candidate_digest
            or Path(manifest_ref.path).absolute() != (script / "story_candidate.json").absolute()
            or current_path.parent.absolute() != script.absolute()
            or current_ref.sha256 != approved_sha256
            or hashlib.sha256(_read(root, current_path)).hexdigest() != approved_sha256
        ):
            raise ValueError("story_visual_source_invalid")
        components = {
            "metadata": candidate.metadata.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in candidate.evidence],
            "sections": [item.model_dump(mode="json") for item in candidate.sections],
            "factual_review": candidate.factual_review.model_dump(mode="json"),
        }
        for name, value in components.items():
            digest = _hash(value)
            if getattr(candidate, name + "_sha256") != digest or getattr(approval, name + "_sha256") != digest:
                raise ValueError("story_visual_source_invalid")
        _validate_contract(approved_text, candidate.metadata, candidate.evidence, candidate.sections, candidate.factual_review)
        known_evidence = {item.evidence_id for item in candidate.evidence}
        if any(not set(item.evidence_ids) <= known_evidence for item in candidate.metadata.characters):
            raise ValueError("story_visual_source_invalid")
        sections = tuple(candidate.sections)
        sources = {item.evidence_id: item for item in candidate.evidence}
        fields = dict(
            episode_id=episode_id,
            topic_identity_sha256=candidate.metadata_sha256,
            value_thesis=candidate.metadata.selected_passage_reason,
            target_reader="跟随书中人物经历关键选择与后果的观众",
            reader_before=sections[0].beat,
            reader_after=sections[-1].beat,
            central_life_tension=candidate.metadata.title,
            life_connection="按已批准故事的角色、事件和时间线呈现；不插入现代读者替身或虚构亲历。",
            required_cluster_ids=tuple(item.section_id for item in sections),
            allowed_claim_ids_by_cluster={item.section_id: tuple(item.evidence_ids) for item in sections},
            chapter_regions_by_cluster={item.section_id: tuple(sources[eid].source_locator for eid in item.evidence_ids) for item in sections},
            cluster_coverage_terms={item.section_id: item.beat for item in sections},
            practical_boundary="人物转述与解释遵守已批准的逐段事实复查；不得新增对白、事件或让人物提前知道后续信息。",
            reading_reason=candidate.metadata.selected_passage_reason,
            allowed_numbers=(),
            allowed_negations=(),
        )
        lock = SemanticLock(**fields, sha256=_hash(fields))
        return StoryVisualContext(
            semantic_lock=lock,
            sections=sections,
            important_events=tuple(item.beat for item in sections),
            point_of_view=candidate.metadata.point_of_view,
            source_contract_sha256=_hash({"candidate": candidate_digest, "approval": hashlib.sha256(approval_bytes).hexdigest()}),
            characters=tuple(candidate.metadata.characters),
        )
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        raise StoryVisualContextError("story_visual_source_invalid") from None
