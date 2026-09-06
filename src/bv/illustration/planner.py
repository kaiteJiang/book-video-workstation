from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from bv.asr.alignment import AlignedScript
from bv.content.scripts import SemanticLock
from bv.models.contracts import (
    ModelCompletionError,
    PromptAsset,
    StructuredModel,
    empty_request_directory,
    is_redirected,
)
from bv.models.prompts import compose_source_prompt
from bv.production.profile import VisualSequenceMode
from bv.subtitles.generate import SubtitleCue

from .contracts import (
    CharacterBible,
    CharacterLock,
    IllustrationScene,
    IllustrationStoryboard,
    StyleDecision,
)


_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "developer message",
    "忽略以上",
    "忽略前文",
    "系统提示词",
    "开发者消息",
)
_PICTURE_TEXT_MARKERS = (
    "写着",
    "显示文字",
    "可读文字",
    "字幕",
    "书名",
    "假封面",
    "logo",
    "watermark",
    "水印",
)
_FIXED_NEGATIVE_CONSTRAINTS = (
    "禁止画内文字、字母和数字",
    "禁止书名、书中文字页和假书封",
    "禁止Logo和水印",
    "禁止裁切人物头部、手部和关键道具",
)


class IllustrationPlanError(RuntimeError):
    """Stable failure raised by audio-timed illustration planning."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SceneWindow(_PlanModel):
    scene_id: str
    cue_indices: tuple[int, ...]
    start_ms: int
    end_ms: int
    from_frame: int
    to_frame: int


class _CharacterDraft(_PlanModel):
    character_id: str
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


class _SceneDraft(_PlanModel):
    scene_id: str
    key_line: str
    visual_purpose: str
    setting: str
    character_action: str
    metaphor: str | None
    composition: str
    character_refs: tuple[str, ...]
    image_prompt: str
    negative_constraints: tuple[str, ...]
    semantic_turn_offset: int | None
    continuation_action: str | None
    continuation_prompt: str | None
    continuity_constraints: tuple[str, ...] | None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_scene(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        migrated = dict(value)
        for field in (
            "semantic_turn_offset",
            "continuation_action",
            "continuation_prompt",
            "continuity_constraints",
        ):
            migrated.setdefault(field, None)
        return migrated


class _VisualPlanDraft(_PlanModel):
    characters: tuple[_CharacterDraft, ...]
    scenes: tuple[_SceneDraft, ...]
    narrative_deviation_reason: str | None
    legacy_single_character: bool

    @model_validator(mode="before")
    @classmethod
    def _migrate_single_character(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        migrated = dict(value)
        if "character" in migrated and "characters" not in migrated:
            character = migrated.pop("character")
            if isinstance(character, Mapping):
                character = {"character_id": "reader-01", **character}
            migrated["characters"] = [character]
            migrated["legacy_single_character"] = True
        elif "legacy_single_character" not in migrated:
            migrated["legacy_single_character"] = False
        return migrated


def display_width(value: str) -> int:
    """Return a compact display-unit count where two ASCII glyphs equal one CJK glyph."""
    units = 0.0
    for character in value.strip():
        if character.isspace():
            continue
        units += 1.0 if unicodedata.east_asian_width(character) in {"W", "F", "A"} else 0.5
    return math.ceil(units)


def partition_illustration_timeline(
    cues: Sequence[SubtitleCue],
    *,
    master_duration_ms: int,
    fps: int = 30,
    seconds_per_scene_min: float | None = None,
    seconds_per_scene_max: float | None = None,
    target_scene_count: int | None = None,
) -> tuple[SceneWindow, ...]:
    if (
        not isinstance(master_duration_ms, int)
        or isinstance(master_duration_ms, bool)
        or master_duration_ms <= 0
        or fps != 30
        or not cues
    ):
        raise IllustrationPlanError("invalid_illustration_timeline")
    ordered = tuple(cues)
    _validate_cues(ordered, master_duration_ms)
    if (seconds_per_scene_min is None) != (seconds_per_scene_max is None):
        raise IllustrationPlanError("invalid_scene_duration_profile")
    if seconds_per_scene_min is not None and seconds_per_scene_max is not None:
        if (
            isinstance(seconds_per_scene_min, bool)
            or isinstance(seconds_per_scene_max, bool)
            or not math.isfinite(seconds_per_scene_min)
            or not math.isfinite(seconds_per_scene_max)
            or seconds_per_scene_min <= 0
            or seconds_per_scene_max < seconds_per_scene_min
        ):
            raise IllustrationPlanError("invalid_scene_duration_profile")
    if target_scene_count is not None:
        if (
            not isinstance(target_scene_count, int)
            or isinstance(target_scene_count, bool)
            or not 3 <= target_scene_count <= 24
        ):
            raise IllustrationPlanError("invalid_target_scene_count")
        target_count = target_scene_count
    elif seconds_per_scene_min is not None and seconds_per_scene_max is not None:
        target_count = (
            3
            if master_duration_ms <= 20_000
            else _profile_scene_count(
                master_duration_ms / 1000,
                seconds_per_scene_min,
                seconds_per_scene_max,
            )
        )
    else:
        target_count = 3 if master_duration_ms <= 20_000 else min(
            12, max(8, round(master_duration_ms / 5_500))
        )
    if len(ordered) < target_count:
        raise IllustrationPlanError("insufficient_cue_boundaries")

    starts = [0]
    previous = 0
    for scene_number in range(1, target_count):
        remaining_scenes = target_count - scene_number
        lower = previous + 1
        upper = len(ordered) - remaining_scenes
        target_ms = round(master_duration_ms * scene_number / target_count)
        candidates = range(lower, upper + 1)
        boundary = min(
            candidates,
            key=lambda index: (
                _boundary_cost(ordered, index, target_ms),
                index,
            ),
        )
        starts.append(boundary)
        previous = boundary
    starts.append(len(ordered))

    windows: list[SceneWindow] = []
    previous_frame = 0
    for index, (cue_start, cue_end) in enumerate(zip(starts, starts[1:])):
        start_ms = 0 if index == 0 else ordered[cue_start].start_ms
        end_ms = (
            master_duration_ms
            if index == target_count - 1
            else ordered[cue_end].start_ms
        )
        from_frame = previous_frame
        to_frame = (
            round(master_duration_ms * fps / 1000)
            if index == target_count - 1
            else round(end_ms * fps / 1000)
        )
        if end_ms <= start_ms or to_frame <= from_frame:
            raise IllustrationPlanError("invalid_illustration_timeline")
        windows.append(
            SceneWindow(
                scene_id=f"S{index + 1:02d}",
                cue_indices=tuple(cue.index for cue in ordered[cue_start:cue_end]),
                start_ms=start_ms,
                end_ms=end_ms,
                from_frame=from_frame,
                to_frame=to_frame,
            )
        )
        previous_frame = to_frame
    return tuple(windows)


def _profile_scene_count(
    duration_seconds: float,
    low: float,
    high: float,
) -> int:
    minimum_count = math.ceil(duration_seconds / high)
    maximum_count = math.floor(duration_seconds / low)
    if maximum_count < minimum_count or maximum_count < 1:
        raise IllustrationPlanError("invalid_scene_duration_profile")
    target_count = round(duration_seconds / ((low + high) / 2))
    return min(max(target_count, minimum_count), maximum_count)


def _has_source_protagonist(approved_text: str) -> bool:
    return "福贵" in approved_text or re.search(
        r"我叫[\u3400-\u4dbf\u4e00-\u9fff·]{2,8}[。！？!?]",
        approved_text,
    ) is not None


def _allowed_character_ids(approved_text: str) -> tuple[str, ...]:
    identifiers = ["reader-01"]
    if _has_source_protagonist(approved_text):
        identifiers.append("source-protagonist")
    if any(
        name in approved_text
        for name in ("家珍", "凤霞", "有庆", "二喜", "苦根")
    ):
        identifiers.append("source-family-01")
    return tuple(identifiers)


def _visual_allocation(allowed_character_ids: tuple[str, ...]) -> dict[str, str]:
    if "source-protagonist" in allowed_character_ids:
        return {
            "source_protagonist_scenes": "approximately 60% to 80%",
            "current_reader_scenes": "approximately 20% to 40%",
        }
    return {
        "current_reader_scenes": "approximately 60%",
        "source_book_or_controlled_metaphor_scenes": "approximately 40%",
    }


def _representative_indexes(
    scenes: Sequence[_SceneDraft], *, sequence_mode: VisualSequenceMode
) -> set[int]:
    count = len(scenes)
    anchors = (0, count // 2, count - 1)
    if sequence_mode == "color-story-pair":
        return set(anchors)
    protagonist_indexes = tuple(
        index
        for index, scene in enumerate(scenes)
        if "source-protagonist" in scene.character_refs
    )
    if not protagonist_indexes:
        return set(anchors)

    boundaries = (0, math.ceil(count / 3), math.ceil(count * 2 / 3), count)
    selected: set[int] = set()
    for anchor, lower, upper in zip(
        anchors, boundaries[:-1], boundaries[1:], strict=True
    ):
        candidates = tuple(
            index for index in protagonist_indexes if lower <= index < upper
        )
        selected.add(min(candidates, key=lambda index: (abs(index - anchor), index)) if candidates else anchor)
    return selected


def plan_illustrations(
    *,
    book_id: str,
    approved_text: str,
    aligned_script: AlignedScript,
    cues: Sequence[SubtitleCue],
    master_duration_ms: int,
    semantic_lock: SemanticLock,
    style_decision: StyleDecision,
    model: StructuredModel,
    request_root: Path,
    prompt_asset: PromptAsset,
    seconds_per_scene_min: float | None = None,
    seconds_per_scene_max: float | None = None,
    target_scene_count: int | None = None,
    book_title: str | None = None,
    sequence_mode: VisualSequenceMode = "legacy-monochrome-reveal",
) -> tuple[CharacterBible, IllustrationStoryboard]:
    _validate_sources(
        approved_text=approved_text,
        aligned_script=aligned_script,
        cues=cues,
        master_duration_ms=master_duration_ms,
        semantic_lock=semantic_lock,
        style_decision=style_decision,
        prompt_asset=prompt_asset,
    )
    windows = partition_illustration_timeline(
        cues,
        master_duration_ms=master_duration_ms,
        seconds_per_scene_min=seconds_per_scene_min,
        seconds_per_scene_max=seconds_per_scene_max,
        target_scene_count=target_scene_count,
    )
    narration = _narration_windows(approved_text, tuple(cues), windows)
    root = _prepare_request_root(Path(request_root))
    try:
        request_dir = Path(tempfile.mkdtemp(prefix="illustration-plan-", dir=root))
    except OSError:
        raise IllustrationPlanError("unsafe_request_directory") from None
    if (
        not empty_request_directory(request_dir)
        or not request_dir.is_relative_to(root)
        or request_dir == root
    ):
        raise IllustrationPlanError("unsafe_request_directory")

    allowed_character_ids = _allowed_character_ids(approved_text)
    payload: dict[str, object] = {
        "book_title": (book_title or book_id).strip(),
        "approved_text": approved_text,
        "semantic_lock": semantic_lock.model_dump(mode="json"),
        "style_decision": style_decision.model_dump(mode="json"),
        "allowed_character_ids": allowed_character_ids,
        "visual_allocation": _visual_allocation(allowed_character_ids),
        "fixed_windows": [
            {
                **window.model_dump(mode="json"),
                "narration": text,
                "narration_span": span,
            }
            for window, (text, span) in zip(windows, narration, strict=True)
        ],
    }
    if sequence_mode == "color-story-pair":
        payload["sequence_mode"] = sequence_mode
    prompt = compose_source_prompt(
        prompt_asset.text,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    try:
        draft = model.complete(prompt, _VisualPlanDraft, request_dir)
    except ModelCompletionError:
        raise IllustrationPlanError("visual_plan_model_failed") from None
    except Exception:
        raise IllustrationPlanError("visual_plan_model_failed") from None
    if not isinstance(draft, _VisualPlanDraft):
        raise IllustrationPlanError("invalid_visual_plan")
    return _validate_visual_plan(
        draft,
        book_id=book_id,
        approved_text=approved_text,
        aligned_script=aligned_script,
        cues=tuple(cues),
        master_duration_ms=master_duration_ms,
        semantic_lock=semantic_lock,
        style_decision=style_decision,
        windows=windows,
        narration=narration,
        sequence_mode=sequence_mode,
    )


def _validate_visual_plan(
    draft: _VisualPlanDraft,
    *,
    book_id: str,
    approved_text: str,
    aligned_script: AlignedScript,
    cues: tuple[SubtitleCue, ...],
    master_duration_ms: int,
    semantic_lock: SemanticLock,
    style_decision: StyleDecision,
    windows: tuple[SceneWindow, ...],
    narration: tuple[tuple[str, tuple[int, int]], ...],
    sequence_mode: VisualSequenceMode,
) -> tuple[CharacterBible, IllustrationStoryboard]:
    if len(draft.scenes) != len(windows) or any(
        scene.scene_id != window.scene_id
        for scene, window in zip(draft.scenes, windows)
    ):
        raise IllustrationPlanError("visual_plan_window_mismatch")
    if _unsafe_model_output(draft):
        raise IllustrationPlanError("unsafe_visual_plan")
    if sequence_mode == "color-story-pair" and (
        len(windows) not in {3, 4}
        or any(
            scene.semantic_turn_offset is None
            or scene.continuation_action is None
            or scene.continuation_prompt is None
            or scene.continuity_constraints is None
            for scene in draft.scenes
        )
    ):
        raise IllustrationPlanError("invalid_visual_plan")
    character_values = tuple(
        character.model_dump(mode="python") for character in draft.characters
    )
    if not 1 <= len(character_values) <= 3 or any(
        not value.strip()
        for character in character_values
        for value in character.values()
        if isinstance(value, str)
    ):
        raise IllustrationPlanError("invalid_character_lock")
    try:
        characters = tuple(
            CharacterLock(
                **values,
                source_script_sha256=_sha(approved_text),
                style_fingerprint=style_decision.style_fingerprint,
            )
            for values in character_values
        )
        character_bible = CharacterBible(
            source_script_sha256=_sha(approved_text),
            style_fingerprint=style_decision.style_fingerprint,
            characters=characters,
        )
    except ValidationError:
        raise IllustrationPlanError("invalid_character_lock") from None

    allowed_character_refs = {
        character.character_id for character in character_bible.characters
    }
    if draft.legacy_single_character:
        if "史铁生" in approved_text:
            allowed_character_refs.add("source-author")
        if "母亲" in approved_text:
            allowed_character_refs.add("source-mother")
    else:
        if (
            "source-protagonist" in allowed_character_refs
            and not _has_source_protagonist(approved_text)
        ):
            raise IllustrationPlanError("source_character_not_grounded")
        if (
            "source-family-01" in allowed_character_refs
            and not any(
                name in approved_text
                for name in ("家珍", "凤霞", "有庆", "二喜", "苦根")
            )
        ):
            raise IllustrationPlanError("source_character_not_grounded")
        if any(
            {"reader-01", "source-protagonist"}.issubset(
                set(scene.character_refs)
            )
            for scene in draft.scenes
        ):
            raise IllustrationPlanError("character_role_conflation")
    if any(
        not set(scene.character_refs).issubset(allowed_character_refs)
        for scene in draft.scenes
    ):
        raise IllustrationPlanError("unknown_character_reference")
    deviation = (
        draft.narrative_deviation_reason.strip()
        if isinstance(draft.narrative_deviation_reason, str)
        else None
    )
    if "source-protagonist" in allowed_character_refs:
        protagonist_count = sum(
            "source-protagonist" in scene.character_refs for scene in draft.scenes
        )
        protagonist_ratio = protagonist_count / len(draft.scenes)
        if not 0.5 <= protagonist_ratio <= 0.8 and not deviation:
            raise IllustrationPlanError("source_protagonist_ratio_unexplained")
    else:
        reader_count = sum(
            "reader-01" in scene.character_refs for scene in draft.scenes
        )
        reader_ratio = reader_count / len(draft.scenes)
        if not 0.5 <= reader_ratio <= 0.7 and not deviation:
            raise IllustrationPlanError("narrative_ratio_unexplained")

    representative_indexes = _representative_indexes(
        draft.scenes, sequence_mode=sequence_mode
    )
    scenes: list[IllustrationScene] = []
    for index, (scene, window, (scene_narration, span)) in enumerate(
        zip(draft.scenes, windows, narration, strict=True)
    ):
        if not 6 <= display_width(scene.key_line) <= 14:
            raise IllustrationPlanError("invalid_key_line")
        if _visible(scene.key_line) == _visible(scene_narration):
            raise IllustrationPlanError("invalid_key_line")
        negative = tuple(dict.fromkeys((*scene.negative_constraints, *_FIXED_NEGATIVE_CONSTRAINTS)))
        pair_fields: dict[str, object] = {}
        if sequence_mode == "color-story-pair":
            assert scene.semantic_turn_offset is not None
            assert scene.continuation_action is not None
            assert scene.continuation_prompt is not None
            assert scene.continuity_constraints is not None
            semantic_turn_span, semantic_turn_ms, semantic_turn_frame = resolve_semantic_turn(
                scene_span=span,
                semantic_turn_offset=scene.semantic_turn_offset,
                aligned_script=aligned_script,
            )
            pair_fields = {
                "semantic_turn_span": semantic_turn_span,
                "semantic_turn_ms": semantic_turn_ms,
                "semantic_turn_frame": semantic_turn_frame,
                "continuation_action": scene.continuation_action,
                "continuation_prompt": scene.continuation_prompt,
                "continuity_constraints": scene.continuity_constraints,
            }
        try:
            scenes.append(
                IllustrationScene(
                    scene_id=window.scene_id,
                    start_ms=window.start_ms,
                    end_ms=window.end_ms,
                    from_frame=window.from_frame,
                    to_frame=window.to_frame,
                    narration=scene_narration,
                    narration_span=span,
                    key_line=scene.key_line,
                    visual_purpose=scene.visual_purpose,
                    setting=scene.setting,
                    character_action=scene.character_action,
                    metaphor=scene.metaphor,
                    composition=scene.composition,
                    character_refs=scene.character_refs,
                    image_prompt=scene.image_prompt,
                    negative_constraints=negative,
                    representative_frame=index in representative_indexes,
                    asset_status="planned",
                    **pair_fields,
                )
            )
        except ValidationError:
            raise IllustrationPlanError("invalid_visual_plan") from None
    if "".join(scene.narration for scene in scenes) != approved_text:
        raise IllustrationPlanError("approved_narration_mismatch")

    character_sha = _canonical_sha(character_bible.model_dump(mode="json"))
    style_sha = _canonical_sha(style_decision.model_dump(mode="json"))
    subtitle_sha = _canonical_sha(
        [cue.model_dump(mode="json") for cue in cues]
    )
    try:
        storyboard = IllustrationStoryboard(
            book_id=book_id,
            episode_id=semantic_lock.episode_id,
            width=1080,
            height=1920,
            fps=30,
            master_duration_ms=master_duration_ms,
            total_frames=round(master_duration_ms * 30 / 1000),
            script_sha256=_sha(approved_text),
            audio_sha256=aligned_script.report.audio_sha256,
            subtitle_sha256=subtitle_sha,
            style_decision_sha256=style_sha,
            character_lock_sha256=character_sha,
            character_bible_sha256=character_sha,
            narrative_deviation_reason=deviation,
            sequence_mode=sequence_mode,
            scenes=tuple(scenes),
        )
    except ValidationError:
        raise IllustrationPlanError("invalid_visual_plan") from None
    return character_bible, storyboard


def resolve_semantic_turn(
    *,
    scene_span: tuple[int, int],
    semantic_turn_offset: int,
    aligned_script: AlignedScript,
) -> tuple[tuple[int, int], int, int]:
    start, end = scene_span
    absolute_turn_index = start + semantic_turn_offset
    if absolute_turn_index <= start or absolute_turn_index >= end:
        raise IllustrationPlanError("semantic_turn_outside_scene")
    timed = next(
        (
            character
            for character in aligned_script.characters[absolute_turn_index:end]
            if character.start_ms is not None
        ),
        None,
    )
    if timed is None:
        raise IllustrationPlanError("semantic_turn_alignment_missing")
    assert timed.start_ms is not None
    return (
        (start, absolute_turn_index),
        timed.start_ms,
        round(timed.start_ms * 30 / 1000),
    )


def _narration_windows(
    approved_text: str,
    cues: tuple[SubtitleCue, ...],
    windows: tuple[SceneWindow, ...],
) -> tuple[tuple[str, tuple[int, int]], ...]:
    visible_positions = [
        index for index, character in enumerate(approved_text) if not character.isspace()
    ]
    cue_by_index = {cue.index: cue for cue in cues}
    combined = "".join(_visible(cue.text) for cue in cues)
    if combined != _visible(approved_text):
        raise IllustrationPlanError("approved_narration_mismatch")
    result: list[tuple[str, tuple[int, int]]] = []
    consumed = 0
    start = 0
    for index, window in enumerate(windows):
        count = sum(len(_visible(cue_by_index[item].text)) for item in window.cue_indices)
        consumed += count
        end = (
            len(approved_text)
            if index == len(windows) - 1
            else visible_positions[consumed]
        )
        if end <= start:
            raise IllustrationPlanError("approved_narration_mismatch")
        result.append((approved_text[start:end], (start, end)))
        start = end
    if consumed != len(visible_positions) or start != len(approved_text):
        raise IllustrationPlanError("approved_narration_mismatch")
    return tuple(result)


def _validate_sources(
    *,
    approved_text: str,
    aligned_script: AlignedScript,
    cues: Sequence[SubtitleCue],
    master_duration_ms: int,
    semantic_lock: SemanticLock,
    style_decision: StyleDecision,
    prompt_asset: PromptAsset,
) -> None:
    if (
        not isinstance(approved_text, str)
        or not approved_text
        or aligned_script.approved_text != approved_text
        or aligned_script.report.level != "pass"
        or aligned_script.report.violations
        or aligned_script.report.approved_sha256 != _sha(approved_text)
        or "".join(item.character for item in aligned_script.characters) != approved_text
        or not _alignment_is_valid(aligned_script, master_duration_ms)
        or style_decision.source_script_sha256 != _sha(approved_text)
        or _sha(prompt_asset.text) != prompt_asset.sha256
        or not _semantic_lock_is_intact(semantic_lock)
    ):
        raise IllustrationPlanError("invalid_visual_plan_source")
    _validate_cues(tuple(cues), master_duration_ms)


def _alignment_is_valid(aligned: AlignedScript, duration_ms: int) -> bool:
    if len(aligned.characters) != len(aligned.approved_text):
        return False
    previous_start = -1
    for index, character in enumerate(aligned.characters):
        if (
            character.index != index
            or character.start_ms is None
            or character.end_ms is None
            or character.start_ms < 0
            or character.end_ms <= character.start_ms
            or character.start_ms < previous_start
            or character.end_ms > duration_ms
        ):
            return False
        previous_start = character.start_ms
    return True


def _validate_cues(cues: tuple[SubtitleCue, ...], duration_ms: int) -> None:
    if not cues:
        raise IllustrationPlanError("invalid_illustration_timeline")
    previous_end = 0
    for index, cue in enumerate(cues):
        if (
            cue.index != index
            or cue.start_ms < previous_end
            or cue.end_ms <= cue.start_ms
            or cue.end_ms > duration_ms
            or not cue.text
        ):
            raise IllustrationPlanError("invalid_illustration_timeline")
        previous_end = cue.end_ms


def _boundary_cost(cues: tuple[SubtitleCue, ...], index: int, target_ms: int) -> int:
    previous = cues[index - 1]
    following = cues[index]
    punctuation_bonus = 1_200 if previous.text.rstrip().endswith(tuple("。！？!?…")) else 0
    pause_bonus = min(max(following.start_ms - previous.end_ms, 0), 800)
    return abs(following.start_ms - target_ms) - punctuation_bonus - pause_bonus


def _unsafe_model_output(draft: _VisualPlanDraft) -> bool:
    values = _all_strings(draft.model_dump(mode="python"))
    if any(marker in value.casefold() for value in values for marker in _INJECTION_MARKERS):
        return True
    return any(
        marker in scene.image_prompt.casefold()
        for scene in draft.scenes
        for marker in _PICTURE_TEXT_MARKERS
    )


def _all_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(item for child in value.values() for item in _all_strings(child))
    if isinstance(value, (list, tuple)):
        return tuple(item for child in value for item in _all_strings(child))
    return ()


def _semantic_lock_is_intact(lock: SemanticLock) -> bool:
    return _canonical_sha(lock.model_dump(mode="json", exclude={"sha256"})) == lock.sha256


def _prepare_request_root(path: Path) -> Path:
    root = Path(os.path.abspath(path))
    if _redirect_in_chain(root):
        raise IllustrationPlanError("unsafe_request_directory")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise IllustrationPlanError("unsafe_request_directory") from None
    if not root.is_dir() or is_redirected(root) or _redirect_in_chain(root):
        raise IllustrationPlanError("unsafe_request_directory")
    return root


def _redirect_in_chain(path: Path) -> bool:
    current = path
    while True:
        try:
            if current.is_symlink():
                return True
            if current.exists():
                attributes = getattr(current.stat(follow_symlinks=False), "st_file_attributes", 0)
                if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                    return True
        except OSError:
            return True
        if current.parent == current:
            return False
        current = current.parent


def _visible(value: str) -> str:
    return "".join(character for character in value if not character.isspace())


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
