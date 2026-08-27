from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from bv.models.contracts import (
    ModelCompletionError,
    PromptAsset,
    ReviewSkipped,
    StructuredModel,
    empty_request_directory,
    is_redirected,
)
from bv.models.prompts import compose_source_prompt
from bv.storyboard.generate import ReaderValueContract

from .contracts import StyleDecision


_LIBRARY_VERSION = "1"


class StyleSelectionError(RuntimeError):
    """Stable failure raised by bounded illustration style selection."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _StyleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StyleReference(_StyleModel):
    path: str
    role: str


class StyleRecipe(_StyleModel):
    style_id: str
    order: int = Field(gt=0)
    name_zh: str
    group: str
    best_for: tuple[str, ...]
    summary: str
    caption_prompt: str
    color_hint: str
    avoid: str
    reference_images: tuple[StyleReference, ...]
    prompt_blocks: tuple[str, ...] = ()
    example_image: str | None = None

    @field_validator(
        "style_id", "name_zh", "group", "summary", "caption_prompt", "color_hint", "avoid"
    )
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("blank_style_field")
        return value.strip()


class NarrativeProfile(_StyleModel):
    narrative_types: tuple[str, ...]
    emotional_temperature: Literal["温暖", "克制", "沉重", "轻快"]
    abstraction: Literal["现实", "混合", "隐喻"]
    relationships: tuple[str, ...]
    era: str
    target_reader: str


class RankedStyle(_StyleModel):
    style_id: str
    order: int = Field(gt=0)
    score: int
    recipe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    matched_dimensions: tuple[str, ...]
    rejected_dimensions: tuple[str, ...]


class _RejectedStyleReason(_StyleModel):
    style_id: str
    reason: str


class _StyleSelectionDraft(_StyleModel):
    selected_style: str
    selection_reasons: tuple[str, ...]
    rejected_reasons: tuple[_RejectedStyleReason, ...]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("rejected_reasons", mode="before")
    @classmethod
    def _normalize_rejected_reasons(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return [
                {"style_id": style_id, "reason": reason}
                for style_id, reason in value.items()
            ]
        return value


def load_style_catalog(path: Path) -> tuple[StyleRecipe, ...]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        styles = payload["styles"]
        if not isinstance(styles, list):
            raise TypeError
        catalog = tuple(
            StyleRecipe.model_validate(
                {
                    "style_id": item["id"],
                    "order": item["order"],
                    "name_zh": item["name_zh"],
                    "group": item["group"],
                    "best_for": item["best_for"],
                    "summary": item["summary"],
                    "caption_prompt": item["caption_prompt"],
                    "color_hint": item["color_hint"],
                    "avoid": item["avoid"],
                    "reference_images": item.get("reference_images", []),
                    "prompt_blocks": item.get("prompt_blocks", []),
                    "example_image": item.get("example_image"),
                }
            )
            for item in styles
        )
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, ValidationError):
        raise StyleSelectionError("invalid_style_catalog") from None
    if not catalog or len({item.style_id for item in catalog}) != len(catalog):
        raise StyleSelectionError("invalid_style_catalog")
    orders = [item.order for item in catalog]
    if len(set(orders)) != len(orders):
        raise StyleSelectionError("invalid_style_catalog")
    return tuple(sorted(catalog, key=lambda item: item.order))


def build_narrative_profile(
    approved_script: str,
    value_contract: ReaderValueContract,
) -> NarrativeProfile:
    if not isinstance(approved_script, str) or not approved_script.strip():
        raise StyleSelectionError("invalid_style_source")
    source = " ".join(
        (
            approved_script,
            value_contract.target_reader,
            value_contract.reader_before,
            value_contract.central_life_tension,
            value_contract.value_thesis,
            value_contract.life_connection,
            value_contract.reader_after,
        )
    )
    narrative_types: list[str] = []
    for label, words in (
        ("关系", ("关系", "伴侣", "爱人", "婚姻", "父母", "孩子", "家人", "朋友")),
        ("感悟", ("人生", "生活", "选择", "责任", "自己", "成长", "价值")),
        ("故事", ("故事", "经历", "曾经", "后来", "那天", "回忆")),
        ("知识", ("方法", "步骤", "原理", "因果", "研究", "概念")),
    ):
        if any(word in source for word in words):
            narrative_types.append(label)
    if not narrative_types:
        narrative_types.append("感悟")

    if any(word in source for word in ("沉重", "创伤", "死亡", "绝望", "失去")):
        temperature: Literal["温暖", "克制", "沉重", "轻快"] = "沉重"
    elif any(word in source for word in ("轻松", "幽默", "好笑", "快乐", "活泼")):
        temperature = "轻快"
    elif any(word in source for word in ("治愈", "温暖", "陪伴", "希望", "拥抱")):
        temperature = "温暖"
    else:
        temperature = "克制"

    metaphor_hits = sum(source.count(word) for word in ("像", "仿佛", "隐喻", "寓言", "梦境", "象征"))
    reality_hits = sum(source.count(word) for word in ("成年人", "关系", "生活", "工作", "家庭", "选择", "责任"))
    if metaphor_hits and reality_hits:
        abstraction: Literal["现实", "混合", "隐喻"] = "混合"
    elif metaphor_hits:
        abstraction = "隐喻"
    else:
        abstraction = "现实"

    relationships = tuple(
        label
        for label, words in (
            ("伴侣", ("伴侣", "爱人", "婚姻", "恋人")),
            ("亲子", ("孩子", "父母", "亲子")),
            ("家庭", ("家人", "家庭")),
            ("朋友", ("朋友", "友谊")),
            ("同事", ("同事", "职场", "老板")),
        )
        if any(word in source for word in words)
    )
    era = "历史" if any(word in source for word in ("古代", "历史", "王朝", "民国")) else "现代"
    return NarrativeProfile(
        narrative_types=tuple(narrative_types),
        emotional_temperature=temperature,
        abstraction=abstraction,
        relationships=relationships,
        era=era,
        target_reader=value_contract.target_reader.strip(),
    )


def rank_style_candidates(
    profile: NarrativeProfile,
    catalog: Sequence[StyleRecipe],
) -> tuple[RankedStyle, ...]:
    if not catalog:
        raise StyleSelectionError("invalid_style_catalog")
    ranked = [_score_style(profile, style) for style in catalog]
    return tuple(sorted(ranked, key=lambda item: (-item.score, item.order)))


def style_candidates_with_override(
    ranked: Sequence[RankedStyle],
    style_id: str,
) -> tuple[RankedStyle, RankedStyle, RankedStyle]:
    selected = next((item for item in ranked if item.style_id == style_id), None)
    alternatives = tuple(item for item in ranked if item.style_id != style_id)[:2]
    if selected is None or len(alternatives) != 2:
        raise StyleSelectionError("style_not_in_catalog")
    return selected, alternatives[0], alternatives[1]


def select_style(
    profile: NarrativeProfile,
    approved_script: str,
    value_contract: ReaderValueContract,
    candidates: tuple[RankedStyle, RankedStyle, RankedStyle],
    model: StructuredModel,
    request_root: Path,
    prompt_asset: PromptAsset,
    manual_override: str | None = None,
    selection_context_script: str | None = None,
) -> StyleDecision:
    candidate_ids = tuple(item.style_id for item in candidates)
    if len(candidates) != 3 or len(set(candidate_ids)) != 3:
        raise StyleSelectionError("invalid_style_candidates")
    if not approved_script.strip() or _sha(prompt_asset.text) != prompt_asset.sha256:
        raise StyleSelectionError("invalid_style_source")

    if manual_override is not None:
        if manual_override not in candidate_ids:
            raise StyleSelectionError("style_not_in_candidates")
        return _decision(
            selected_style=manual_override,
            candidate_ids=candidate_ids,
            selection_reasons=("用户在三个已验证候选中手动指定",),
            rejected_reasons={
                item: "未采用，用户已明确指定其他候选"
                for item in candidate_ids
                if item != manual_override
            },
            confidence=1.0,
            approved_script=approved_script,
            manual_override=True,
            recipe_sha256=next(
                item.recipe_sha256 for item in candidates
                if item.style_id == manual_override
            ),
        )

    root = _prepare_request_root(Path(request_root))
    try:
        request_dir = Path(tempfile.mkdtemp(prefix="style-select-", dir=root))
    except OSError:
        raise StyleSelectionError("unsafe_request_directory") from None
    if (
        not empty_request_directory(request_dir)
        or not request_dir.is_relative_to(root)
        or request_dir == root
    ):
        raise StyleSelectionError("unsafe_request_directory")

    context_script = (
        approved_script if selection_context_script is None else selection_context_script
    )
    if not isinstance(context_script, str) or not context_script.strip():
        raise StyleSelectionError("invalid_style_source")
    source = {
        "approved_script": context_script,
        "reader_value_contract": value_contract.model_dump(mode="json"),
        "narrative_profile": profile.model_dump(mode="json"),
        "candidates": [item.model_dump(mode="json") for item in candidates],
    }
    prompt = compose_source_prompt(
        prompt_asset.text,
        json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    try:
        response = model.complete(prompt, _StyleSelectionDraft, request_dir)
    except ModelCompletionError:
        raise StyleSelectionError("style_model_failed") from None
    except Exception:
        raise StyleSelectionError("style_model_failed") from None
    if isinstance(response, ReviewSkipped) or not isinstance(response, _StyleSelectionDraft):
        raise StyleSelectionError("invalid_style_response")
    if response.selected_style not in candidate_ids:
        raise StyleSelectionError("style_not_in_candidates")
    rejected_reasons = {
        item.style_id: item.reason for item in response.rejected_reasons
    }
    expected_rejected = set(candidate_ids) - {response.selected_style}
    if (
        not response.selection_reasons
        or any(not reason.strip() for reason in response.selection_reasons)
        or len(rejected_reasons) != len(response.rejected_reasons)
        or set(rejected_reasons) != expected_rejected
        or any(not reason.strip() for reason in rejected_reasons.values())
    ):
        raise StyleSelectionError("invalid_style_reasons")
    return _decision(
        selected_style=response.selected_style,
        candidate_ids=candidate_ids,
        selection_reasons=response.selection_reasons,
        rejected_reasons=rejected_reasons,
        confidence=response.confidence,
        approved_script=approved_script,
        manual_override=False,
        recipe_sha256=next(
            item.recipe_sha256 for item in candidates
            if item.style_id == response.selected_style
        ),
    )


def _score_style(profile: NarrativeProfile, style: StyleRecipe) -> RankedStyle:
    source = " ".join(
        (style.name_zh, style.group, *style.best_for, style.summary, style.color_hint)
    )
    dimensions: list[tuple[str, tuple[str, ...], int]] = []
    for narrative_type in profile.narrative_types:
        keywords = {
            "关系": ("关系", "情感", "家庭", "亲情", "群像"),
            "感悟": ("人生感悟", "感悟", "宁静叙事", "内心独白", "情绪"),
            "故事": ("故事", "剧情", "角色", "纪实", "叙事"),
            "知识": ("知识", "讲解", "流程", "观点", "科普", "因果", "教程"),
        }.get(narrative_type, (narrative_type,))
        dimensions.append((narrative_type, keywords, 7))
    dimensions.append(
        (
            profile.emotional_temperature,
            {
                "克制": ("克制", "安静", "宁静", "纪实", "留白"),
                "温暖": ("温暖", "暖光", "治愈", "亲情", "暖色"),
                "沉重": ("社会议题", "有力量", "催泪", "历史", "版画"),
                "轻快": ("轻松", "轻喜剧", "幽默", "活泼", "病毒感"),
            }[profile.emotional_temperature],
            8,
        )
    )
    dimensions.append(
        (
            profile.abstraction,
            {
                "现实": ("纪实", "生活", "人物", "日常", "城市"),
                "混合": ("概念", "寓言", "情绪", "故事"),
                "隐喻": ("寓言", "梦境", "概念", "象征"),
            }[profile.abstraction],
            4,
        )
    )
    if profile.relationships:
        dimensions.append(
            ("关系", ("关系", "人物关系", "家庭", "亲情", "群像", "情感"), 6)
        )
    if profile.era == "历史":
        dimensions.append(("时代", ("历史", "传统", "复古", "民国"), 4))

    matched: list[str] = []
    rejected: list[str] = []
    score = 0
    for label, keywords, weight in dimensions:
        if any(keyword in source for keyword in keywords):
            score += weight
            if label not in matched:
                matched.append(label)
        elif label not in rejected:
            rejected.append(label)
    return RankedStyle(
        style_id=style.style_id,
        order=style.order,
        score=score,
        recipe_sha256=_canonical_sha(style.model_dump(mode="json")),
        matched_dimensions=tuple(matched),
        rejected_dimensions=tuple(rejected),
    )


def _decision(
    *,
    selected_style: str,
    candidate_ids: tuple[str, str, str],
    selection_reasons: tuple[str, ...],
    rejected_reasons: Mapping[str, str],
    confidence: float,
    approved_script: str,
    manual_override: bool,
    recipe_sha256: str,
) -> StyleDecision:
    fingerprint_payload = {
        "library_version": _LIBRARY_VERSION,
        "selected_style": selected_style,
        "recipe_sha256": recipe_sha256,
    }
    return StyleDecision(
        library_version=_LIBRARY_VERSION,
        selected_style=selected_style,
        candidate_styles=candidate_ids,
        selection_reasons=selection_reasons,
        rejected_reasons=dict(rejected_reasons),
        confidence=confidence,
        manual_override=manual_override,
        source_script_sha256=_sha(approved_script),
        style_fingerprint=_canonical_sha(fingerprint_payload),
    )


def _prepare_request_root(path: Path) -> Path:
    root = Path(os.path.abspath(path))
    if _redirect_in_chain(root):
        raise StyleSelectionError("unsafe_request_directory")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise StyleSelectionError("unsafe_request_directory") from None
    if not root.is_dir() or is_redirected(root) or _redirect_in_chain(root):
        raise StyleSelectionError("unsafe_request_directory")
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


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
