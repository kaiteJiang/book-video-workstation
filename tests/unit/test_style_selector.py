from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bv.models.contracts import PromptAsset
from bv.storyboard.generate import ReaderValueContract
from bv.illustration.style_selector import (
    NarrativeProfile,
    RankedStyle,
    StyleRecipe,
    StyleSelectionError,
    build_narrative_profile,
    load_style_catalog,
    rank_style_candidates,
    select_style,
    style_candidates_with_override,
)


CATALOG = Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json")
SCRIPT = (
    "很多成年人在关系里反复解释，不是因为不知道答案，而是害怕自己的选择不被理解。"
    "这本书让人重新看见：生活不是赢得所有人的认可，而是分清哪些责任该由自己承担。"
    "当你不再拿伴侣和旁人的评价替代判断，才有可能把日子过回自己手里。"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _value_contract() -> ReaderValueContract:
    return ReaderValueContract(
        target_reader="正在关系中内耗的成年人",
        reader_before="总想靠解释换来伴侣和旁人的理解",
        central_life_tension="想忠于自己，又害怕关系因此破裂",
        value_thesis="把他人的评价和自己的责任重新分开",
        life_connection="从反复解释转向承担自己的选择",
        reader_after="允许关系有分歧，也能保留自己的判断",
        practical_boundary="提供理解关系的视角，不替代专业咨询",
        required_cluster_ids=("problem", "reframe", "application"),
        cluster_coverage_terms={
            "problem": "关系里的解释和内耗",
            "reframe": "评价与责任的边界",
            "application": "承担自己的选择",
        },
        allowed_claim_ids_by_cluster={
            "problem": ("C01",),
            "reframe": ("C02",),
            "application": ("C03",),
        },
        source_semantic_lock_sha256="a" * 64,
        sha256="b" * 64,
    )


def _prompt() -> PromptAsset:
    text = "BV_ILLUSTRATION_STYLE_SELECT_V1"
    return PromptAsset(name="style_select", text=text, sha256=_sha(text))


class ScriptedStructuredModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Path]] = []

    def complete(self, prompt: str, schema_type: type, request_dir: Path):
        self.calls.append((prompt, request_dir))
        response = self.responses.pop(0)
        return schema_type.model_validate(response)


def _profile() -> NarrativeProfile:
    return NarrativeProfile(
        narrative_types=("关系", "感悟"),
        emotional_temperature="克制",
        abstraction="现实",
        relationships=("伴侣",),
        era="现代",
        target_reader="正在关系中内耗的成年人",
    )


def _candidates() -> tuple[RankedStyle, RankedStyle, RankedStyle]:
    ranked = rank_style_candidates(_profile(), load_style_catalog(CATALOG))
    return ranked[0], ranked[1], ranked[2]


def test_catalog_loads_all_twenty_pinned_styles() -> None:
    catalog = load_style_catalog(CATALOG)

    assert len(catalog) == 20
    assert len({style.style_id for style in catalog}) == 20
    assert [style.order for style in catalog] == list(range(1, 21))


def test_fixed_style_is_injected_ahead_of_ranked_alternatives() -> None:
    ranked = rank_style_candidates(_profile(), load_style_catalog(CATALOG))

    candidates = style_candidates_with_override(
        ranked,
        "retro-gouache-concept",
    )

    assert len(candidates) == 3
    assert candidates[0].style_id == "retro-gouache-concept"
    assert len({candidate.style_id for candidate in candidates}) == 3


def test_style_selection_response_schema_uses_fixed_rejection_items(tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []

    class CapturingModel(ScriptedStructuredModel):
        def complete(self, prompt: str, schema_type: type, request_dir: Path):
            captured.append(schema_type.model_json_schema())
            return super().complete(prompt, schema_type, request_dir)

    candidates = _candidates()
    selected = candidates[0].style_id
    model = CapturingModel([{
        "selected_style": selected,
        "selection_reasons": ["匹配"],
        "rejected_reasons": {
            item.style_id: "不如首选贴合" for item in candidates[1:]
        },
        "confidence": 0.9,
    }])

    select_style(
        _profile(), SCRIPT, _value_contract(), candidates, model,
        tmp_path / "requests", _prompt(),
    )

    rejected = captured[0]["properties"]["rejected_reasons"]
    assert rejected["type"] == "array"


def test_emotional_relationship_script_shortlists_watercolor_before_whiteboard() -> None:
    ranked = rank_style_candidates(_profile(), load_style_catalog(CATALOG))

    assert ranked[0].style_id == "emotional-watercolor-sketch"
    assert "whiteboard-explainer" not in [item.style_id for item in ranked[:3]]
    assert "克制" in ranked[0].matched_dimensions
    assert "关系" in ranked[0].matched_dimensions


def test_equal_scores_break_by_upstream_order() -> None:
    recipes = tuple(
        StyleRecipe(
            style_id=f"style-{order}",
            order=order,
            name_zh=f"风格{order}",
            group="未匹配",
            best_for=("未匹配",),
            summary="未匹配",
            caption_prompt="无",
            color_hint="无",
            avoid="无",
            reference_images=(),
        )
        for order in (3, 1, 2)
    )

    ranked = rank_style_candidates(_profile(), recipes)

    assert [item.order for item in ranked] == [1, 2, 3]


def test_profile_uses_whole_script_and_reader_value_contract() -> None:
    profile = build_narrative_profile(SCRIPT, _value_contract())

    assert "关系" in profile.narrative_types
    assert "感悟" in profile.narrative_types
    assert profile.emotional_temperature == "克制"
    assert profile.abstraction == "现实"
    assert "伴侣" in profile.relationships
    assert profile.target_reader == "正在关系中内耗的成年人"


def test_model_can_choose_only_from_three_candidates(tmp_path: Path) -> None:
    model = ScriptedStructuredModel(
        [{
            "selected_style": "kid-crayon",
            "selection_reasons": ["看起来活泼"],
            "rejected_reasons": {},
            "confidence": 0.8,
        }]
    )

    with pytest.raises(StyleSelectionError, match="style_not_in_candidates"):
        select_style(
            profile=_profile(),
            approved_script=SCRIPT,
            value_contract=_value_contract(),
            candidates=_candidates(),
            model=model,
            request_root=tmp_path / "requests",
            prompt_asset=_prompt(),
        )


def test_selection_binds_reasons_and_hashes_to_exact_script(tmp_path: Path) -> None:
    candidates = _candidates()
    selected = candidates[0].style_id
    rejected = {item.style_id: f"不如{selected}贴合" for item in candidates[1:]}
    model = ScriptedStructuredModel(
        [{
            "selected_style": selected,
            "selection_reasons": ["关系叙事与克制情绪都匹配"],
            "rejected_reasons": rejected,
            "confidence": 0.91,
        }]
    )

    decision = select_style(
        profile=_profile(),
        approved_script=SCRIPT,
        value_contract=_value_contract(),
        candidates=candidates,
        model=model,
        request_root=tmp_path / "requests",
        prompt_asset=_prompt(),
    )

    assert decision.selected_style == selected
    assert decision.candidate_styles == tuple(item.style_id for item in candidates)
    assert decision.source_script_sha256 == _sha(SCRIPT)
    assert len(decision.style_fingerprint) == 64
    assert decision.manual_override is False
    assert model.calls[0][1].parent == tmp_path / "requests"
    assert SCRIPT in model.calls[0][0]


def test_selection_can_use_whole_script_context_while_binding_sample_hash(
    tmp_path: Path,
) -> None:
    candidates = _candidates()
    selected = candidates[0].style_id
    sample = "这是未经改写的技术样片片段。"
    whole = SCRIPT + "这是只有整篇批准文稿才包含的生活联系。"
    model = ScriptedStructuredModel([{
        "selected_style": selected,
        "selection_reasons": ["整篇语境匹配"],
        "rejected_reasons": {
            item.style_id: "不如首选贴合" for item in candidates[1:]
        },
        "confidence": 0.9,
    }])

    decision = select_style(
        _profile(), sample, _value_contract(), candidates, model,
        tmp_path / "requests", _prompt(), selection_context_script=whole,
    )

    assert whole in model.calls[0][0]
    assert decision.source_script_sha256 == _sha(sample)


def test_missing_rejection_reason_is_rejected(tmp_path: Path) -> None:
    candidates = _candidates()
    model = ScriptedStructuredModel(
        [{
            "selected_style": candidates[0].style_id,
            "selection_reasons": ["匹配"],
            "rejected_reasons": {},
            "confidence": 0.7,
        }]
    )

    with pytest.raises(StyleSelectionError, match="invalid_style_reasons"):
        select_style(
            profile=_profile(),
            approved_script=SCRIPT,
            value_contract=_value_contract(),
            candidates=candidates,
            model=model,
            request_root=tmp_path / "requests",
            prompt_asset=_prompt(),
        )


def test_manual_override_skips_model_and_is_explicit(tmp_path: Path) -> None:
    candidates = _candidates()
    model = ScriptedStructuredModel([])

    decision = select_style(
        profile=_profile(),
        approved_script=SCRIPT,
        value_contract=_value_contract(),
        candidates=candidates,
        model=model,
        request_root=tmp_path / "requests",
        prompt_asset=_prompt(),
        manual_override=candidates[1].style_id,
    )

    assert decision.selected_style == candidates[1].style_id
    assert decision.manual_override is True
    assert model.calls == []


def test_style_fingerprint_changes_when_selected_recipe_changes(tmp_path: Path) -> None:
    catalog = list(load_style_catalog(CATALOG))
    original_candidates = rank_style_candidates(_profile(), catalog)[:3]
    selected_id = original_candidates[0].style_id
    selected_index = next(
        index for index, recipe in enumerate(catalog) if recipe.style_id == selected_id
    )
    catalog[selected_index] = catalog[selected_index].model_copy(
        update={"summary": catalog[selected_index].summary + " 本地修订"}
    )
    changed_candidates = rank_style_candidates(_profile(), catalog)[:3]

    original = select_style(
        _profile(), SCRIPT, _value_contract(), tuple(original_candidates),
        ScriptedStructuredModel([]), tmp_path / "one", _prompt(),
        manual_override=selected_id,
    )
    changed = select_style(
        _profile(), SCRIPT, _value_contract(), tuple(changed_candidates),
        ScriptedStructuredModel([]), tmp_path / "two", _prompt(),
        manual_override=selected_id,
    )

    assert original.style_fingerprint != changed.style_fingerprint
