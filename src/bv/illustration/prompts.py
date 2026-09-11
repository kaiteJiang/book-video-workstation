from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import (
    CharacterBible,
    CharacterLock,
    IllustrationScene,
    load_character_bible,
)
from .style_selector import StyleRecipe


class CompiledImagePrompt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scene_id: str
    prompt: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    style_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    character_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_image_sha256s: tuple[str, ...]


def compile_image_prompt(
    scene: IllustrationScene,
    *,
    style: StyleRecipe,
    character_lock: CharacterLock | CharacterBible,
    reference_scene_ids: tuple[str, ...] = (),
    reference_image_sha256s: tuple[str, ...] = (),
    phase: Literal["legacy", "anchor", "continuation"] = "legacy",
) -> CompiledImagePrompt:
    if any(not _is_sha256(value) for value in reference_image_sha256s):
        raise ValueError("invalid_reference_image_hash")
    bible = load_character_bible(character_lock)
    character_sha = canonical_model_sha256(character_lock)
    referenced = tuple(
        character
        for character in bible.characters
        if character.character_id in scene.character_refs
    )
    continuity = "\n".join(
        (
            f"- 角色ID：{character.character_id}\n"
            f"  身份：{character.role}\n"
            f"  年龄：{character.age_range}\n"
            f"  脸部：{character.face}\n"
            f"  发型：{character.hair}\n"
            f"  体态：{character.body}\n"
            f"  基础服装：{character.base_clothing}\n"
            f"  色彩标记：{'、'.join(character.color_markers) or '无'}\n"
            f"  随身物件：{'、'.join(character.personal_objects) or '无'}\n"
            f"  禁止变化：{'、'.join(character.forbidden_changes) or '无'}"
        )
        for character in referenced
    ) or "- 本场景不出现需要锁定外观的角色"
    style_blocks = "\n".join(f"- {item}" for item in style.prompt_blocks)
    phase_instructions = ""
    if phase == "anchor":
        phase_instructions = "\n剧情对阶段：A（故事起点）。建立本叙事小节开始时的人物处境与悬念，不提前显示小节结果。\n"
    elif phase == "continuation":
        phase_instructions = (
            "\n剧情对阶段：B（小节高潮或结果）。第一张参考图用于人物身份和画风连续性，"
            "不是必须复制的镜头。按本小节已经讲出的事件发展呈现改变后的处境；"
            "时间、地点、年龄、服饰、机位及在场人物可依据正文改变。"
            "不得只改手势、表情或道具位置充当故事发展。\n"
            f"结果或新处境：{scene.continuation_action}\n"
            f"延续画面：{scene.continuation_prompt}\n"
            f"连续性约束：{'；'.join(scene.continuity_constraints or ())}\n"
        )
    prompt = (
        "生成一张 1080×1920、9:16 全屏竖版的单幅手绘插画。水彩纸画布铺满整个画面，"
        "构图从边缘延伸到边缘，不要正方形画芯、上下留白条、边框或画中画。\n"
        f"锁定画风 ID：{style.style_id}\n"
        f"画风摘要：{style.summary}\n"
        f"画风细则：\n{style_blocks or '- 严格遵守画风摘要和配色提示'}\n"
        f"配色：{style.color_hint}\n"
        f"避免：{style.avoid}\n\n"
        f"人物连续性锁定：\n{continuity}\n\n"
        f"当前场景：{scene.scene_id}\n"
        f"视觉目的：{scene.visual_purpose}\n"
        f"场所：{scene.setting}\n"
        f"人物动作：{scene.character_action}\n"
        f"生活隐喻：{scene.metaphor or '无'}\n"
        f"构图：{scene.composition}\n"
        f"画面内容：{scene.image_prompt}\n"
        f"只出现这些角色：{'、'.join(scene.character_refs) or '不出现固定主人公'}\n"
        f"身份参考场景：{'、'.join(reference_scene_ids) or '无，建立初始身份'}\n\n"
        f"{phase_instructions}"
        "硬性限制：主体和所有可见肢体、道具必须完整落在画布内；顶部、底部和右侧留出安全空间。"
        "画中不得出现任何文字、字母、数字、字幕、书名、书中文字页、假书封、Logo 或水印。"
        "不要增加旁白中没有的事实、引语、疗效、保证或人物经历。\n"
        f"补充禁止项：{'；'.join(scene.negative_constraints)}"
    )
    return CompiledImagePrompt(
        scene_id=scene.scene_id,
        prompt=prompt,
        prompt_sha256=image_prompt_identity_sha256(
            prompt=prompt,
            style_fingerprint=bible.style_fingerprint,
            character_lock_sha256=character_sha,
            reference_image_sha256s=reference_image_sha256s,
        ),
        style_fingerprint=bible.style_fingerprint,
        character_lock_sha256=character_sha,
        reference_image_sha256s=reference_image_sha256s,
    )


def canonical_model_sha256(value: BaseModel) -> str:
    return _canonical_sha(value.model_dump(mode="json"))


def image_prompt_identity_sha256(
    *,
    prompt: str,
    style_fingerprint: str,
    character_lock_sha256: str,
    reference_image_sha256s: tuple[str, ...],
) -> str:
    if (
        not _is_sha256(style_fingerprint)
        or not _is_sha256(character_lock_sha256)
        or any(not _is_sha256(value) for value in reference_image_sha256s)
    ):
        raise ValueError("invalid_image_prompt_identity")
    return _canonical_sha(
        {
            "prompt": prompt,
            "style_fingerprint": style_fingerprint,
            "character_lock_sha256": character_lock_sha256,
            "reference_image_sha256s": reference_image_sha256s,
        }
    )


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
