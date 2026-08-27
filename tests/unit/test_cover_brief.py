from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bv.video.cover_brief import (
    CoverBriefBuilder,
    CoverBriefError,
    CoverDerivedFields,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _style(tmp_path: Path) -> tuple[Path, Path]:
    style = tmp_path / "styles" / "french-minimal-ink-poster"
    style.mkdir(parents=True)
    meta = style / "META.md"
    atom = style / "STYLE.md"
    meta.write_text(
        "# 法式极简墨线海报\n\n```yaml\n"
        "id: french-minimal-ink-poster\n"
        "name: 法式极简墨线海报\n"
        "outputs: [cover, poster]\n"
        "style_anchors: [warm ivory paper, sparse black ink, one metaphor]\n"
        "cover_shape_adaptation: [portrait title safe area]\n"
        "must_preserve: [one metaphor, readable title]\n"
        "avoid_when_applying_to_cover: [generic icon, crowded scene]\n"
        "```\n",
        encoding="utf-8",
    )
    atom.write_text("大片留白，稀疏黑色墨线，只保留一个准确隐喻。", encoding="utf-8")
    return meta, atom


def _source(tmp_path: Path) -> tuple[Path, Path, Path]:
    episode = tmp_path / "episode"
    script = episode / "script" / "approved.txt"
    lock = episode / "script" / "semantic-lock.json"
    blueprint = tmp_path / "cover-prompt-blueprint.md"
    script.parent.mkdir(parents=True)
    script.write_text("《活着》让人看见，生活不断失去以后，人仍然怎样把日子过下去。", encoding="utf-8")
    lock.write_text('{"value_thesis":"把一天交给下一天"}', encoding="utf-8")
    blueprint.write_text("required cover blueprint", encoding="utf-8")
    return episode, script, lock


def _prompt() -> str:
    return """# 法式极简墨线海报 cover prompt

Create one social video cover at 3:4.
Selected style: 法式极简墨线海报 / french-minimal-ink-poster.
Main title: 活着. Keep the complete Chinese title centered, enlarged and readable.
Use warm ivory paper, sparse black ink and one restrained metaphor: an old man and an old ox standing side by side in a vast quiet field of negative space.
The image speaks to adults carrying real losses. It should feel restrained, dignified and alive, never sensational.
Avoid fake book covers, actor likenesses, blood, tears, generic roads, birds, cages, advertising, logos and watermarks.
Generate one single 1080x1440 cover image, no grid and no alternatives.
"""


def test_cover_brief_binds_one_style_and_approved_meaning(tmp_path: Path) -> None:
    meta, atom = _style(tmp_path)
    episode, script, lock = _source(tmp_path)
    builder = CoverBriefBuilder(blueprint_path=tmp_path / "cover-prompt-blueprint.md")

    brief = builder.build(
        episode_root=episode,
        slug="huo-zhe-e001",
        fields=CoverDerivedFields(
            title="活着",
            subtitle="把一天交给下一天",
            summary="生活不断失去以后，一个普通人如何继续活下去。",
            visual_subject="一位老人和一头老牛",
            audience="正在承受现实压力和失去的成年人",
            mood="克制，沉静，有尊严",
            visual_metaphor="老人和老牛在大片留白中并肩站立",
            banned_elements=("假书封", "影视演员脸", "苦难奇观"),
        ),
        compiled_prompt=_prompt(),
        approved_script_path=script,
        approved_script_sha256=_sha(script),
        semantic_lock_path=lock,
        semantic_lock_sha256=_sha(lock),
        production_profile_sha256="a" * 64,
        style_id="french-minimal-ink-poster",
        style_meta_path=meta,
        style_atom_path=atom,
    )

    assert brief.style_id == "french-minimal-ink-poster"
    assert brief.title == "活着"
    assert (brief.aspect_ratio, brief.width, brief.height) == ("3:4", 1080, 1440)
    assert brief.prompt_path == (
        episode / "punk-assets" / "punk-cover" / "huo-zhe-e001"
        / "prompts" / "cover.md"
    )
    assert brief.prompt_path.read_text(encoding="utf-8") == _prompt()
    assert brief.prompt_sha256 == _sha(brief.prompt_path)
    assert "{{" not in brief.prompt_path.read_text(encoding="utf-8")


def test_cover_brief_rejects_ineligible_style_or_source_body_copy(
    tmp_path: Path,
) -> None:
    meta, atom = _style(tmp_path)
    episode, script, lock = _source(tmp_path)
    builder = CoverBriefBuilder(blueprint_path=tmp_path / "cover-prompt-blueprint.md")
    bad_meta = meta.parent / "BAD.md"
    bad_meta.write_text(
        "```yaml\nid: french-minimal-ink-poster\noutputs: [avatar]\n```",
        encoding="utf-8",
    )
    common = dict(
        episode_root=episode,
        slug="huo-zhe-e001",
        fields=CoverDerivedFields(
            title="活着",
            summary="摘要",
            visual_subject="老人和老牛",
            audience="成年人",
            mood="克制",
            visual_metaphor="并肩站立",
        ),
        approved_script_path=script,
        approved_script_sha256=_sha(script),
        semantic_lock_path=lock,
        semantic_lock_sha256=_sha(lock),
        production_profile_sha256="a" * 64,
        style_id="french-minimal-ink-poster",
        style_atom_path=atom,
    )

    with pytest.raises(CoverBriefError, match="cover_style_ineligible"):
        builder.build(
            **common,
            compiled_prompt=_prompt(),
            style_meta_path=bad_meta,
        )
    with pytest.raises(CoverBriefError, match="cover_prompt_copies_source"):
        builder.build(
            **common,
            compiled_prompt=_prompt() + script.read_text(encoding="utf-8"),
            style_meta_path=meta,
        )
