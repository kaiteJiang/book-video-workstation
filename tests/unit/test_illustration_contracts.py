from pathlib import Path

import pytest
from pydantic import ValidationError

from bv.illustration.contracts import (
    CharacterBible,
    CharacterLock,
    IllustrationScene,
    IllustrationStoryboard,
    SafeArea,
    SilentRenderRequest,
    SilentRenderResult,
    StyleDecision,
    load_character_bible,
)


def _scene(
    scene_id: str,
    from_frame: int,
    to_frame: int,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> IllustrationScene:
    return IllustrationScene(
        scene_id=scene_id,
        start_ms=from_frame * 1000 // 30 if start_ms is None else start_ms,
        end_ms=to_frame * 1000 // 30 if end_ms is None else end_ms,
        from_frame=from_frame,
        to_frame=to_frame,
        narration="这本书让人重新看见自己的生活。",
        narration_span=(0, 16),
        key_line="把人生还给自己",
        visual_purpose="建立真实生活困境",
        setting="清晨的通勤地铁",
        character_action="主人公放下反复查看的手机",
        metaphor=None,
        composition="人物居中，右侧交互区留空",
        character_refs=("reader-01",),
        image_prompt="同一位普通读者，手绘插画，无画内文字",
        negative_constraints=("no_text", "no_logo", "no_watermark"),
        representative_frame=True,
        asset_status="planned",
    )


def _storyboard(**updates: object) -> IllustrationStoryboard:
    payload: dict[str, object] = {
        "book_id": "book-demo",
        "episode_id": "E001",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "master_duration_ms": 10_000,
        "total_frames": 300,
        "script_sha256": "a" * 64,
        "audio_sha256": "b" * 64,
        "subtitle_sha256": "c" * 64,
        "style_decision_sha256": "d" * 64,
        "character_lock_sha256": "e" * 64,
        "scenes": (
            _scene("S01", 0, 150, start_ms=0, end_ms=5_000),
            _scene("S02", 150, 300, start_ms=5_000, end_ms=10_000),
        ),
    }
    payload.update(updates)
    return IllustrationStoryboard.model_validate(payload)


def test_storyboard_requires_gapless_absolute_frames() -> None:
    with pytest.raises(ValidationError, match="scene_frame_gap"):
        _storyboard(
            scenes=(
                _scene("S01", 0, 149),
                _scene("S02", 151, 300),
            )
        )


@pytest.mark.parametrize(
    ("updates", "error_code"),
    [
        (
            {
                "scenes": (
                    _scene("S01", 0, 151),
                    _scene("S02", 150, 300),
                )
            },
            "scene_frame_overlap",
        ),
        (
            {
                "scenes": (
                    _scene("S01", 1, 150),
                    _scene("S02", 150, 300),
                )
            },
            "first_scene_frame",
        ),
        (
            {
                "scenes": (
                    _scene("S01", 0, 150),
                    _scene("S02", 150, 299),
                )
            },
            "final_scene_frame",
        ),
        ({"total_frames": 299}, "total_frames_mismatch"),
    ],
)
def test_storyboard_rejects_broken_master_timeline(
    updates: dict[str, object],
    error_code: str,
) -> None:
    with pytest.raises(ValidationError, match=error_code):
        _storyboard(**updates)


@pytest.mark.parametrize("field", ["book_id", "episode_id"])
@pytest.mark.parametrize("unsafe", ["../escape", "nested/name", r"nested\name", "."])
def test_storyboard_rejects_unsafe_identifiers(field: str, unsafe: str) -> None:
    with pytest.raises(ValidationError, match="unsafe_identifier"):
        _storyboard(**{field: unsafe})


@pytest.mark.parametrize(
    "field",
    [
        "script_sha256",
        "audio_sha256",
        "subtitle_sha256",
        "style_decision_sha256",
        "character_lock_sha256",
    ],
)
@pytest.mark.parametrize("invalid", ["A" * 64, "a" * 63, "g" * 64])
def test_storyboard_rejects_noncanonical_sha256(field: str, invalid: str) -> None:
    with pytest.raises(ValidationError, match="invalid_sha256"):
        _storyboard(**{field: invalid})


def test_illustration_models_are_frozen_and_forbid_extra_fields() -> None:
    safe_area = SafeArea()

    with pytest.raises(ValidationError, match="frozen_instance"):
        safe_area.subtitle_bottom = 1700
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SafeArea.model_validate({"unexpected": 1})


def test_safe_area_rejects_reversed_vertical_regions() -> None:
    with pytest.raises(ValidationError, match="invalid_safe_area"):
        SafeArea(keyline_bottom=100, illustration_top=100)


def test_style_decision_validates_fingerprint_and_confidence() -> None:
    valid = {
        "library_version": "1.0",
        "selected_style": "12-sunlit-storybook",
        "candidate_styles": (
            "12-sunlit-storybook",
            "15-warm-flat-storybook",
            "10-emotional-watercolor-sketch",
        ),
        "selection_reasons": ("温暖克制，适合生活感悟",),
        "rejected_reasons": {"03-kid-crayon": "童稚感过强"},
        "confidence": 0.86,
        "manual_override": False,
        "source_script_sha256": "a" * 64,
        "style_fingerprint": "b" * 64,
    }
    assert StyleDecision.model_validate(valid).confidence == 0.86

    with pytest.raises(ValidationError, match="invalid_sha256"):
        StyleDecision.model_validate({**valid, "style_fingerprint": "B" * 64})
    with pytest.raises(ValidationError):
        StyleDecision.model_validate({**valid, "confidence": 1.01})


def test_legacy_single_character_lock_loads_as_reader() -> None:
    legacy = CharacterLock(
        role="普通读者主人公",
        age_range="30至39岁",
        face="椭圆脸",
        hair="黑色短发",
        body="中等身材",
        base_clothing="深蓝衬衫",
        allowed_variations=(),
        color_markers=("深蓝",),
        personal_objects=("旧笔记本",),
        forbidden_changes=("发型变化",),
        source_script_sha256="a" * 64,
        style_fingerprint="b" * 64,
    )

    bible = load_character_bible(legacy.model_dump(mode="json"))

    assert isinstance(bible, CharacterBible)
    assert [item.character_id for item in bible.characters] == ["reader-01"]


def test_character_bible_rejects_duplicate_or_unsupported_ids() -> None:
    base = load_character_bible(
        CharacterLock(
            role="读者",
            age_range="30至39岁",
            face="椭圆脸",
            hair="黑色短发",
            body="中等身材",
            base_clothing="深蓝衬衫",
            allowed_variations=(),
            color_markers=(),
            personal_objects=(),
            forbidden_changes=(),
            source_script_sha256="a" * 64,
            style_fingerprint="b" * 64,
        )
    )
    duplicate = base.characters[0].model_copy(update={"role": "另一个读者"})
    with pytest.raises(ValidationError, match="duplicate_character_id"):
        CharacterBible(
            source_script_sha256="a" * 64,
            style_fingerprint="b" * 64,
            characters=(base.characters[0], duplicate),
        )
    unsupported = base.characters[0].model_copy(update={"character_id": "actor-01"})
    with pytest.raises(ValidationError, match="unsupported_character_id"):
        CharacterBible(
            source_script_sha256="a" * 64,
            style_fingerprint="b" * 64,
            characters=(unsupported,),
        )


@pytest.mark.parametrize("scene_id", ["../S01", "scene/01", r"scene\01"])
def test_scene_rejects_unsafe_identifier(scene_id: str) -> None:
    with pytest.raises(ValidationError, match="unsafe_identifier"):
        _scene(scene_id, 0, 150)


@pytest.mark.parametrize("key_line", ["太短", "这是一条明显超过十四个汉字的主题句"])
def test_scene_rejects_unreadable_key_line_length(key_line: str) -> None:
    payload = _scene("S01", 0, 150).model_dump()
    payload["key_line"] = key_line

    with pytest.raises(ValidationError, match="invalid_key_line_length"):
        IllustrationScene.model_validate(payload)


def test_silent_render_request_normalizes_episode_paths(tmp_path: Path) -> None:
    episode_root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    storyboard = episode_root / "media" / "illustration" / "storyboard.json"
    output = episode_root / "media" / "render" / "picture_silent.mp4"
    vendor = tmp_path / "vendor" / "story_to_handdrawn_video"

    request = SilentRenderRequest(
        episode_root=episode_root,
        storyboard_path=storyboard,
        storyboard_sha256="a" * 64,
        output_path=output,
        vendor_dir=vendor,
    )

    assert request.episode_root == episode_root.resolve()
    assert request.storyboard_path == storyboard.resolve()
    assert request.output_path == output.resolve()
    assert request.vendor_dir == vendor.resolve()


@pytest.mark.parametrize("field", ["storyboard_path", "output_path"])
def test_silent_render_request_rejects_paths_outside_episode(
    tmp_path: Path,
    field: str,
) -> None:
    episode_root = tmp_path / "episode"
    payload: dict[str, object] = {
        "episode_root": episode_root,
        "storyboard_path": episode_root / "media" / "storyboard.json",
        "storyboard_sha256": "a" * 64,
        "output_path": episode_root / "media" / "picture_silent.mp4",
        "vendor_dir": tmp_path / "vendor",
    }
    payload[field] = tmp_path / "outside" / Path(str(payload[field])).name

    with pytest.raises(ValidationError, match="unsafe_episode_path"):
        SilentRenderRequest.model_validate(payload)


def test_silent_render_request_revalidates_tampered_model_copy(tmp_path: Path) -> None:
    episode_root = tmp_path / "episode"
    request = SilentRenderRequest(
        episode_root=episode_root,
        storyboard_path=episode_root / "storyboard.json",
        storyboard_sha256="a" * 64,
        output_path=episode_root / "picture_silent.mp4",
        vendor_dir=tmp_path / "vendor",
    )
    tampered = request.model_copy(
        update={"output_path": tmp_path / "outside" / "picture_silent.mp4"}
    )

    with pytest.raises(ValidationError, match="unsafe_episode_path"):
        SilentRenderRequest.model_validate(tampered)


def test_silent_render_request_rejects_symlinked_episode_path(tmp_path: Path) -> None:
    episode_root = tmp_path / "episode"
    episode_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = episode_root / "redirected"
    try:
        redirected.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation requires additional privilege")
    if not redirected.is_symlink():
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(ValidationError, match="unsafe_episode_path"):
        SilentRenderRequest(
            episode_root=episode_root,
            storyboard_path=redirected / "storyboard.json",
            storyboard_sha256="a" * 64,
            output_path=episode_root / "picture_silent.mp4",
            vendor_dir=tmp_path / "vendor",
        )


def test_silent_render_result_rejects_noncanonical_output_hash(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="invalid_sha256"):
        SilentRenderResult(
            output_path=tmp_path / "picture_silent.mp4",
            output_sha256="F" * 64,
            duration_ms=15_000,
            width=1080,
            height=1920,
            frame_rate=30.0,
            audio_present=False,
        )
