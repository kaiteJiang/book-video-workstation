from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bv.production.profile import (
    ProductionProfile,
    ProductionProfileError,
    load_production_profile,
    production_profile_sha256,
)


def _write_profile(root: Path, **changes: object) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "duration": {
            "hard_min_seconds": 120.0,
            "ideal_min_seconds": 128.0,
            "ideal_max_seconds": 142.0,
            "hard_max_seconds": 150.0,
        },
        "tts": {
            "provider": "doubao",
            "resource_id": "seed-tts-2.0",
            "voice_type": None,
            "speed": 1.0,
        },
        "visual": {
            "seconds_per_scene_min": 7.0,
            "seconds_per_scene_max": 11.0,
            "representative_count": 3,
            "transition": "cross-dissolve",
        },
        "delivery": {
            "timezone": "Asia/Shanghai",
            "include": [
                "video",
                "cover",
                "script",
                "voice",
                "subtitles",
                "manifest",
            ],
        },
    }
    for key, value in changes.items():
        section, field = key.split("__", maxsplit=1)
        assert isinstance(payload[section], dict)
        payload[section][field] = value  # type: ignore[index]
    root.mkdir(parents=True, exist_ok=True)
    (root / "production_profile.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_missing_profile_uses_legacy_defaults(tmp_path: Path) -> None:
    profile = load_production_profile(tmp_path)

    assert profile.tts.provider == "indextts2"
    assert profile.tts.voice_type == "黑金3"
    assert profile.visual.sequence_mode == "legacy-monochrome-reveal"
    assert profile.duration.model_dump() == {
        "hard_min_seconds": 45.0,
        "ideal_min_seconds": 48.0,
        "ideal_max_seconds": 56.0,
        "hard_max_seconds": 60.0,
    }


def test_short_book_default_locks_new_video_contract() -> None:
    profile = ProductionProfile.short_book_default()

    assert profile.duration.model_dump() == {
        "hard_min_seconds": 30.0,
        "ideal_min_seconds": 35.0,
        "ideal_max_seconds": 40.0,
        "hard_max_seconds": 45.0,
    }
    assert profile.visual.scene_count == 4
    assert profile.visual.sequence_mode == "color-story-pair"
    assert profile.visual.style_id == "retro-gouache-concept"
    assert profile.visual.representative_count == 3


def test_living_profile_parses_approved_schema(tmp_path: Path) -> None:
    _write_profile(tmp_path)

    profile = load_production_profile(tmp_path)

    assert profile.tts.provider == "doubao"
    assert profile.tts.resource_id == "seed-tts-2.0"
    assert profile.tts.voice_type is None
    assert profile.visual.seconds_per_scene_min == 7.0
    assert profile.visual.seconds_per_scene_max == 11.0
    assert profile.delivery.include == (
        "video",
        "cover",
        "script",
        "voice",
        "subtitles",
        "manifest",
    )


def test_living_default_uses_color_story_pairs() -> None:
    visual = ProductionProfile.living_default().visual

    assert visual.sequence_mode == "color-story-pair"
    assert visual.scene_count == 4


def test_longform_story_profile_is_advisory_and_dynamic() -> None:
    profile = ProductionProfile.longform_story_default()

    assert profile.narrative_mode == "story"
    assert profile.duration.ideal_min_seconds == 360
    assert profile.duration.ideal_max_seconds == 480
    assert profile.duration.soft_max_seconds == 600
    assert profile.visual.scene_count is None
    assert profile.duration.advisory_only is True
    assert profile.duration_advisories(420) == ()
    assert profile.duration_advisories(300) == ("outside_recommended_story_duration",)
    assert profile.duration_advisories(601) == (
        "outside_recommended_story_duration",
        "story_duration_above_soft_max",
    )


def test_doubao_profile_can_persist_approved_voice_id(tmp_path: Path) -> None:
    _write_profile(tmp_path, tts__voice_type="zh_female_reader")

    profile = load_production_profile(tmp_path)

    assert profile.tts.voice_type == "zh_female_reader"


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"duration__hard_min_seconds": 151.0}, "invalid_duration_profile"),
        ({"visual__seconds_per_scene_min": 12.0}, "invalid_scene_duration_profile"),
        ({"tts__resource_id": ""}, "blank_tts_resource_id"),
        ({"tts__voice_type": "  "}, "blank_voice_type"),
        ({"delivery__timezone": "UTC"}, "literal_error"),
    ],
)
def test_profile_rejects_invalid_contract(
    tmp_path: Path,
    changes: dict[str, object],
    error: str,
) -> None:
    _write_profile(tmp_path, **changes)

    with pytest.raises((ValidationError, ProductionProfileError), match=error):
        load_production_profile(tmp_path)


def test_profile_hash_changes_when_duration_changes(tmp_path: Path) -> None:
    _write_profile(tmp_path)
    first = production_profile_sha256(tmp_path)

    _write_profile(tmp_path, duration__ideal_max_seconds=141.0)

    assert production_profile_sha256(tmp_path) != first


def test_missing_profile_hash_preserves_legacy_shape(tmp_path: Path) -> None:
    assert production_profile_sha256(tmp_path) == (
        "0c200f6028fd4daba3087e7c3d4210ac7bf8c2b185b8ce26a75ca3da4c4d5019"
    )


def test_redirected_profile_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    _write_profile(tmp_path / "source-root")
    source.write_bytes((tmp_path / "source-root" / "production_profile.json").read_bytes())
    profile_path = tmp_path / "production_profile.json"
    try:
        profile_path.symlink_to(source)
    except OSError:
        pytest.skip("symlink creation requires additional privilege")

    with pytest.raises(ProductionProfileError, match="unsafe_production_profile_path"):
        load_production_profile(tmp_path)
