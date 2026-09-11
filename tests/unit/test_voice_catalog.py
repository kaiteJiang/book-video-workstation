from __future__ import annotations

from bv.voice.catalog import (
    load_voice_catalog,
    select_story_voices,
    supported_voice_ids,
)


def test_checked_in_catalog_is_bounded_and_truthful() -> None:
    catalog = load_voice_catalog()

    assert 8 <= len(catalog.entries) <= 12
    assert len({entry.voice_id for entry in catalog.entries}) == len(catalog.entries)
    assert all(entry.audition_status == "pending" for entry in catalog.entries)
    assert all(entry.popularity_evidence is None for entry in catalog.entries)
    assert all(entry.source_verified_at == "2026-09-08" for entry in catalog.entries)
    assert all(
        entry.age_group == "unknown" or entry.display_name == "爽朗少年"
        for entry in catalog.entries
    )
    assert all(entry.source_url.startswith("https://www.volcengine.com/docs/") for entry in catalog.entries)


def test_story_selection_filters_gender_and_is_deterministic() -> None:
    catalog = load_voice_catalog()

    selected = select_story_voices(
        catalog,
        narrator_gender="male",
        desired_story_tags=("natural", "historical", "audiobook"),
    )

    assert 1 <= len(selected) <= 3
    assert selected[0].voice.voice_id == "zh_male_ruyayichen_saturn_bigtts"
    assert all(item.voice.gender == "male" for item in selected)
    assert any("异步端点仍待实际验证" in reason for reason in selected[0].reasons)
    assert selected == select_story_voices(
        catalog,
        narrator_gender="male",
        desired_story_tags=("natural", "historical", "audiobook"),
    )


def test_unavailable_or_wrong_resource_voice_is_not_supported() -> None:
    catalog = load_voice_catalog()
    first = catalog.entries[0]
    unavailable = first.model_copy(
        update={
            "compatibility": first.compatibility.model_copy(
                update={"async_v3": "unavailable"}
            )
        }
    )
    modified = catalog.model_copy(update={"entries": (unavailable, *catalog.entries[1:])})

    assert first.voice_id not in supported_voice_ids(modified)
    assert supported_voice_ids(modified, resource_id="other") == ()


def test_candidate_compatibility_can_be_excluded_for_strict_execution() -> None:
    catalog = load_voice_catalog()

    assert supported_voice_ids(catalog, allow_candidate_compatibility=False) == ()
