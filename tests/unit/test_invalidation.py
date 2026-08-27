from pathlib import Path

import pytest

from bv.state.invalidation import invalidate_from
from bv.state.models import EpisodeState, StageManifest


def test_episode_state_accepts_multiclip_progress_statuses() -> None:
    assert (
        EpisodeState(
            book_id="book-demo",
            episode_id="E001",
            status="visual_sample_ready",
        ).status
        == "visual_sample_ready"
    )
    assert (
        EpisodeState(
            book_id="book-demo",
            episode_id="E001",
            status="video_partial",
        ).status
        == "video_partial"
    )


def test_script_change_invalidates_all_multiclip_artifacts() -> None:
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        completed_stages=[
            "script",
            "tts",
            "asr",
            "subtitles",
            "storyboard",
            "continuity",
            "h3_prompts",
            "h3_clips",
            "render",
            "qc",
        ],
    )

    result = invalidate_from(episode, "script")

    assert result.stale_stages == [
        "tts",
        "asr",
        "subtitles",
        "storyboard",
        "continuity",
        "h3_prompts",
        "h3_clips",
        "render",
        "qc",
    ]
    assert result.completed_stages == ["script"]


def test_script_change_invalidates_social_cover_and_delivery() -> None:
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        completed_stages=["script", "social_cover", "delivery", "final_approval"],
    )

    result = invalidate_from(episode, "script")

    assert result.stale_stages == ["social_cover", "delivery", "final_approval"]
    assert result.completed_stages == ["script"]


def test_continuity_change_invalidates_multiclip_downstream_artifacts() -> None:
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        completed_stages=[
            "continuity",
            "h3_prompts",
            "h3_clips",
            "render",
            "qc",
        ],
    )

    result = invalidate_from(episode, "continuity")

    assert result.stale_stages == ["h3_prompts", "h3_clips", "render", "qc"]
    assert result.completed_stages == ["continuity"]


@pytest.mark.parametrize(
    ("changed_stage", "expected_stale"),
    [
        (
            "voice_reference",
            [
                "tts",
                "asr",
                "subtitles",
                "storyboard",
                "continuity",
                "h3_prompts",
                "h3_clips",
                "render",
                "qc",
            ],
        ),
        (
            "tts",
            [
                "asr",
                "subtitles",
                "storyboard",
                "continuity",
                "h3_prompts",
                "h3_clips",
                "render",
                "qc",
            ],
        ),
        (
            "asr",
            [
                "subtitles",
                "storyboard",
                "continuity",
                "h3_prompts",
                "h3_clips",
                "render",
                "qc",
            ],
        ),
        (
            "storyboard",
            ["continuity", "h3_prompts", "h3_clips", "render", "qc"],
        ),
        ("h3_prompts", ["h3_clips", "render", "qc"]),
        ("h3_clips", ["render", "qc"]),
    ],
)
def test_long_form_stage_changes_invalidate_eligible_downstream_artifacts(
    changed_stage: str,
    expected_stale: list[str],
) -> None:
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        completed_stages=[
            "tts",
            "asr",
            "subtitles",
            "storyboard",
            "continuity",
            "h3_prompts",
            "h3_clips",
            "render",
            "qc",
        ],
    )

    result = invalidate_from(episode, changed_stage)

    assert result.stale_stages == expected_stale


def test_invalidation_preserves_graph_order_without_duplicate_stages() -> None:
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        completed_stages=[
            "script",
            "tts",
            "asr",
            "storyboard",
            "continuity",
            "h3_prompts",
            "h3_clips",
            "render",
            "qc",
        ],
        stale_stages=["render"],
        stage_manifests={
            "render": StageManifest(
                stage="render",
                status="completed",
                inputs={},
                outputs={},
                config_sha256="config-hash",
            )
        },
    )

    result = invalidate_from(episode, "script")

    assert result.stale_stages == [
        "render",
        "tts",
        "asr",
        "storyboard",
        "continuity",
        "h3_prompts",
        "h3_clips",
        "qc",
    ]
    assert result.completed_stages == ["script"]
    assert result.stage_manifests["render"].status == "stale"


def test_leaf_invalidation_does_not_create_unproduced_stale_stages() -> None:
    episode = EpisodeState(book_id="book-demo", episode_id="E001")

    result = invalidate_from(episode, "subtitle_style")

    assert result.stale_stages == []
    assert result.completed_stages == []


def test_subtitle_style_change_does_not_invalidate_qc_without_render_output() -> None:
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        completed_stages=["script"],
    )

    result = invalidate_from(episode, "subtitle_style")

    assert result.stale_stages == []


def test_style_change_invalidates_representatives_and_all_visuals() -> None:
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        completed_stages=[
            "style_decision",
            "illustration_plan",
            "representative_review",
            "illustration_images",
            "visual_render",
            "render",
            "qc",
        ],
    )

    result = invalidate_from(episode, "style_decision")

    assert result.stale_stages == [
        "illustration_plan",
        "representative_review",
        "illustration_images",
        "visual_render",
        "render",
        "qc",
    ]
    assert result.completed_stages == ["style_decision"]


def test_visual_render_change_keeps_approved_images_but_invalidates_final() -> None:
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        completed_stages=["illustration_images", "visual_render", "render", "qc"],
    )

    result = invalidate_from(episode, "visual_render")

    assert result.completed_stages == ["illustration_images", "visual_render"]
    assert result.stale_stages == ["render", "qc"]


def test_qc_or_social_cover_change_invalidates_delivery_and_final_approval() -> None:
    for changed_stage in ("qc", "social_cover"):
        episode = EpisodeState(
            book_id="book-demo",
            episode_id="E001",
            completed_stages=[changed_stage, "delivery", "final_approval"],
        )

        result = invalidate_from(episode, changed_stage)

        assert result.stale_stages == ["delivery", "final_approval"]
        assert result.completed_stages == [changed_stage]


def test_batch_prompt_change_preserves_representative_approval() -> None:
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        completed_stages=[
            "representative_review",
            "illustration_images",
            "visual_render",
            "render",
            "qc",
        ],
    )

    result = invalidate_from(episode, "batch_illustration_prompt")

    assert result.completed_stages == ["representative_review"]
    assert result.stale_stages == [
        "illustration_images", "visual_render", "render", "qc"
    ]
