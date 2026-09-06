from bv.state.models import EpisodeState


DEPENDENTS: dict[str, list[str]] = {
    "production_profile": [
        "tts",
        "asr",
        "subtitles",
        "style_decision",
        "illustration_plan",
        "representative_review",
        "illustration_images",
        "visual_render",
        "render",
        "qc",
        "social_cover",
        "delivery",
    ],
    "voice_profile": [
        "tts",
        "asr",
        "subtitles",
        "illustration_plan",
        "representative_review",
        "illustration_images",
        "visual_render",
        "render",
        "qc",
        "delivery",
    ],
    "script": [
        "tts",
        "asr",
        "subtitles",
        "style_decision",
        "illustration_plan",
        "representative_review",
        "illustration_images",
        "visual_render",
        "storyboard",
        "continuity",
        "h3_prompts",
        "h3_clips",
        "render",
        "qc",
        "social_cover",
        "delivery",
    ],
    "voice_reference": [
        "tts",
        "asr",
        "subtitles",
        "illustration_plan",
        "representative_review",
        "illustration_images",
        "visual_render",
        "storyboard",
        "continuity",
        "h3_prompts",
        "h3_clips",
        "render",
        "qc",
    ],
    "tts": [
        "asr",
        "subtitles",
        "illustration_plan",
        "representative_review",
        "illustration_images",
        "visual_render",
        "storyboard",
        "continuity",
        "h3_prompts",
        "h3_clips",
        "render",
        "qc",
    ],
    "asr": [
        "subtitles",
        "illustration_plan",
        "representative_review",
        "illustration_images",
        "visual_render",
        "storyboard",
        "continuity",
        "h3_prompts",
        "h3_clips",
        "render",
        "qc",
    ],
    "storyboard": [
        "continuity",
        "h3_prompts",
        "h3_clips",
        "render",
        "qc",
    ],
    "continuity": ["h3_prompts", "h3_clips", "render", "qc"],
    "h3_prompts": ["h3_clips", "render", "qc"],
    "h3_clips": ["render", "qc"],
    "subtitles": [
        "plan_illustrations",
        "prepare_representatives",
        "illustration_plan",
        "representative_review",
        "illustration_images",
        "visual_render",
        "render",
        "qc",
    ],
    "style_decision": [
        "select_style",
        "plan_illustrations",
        "prepare_representatives",
        "illustration_plan",
        "illustration_anchor",
        "illustration_continuation",
        "representative_review",
        "illustration_images",
        "visual_render",
        "render",
        "qc",
    ],
    "character_lock": [
        "illustration_plan",
        "representative_review",
        "illustration_images",
        "visual_render",
        "render",
        "qc",
    ],
    "illustration_plan": [
        "representative_review",
        "illustration_images",
        "visual_render",
        "render",
        "qc",
    ],
    "illustration_prompt": [
        "representative_review",
        "illustration_images",
        "visual_render",
        "render",
        "qc",
    ],
    "representative_illustration_prompt": [
        "representative_review",
        "illustration_images",
        "visual_render",
        "render",
        "qc",
    ],
    "illustration_anchor": [
        "illustration_continuation",
        "representative_review",
        "illustration_images",
        "visual_render",
        "render",
        "qc",
        "delivery",
        "final_approval",
    ],
    "illustration_continuation": [
        "representative_review",
        "illustration_images",
        "visual_render",
        "render",
        "qc",
    ],
    "batch_illustration_prompt": ["illustration_images", "visual_render", "render", "qc"],
    "representative_review": ["illustration_images", "visual_render", "render", "qc"],
    "illustration_images": ["visual_render", "render", "qc"],
    "visual_render": ["render", "qc"],
    "book_cover": ["render", "qc"],
    "subtitle_style": ["render", "qc"],
    "ambience": ["render", "qc"],
    "qc": ["delivery", "final_approval"],
    "social_cover": ["delivery", "final_approval"],
    "delivery": ["final_approval"],
}


def invalidate_from(episode: EpisodeState, stage: str) -> EpisodeState:
    eligible = set(episode.completed_stages) | set(episode.stale_stages)
    stale: list[str] = []
    for existing in episode.stale_stages:
        if existing not in stale:
            stale.append(existing)

    pending = list(DEPENDENTS.get(stage, []))
    visited: set[str] = set()
    while pending:
        dependent = pending.pop(0)
        if dependent in visited:
            continue
        visited.add(dependent)
        qc_guard = dependent == "qc" and "render" in eligible
        if (dependent in eligible or qc_guard) and dependent not in stale:
            stale.append(dependent)
        for downstream in DEPENDENTS.get(dependent, []):
            if downstream not in visited and downstream not in pending:
                pending.append(downstream)

    episode.stale_stages = stale
    episode.completed_stages = [
        completed for completed in episode.completed_stages if completed not in stale
    ]
    for stale_stage in stale:
        manifest = episode.stage_manifests.get(stale_stage)
        if manifest is not None:
            manifest.status = "stale"
    return episode
