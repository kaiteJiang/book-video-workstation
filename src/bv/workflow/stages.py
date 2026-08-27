"""Small, explicit contracts for the command-line workflow orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol

from bv.state.models import EpisodeState


STAGE_ORDER = [
    "parse_source",
    "analyze_chapters",
    "synthesize_book_value",
    "generate_episode_brief",
    "draft_script",
    "review_script",
    "tts",
    "asr",
    "subtitles",
    "storyboard",
    "await_video_segments",
    "render",
    "qc",
    "delivery",
    "final_approval",
]

GATES = {
    "review_script": "awaiting_script_review",
    "await_video_segments": "awaiting_video_generation",
    "qc": "awaiting_final_review",
}

HANDDRAWN_MEDIA_PREPARE_ORDER = [
    "tts",
    "asr",
    "subtitles",
    "select_style",
    "plan_illustrations",
    "prepare_representatives",
]

HANDDRAWN_MEDIA_STAGE_ORDER = [
    *HANDDRAWN_MEDIA_PREPARE_ORDER,
    "review_representatives",
    "await_illustrations",
    "social_cover",
    "visual_render",
    "render",
    "qc",
]


@dataclass(frozen=True)
class StageContext:
    book_id: str
    episode_id: str
    episode_root: Path
    episode_state: EpisodeState | None = None


@dataclass(frozen=True)
class StageOutcome:
    """The local artifacts a stage made durable before it reports success."""

    outputs: Mapping[str, Path] = field(default_factory=dict)
    inputs: Mapping[str, str] = field(default_factory=dict)


class StageRunner(Protocol):
    def run(self, context: StageContext) -> StageOutcome | Mapping[str, object]: ...
