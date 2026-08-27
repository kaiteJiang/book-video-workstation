"""Per-command external authorization and runtime assembly contracts.

This module is the only place that owns the current command's
``--allow-external`` flag.  It must not persist that flag, and public
errors may carry only a stable code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, TypeVar

from bv.config import AppConfig
from bv.content.human_writing import HumanWritingService
from bv.content.scripts import ScriptService
from bv.content.topics import TopicService
from bv.models.codex_cli import CodexCliModel
from bv.models.grok_cli import GrokCliModel
from bv.models.grok_research import GrokResearchModel
from bv.models.prompts import load_prompt
from bv.state.store import StateStore

from .stages import StageRunner

T = TypeVar("T")


@dataclass(frozen=True)
class RuntimeAuthorization:
    allow_external: bool = False
    book_id: str | None = None
    episode_id: str | None = None
    script_sha256: str | None = None
    provider_voice_id: str | None = None
    audition_voice_ids: tuple[str, ...] = ()
    max_audition_submissions: int = 0
    max_formal_submissions: int = 0
    allow_fallback: bool = False

    def assert_audition_batch(
        self,
        *,
        book_id: str,
        episode_id: str,
        script_sha256: str,
        provider_voice_ids: tuple[str, ...],
    ) -> None:
        if not self.allow_external:
            raise ExternalAuthorizationError()
        if (
            (self.book_id, self.episode_id, self.script_sha256)
            != (book_id, episode_id, script_sha256)
            or not provider_voice_ids
            or provider_voice_ids != self.audition_voice_ids
            or self.max_audition_submissions < len(provider_voice_ids)
            or self.max_formal_submissions != 0
            or self.allow_fallback
        ):
            raise ExternalAuthorizationError("external_scope_mismatch")

    def assert_tts_submission(
        self,
        *,
        book_id: str,
        episode_id: str,
        script_sha256: str,
        provider_voice_id: str,
        kind: Literal["audition", "formal"],
    ) -> None:
        if not self.allow_external:
            raise ExternalAuthorizationError()
        expected = (
            self.book_id,
            self.episode_id,
            self.script_sha256,
            self.provider_voice_id,
        )
        supplied = (book_id, episode_id, script_sha256, provider_voice_id)
        limit = (
            self.max_audition_submissions
            if kind == "audition"
            else self.max_formal_submissions
        )
        if expected != supplied or limit < 1 or self.allow_fallback:
            raise ExternalAuthorizationError("external_scope_mismatch")


class ExternalAuthorizationError(RuntimeError):
    def __init__(self, error_code: str = "external_not_authorized") -> None:
        self.error_code = error_code
        super().__init__(self.error_code)


@dataclass(frozen=True)
class RuntimeBindings:
    stages: Mapping[str, StageRunner]
    gate_approvers: Mapping[str, StageRunner]


def require_external_authorization(authorization: RuntimeAuthorization) -> None:
    if not authorization.allow_external:
        raise ExternalAuthorizationError()


def invoke_external(authorization: RuntimeAuthorization, operation: Callable[[], T]) -> T:
    require_external_authorization(authorization)
    return operation()


def build_runtime_bindings(
    config: AppConfig,
    authorization: RuntimeAuthorization,
) -> RuntimeBindings:
    from bv.workflow.content_runtime import (
        BookAnalysisStage,
        BookValueStage,
        EpisodeBriefStage,
        ScriptApprovalStage,
        ScriptDraftStage,
        TitleOrFileParseStage,
    )

    prompt_root = Path(__file__).resolve().parents[3] / "prompts"
    topic_paths = {
        "topic": prompt_root / "topic_generation" / "v2.md",
        "dedup": prompt_root / "topic_dedup" / "v2.md",
        "grok": prompt_root / "grok_topic_review" / "v2.md",
    }
    script_paths = {
        "draft": prompt_root / "script_draft" / "v2.md",
        "fact_diff": prompt_root / "script_fact_diff" / "v2.md",
        "grok": prompt_root / "grok_script_review" / "v2.md",
    }
    codex = CodexCliModel(
        command=config.codex.command, timeout=float(config.codex.timeout_seconds)
    )
    grok = GrokCliModel(
        command=config.grok.command, timeout=float(config.grok.timeout_seconds)
    )
    research = GrokResearchModel(
        command=config.grok.command, timeout=float(config.grok.timeout_seconds)
    )
    human_writing = HumanWritingService(
        model=codex,
        skill_path=config.human_writing.skill_path,
        checker_path=config.human_writing.checker_path,
    )
    store = StateStore(config.workspace_dir)

    def topic_service(book_root: Path) -> TopicService:
        return TopicService(
            book_root,
            model=codex,
            grok_model=grok,
            topic_prompt_text=load_prompt(topic_paths["topic"]).text,
            dedup_prompt_text=load_prompt(topic_paths["dedup"]).text,
            grok_prompt_text=load_prompt(topic_paths["grok"]).text,
        )

    def script_service(book_root: Path) -> ScriptService:
        prompt_texts = {name: load_prompt(path) for name, path in script_paths.items()}
        return ScriptService(
            book_root,
            model=codex,
            grok_model=grok,
            human_writing=human_writing,
            draft_prompt_text=prompt_texts["draft"].text,
            fact_diff_prompt_text=prompt_texts["fact_diff"].text,
            grok_prompt_text=prompt_texts["grok"].text,
            prompt_hashes={name: prompt.sha256 for name, prompt in prompt_texts.items()},
        )

    stages: dict[str, StageRunner] = {
        "parse_source": TitleOrFileParseStage(store),
        "analyze_chapters": BookAnalysisStage(
            store,
            authorization=authorization,
            research_model=research,
            chapter_model=codex,
        ),
        "synthesize_book_value": BookValueStage(
            store,
            authorization=authorization,
            file_model=codex,
            title_model=codex,
        ),
        "generate_episode_brief": EpisodeBriefStage(
            store,
            authorization=authorization,
            service_factory=topic_service,
            topic_prompt_path=topic_paths["topic"],
            dedup_prompt_path=topic_paths["dedup"],
            grok_prompt_path=topic_paths["grok"],
        ),
        "draft_script": ScriptDraftStage(
            store,
            authorization=authorization,
            service_factory=script_service,
            human_writing_skill_path=config.human_writing.skill_path,
            draft_prompt_path=script_paths["draft"],
            fact_diff_prompt_path=script_paths["fact_diff"],
            grok_prompt_path=script_paths["grok"],
        ),
    }
    return RuntimeBindings(
        stages=stages,
        gate_approvers={
            "script": ScriptApprovalStage(
                store,
                topic_service_factory=topic_service,
                human_writing_skill_path=config.human_writing.skill_path,
                draft_prompt_path=script_paths["draft"],
                fact_diff_prompt_path=script_paths["fact_diff"],
                grok_prompt_path=script_paths["grok"],
            )
        },
    )


def build_media_runtime_bindings(
    config: AppConfig,
    authorization: RuntimeAuthorization,
    *,
    mode: Literal["technical_sample", "full"],
    sample_span: tuple[int, int] | None = None,
) -> "MediaRuntimeBindings":
    """Build the internal Codex-driven hand-drawn media runtime.

    This intentionally has no CLI registration and no Grok dependency. Provider
    calls remain guarded by the per-invocation authorization object.
    """
    import httpx

    from bv.asr.volcengine import VolcCredentials
    from bv.voice.audition import VoiceAuditionService
    from bv.voice.doubao import (
        DoubaoCredentials,
        DoubaoNarrationSynthesizer,
    )
    from bv.voice.providers import IndexTTS2NarrationSynthesizer
    from bv.workflow.media_runtime import (
        LocalIllustrationGateway,
        LocalCoverGateway,
        LocalSocialCoverGateway,
        LocalDeliveryGateway,
        MediaProductionService,
        MediaRuntimeBindings,
    )
    from bv.workflow.media_stages import (
        AsrStage,
        IllustrationPlanningStage,
        FinalQcStage,
        FinalRenderStage,
        PrepareIllustrationsStage,
        StyleSelectionStage,
        SubtitleStage,
        TtsStage,
        VisualRenderStage,
    )

    prompt_root = Path(__file__).resolve().parents[3] / "prompts" / "illustration"
    catalog_path = (
        config.handdrawn.vendor_dir / "references" / "handdrawn-style-library.json"
    )
    codex = CodexCliModel(
        command=config.codex.command,
        timeout=float(config.codex.timeout_seconds),
    )
    credentials = VolcCredentials(
        api_key=os.getenv(config.volc_asr.api_key_env),
        app_key=os.getenv(config.volc_asr.app_key_env),
        access_key=os.getenv(config.volc_asr.access_key_env),
    )
    narration_synthesizers: dict[str, object] = {
        "indextts2": IndexTTS2NarrationSynthesizer(
            config=config.indextts2,
            reference_voice_path=config.voice_path,
        )
    }
    doubao_credentials = DoubaoCredentials.from_environment(config.doubao_tts)
    if all(value == "SET" for value in doubao_credentials.state().values()):
        narration_synthesizers["doubao"] = DoubaoNarrationSynthesizer(
            config=config.doubao_tts,
            credentials=doubao_credentials,
            http=httpx.Client(timeout=config.doubao_tts.timeout_seconds),
        )
    stages: dict[str, StageRunner] = {
        "tts": TtsStage(
            config=config.indextts2,
            reference_voice_path=config.voice_path,
            ffmpeg_command=config.ffmpeg.command,
            mode=mode,
            sample_span=sample_span,
            synthesizers=narration_synthesizers,
            authorization=authorization,
        ),
        "asr": AsrStage(
            credentials=credentials,
            endpoint=config.volc_asr.endpoint,
            authorization=authorization,
        ),
        "subtitles": SubtitleStage(
            font_path=config.subtitle_font_path,
            font_family=config.subtitle_font_family,
        ),
        "select_style": StyleSelectionStage(
            model=codex,
            authorization=authorization,
            catalog_path=catalog_path,
            prompt_path=prompt_root / "style_select.md",
        ),
        "plan_illustrations": IllustrationPlanningStage(
            model=codex,
            authorization=authorization,
            prompt_path=prompt_root / "storyboard.md",
        ),
        "prepare_representatives": PrepareIllustrationsStage(
            catalog_path=catalog_path,
        ),
        "visual_render": VisualRenderStage(
            vendor_dir=config.handdrawn.vendor_dir,
            npm_command=config.handdrawn.node_command,
            ffprobe_command=config.ffmpeg.ffprobe_command,
            safe_area=config.handdrawn.safe_area,
            transition=config.handdrawn.transition,
        ),
        "render": FinalRenderStage(
            ffmpeg_command=config.ffmpeg.command,
            ffprobe_command=config.ffmpeg.ffprobe_command,
        ),
        "qc": FinalQcStage(ffprobe_command=config.ffmpeg.ffprobe_command),
    }
    store = StateStore(config.workspace_dir)
    service = MediaProductionService(
        store=store,
        stages=stages,
        images=LocalIllustrationGateway(ffmpeg_command=config.ffmpeg.command),
        configured_mode=mode,
        cover=LocalCoverGateway(),
        social_cover=LocalSocialCoverGateway(
            ffmpeg_command=config.ffmpeg.command,
        ),
        delivery=LocalDeliveryGateway(store=store),
        voice_auditions=VoiceAuditionService(
            synthesizers=narration_synthesizers,
        ),
    )
    return MediaRuntimeBindings(service=service, stages=stages, gate_approvers={})
