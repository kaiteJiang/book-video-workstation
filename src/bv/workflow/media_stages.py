from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bv.asr.alignment import AlignedScript, align_approved_text
from bv.asr.volcengine import AsrResult, FlashRequest, VolcCredentials, recognize_flash
from bv.config import IndexTTS2Config
from bv.content.scripts import ScriptPackage, SemanticLock
from bv.content.story_visual import load_approved_story_visual_context
from bv.core.atomic import atomic_write_json
from bv.core.hashing import sha256_file
from bv.illustration.assets import prepare_image_jobs
from bv.illustration.assets import (
    IllustrationAssetError,
    IllustrationManifest,
    validate_generated_image_job,
)
from bv.illustration.contracts import (
    CharacterBible,
    CharacterLock,
    IllustrationStoryboard,
    SafeArea,
    SilentRenderRequest,
    StyleDecision,
    load_character_bible,
)
from bv.illustration.planner import plan_illustrations
from bv.illustration.story_units import load_story_units
from bv.illustration.prompts import canonical_model_sha256
from bv.illustration.renderer import render_silent_story
from bv.illustration.style_selector import (
    build_narrative_profile,
    load_style_catalog,
    rank_style_candidates,
    select_style,
    style_candidates_with_override,
)
from bv.models.contracts import StructuredModel
from bv.models.prompts import load_prompt
from bv.storyboard.generate import build_reader_value_contract
from bv.subtitles.generate import (
    SubtitleCue,
    SubtitleManifest,
    build_cues,
    build_subtitle_manifest,
    render_ass,
    render_srt,
)
from bv.subtitles.inputs import SubtitleBreakInputError, load_subtitle_break_input
from bv.voice.indextts2 import SynthesizedVoice, synthesize_voice
from bv.voice.audition import VoiceAuditionError, require_current_voice_profile
from bv.voice.providers import NarrationRequest, NarrationSynthesizer
from bv.production.profile import load_production_profile, production_profile_sha256
from bv.voice.processing import (
    FULL_EPISODE_POLICY,
    TECHNICAL_SAMPLE_POLICY,
    NarrationDurationPolicy,
    VoiceManifest,
    build_voice_master,
    duration_policy_from,
)
from bv.video.cover import CoverManifest
from bv.video.qc import inspect_final
from bv.video.render import (
    CoverOverlay,
    HanddrawnRenderInputs,
    HanddrawnRenderManifest,
    render_handdrawn_final,
)

from .runtime import RuntimeAuthorization, invoke_external, require_external_authorization
from .stages import StageContext, StageOutcome


class MediaStageError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class NarrationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    text: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_span: tuple[int, int]
    projection_path: Path | None = None


def approved_narration_input(
    context: StageContext,
    *,
    mode: Literal["technical_sample", "full"],
    sample_span: tuple[int, int] | None = None,
) -> NarrationInput:
    state = context.episode_state
    approved = context.episode_root / "script" / "approved.txt"
    if state is None or state.script_hash is None or state.status in {
        "source_imported",
        "source_invalid",
        "source_parsed",
        "chapter_analysis_running",
        "chapter_analysis_failed",
        "book_analysis_ready",
        "topic_selected",
        "insufficient_distinct_topic",
        "script_drafting",
        "evidence_invalid",
        "awaiting_script_review",
    }:
        raise MediaStageError("script_not_approved")
    try:
        raw = approved.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError):
        raise MediaStageError("approved_script_invalid") from None
    source_hash = hashlib.sha256(raw).hexdigest()
    if source_hash != state.script_hash or not text.strip():
        raise MediaStageError("approved_script_invalid")
    if mode == "full":
        if sample_span is not None:
            raise MediaStageError("sample_span_not_allowed")
        return NarrationInput(
            path=approved, text=text, sha256=source_hash,
            source_script_sha256=source_hash, source_span=(0, len(text)),
        )
    if mode != "technical_sample" or sample_span is None:
        raise MediaStageError("sample_span_required")
    start, end = sample_span
    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, int) or not isinstance(end, int)
        or start < 0 or end <= start or end > len(text)
    ):
        raise MediaStageError("sample_span_invalid")
    sample = text[start:end]
    if not sample.strip():
        raise MediaStageError("sample_span_invalid")
    media_script = context.episode_root / "media" / "script"
    sample_path = media_script / "sample.txt"
    projection_path = media_script / "sample_projection.json"
    _atomic_text(sample_path, sample)
    sample_hash = hashlib.sha256(sample.encode("utf-8")).hexdigest()
    atomic_write_json(
        projection_path,
        {
            "source_script_sha256": source_hash,
            "source_span": [start, end],
            "sample_sha256": sample_hash,
        },
    )
    return NarrationInput(
        path=sample_path, text=sample, sha256=sample_hash,
        source_script_sha256=source_hash, source_span=(start, end),
        projection_path=projection_path,
    )


class TtsStage:
    def __init__(
        self,
        *,
        config: IndexTTS2Config,
        reference_voice_path: Path,
        ffmpeg_command: str | Path,
        mode: Literal["technical_sample", "full"],
        sample_span: tuple[int, int] | None = None,
        synthesizer: Callable[..., SynthesizedVoice] = synthesize_voice,
        master_builder: Callable[..., VoiceManifest] = build_voice_master,
        synthesizers: Mapping[str, NarrationSynthesizer] | None = None,
        authorization: RuntimeAuthorization | None = None,
    ) -> None:
        self.config = config
        self.reference_voice_path = Path(reference_voice_path)
        self.ffmpeg_command = ffmpeg_command
        self.mode = mode
        self.sample_span = sample_span
        self.synthesizer = synthesizer
        self.master_builder = master_builder
        self.synthesizers = None if synthesizers is None else dict(synthesizers)
        self.authorization = authorization or RuntimeAuthorization()

    def run(self, context: StageContext) -> StageOutcome:
        narration = approved_narration_input(
            context, mode=self.mode, sample_span=self.sample_span
        )
        if self.synthesizers is not None:
            return self._run_provider_neutral(context, narration)
        if self.config.voice_id != "黑金3":
            raise MediaStageError("voice_profile_mismatch")
        voice_root = context.episode_root / "media" / "voice"
        voice_root.mkdir(parents=True, exist_ok=True)
        raw_path = context.episode_root / ".private" / "media" / "voice" / "voice_raw.wav"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        master_path = voice_root / "voice_master.wav"
        manifest_path = master_path.with_suffix(".json")
        master_existed = master_path.exists()
        manifest_existed = manifest_path.exists()
        policy = policy_for(self.mode, episode_root=context.episode_root)
        try:
            synthesized = self.synthesizer(
                episode_root=context.episode_root,
                install_dir=self.config.install_dir,
                model_dir=self.config.model_dir,
                script_path=narration.path,
                approved_script_sha256=narration.sha256,
                reference_voice_path=self.reference_voice_path,
                raw_output_path=raw_path,
                uv_command=self.config.uv_command,
                device=self.config.device,
            )
            manifest = self.master_builder(
                episode_root=context.episode_root,
                raw_path=synthesized.raw_path,
                master_path=master_path,
                script_path=narration.path,
                reference_voice_path=self.reference_voice_path,
                approved_script_sha256=narration.sha256,
                voice_id=self.config.voice_id,
                ffmpeg_command=self.ffmpeg_command,
                duration_policy=policy,
            )
        except Exception:
            raise MediaStageError("tts_stage_failed") from None
        if (
            not master_path.is_file() or not manifest_path.is_file()
            or manifest.master_sha256 != sha256_file(master_path)
            or manifest.script_sha256 != narration.sha256
            or manifest.voice_id != "黑金3"
            or manifest.duration.level != "pass"
        ):
            if not master_existed:
                _remove_file(master_path)
            if not manifest_existed:
                _remove_file(manifest_path)
            raise MediaStageError("tts_artifact_invalid")
        return StageOutcome(
            outputs={"voice_master": master_path, "voice_manifest": manifest_path},
            inputs={
                "script_sha256": narration.sha256,
                "source_script_sha256": narration.source_script_sha256,
                "production_profile_sha256": production_profile_sha256(
                    context.episode_root
                ),
            },
        )

    def _run_provider_neutral(
        self,
        context: StageContext,
        narration: NarrationInput,
    ) -> StageOutcome:
        profile = load_production_profile(context.episode_root)
        provider = profile.tts.provider
        if provider not in self.synthesizers:
            raise MediaStageError("tts_provider_not_configured")
        if provider == "doubao":
            try:
                voice_profile = require_current_voice_profile(
                    context.episode_root,
                    profile,
                    script_sha256=narration.source_script_sha256,
                )
            except VoiceAuditionError as error:
                raise MediaStageError(error.error_code) from None
            voice_id = voice_profile.provider_voice_id
        else:
            voice_id = profile.tts.voice_type
            if voice_id != "黑金3":
                raise MediaStageError("voice_profile_mismatch")
        if voice_id is None:
            raise MediaStageError("voice_profile_not_approved")

        voice_root = context.episode_root / "media" / "voice"
        voice_root.mkdir(parents=True, exist_ok=True)
        private_root = context.episode_root / ".private" / "media" / "voice" / provider
        master_path = voice_root / "voice_master.wav"
        manifest_path = master_path.with_suffix(".json")
        master_existed = master_path.exists()
        manifest_existed = manifest_path.exists()
        request = NarrationRequest(
            book_id=context.book_id,
            episode_id=context.episode_id,
            approved_script_path=narration.path,
            approved_script_sha256=narration.sha256,
            provider_resource_id=profile.tts.resource_id,
            provider_voice_id=voice_id,
            speed=profile.tts.speed,
            output_dir=private_root,
            kind="formal",
        )
        try:
            artifact = self.synthesizers[provider].synthesize(
                request,
                self.authorization,
            )
            kwargs: dict[str, object] = {
                "episode_root": context.episode_root,
                "raw_path": artifact.raw_audio_path,
                "master_path": master_path,
                "script_path": narration.path,
                "approved_script_sha256": narration.sha256,
                "provider": provider,
                "provider_voice_id": voice_id,
                "ffmpeg_command": self.ffmpeg_command,
                "duration_policy": policy_for(
                    self.mode, episode_root=context.episode_root
                ),
            }
            if provider == "doubao":
                kwargs["provider_receipt_path"] = artifact.receipt_path
            else:
                kwargs["reference_voice_path"] = self.reference_voice_path
            manifest = self.master_builder(**kwargs)
        except Exception as error:
            if error.__class__.__name__ in {
                "ExternalAuthorizationError",
                "DoubaoTtsError",
            }:
                raise
            raise MediaStageError("tts_stage_failed") from None
        if (
            not master_path.is_file()
            or not manifest_path.is_file()
            or manifest.master_sha256 != sha256_file(master_path)
            or manifest.script_sha256 != narration.sha256
            or manifest.provider != provider
            or manifest.provider_voice_id != voice_id
            or manifest.duration.level != "pass"
        ):
            if not master_existed:
                _remove_file(master_path)
            if not manifest_existed:
                _remove_file(manifest_path)
            raise MediaStageError("tts_artifact_invalid")
        return StageOutcome(
            outputs={"voice_master": master_path, "voice_manifest": manifest_path},
            inputs={
                "script_sha256": narration.sha256,
                "source_script_sha256": narration.source_script_sha256,
                "provider": provider,
                "provider_voice_id": voice_id,
                "production_profile_sha256": production_profile_sha256(
                    context.episode_root
                ),
            },
        )


class AsrStage:
    def __init__(
        self,
        *,
        credentials: VolcCredentials,
        endpoint: str,
        authorization: RuntimeAuthorization,
        recognizer: Callable[..., AsrResult] = recognize_flash,
        aligner: Callable[..., AlignedScript] = align_approved_text,
    ) -> None:
        self.credentials = credentials
        self.endpoint = endpoint
        self.authorization = authorization
        self.recognizer = recognizer
        self.aligner = aligner

    def run(self, context: StageContext) -> StageOutcome:
        require_external_authorization(self.authorization)
        master = context.episode_root / "media" / "voice" / "voice_master.wav"
        manifest_path = master.with_suffix(".json")
        try:
            manifest = VoiceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError):
            raise MediaStageError("voice_manifest_invalid") from None
        narration = _current_narration(context, manifest.script_sha256)
        if not master.is_file() or sha256_file(master) != manifest.master_sha256:
            raise MediaStageError("voice_manifest_invalid")
        request = FlashRequest(
            audio_path=master,
            audio_sha256=manifest.master_sha256,
            endpoint=self.endpoint,
            longform=load_production_profile(context.episode_root).narrative_mode == "story",
        )
        try:
            result = invoke_external(
                self.authorization,
                lambda: self.recognizer(self.credentials, request, **(
                    {"timeout": 300.0} if request.longform and self.recognizer is recognize_flash else {}
                )),
            )
            aligned = self.aligner(
                narration.text,
                result,
                audio_sha256=manifest.master_sha256,
            )
        except Exception as error:
            if error.__class__.__name__ == "ExternalAuthorizationError":
                raise
            raise MediaStageError("asr_stage_failed") from None
        if aligned.report.level != "pass" or aligned.report.violations:
            raise MediaStageError("tts_asr_mismatch")
        root = context.episode_root / "media" / "alignment"
        alignment_path = root / "alignment.json"
        report_path = root / "pronunciation_report.json"
        atomic_write_json(alignment_path, aligned.model_dump(mode="json"))
        atomic_write_json(report_path, aligned.report.model_dump(mode="json"))
        return StageOutcome(
            outputs={"alignment": alignment_path, "pronunciation_report": report_path},
            inputs={"audio_sha256": manifest.master_sha256, "script_sha256": narration.sha256},
        )


class SubtitleStage:
    def __init__(self, *, font_path: Path, font_family: str) -> None:
        self.font_path = Path(font_path)
        self.font_family = font_family

    def run(self, context: StageContext) -> StageOutcome:
        alignment_path = context.episode_root / "media" / "alignment" / "alignment.json"
        voice_manifest_path = context.episode_root / "media" / "voice" / "voice_master.json"
        try:
            aligned = AlignedScript.model_validate_json(alignment_path.read_text(encoding="utf-8"))
            voice = VoiceManifest.model_validate_json(voice_manifest_path.read_text(encoding="utf-8"))
            break_input = load_subtitle_break_input(context.episode_root)
            semantic_chunks = break_input.chunks
        except (OSError, ValueError, ValidationError, SubtitleBreakInputError):
            raise MediaStageError("subtitle_input_invalid") from None
        if not self.font_path.is_file():
            raise MediaStageError("subtitle_font_invalid")
        font_hash = sha256_file(self.font_path)
        try:
            cues = build_cues(
                aligned,
                audio_duration_ms=voice.master_duration_ms,
                semantic_chunks=semantic_chunks,
            )
            srt = render_srt(cues)
            ass = render_ass(
                cues,
                font_path=self.font_path,
                font_sha256=font_hash,
                font_family=self.font_family,
                width=1080,
                height=1920,
            )
            manifest = build_subtitle_manifest(
                aligned,
                cues,
                audio_duration_ms=voice.master_duration_ms,
                font_path=self.font_path,
                font_sha256=font_hash,
                font_family=self.font_family,
                width=1080,
                height=1920,
            )
        except Exception:
            raise MediaStageError("subtitle_stage_failed") from None
        root = context.episode_root / "media" / "subtitles"
        srt_path = root / "subtitles.srt"
        ass_path = root / "subtitles.ass"
        cues_path = root / "cues.json"
        manifest_path = root / "subtitle_manifest.json"
        _atomic_text(srt_path, srt)
        _atomic_text(ass_path, ass)
        atomic_write_json(cues_path, [cue.model_dump(mode="json") for cue in cues])
        atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
        inputs = {
            "script_sha256": aligned.report.approved_sha256,
            "audio_sha256": aligned.report.audio_sha256,
            "subtitle_sequence_mode": break_input.sequence_mode,
            "subtitle_breaks_presence": (
                "present" if break_input.present else "absent"
            ),
        }
        if break_input.sha256 is not None:
            inputs["subtitle_breaks_sha256"] = break_input.sha256
        return StageOutcome(
            outputs={
                "subtitles_srt": srt_path,
                "subtitles_ass": ass_path,
                "subtitle_cues": cues_path,
                "subtitle_manifest": manifest_path,
            },
            inputs=inputs,
        )


class StyleSelectionStage:
    def __init__(
        self,
        *,
        model: StructuredModel,
        authorization: RuntimeAuthorization,
        catalog_path: Path,
        prompt_path: Path,
    ) -> None:
        self.model = model
        self.authorization = authorization
        self.catalog_path = Path(catalog_path)
        self.prompt_path = Path(prompt_path)

    def run(self, context: StageContext) -> StageOutcome:
        require_external_authorization(self.authorization)
        narration, semantic_lock = _visual_sources(context)
        try:
            whole_narration = approved_narration_input(context, mode="full")
            catalog = load_style_catalog(self.catalog_path)
            profile = build_narrative_profile(
                whole_narration.text,
                build_reader_value_contract(semantic_lock),
            )
            ranked = rank_style_candidates(profile, catalog)
            if len(ranked) < 3:
                raise ValueError
            production_profile = load_production_profile(context.episode_root)
            style_override = production_profile.visual.style_id
            candidates = (
                style_candidates_with_override(ranked, style_override)
                if style_override is not None
                else tuple(ranked[:3])
            )
            prompt = load_prompt(self.prompt_path)
            decision = select_style(
                profile,
                narration.text,
                build_reader_value_contract(semantic_lock),
                candidates,  # type: ignore[arg-type]
                self.model,
                context.episode_root / ".private" / "requests" / "style",
                prompt,
                manual_override=style_override,
                selection_context_script=whole_narration.text,
            )
        except Exception as error:
            if error.__class__.__name__ == "ExternalAuthorizationError":
                raise
            raise MediaStageError("style_selection_failed") from None
        output = context.episode_root / "media" / "illustration" / "style_decision.json"
        atomic_write_json(output, decision.model_dump(mode="json"))
        return StageOutcome(
            outputs={"style_decision": output},
            inputs={
                "script_sha256": narration.sha256,
                "semantic_lock_sha256": semantic_lock.sha256,
                "prompt_sha256": prompt.sha256,
            },
        )


def _approved_story_context(context: StageContext):
    # Validation always binds the complete approved manuscript, even for a sample projection.
    whole = approved_narration_input(context, mode="full")
    return load_approved_story_visual_context(
        context.episode_root, book_id=context.book_id, episode_id=context.episode_id,
        approved_text=whole.text, approved_sha256=whole.sha256,
    )


def _story_scene_boundaries(sections, aligned: AlignedScript, cues: tuple[SubtitleCue, ...], master_duration_ms: int, *, strict: bool = False) -> tuple[int, ...]:
    """Map approved story beats to ASR cue starts, merging unusably short windows."""
    starts = sorted({cue.start_ms for cue in cues if 0 < cue.start_ms < master_duration_ms})
    minimum_ms = 4_000  # A >= 1s plus B reveal/hold/transition >= 3s.
    boundaries: list[int] = []
    for section in sections[1:]:
        timed = next((item for item in aligned.characters[section.start:] if item.start_ms is not None), None)
        if timed is None or not starts:
            continue
        boundary = min(starts, key=lambda value: (abs(value - timed.start_ms), value))
        if boundary - (boundaries[-1] if boundaries else 0) >= minimum_ms and master_duration_ms - boundary >= minimum_ms:
            boundaries.append(boundary)
    if strict:
        if len(sections) < 3 or len(boundaries) != len(sections) - 1:
            raise ValueError("ab_units_need_replanning_not_mechanical_split")
        return tuple(boundaries)
    # A coarse three-act manuscript can still need a minimum of three reviewed pairs.
    # Split the largest surviving window at a real sentence-ending cue where possible.
    cue_by_start = {cue.start_ms: index for index, cue in enumerate(cues)}
    while len(boundaries) < 2:
        edges = [0, *boundaries, master_duration_ms]
        spans = sorted(zip(edges, edges[1:]), key=lambda pair: pair[1] - pair[0], reverse=True)
        chosen = None
        for left, right in spans:
            options = [value for value in starts if value - left >= minimum_ms and right - value >= minimum_ms]
            if options:
                def score(value):
                    index = cue_by_start[value]
                    sentence_end = index > 0 and cues[index - 1].text.rstrip().endswith(("。", "！", "？", ".", "!", "?"))
                    return (not sentence_end, abs(value - (left + right) / 2), value)
                chosen = min(options, key=score)
                break
        if chosen is None:
            raise ValueError("story_semantic_windows_too_short")
        boundaries.append(chosen)
        boundaries.sort()
    while len(boundaries) > 47:
        edges = [0, *boundaries, master_duration_ms]
        remove = min(range(len(boundaries)), key=lambda index: edges[index + 2] - edges[index])
        boundaries.pop(remove)
    return tuple(boundaries)


class IllustrationPlanningStage:
    def __init__(
        self,
        *,
        model: StructuredModel,
        authorization: RuntimeAuthorization,
        prompt_path: Path,
    ) -> None:
        self.model = model
        self.authorization = authorization
        self.prompt_path = Path(prompt_path)

    def run(self, context: StageContext) -> StageOutcome:
        require_external_authorization(self.authorization)
        narration, semantic_lock = _visual_sources(context)
        root = context.episode_root / "media"
        try:
            aligned = AlignedScript.model_validate_json(
                (root / "alignment" / "alignment.json").read_text(encoding="utf-8")
            )
            voice = VoiceManifest.model_validate_json(
                (root / "voice" / "voice_master.json").read_text(encoding="utf-8")
            )
            cues_payload = json.loads(
                (root / "subtitles" / "cues.json").read_text(encoding="utf-8")
            )
            cues = tuple(SubtitleCue.model_validate(item) for item in cues_payload)
            decision = StyleDecision.model_validate_json(
                (root / "illustration" / "style_decision.json").read_text(encoding="utf-8")
            )
            prompt = load_prompt(self.prompt_path)
            production_profile = load_production_profile(context.episode_root)
            book_payload = json.loads(
                (context.episode_root.parents[1] / "book.json").read_text(
                    encoding="utf-8"
                )
            )
            book_title = book_payload.get("title")
            if not isinstance(book_title, str) or not book_title.strip():
                raise ValueError("book_title_invalid")
            story = _approved_story_context(context)
            story_options: dict[str, object] = {}
            scene_count = production_profile.visual.scene_count
            if story is not None:
                source_start, source_end = narration.source_span
                full_text = (context.episode_root / "script" / "approved.txt").read_text(encoding="utf-8")
                units = load_story_units(context.episode_root / "script" / "ab_units.json", full_text)
                selected = tuple(unit for unit in units if unit.start < source_end and unit.end > source_start)
                if any(unit.start < source_start or unit.end > source_end for unit in selected):
                    raise ValueError("ab_sample_must_cover_whole_units")
                sections = tuple(replace(unit, start=unit.start-source_start, end=unit.end-source_start) for unit in selected)
                boundaries = _story_scene_boundaries(sections, aligned, cues, voice.master_duration_ms, strict=True)
                scene_count = len(boundaries) + 1
                cast = tuple(item.model_dump(mode="json") for item in story.characters)
                if not cast:
                    cast = ({"character_id": "source-protagonist", "name": "故事中心人物", "description": "依据批准文稿中明确的人物身份绘制；不得将第三人称改为第一人称", "evidence_ids": []},)
                story_options = {
                    "semantic_boundary_ms": boundaries,
                    "important_events": tuple(json.dumps(section.data, ensure_ascii=False) for section in sections),
                    "representative_scene_ids": ("S01", f"S{scene_count // 2 + 1:02d}", f"S{scene_count:02d}"),
                    "story_characters": cast,
                    "story_point_of_view": story.point_of_view,
                }
            character, storyboard = plan_illustrations(
                book_id=context.book_id,
                approved_text=narration.text,
                aligned_script=aligned,
                cues=cues,
                master_duration_ms=voice.master_duration_ms,
                semantic_lock=semantic_lock,
                style_decision=decision,
                model=self.model,
                request_root=context.episode_root / ".private" / "requests" / "illustration",
                prompt_asset=prompt,
                seconds_per_scene_min=(
                    production_profile.visual.seconds_per_scene_min
                ),
                seconds_per_scene_max=(
                    production_profile.visual.seconds_per_scene_max
                ),
                target_scene_count=scene_count,
                **story_options,
                book_title=book_title,
                sequence_mode=production_profile.visual.sequence_mode,
            )
            if story is not None:
                if len(storyboard.scenes) != len(selected):
                    raise ValueError("ab_unit_count_changed")
                for scene, unit in zip(storyboard.scenes, selected, strict=True):
                    expected = unit.start - source_start + full_text[unit.start:unit.end].index(unit.data["b_entry_text"])
                    if scene.semantic_turn_span is None or scene.semantic_turn_span[1] != expected:
                        raise ValueError("ab_result_reveal_drift")
        except Exception as error:
            if error.__class__.__name__ == "ExternalAuthorizationError":
                raise
            raise MediaStageError("illustration_plan_failed") from None
        illustration_root = root / "illustration"
        character_path = illustration_root / "character_bible.json"
        storyboard_path = illustration_root / "illustration_storyboard.json"
        atomic_write_json(character_path, character.model_dump(mode="json"))
        atomic_write_json(storyboard_path, storyboard.model_dump(mode="json"))
        return StageOutcome(
            outputs={"character_bible": character_path, "illustration_storyboard": storyboard_path},
            inputs={
                **({"source_contract_sha256": story.source_contract_sha256,
                    "ab_units_sha256": sha256_file(context.episode_root / 'script' / 'ab_units.json')} if story is not None else {}),
                "script_sha256": narration.sha256,
                "audio_sha256": voice.master_sha256,
                "prompt_sha256": prompt.sha256,
            },
        )


class PrepareIllustrationsStage:
    def __init__(self, *, catalog_path: Path) -> None:
        self.catalog_path = Path(catalog_path)

    def run(self, context: StageContext) -> StageOutcome:
        root = context.episode_root / "media" / "illustration"
        try:
            storyboard = IllustrationStoryboard.model_validate_json(
                (root / "illustration_storyboard.json").read_text(encoding="utf-8")
            )
            bible_path = root / "character_bible.json"
            legacy_path = root / "character_lock.json"
            payload = (bible_path if bible_path.is_file() else legacy_path).read_text(
                encoding="utf-8"
            )
            character = load_character_bible(json.loads(payload))
            decision = StyleDecision.model_validate_json(
                (root / "style_decision.json").read_text(encoding="utf-8")
            )
            catalog = load_style_catalog(self.catalog_path)
            style = next(item for item in catalog if item.style_id == decision.selected_style)
            manifest = prepare_image_jobs(
                storyboard, character, style, decision, context.episode_root
            )
        except Exception:
            raise MediaStageError("illustration_jobs_failed") from None
        output = root / "illustration_manifest.json"
        atomic_write_json(output, manifest.model_dump(mode="json"))
        return StageOutcome(
            outputs={"illustration_manifest": output},
            inputs={
                "storyboard_sha256": manifest.storyboard_sha256,
                "style_fingerprint": manifest.style_fingerprint,
                "character_lock_sha256": manifest.character_lock_sha256,
            },
        )


def build_renderer_storyboard(
    storyboard: IllustrationStoryboard,
    manifest: IllustrationManifest,
    *,
    safe_area: SafeArea,
    transition: Literal["cut", "page-flip", "cross-dissolve"],
    title_overlay: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if manifest.storyboard_sha256 != canonical_model_sha256(storyboard):
        raise MediaStageError("illustration_dependency_mismatch")
    legacy_jobs = {job.scene_id: job for job in manifest.jobs if job.phase == "legacy"}
    pair_jobs = {
        (job.scene_id, job.phase): job
        for job in manifest.jobs
        if job.phase in {"anchor", "continuation"}
    }
    scenes: list[dict[str, object]] = []
    for scene in storyboard.scenes:
        base = {
            "id": scene.scene_id,
            "start_ms": scene.start_ms,
            "end_ms": scene.end_ms,
            "from_frame": scene.from_frame,
            "to_frame": scene.to_frame,
            "key_line": scene.key_line,
            "narration": scene.narration,
        }
        if storyboard.sequence_mode == "color-story-pair":
            anchor = pair_jobs.get((scene.scene_id, "anchor"))
            continuation = pair_jobs.get((scene.scene_id, "continuation"))
            if anchor is None or continuation is None:
                raise MediaStageError("illustration_job_missing")
            try:
                validate_generated_image_job(anchor)
                validate_generated_image_job(continuation, manifest=manifest)
                anchor_path = anchor.output_master.relative_to(manifest.episode_root).as_posix()
                continuation_path = continuation.output_master.relative_to(
                    manifest.episode_root
                ).as_posix()
            except (IllustrationAssetError, ValueError, TypeError):
                raise MediaStageError("illustration_asset_invalid") from None
            scenes.append({
                **base,
                "sequence_mode": "color-story-pair",
                "semantic_turn_frame": scene.semantic_turn_frame,
                "assets": {
                    "anchor": anchor_path,
                    "continuation": continuation_path,
                },
            })
        else:
            job = legacy_jobs.get(scene.scene_id)
            if job is None:
                raise MediaStageError("illustration_job_missing")
            try:
                validate_generated_image_job(job)
                if job.output_bw is None:
                    raise ValueError
                bw = job.output_bw.relative_to(manifest.episode_root).as_posix()
                color = job.output_master.relative_to(manifest.episode_root).as_posix()
            except (IllustrationAssetError, ValueError, TypeError):
                raise MediaStageError("illustration_asset_invalid") from None
            scenes.append({**base, "assets": {"bw": bw, "color": color}})
    payload: dict[str, object] = {
        "project": {
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "ratio": "9:16",
            "total_frames": storyboard.total_frames,
            "transition": transition,
            "transition_frames": (
                18 if transition == "page-flip"
                else 15 if transition == "cross-dissolve"
                else 0
            ),
            "show_key_line": title_overlay is None,
        },
        "safe_area": safe_area.model_dump(mode="json"),
        "scenes": scenes,
    }
    if storyboard.sequence_mode == "color-story-pair":
        payload["project"]["ink_reveal_frames"] = 45
    if title_overlay is not None:
        title = title_overlay.get("title", "").strip()
        author = title_overlay.get("author", "").strip()
        if not title or not author:
            raise MediaStageError("book_title_overlay_invalid")
        payload["title_overlay"] = {"title": title, "author": author}
    return payload


def build_renderer_asset_sha256s(
    storyboard: IllustrationStoryboard,
    manifest: IllustrationManifest,
) -> dict[str, str]:
    """Bind the renderer request to the hashes recorded in validated image jobs."""
    if manifest.storyboard_sha256 != canonical_model_sha256(storyboard):
        raise MediaStageError("illustration_dependency_mismatch")
    legacy_jobs = {job.scene_id: job for job in manifest.jobs if job.phase == "legacy"}
    pair_jobs = {
        (job.scene_id, job.phase): job
        for job in manifest.jobs
        if job.phase in {"anchor", "continuation"}
    }
    asset_sha256s: dict[str, str] = {}

    def record(path: Path, expected_hash: str | None) -> None:
        if expected_hash is None:
            raise ValueError
        relative_path = path.relative_to(manifest.episode_root).as_posix()
        existing = asset_sha256s.get(relative_path)
        if existing is not None and existing != expected_hash:
            raise ValueError
        asset_sha256s[relative_path] = expected_hash

    for scene in storyboard.scenes:
        try:
            if storyboard.sequence_mode == "color-story-pair":
                anchor = pair_jobs.get((scene.scene_id, "anchor"))
                continuation = pair_jobs.get((scene.scene_id, "continuation"))
                if anchor is None or continuation is None:
                    raise MediaStageError("illustration_job_missing")
                validate_generated_image_job(anchor)
                validate_generated_image_job(continuation, manifest=manifest)
                record(anchor.output_master, anchor.master_sha256)
                record(continuation.output_master, continuation.master_sha256)
            else:
                job = legacy_jobs.get(scene.scene_id)
                if job is None:
                    raise MediaStageError("illustration_job_missing")
                validate_generated_image_job(job)
                if job.output_bw is None:
                    raise ValueError
                record(job.output_bw, job.bw_sha256)
                record(job.output_master, job.master_sha256)
        except MediaStageError:
            raise
        except (IllustrationAssetError, ValueError, TypeError):
            raise MediaStageError("illustration_asset_invalid") from None
    return asset_sha256s


def _book_title_overlay(episode_root: Path) -> dict[str, str] | None:
    book_path = episode_root.parents[1] / "book.json"
    if not book_path.is_file():
        return None
    payload = json.loads(book_path.read_text(encoding="utf-8"))
    title = payload.get("title")
    authors = payload.get("authors")
    if (
        not isinstance(title, str)
        or not title.strip()
        or not isinstance(authors, list)
        or not authors
        or any(not isinstance(author, str) or not author.strip() for author in authors)
    ):
        raise MediaStageError("book_title_overlay_invalid")
    return {
        "title": title.strip(),
        "author": "、".join(author.strip() for author in authors),
    }


class VisualRenderStage:
    def __init__(
        self,
        *,
        vendor_dir: Path,
        npm_command: str,
        ffprobe_command: str | Path,
        safe_area: SafeArea,
        transition: Literal["cut", "page-flip", "cross-dissolve"],
    ) -> None:
        self.vendor_dir = Path(vendor_dir)
        self.npm_command = npm_command
        self.ffprobe_command = str(ffprobe_command)
        self.safe_area = safe_area
        self.transition = transition

    def run(self, context: StageContext) -> StageOutcome:
        root = context.episode_root / "media" / "illustration"
        try:
            storyboard = IllustrationStoryboard.model_validate_json(
                (root / "illustration_storyboard.json").read_text(encoding="utf-8")
            )
            manifest = IllustrationManifest.model_validate_json(
                (root / "illustration_manifest.json").read_text(encoding="utf-8")
            )
            payload = build_renderer_storyboard(
                storyboard,
                manifest,
                safe_area=self.safe_area,
                transition=self.transition,
                title_overlay=_book_title_overlay(context.episode_root),
            )
            asset_sha256s = build_renderer_asset_sha256s(storyboard, manifest)
            props_path = root / "render_storyboard.json"
            atomic_write_json(props_path, payload)
            output = context.episode_root / "media" / "render" / "picture_silent.mp4"
            result = render_silent_story(
                SilentRenderRequest(
                    episode_root=context.episode_root,
                    storyboard_path=props_path,
                    storyboard_sha256=sha256_file(props_path),
                    output_path=output,
                    vendor_dir=self.vendor_dir,
                    asset_sha256s=asset_sha256s,
                ),
                npm_command=self.npm_command,
                ffprobe_command=self.ffprobe_command,
            )
        except Exception:
            raise MediaStageError("visual_render_failed") from None
        return StageOutcome(
            outputs={"render_storyboard": props_path, "picture_silent": result.output_path},
            inputs={
                "storyboard_sha256": manifest.storyboard_sha256,
                "visual_sha256": result.output_sha256,
            },
        )


class FinalRenderStage:
    def __init__(self, *, ffmpeg_command: str | Path, ffprobe_command: str | Path) -> None:
        self.ffmpeg_command = ffmpeg_command
        self.ffprobe_command = ffprobe_command

    def run(self, context: StageContext) -> StageOutcome:
        try:
            inputs = _handdrawn_render_inputs(context)
            result = render_handdrawn_final(
                inputs,
                ffmpeg_command=self.ffmpeg_command,
                ffprobe_command=self.ffprobe_command,
            )
        except Exception:
            raise MediaStageError("final_render_failed") from None
        stage_inputs = {
            "visual_sha256": inputs.visual_sha256,
            "audio_sha256": inputs.audio_sha256,
            "subtitle_sha256": inputs.subtitle_sha256,
            "production_profile_sha256": production_profile_sha256(
                context.episode_root
            ),
        }
        if inputs.cover is not None:
            stage_inputs["cover_sha256"] = inputs.cover.cover_sha256
        return StageOutcome(
            outputs={"final_video": result.final_path, "render_manifest": result.manifest_path},
            inputs=stage_inputs,
        )


class FinalQcStage:
    def __init__(self, *, ffprobe_command: str | Path) -> None:
        self.ffprobe_command = ffprobe_command

    def run(self, context: StageContext) -> StageOutcome:
        try:
            inputs = _handdrawn_render_inputs(context)
            manifest_path = inputs.final_path.with_suffix(".render.json")
            manifest = HanddrawnRenderManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            report = inspect_final(
                inputs.final_path,
                inputs=inputs,
                render_manifest=manifest,
                production_profile=load_production_profile(context.episode_root),
                episode_root=context.episode_root,
                ffprobe_command=self.ffprobe_command,
            )
        except Exception:
            raise MediaStageError("final_qc_failed") from None
        report_path = inputs.final_path.with_suffix(".qc.json")
        atomic_write_json(report_path, report.model_dump(mode="json"))
        return StageOutcome(
            outputs={"quality_report": report_path},
            inputs={
                "final_sha256": report.facts.sha256,
                "production_profile_sha256": production_profile_sha256(
                    context.episode_root
                ),
            },
        )


def policy_for(
    mode: Literal["technical_sample", "full"],
    *,
    episode_root: Path | None = None,
) -> NarrationDurationPolicy:
    if mode == "technical_sample":
        return TECHNICAL_SAMPLE_POLICY
    if episode_root is None:
        return FULL_EPISODE_POLICY
    return duration_policy_from(load_production_profile(episode_root).duration)


def _current_narration(context: StageContext, expected_hash: str) -> NarrationInput:
    projection = context.episode_root / "media" / "script" / "sample_projection.json"
    if projection.is_file():
        try:
            payload = json.loads(projection.read_text(encoding="utf-8"))
            span = tuple(payload["source_span"])
        except (OSError, ValueError, KeyError, TypeError):
            raise MediaStageError("sample_projection_invalid") from None
        narration = approved_narration_input(
            context,
            mode="technical_sample",
            sample_span=span,  # type: ignore[arg-type]
        )
    else:
        narration = approved_narration_input(context, mode="full")
    if narration.sha256 != expected_hash:
        raise MediaStageError("narration_hash_mismatch")
    return narration


def _visual_sources(context: StageContext) -> tuple[NarrationInput, SemanticLock]:
    voice_path = context.episode_root / "media" / "voice" / "voice_master.json"
    package_path = context.episode_root / "script" / "script_package.json"
    try:
        voice = VoiceManifest.model_validate_json(voice_path.read_text(encoding="utf-8"))
        narration = _current_narration(context, voice.script_sha256)
        story = _approved_story_context(context)
        if story is not None:
            lock = story.semantic_lock
        elif os.path.lexists(package_path):
            if _is_reparse_or_symlink(package_path) or not package_path.is_file():
                raise ValueError
            lock = ScriptPackage.model_validate_json(
                package_path.read_text(encoding="utf-8")
            ).semantic_lock
        else:
            script_root = context.episode_root / "script"
            lock_path = script_root / "semantic-lock.json"
            manifest_path = script_root / "script_manifest.json"
            if (
                not lock_path.is_file()
                or _is_reparse_or_symlink(lock_path)
                or not manifest_path.is_file()
                or _is_reparse_or_symlink(manifest_path)
            ):
                raise ValueError
            lock = SemanticLock.model_validate_json(
                lock_path.read_text(encoding="utf-8")
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("script_sha256") != narration.sha256
                or manifest.get("semantic_lock_sha256") != lock.sha256
            ):
                raise ValueError
    except (OSError, ValueError, ValidationError):
        raise MediaStageError("visual_source_invalid") from None
    lock_payload = lock.model_dump(mode="json", exclude={"sha256"})
    lock_sha256 = hashlib.sha256(
        json.dumps(
            lock_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if lock.episode_id != context.episode_id or lock.sha256 != lock_sha256:
        raise MediaStageError("visual_source_invalid")
    return narration, lock


def _handdrawn_render_inputs(context: StageContext) -> HanddrawnRenderInputs:
    media = context.episode_root / "media"
    try:
        _, semantic_lock = _visual_sources(context)
        storyboard = IllustrationStoryboard.model_validate_json(
            (media / "illustration" / "illustration_storyboard.json").read_text(
                encoding="utf-8"
            )
        )
        voice = VoiceManifest.model_validate_json(
            (media / "voice" / "voice_master.json").read_text(encoding="utf-8")
        )
        subtitle_manifest = SubtitleManifest.model_validate_json(
            (media / "subtitles" / "subtitle_manifest.json").read_text(encoding="utf-8")
        )
        cue_payload = json.loads(
            (media / "subtitles" / "cues.json").read_text(encoding="utf-8")
        )
        cues = tuple(SubtitleCue.model_validate(item) for item in cue_payload)
        visual = media / "render" / "picture_silent.mp4"
        ass = media / "subtitles" / "subtitles.ass"
        voice_path = media / "voice" / "voice_master.wav"
    except (OSError, ValueError, ValidationError, AttributeError):
        raise MediaStageError("final_render_input_invalid") from None
    cover = _optional_handdrawn_cover(context, voice.master_duration_ms)
    return HanddrawnRenderInputs(
        book_id=context.book_id,
        episode_id=context.episode_id,
        visual_path=visual,
        visual_sha256=sha256_file(visual),
        storyboard_sha256=canonical_model_sha256(storyboard),
        semantic_lock_sha256=semantic_lock.sha256,
        script_sha256=storyboard.script_sha256,
        audio_sha256=storyboard.audio_sha256,
        subtitle_sha256=storyboard.subtitle_sha256,
        voice_master_path=voice_path,
        voice_master_sha256=voice.master_sha256,
        voice_manifest=voice,
        master_duration_ms=voice.master_duration_ms,
        production_profile_sha256=production_profile_sha256(
            context.episode_root
        ),
        ass_path=ass,
        ass_sha256=sha256_file(ass),
        asr_sha256=subtitle_manifest.asr_sha256,
        subtitle_manifest=subtitle_manifest,
        subtitle_cues=cues,
        cover=cover,
        render_root=context.episode_root / ".private" / "media" / "final-render",
        final_path=media / "final" / "final.mp4",
    )


def _optional_handdrawn_cover(
    context: StageContext,
    master_duration_ms: int,
) -> CoverOverlay | None:
    cover_root = context.episode_root / "cover"
    if _is_reparse_or_symlink(cover_root):
        raise MediaStageError("final_render_input_invalid")
    if not cover_root.exists():
        return None
    try:
        if not cover_root.is_dir():
            raise ValueError
        manifest_path = cover_root / "manifest.json"
        if _is_reparse_or_symlink(manifest_path) or not manifest_path.is_file():
            raise ValueError
        cover_manifest = CoverManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        cover_candidates = tuple(
            path for path in cover_root.iterdir() if path.name != "manifest.json"
        )
        if len(cover_candidates) != 1:
            raise ValueError
        cover_path = cover_candidates[0]
        if _is_reparse_or_symlink(cover_path) or not cover_path.is_file():
            raise ValueError
    except (OSError, ValueError, ValidationError, AttributeError):
        raise MediaStageError("final_render_input_invalid") from None
    overlay_text = "最终十五秒右上角真实商品书封区"
    return CoverOverlay(
        cover_path=cover_path,
        cover_sha256=cover_manifest.copied_sha256,
        manifest=cover_manifest,
        start_ms=max(0, master_duration_ms - 3_000),
        end_ms=master_duration_ms,
        overlay_zone_text=overlay_text,
        overlay_zone_sha256=hashlib.sha256(overlay_text.encode("utf-8")).hexdigest(),
        x=760,
        y=100,
        width=240,
        height=360,
    )


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        if not os.path.lexists(path):
            return False
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
        return path.is_symlink() or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    except OSError:
        return True


def _has_reparse_or_symlink_in_existing_chain(path: Path) -> bool:
    current = Path(path)
    while True:
        if os.path.lexists(current) and _is_reparse_or_symlink(current):
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _remove_file(path: Path) -> None:
    try:
        if path.is_file() and not path.is_symlink():
            path.unlink()
    except OSError:
        pass
