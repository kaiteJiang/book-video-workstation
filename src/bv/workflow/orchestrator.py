"""Stage-gated, local-only command workflow coordination.

This module deliberately does not construct model, ASR, TTS, video, or renderer
providers.  Those operations are supplied as stage runners so command handling
can be tested without a network, quota, private asset, or GUI.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from bv.core.hashing import sha256_file
from bv.state.locks import EpisodeLock
from bv.state.models import ArtifactRef, EpisodeState, StageManifest
from bv.state.store import StateStore
from bv.video.contracts import VideoGateway

from .stages import GATES, STAGE_ORDER, StageContext, StageOutcome, StageRunner

H3Gateway = VideoGateway


class WorkflowError(RuntimeError):
    """Stable public error codes; never retain private paths or tool output."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


@dataclass(frozen=True)
class WorkflowView:
    status: str
    next_command: str
    present_shot_ids: tuple[str, ...] = ()
    missing_shot_ids: tuple[str, ...] = ()
    failure_code: str | None = None


@dataclass(frozen=True)
class OpenTarget:
    target: str
    path: Path
    listing: str


TopicAdder = Callable[[str], str]
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class Orchestrator:
    """Advance one E001 only through its ordered durable stages and human gates."""

    def __init__(
        self,
        *,
        store: StateStore,
        stages: Mapping[str, StageRunner] | None = None,
        video_gateway: VideoGateway | None = None,
        h3_gateway: VideoGateway | None = None,
        topic_adder: TopicAdder | None = None,
        gate_approvers: Mapping[str, StageRunner] | None = None,
        config_sha256: str | None = None,
    ) -> None:
        if video_gateway is not None and h3_gateway is not None:
            raise WorkflowError("video_gateway_conflict")
        self.store = store
        self.stages = dict(stages or {})
        self.video_gateway = video_gateway if video_gateway is not None else h3_gateway
        self.topic_adder = topic_adder
        self.gate_approvers = dict(gate_approvers or {})
        self.config_sha256 = config_sha256 or ("0" * 64)
        if _SHA256.fullmatch(self.config_sha256) is None:
            raise WorkflowError("invalid_config_identity")

    @property
    def h3_gateway(self) -> VideoGateway | None:
        return self.video_gateway

    @h3_gateway.setter
    def h3_gateway(self, value: VideoGateway | None) -> None:
        self.video_gateway = value

    def next(self, book_id: str, episode_id: str = "E001") -> WorkflowView:
        return self._locked(book_id, episode_id, self._advance)

    def retry(self, book_id: str, episode_id: str = "E001") -> WorkflowView:
        def action(episode: EpisodeState) -> WorkflowView:
            failed = [
                name
                for name, manifest in episode.stage_manifests.items()
                if manifest.status == "failed"
            ]
            if not failed and episode.failure_summary is None:
                raise WorkflowError("nothing_to_retry")
            for name in failed:
                episode.stage_manifests.pop(name, None)
                if name in episode.completed_stages:
                    episode.completed_stages.remove(name)
            episode.failure_summary = None
            episode.status = "source_imported"
            self.store.save_episode(episode)
            return self._advance(episode)

        return self._locked(book_id, episode_id, action)

    def status_view(self, book_id: str, episode_id: str = "E001") -> WorkflowView:
        return self._locked(book_id, episode_id, self._status)

    def open_target(self, book_id: str, episode_id: str, target: str) -> OpenTarget:
        def action(episode: EpisodeState) -> OpenTarget:
            root = self._episode_root(book_id, episode_id)
            normalized = target.strip().lower()
            if normalized in {"video", "h3"}:
                if not self._manifest_current(episode, "storyboard"):
                    raise WorkflowError("target_not_ready")
                video = self._video_status(book_id, episode_id)
                prompt_root = root / "storyboard"
                prompts = sorted(
                    {
                        *prompt_root.glob("video_prompt_*.md"),
                        *prompt_root.glob("h3_prompt_*.md"),
                    }
                )
                prompt_lines = [str(path) for path in prompts]
                listing = "\n".join(
                    [
                        f"Present shots: {self._join_ids(video.present)}",
                        f"Missing shots: {self._join_ids(video.missing)}",
                        *prompt_lines,
                    ]
                )
                return OpenTarget("video", prompt_root, listing)
            paths = {
                "script": root / "script" / "review.md",
                "storyboard": root / "storyboard" / "storyboard.md",
                "final": root / "output" / "final.mp4",
            }
            if normalized not in paths:
                raise WorkflowError("unknown_open_target")
            path = paths[normalized]
            if not path.is_file() or self._redirect_in_existing_chain(path):
                raise WorkflowError("target_not_ready")
            return OpenTarget(normalized, path, str(path))

        return self._locked(book_id, episode_id, action)

    def approve_gate(self, book_id: str, episode_id: str, gate: str) -> WorkflowView:
        def action(episode: EpisodeState) -> WorkflowView:
            normalized = gate.strip().lower()
            expected = {
                "script": "awaiting_script_review",
                "final": "awaiting_final_review",
            }.get(normalized)
            if expected is None:
                raise WorkflowError("unknown_approval_gate")
            if episode.status != expected:
                raise WorkflowError("gate_not_ready")
            if normalized == "final" and not self._manifest_current(episode, "qc"):
                raise WorkflowError("stale_qc")
            if normalized == "script":
                self._run_gate_approver(episode, normalized, "review_script")
            else:
                self._complete_gate(episode, "final_review")
            episode.status = "script_approved" if normalized == "script" else "final_approved"
            episode.failure_summary = None
            self.store.save_episode(episode)
            return self._status(episode)

        return self._locked(book_id, episode_id, action)

    def import_video(
        self, book_id: str, episode_id: str, shot_id: str, source: Path
    ) -> WorkflowView:
        def action(episode: EpisodeState) -> WorkflowView:
            if self.video_gateway is None:
                raise WorkflowError("video_gateway_not_configured")
            if episode.status not in {
                "awaiting_video_generation",
                "visual_sample_ready",
                "video_partial",
            }:
                raise WorkflowError("video_gate_not_ready")
            try:
                self.video_gateway.import_video(book_id, episode_id, shot_id, Path(source))
            except WorkflowError:
                raise
            except Exception as exc:
                raise WorkflowError(self._error_code(exc, "video_import_failed")) from None
            video = self._video_status(book_id, episode_id)
            episode.status = self._video_episode_status(video)
            self._invalidate_later_stages(episode, "await_video_segments")
            if not video.missing:
                self._complete_gate(episode, "await_video_segments")
            self.store.save_episode(episode)
            return self._status(episode, video)

        return self._locked(book_id, episode_id, action)

    def add(self, book_id: str) -> str:
        """Create one later value angle only through a supplied Task 13 dedup flow."""
        def action(episode: EpisodeState) -> str:
            if episode.episode_id != "E001":
                raise WorkflowError("e001_required_before_add")
            if self.topic_adder is None:
                raise WorkflowError("topic_adder_not_configured")
            try:
                result = self.topic_adder(book_id)
            except Exception as exc:
                raise WorkflowError(self._error_code(exc, "topic_generation_failed")) from None
            if not isinstance(result, str) or _IDENTIFIER.fullmatch(result) is None or result == "E001":
                raise WorkflowError("invalid_topic_result")
            return result

        try:
            return self._locked(book_id, "E001", action)
        except WorkflowError as exc:
            if exc.error_code == "workflow_state_unavailable":
                raise WorkflowError("e001_required_before_add") from None
            raise

    def _advance(self, episode: EpisodeState) -> WorkflowView:
        for stage in STAGE_ORDER:
            if stage == "review_script":
                if not self._manifest_current(episode, stage):
                    episode.status = GATES[stage]
                    self.store.save_episode(episode)
                    return self._status(episode)
                continue
            if stage == "await_video_segments":
                video = self._video_status(episode.book_id, episode.episode_id)
                if video.missing:
                    self._invalidate_later_stages(episode, stage)
                    episode.status = self._video_episode_status(video)
                    self.store.save_episode(episode)
                    return self._status(episode, video)
                if not self._manifest_current(episode, stage):
                    self._complete_gate(episode, stage)
                    self.store.save_episode(episode)
                continue
            if self._manifest_current(episode, stage):
                continue
            self._run_stage(episode, stage)
            if stage == "qc":
                episode.status = GATES[stage]
                self.store.save_episode(episode)
                return self._status(episode)
        return self._status(episode)

    def _run_stage(self, episode: EpisodeState, stage: str) -> None:
        runner = self.stages.get(stage)
        if runner is None:
            self._fail(episode, stage, "stage_not_configured")
        self._invalidate_later_stages(episode, stage)
        episode.stage_manifests[stage] = StageManifest(
            stage=stage,
            status="running",
            inputs={},
            outputs={},
            config_sha256=self.config_sha256,
        )
        episode.status = self._running_status(stage)
        self.store.save_episode(episode)

        try:
            result = runner.run(self._stage_context(episode))
            outcome = self._outcome(result)
            outputs = self._artifact_refs(outcome.outputs)
        except WorkflowError as exc:
            self._fail(episode, stage, exc.error_code)
            raise AssertionError("unreachable")
        except Exception as exc:
            self._fail(episode, stage, self._error_code(exc, "stage_failed"))
            raise AssertionError("unreachable")
        episode.stage_manifests[stage] = StageManifest(
            stage=stage,
            status="completed",
            inputs={str(key): str(value) for key, value in outcome.inputs.items()},
            outputs=outputs,
            config_sha256=self.config_sha256,
        )
        if stage not in episode.completed_stages:
            episode.completed_stages.append(stage)
        if stage in episode.stale_stages:
            episode.stale_stages.remove(stage)
        episode.failure_summary = None
        episode.status = self._completed_status(stage)
        self.store.save_episode(episode)

    @staticmethod
    def _invalidate_later_stages(episode: EpisodeState, stage: str) -> None:
        """A regenerated upstream artifact makes every later completed stage stale."""
        try:
            later = STAGE_ORDER[STAGE_ORDER.index(stage) + 1 :]
        except ValueError:
            return
        for dependent in later:
            if dependent in episode.completed_stages:
                episode.completed_stages.remove(dependent)
            manifest = episode.stage_manifests.get(dependent)
            if manifest is not None and manifest.status == "completed":
                manifest.status = "stale"
            if dependent in episode.stage_manifests and dependent not in episode.stale_stages:
                episode.stale_stages.append(dependent)

    def _complete_gate(self, episode: EpisodeState, stage: str) -> None:
        episode.stage_manifests[stage] = StageManifest(
            stage=stage,
            status="completed",
            inputs={},
            outputs={},
            config_sha256=self.config_sha256,
        )
        if stage not in episode.completed_stages:
            episode.completed_stages.append(stage)
        if stage in episode.stale_stages:
            episode.stale_stages.remove(stage)

    def _run_gate_approver(self, episode: EpisodeState, gate: str, stage: str) -> None:
        runner = self.gate_approvers.get(gate)
        if runner is None:
            raise WorkflowError("approval_not_configured")
        self._invalidate_later_stages(episode, stage)
        try:
            outcome = self._outcome(runner.run(self._stage_context(episode)))
            outputs = self._artifact_refs(outcome.outputs)
        except WorkflowError:
            raise
        except Exception as exc:
            raise WorkflowError(self._error_code(exc, "approval_failed")) from None
        episode.stage_manifests[stage] = StageManifest(
            stage=stage, status="completed",
            inputs={str(key): str(value) for key, value in outcome.inputs.items()},
            outputs=outputs, config_sha256=self.config_sha256,
        )
        if stage not in episode.completed_stages:
            episode.completed_stages.append(stage)
        if stage in episode.stale_stages:
            episode.stale_stages.remove(stage)

    def _fail(self, episode: EpisodeState, stage: str, error_code: str) -> None:
        episode.stage_manifests[stage] = StageManifest(
            stage=stage,
            status="failed",
            inputs={},
            outputs={},
            config_sha256=self.config_sha256,
            error_code=error_code,
        )
        episode.failure_summary = error_code
        self.store.save_episode(episode)
        raise WorkflowError(error_code)

    def _status(self, episode: EpisodeState, video: "_VideoStatus | None" = None) -> WorkflowView:
        video_relevant = episode.status in {
            "awaiting_video_generation", "visual_sample_ready", "video_partial", "video_imported",
            "render_running", "render_failed", "awaiting_final_review", "final_approved",
        }
        if video is None and video_relevant:
            video = self._video_status(episode.book_id, episode.episode_id, required=False)
        status = episode.status
        if video is not None and status in {
            "awaiting_video_generation", "visual_sample_ready", "video_partial", "video_imported",
        }:
            status = self._video_episode_status(video)
        return WorkflowView(
            status=status,
            next_command=self._next_command(episode.book_id, episode.episode_id, status, episode.failure_summary),
            present_shot_ids=() if video is None else video.present,
            missing_shot_ids=() if video is None else video.missing,
            failure_code=episode.failure_summary,
        )

    def _locked(self, book_id: str, episode_id: str, action: Callable[[EpisodeState], object]):
        self._require_identifier(book_id)
        self._require_identifier(episode_id)
        episode_root = self._episode_root(book_id, episode_id)
        state_path = episode_root / "episode.json"
        if (
            not episode_root.is_dir() or not state_path.is_file()
            or self._redirect_in_existing_chain(episode_root)
            or self._redirect_in_existing_chain(state_path)
        ):
            raise WorkflowError("workflow_state_unavailable")
        lock_path = episode_root / ".workflow.lock"
        try:
            with EpisodeLock(lock_path):
                episode = self.store.load_episode(book_id, episode_id)
                return action(episode)
        except WorkflowError:
            raise
        except Exception as exc:
            raise WorkflowError(self._error_code(exc, "workflow_state_unavailable")) from None

    def _video_status(self, book_id: str, episode_id: str, *, required: bool = True) -> "_VideoStatus | None":
        if self.video_gateway is None:
            if required:
                raise WorkflowError("video_gateway_not_configured")
            return None
        try:
            result = self.video_gateway.status(book_id, episode_id)
        except Exception as exc:
            raise WorkflowError(self._error_code(exc, "video_manifest_invalid")) from None
        return _VideoStatus.from_result(result)

    def _manifest_current(self, episode: EpisodeState, stage: str) -> bool:
        manifest = episode.stage_manifests.get(stage)
        if (
            manifest is None
            or manifest.status != "completed"
            or stage not in episode.completed_stages
            or stage in episode.stale_stages
            or manifest.config_sha256 != self.config_sha256
        ):
            return False
        for artifact in manifest.outputs.values():
            path = Path(artifact.path)
            try:
                if (
                    not self._safe_workspace_file(path)
                    or path.stat().st_size != artifact.size_bytes
                    or sha256_file(path) != artifact.sha256
                ):
                    return False
            except OSError:
                return False
        return True

    @staticmethod
    def _outcome(result: object) -> StageOutcome:
        if isinstance(result, StageOutcome):
            return result
        if not isinstance(result, Mapping):
            raise WorkflowError("invalid_stage_result")
        raw_outputs = result.get("outputs", {})
        raw_inputs = result.get("inputs", {})
        if not isinstance(raw_outputs, Mapping) or not isinstance(raw_inputs, Mapping):
            raise WorkflowError("invalid_stage_result")
        try:
            outcome = StageOutcome(
                outputs={str(key): Path(value) for key, value in raw_outputs.items()},
                inputs={str(key): str(value) for key, value in raw_inputs.items()},
            )
        except (TypeError, ValueError) as exc:
            raise WorkflowError("invalid_stage_result") from exc
        if any(
            _IDENTIFIER.fullmatch(key) is None or _SHA256.fullmatch(value) is None
            for key, value in outcome.inputs.items()
        ):
            raise WorkflowError("invalid_stage_result")
        return outcome

    def _artifact_refs(self, outputs: Mapping[str, Path]) -> dict[str, ArtifactRef]:
        if not outputs:
            raise WorkflowError("stage_output_missing")
        refs: dict[str, ArtifactRef] = {}
        for name, path in outputs.items():
            if _IDENTIFIER.fullmatch(str(name)) is None:
                raise WorkflowError("invalid_stage_result")
            candidate = Path(path)
            try:
                if not self._safe_workspace_file(candidate):
                    raise WorkflowError("stage_output_unsafe")
                refs[str(name)] = ArtifactRef(
                    path=str(candidate), sha256=sha256_file(candidate), size_bytes=candidate.stat().st_size
                )
            except WorkflowError:
                raise
            except OSError as exc:
                raise WorkflowError("stage_output_missing") from exc
        return refs

    @staticmethod
    def _error_code(exc: Exception, fallback: str) -> str:
        code = getattr(exc, "error_code", None)
        if isinstance(code, str) and _ERROR_CODE.fullmatch(code) is not None:
            return code
        return fallback

    @staticmethod
    def _running_status(stage: str) -> str:
        return {
            "parse_source": "source_imported",
            "analyze_chapters": "chapter_analysis_running",
            "draft_script": "script_drafting",
            "tts": "tts_running",
            "asr": "asr_running",
            "render": "render_running",
        }.get(stage, "source_imported")

    @staticmethod
    def _completed_status(stage: str) -> str:
        return {
            "parse_source": "source_parsed",
            "analyze_chapters": "book_analysis_ready",
            "synthesize_book_value": "book_analysis_ready",
            "generate_episode_brief": "topic_selected",
            "draft_script": "script_drafting",
            "tts": "tts_ready",
            "asr": "audio_ready",
            "subtitles": "audio_ready",
            "storyboard": "storyboard_ready",
            "render": "render_running",
            "qc": "awaiting_final_review",
        }[stage]

    @staticmethod
    def _video_episode_status(video: "_VideoStatus") -> str:
        if not video.missing:
            return "video_imported"
        if video.present == ("S01",):
            return "visual_sample_ready"
        if video.present:
            return "video_partial"
        return "awaiting_video_generation"

    @staticmethod
    def _next_command(book_id: str, episode_id: str, status: str, failure: str | None) -> str:
        episode_argument = "" if episode_id == "E001" else f" {episode_id}"
        if failure:
            return f"bv retry {book_id}{episode_argument}"
        if status == "awaiting_script_review":
            return f"bv approve {book_id} {episode_id} script"
        if status in {"awaiting_video_generation", "visual_sample_ready", "video_partial"}:
            return f"bv open {book_id} {episode_id} video"
        if status == "video_imported":
            return f"bv next {book_id}{episode_argument}"
        if status == "awaiting_final_review":
            return f"bv approve {book_id} {episode_id} final"
        return f"bv next {book_id}{episode_argument}"

    @staticmethod
    def _join_ids(values: tuple[str, ...]) -> str:
        return ", ".join(values) if values else "none"

    def _episode_root(self, book_id: str, episode_id: str) -> Path:
        return self.store.root / "books" / book_id / "episodes" / episode_id

    def _stage_context(self, episode: EpisodeState) -> StageContext:
        return StageContext(
            book_id=episode.book_id,
            episode_id=episode.episode_id,
            episode_root=self._episode_root(episode.book_id, episode.episode_id),
            episode_state=episode,
        )

    @staticmethod
    def _require_identifier(value: str) -> None:
        if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
            raise WorkflowError("unsafe_identifier")

    def _safe_workspace_file(self, path: Path) -> bool:
        candidate = Path(path)
        root = Path(os.path.abspath(os.fspath(self.store.root)))
        absolute = Path(os.path.abspath(os.fspath(candidate)))
        try:
            absolute.relative_to(root)
        except ValueError:
            return False
        return (
            candidate.is_absolute() and candidate.is_file()
            and not self._redirect_in_existing_chain(candidate)
        )

    @staticmethod
    def _redirect_in_existing_chain(path: Path) -> bool:
        candidate = Path(path)
        while True:
            if candidate.exists() or candidate.is_symlink():
                try:
                    attributes = candidate.stat(follow_symlinks=False).st_file_attributes
                except (AttributeError, OSError):
                    attributes = 0
                if candidate.is_symlink() or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    return True
            if candidate.parent == candidate:
                return False
            candidate = candidate.parent


@dataclass(frozen=True)
class _VideoStatus:
    status: str
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @classmethod
    def from_result(cls, result: object) -> "_VideoStatus":
        if isinstance(result, Mapping):
            status = result.get("status")
            present = result.get("present_shot_ids", result.get("present_segment_ids"))
            missing = result.get("missing_shot_ids", result.get("missing_segment_ids"))
        else:
            status = getattr(result, "status", None)
            present = getattr(result, "present_shot_ids", getattr(result, "present_segment_ids", None))
            missing = getattr(result, "missing_shot_ids", getattr(result, "missing_segment_ids", None))
        if (
            not isinstance(status, str)
            or not isinstance(present, tuple)
            or not isinstance(missing, tuple)
            or any(not isinstance(item, str) for item in present + missing)
            or set(present) & set(missing)
        ):
            raise WorkflowError("video_manifest_invalid")
        required = present + missing
        if len(required) not in {4, 5}:
            raise WorkflowError("video_manifest_invalid")
        partition = tuple(f"S{index:02d}" for index in range(1, len(required) + 1))
        if (
            tuple(item for item in partition if item in present) != present
            or tuple(item for item in partition if item in missing) != missing
            or set(required) != set(partition)
        ):
            raise WorkflowError("video_manifest_invalid")
        if not missing:
            accepted = {"video_imported", "h3_imported"}
            canonical_status = "video_imported"
        elif present == ("S01",):
            accepted = {"visual_sample_ready"}
            canonical_status = "visual_sample_ready"
        elif present:
            accepted = {"video_partial", "h3_partial", "awaiting_h3_segments"}
            canonical_status = "video_partial"
        else:
            accepted = {
                "awaiting_video_generation",
                "awaiting_h3_generation",
                "awaiting_h3_segments",
            }
            canonical_status = "awaiting_video_generation"
        if status not in accepted:
            raise WorkflowError("video_manifest_invalid")
        return cls(status=canonical_status, present=present, missing=missing)
