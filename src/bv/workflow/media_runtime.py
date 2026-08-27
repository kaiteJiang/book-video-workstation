from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from bv.core.atomic import atomic_write_json
from bv.core.hashing import sha256_file
from bv.delivery.exporter import ApprovalRecord, DeliveryBundle, DeliveryExporter, DeliveryRequest
from bv.illustration.assets import (
    IllustrationManifest,
    import_image_master,
    record_imported_image,
)
from bv.illustration.contracts import IllustrationStoryboard
from bv.illustration.reviews import (
    RepresentativeApproval,
    RepresentativeReview,
    approve_representatives,
    build_representative_review,
    unlock_batch_jobs,
)
from bv.state.models import ArtifactRef, EpisodeState, StageManifest
from bv.state.invalidation import invalidate_from
from bv.state.store import StateStore
from bv.video.cover import CoverManifest, approve_cover
from bv.video.social_cover import (
    SocialCoverManifest,
    SocialCoverRenderer,
    SocialCoverRequest,
    validate_social_cover_manifest_current,
)
from bv.voice.audition import (
    VoiceAuditionError,
    VoiceAuditionManifest,
    VoiceProfile,
    require_current_voice_profile,
)
from bv.production.profile import load_production_profile
from bv.workflow.runtime import RuntimeAuthorization

from .stages import (
    HANDDRAWN_MEDIA_PREPARE_ORDER,
    HANDDRAWN_MEDIA_STAGE_ORDER,
    StageContext,
    StageOutcome,
    StageRunner,
)


class MediaWorkflowError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class MediaProductionView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    book_id: str
    episode_id: str
    status: str
    next_action: str
    present_scene_ids: tuple[str, ...] = ()
    missing_scene_ids: tuple[str, ...] = ()
    batch_unlocked: bool = False
    cover_ready: bool = False
    social_cover_ready: bool = False
    delivery_root: Path | None = None
    delivery_version: int | None = None
    approval_record_path: Path | None = None


class IllustrationGateway(Protocol):
    def status(self, context: StageContext) -> tuple[tuple[str, ...], tuple[str, ...]]: ...
    def batch_status(self, context: StageContext) -> tuple[tuple[str, ...], tuple[str, ...]]: ...
    def import_master(
        self,
        context: StageContext,
        scene_id: str,
        source: Path,
        *,
        representative_only: bool,
    ) -> None: ...
    def approve(self, context: StageContext) -> None: ...
    def unlock(self, context: StageContext) -> tuple[object, ...]: ...


class CoverGateway(Protocol):
    def ready(self, context: StageContext) -> bool: ...
    def import_cover(
        self,
        context: StageContext,
        source: Path,
        *,
        source_type: str,
        matches_product_version: bool,
    ) -> None: ...


class SocialCoverGateway(Protocol):
    def ready(self, context: StageContext) -> bool: ...
    def import_current_run(
        self,
        context: StageContext,
        *,
        cover_brief_path: Path,
        source: Path,
        source_sha256: str,
    ) -> StageOutcome: ...


class DeliveryGateway(Protocol):
    def export_candidate(self, context: StageContext) -> DeliveryBundle: ...
    def record_final_approval(
        self,
        context: StageContext,
        *,
        manifest_path: Path,
        version: int,
    ) -> ApprovalRecord: ...


class VoiceAuditionGateway(Protocol):
    def prepare_candidates(
        self,
        context: StageContext,
        *,
        voice_ids: tuple[str, ...],
        supported_voice_ids: tuple[str, ...],
        authorization: RuntimeAuthorization,
    ) -> VoiceAuditionManifest: ...

    def approve_candidate(
        self,
        context: StageContext,
        candidate_id: str,
    ) -> VoiceProfile: ...


@dataclass(frozen=True, slots=True)
class MediaRuntimeBindings:
    service: "MediaProductionService"
    stages: Mapping[str, StageRunner]
    gate_approvers: Mapping[str, StageRunner]


class MediaProductionService:
    def __init__(
        self,
        *,
        store: StateStore,
        stages: Mapping[str, StageRunner],
        images: IllustrationGateway,
        config_sha256: str = "0" * 64,
        configured_mode: Literal["technical_sample", "full"] | None = None,
        cover: CoverGateway | None = None,
        social_cover: SocialCoverGateway | None = None,
        delivery: DeliveryGateway | None = None,
        voice_auditions: VoiceAuditionGateway | None = None,
    ) -> None:
        self.store = store
        self.stages = dict(stages)
        self.images = images
        self.config_sha256 = config_sha256
        self.configured_mode = configured_mode
        self.cover = cover
        self.social_cover = social_cover
        self.delivery = delivery
        self.voice_auditions = voice_auditions

    def prepare(
        self,
        book_id: str,
        episode_id: str,
        *,
        mode: Literal["technical_sample", "full"],
    ) -> MediaProductionView:
        if self.configured_mode is not None and mode != self.configured_mode:
            raise MediaWorkflowError("media_runtime_mode_mismatch")
        episode = self._load(book_id, episode_id)
        self._switch_mode_if_needed(episode, mode)
        if episode.status not in {
            "script_approved", "voice_profile_approved", "illustration_planning", "representative_generation_running",
            "awaiting_representative_review", "representatives_approved",
        }:
            raise MediaWorkflowError("media_prepare_not_ready")
        profile = load_production_profile(self._episode_root(book_id, episode_id))
        if profile.tts.provider == "doubao":
            try:
                require_current_voice_profile(
                    self._episode_root(book_id, episode_id),
                    profile,
                    script_sha256=episode.script_hash,
                )
            except VoiceAuditionError as error:
                raise MediaWorkflowError(
                    "voice_profile_not_approved"
                    if error.error_code != "voice_profile_stale"
                    else "voice_profile_stale"
                ) from None
        for name in HANDDRAWN_MEDIA_PREPARE_ORDER:
            if self._manifest_current(episode, name, mode=mode):
                continue
            if name not in self.stages:
                raise MediaWorkflowError("media_stage_not_configured")
            self._run_stage(episode, name, mode=mode)
        present, missing = self.images.status(self._context(episode))
        episode.status = (
            "representative_generation_running" if missing
            else "awaiting_representative_review"
        )
        self.store.save_episode(episode)
        return self._view(episode, present=present, missing=missing)

    def prepare_voice_audition(
        self,
        book_id: str,
        episode_id: str,
        *,
        voice_ids: tuple[str, ...],
        supported_voice_ids: tuple[str, ...],
        authorization: RuntimeAuthorization,
    ) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        if episode.status != "script_approved" or self.voice_auditions is None:
            raise MediaWorkflowError("voice_audition_not_ready")
        try:
            self.voice_auditions.prepare_candidates(
                self._context(episode),
                voice_ids=voice_ids,
                supported_voice_ids=supported_voice_ids,
                authorization=authorization,
            )
        except VoiceAuditionError as error:
            raise MediaWorkflowError(error.error_code) from None
        episode.status = "awaiting_voice_audition_review"
        self.store.save_episode(episode)
        return self._view(episode)

    def approve_voice_audition(
        self,
        book_id: str,
        episode_id: str,
        candidate_id: str,
    ) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        if (
            episode.status != "awaiting_voice_audition_review"
            or self.voice_auditions is None
        ):
            raise MediaWorkflowError("voice_audition_gate_not_ready")
        try:
            profile = self.voice_auditions.approve_candidate(
                self._context(episode), candidate_id
            )
        except VoiceAuditionError as error:
            raise MediaWorkflowError(error.error_code) from None
        episode.voice_id = profile.provider_voice_id
        invalidate_from(episode, "voice_profile")
        episode.status = "voice_profile_approved"
        self.store.save_episode(episode)
        return self._view(episode)

    def _switch_mode_if_needed(
        self,
        episode: EpisodeState,
        mode: Literal["technical_sample", "full"],
    ) -> None:
        recorded = {
            manifest.inputs["media_mode"]
            for name, manifest in episode.stage_manifests.items()
            if name in HANDDRAWN_MEDIA_PREPARE_ORDER
            and manifest.inputs.get("media_mode") in {"technical_sample", "full"}
        }
        if not recorded or recorded == {mode}:
            return
        if len(recorded) != 1:
            raise MediaWorkflowError("media_mode_state_invalid")
        previous = next(iter(recorded))
        root = self._episode_root(episode.book_id, episode.episode_id)
        recovery = (
            root / ".recovery" / "media-mode-switch"
            / f"{previous}-to-{mode}-{time.time_ns()}"
        )
        self._validate_episode_child(root, recovery)
        recovery.mkdir(parents=True, exist_ok=False)
        for source, label in (
            (root / "media", "media"),
            (root / ".private" / "media", "private_media"),
            (root / ".private" / "requests", "private_requests"),
        ):
            if not source.exists():
                continue
            self._validate_episode_child(root, source)
            if source.is_symlink() or not source.is_dir():
                raise MediaWorkflowError("unsafe_media_mode_switch")
            os.replace(source, recovery / label)
        media_stages = set(HANDDRAWN_MEDIA_STAGE_ORDER)
        episode.stage_manifests = {
            name: manifest
            for name, manifest in episode.stage_manifests.items()
            if name not in media_stages
        }
        episode.completed_stages = [
            name for name in episode.completed_stages if name not in media_stages
        ]
        episode.stale_stages = [
            name for name in episode.stale_stages if name not in media_stages
        ]
        episode.status = "script_approved"
        self.store.save_episode(episode)

    @staticmethod
    def _validate_episode_child(root: Path, path: Path) -> None:
        try:
            canonical_root = root.resolve(strict=True)
            canonical_path = path.resolve(strict=False)
        except OSError:
            raise MediaWorkflowError("unsafe_media_mode_switch") from None
        if canonical_path == canonical_root or not canonical_path.is_relative_to(canonical_root):
            raise MediaWorkflowError("unsafe_media_mode_switch")

    def import_image(
        self,
        book_id: str,
        episode_id: str,
        scene_id: str,
        source: Path,
    ) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        representative = episode.status in {
            "representative_generation_running", "awaiting_representative_review"
        }
        batch = episode.status == "illustration_batch_running"
        if not representative and not batch:
            raise MediaWorkflowError("representative_import_not_ready")
        self.images.import_master(
            self._context(episode), scene_id, Path(source),
            representative_only=representative,
        )
        if representative:
            present, missing = self.images.status(self._context(episode))
            episode.status = (
                "representative_generation_running" if missing
                else "awaiting_representative_review"
            )
        else:
            present, missing = self.images.batch_status(self._context(episode))
            episode.status = "illustration_batch_running" if missing else "illustrations_ready"
        self.store.save_episode(episode)
        return self._view(episode, present=present, missing=missing)

    def approve_representatives(self, book_id: str, episode_id: str) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        if episode.status != "awaiting_representative_review":
            raise MediaWorkflowError("representative_gate_not_ready")
        self.images.approve(self._context(episode))
        episode.status = "representatives_approved"
        self.store.save_episode(episode)
        present, missing = self.images.status(self._context(episode))
        return self._view(episode, present=present, missing=missing)

    def import_cover(
        self,
        book_id: str,
        episode_id: str,
        source: Path,
        *,
        source_type: str = "user_provided",
        matches_product_version: bool = True,
    ) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        if self.cover is None or episode.status == "final_approved":
            raise MediaWorkflowError("cover_import_not_ready")
        try:
            self.cover.import_cover(
                self._context(episode),
                Path(source),
                source_type=source_type,
                matches_product_version=matches_product_version,
            )
        except MediaWorkflowError:
            raise
        except Exception:
            raise MediaWorkflowError("cover_import_failed") from None
        return self._view(episode)

    def prepare_batch(self, book_id: str, episode_id: str) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        if episode.status != "representatives_approved":
            raise MediaWorkflowError("representatives_not_approved")
        self.images.unlock(self._context(episode))
        present, missing = self.images.batch_status(self._context(episode))
        episode.status = "illustration_batch_running" if missing else "illustrations_ready"
        self.store.save_episode(episode)
        return self._view(
            episode, present=present, missing=missing, batch_unlocked=True
        )

    def import_social_cover(
        self,
        book_id: str,
        episode_id: str,
        *,
        cover_brief_path: Path,
        source: Path,
        source_sha256: str,
    ) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        if self.social_cover is None or episode.status not in {
            "script_approved",
            "voice_profile_approved",
            "representative_generation_running",
            "awaiting_representative_review",
            "representatives_approved",
            "illustration_batch_running",
            "illustrations_ready",
        }:
            raise MediaWorkflowError("social_cover_import_not_ready")
        try:
            outcome = self.social_cover.import_current_run(
                self._context(episode),
                cover_brief_path=Path(cover_brief_path),
                source=Path(source),
                source_sha256=source_sha256,
            )
        except MediaWorkflowError:
            raise
        except Exception as error:
            code = getattr(error, "error_code", "social_cover_import_failed")
            raise MediaWorkflowError(code) from None
        self._record_outcome(episode, "social_cover", outcome, mode=None)
        return self._view(episode)

    def render(self, book_id: str, episode_id: str) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        if episode.status != "illustrations_ready":
            raise MediaWorkflowError("render_not_ready")
        if (
            self.social_cover is None
            or not self.social_cover.ready(self._context(episode))
            or not self._manifest_current(episode, "social_cover", mode=None)
        ):
            raise MediaWorkflowError("social_cover_not_ready")
        for name in ("visual_render", "render", "qc"):
            if not self._manifest_current(episode, name, mode=None):
                if name not in self.stages:
                    raise MediaWorkflowError("media_stage_not_configured")
                self._run_stage(episode, name, mode=None)
        if not self._manifest_current(episode, "delivery", mode=None):
            if self.delivery is None:
                raise MediaWorkflowError("delivery_not_configured")
            try:
                bundle = self.delivery.export_candidate(self._context(episode))
            except Exception as error:
                code = getattr(error, "error_code", "delivery_export_failed")
                raise MediaWorkflowError(code) from None
            self._record_outcome(
                episode,
                "delivery",
                StageOutcome(
                    outputs={
                        **{
                            f"delivery_file_{index}": path
                            for index, path in enumerate(bundle.files[:-1], start=1)
                        },
                        "delivery_manifest": bundle.manifest_path,
                    },
                    inputs={
                        "delivery_manifest_sha256": bundle.manifest_sha256,
                        "delivery_version": str(bundle.version),
                        "delivery_root": str(bundle.root),
                    },
                ),
                mode=None,
            )
        episode.status = "awaiting_final_review"
        self.store.save_episode(episode)
        return self._view(episode)

    def approve_final(self, book_id: str, episode_id: str) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        if episode.status != "awaiting_final_review" or not self._manifest_current(
            episode, "qc", mode=None
        ) or not self._manifest_current(
            episode, "social_cover", mode=None
        ) or not self._manifest_current(episode, "delivery", mode=None):
            raise MediaWorkflowError("final_gate_not_ready")
        if self.delivery is None:
            raise MediaWorkflowError("delivery_not_configured")
        delivery_manifest = episode.stage_manifests["delivery"]
        try:
            manifest_path = Path(delivery_manifest.outputs["delivery_manifest"].path)
            version = int(delivery_manifest.inputs["delivery_version"])
            approval = self.delivery.record_final_approval(
                self._context(episode),
                manifest_path=manifest_path,
                version=version,
            )
            self._record_outcome(
                episode,
                "final_approval",
                StageOutcome(
                    outputs={"approval_record": approval.path},
                    inputs={
                        "delivery_manifest_sha256": approval.delivery_manifest_sha256,
                        "video_sha256": approval.video_sha256,
                        "cover_sha256": approval.cover_sha256,
                    },
                ),
                mode=None,
            )
        except Exception as error:
            code = getattr(error, "error_code", "final_approval_failed")
            raise MediaWorkflowError(code) from None
        episode.status = "final_approved"
        self.store.save_episode(episode)
        return self._view(episode)

    def status(self, book_id: str, episode_id: str) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        try:
            if episode.status in {"illustration_batch_running", "illustrations_ready"}:
                present, missing = self.images.batch_status(self._context(episode))
            else:
                present, missing = self.images.status(self._context(episode))
        except Exception:
            present, missing = (), ()
        return self._view(episode, present=present, missing=missing)

    def _run_stage(
        self,
        episode: EpisodeState,
        name: str,
        *,
        mode: Literal["technical_sample", "full"] | None,
    ) -> None:
        try:
            outcome = self.stages[name].run(self._context(episode))
        except Exception as error:
            if isinstance(error, MediaWorkflowError):
                raise
            raise MediaWorkflowError(f"{name}_failed") from None
        if not isinstance(outcome, StageOutcome) or not outcome.outputs:
            raise MediaWorkflowError("media_stage_artifacts_invalid")
        self._record_outcome(episode, name, outcome, mode=mode)

    def _record_outcome(
        self,
        episode: EpisodeState,
        name: str,
        outcome: StageOutcome,
        *,
        mode: Literal["technical_sample", "full"] | None,
    ) -> None:
        if not isinstance(outcome, StageOutcome) or not outcome.outputs:
            raise MediaWorkflowError("media_stage_artifacts_invalid")
        outputs: dict[str, ArtifactRef] = {}
        root = self._episode_root(episode.book_id, episode.episode_id)
        for label, path in outcome.outputs.items():
            target = Path(path)
            try:
                target.relative_to(root)
            except ValueError:
                raise MediaWorkflowError("media_stage_artifacts_invalid") from None
            if not target.is_file() or target.is_symlink():
                raise MediaWorkflowError("media_stage_artifacts_invalid")
            outputs[label] = ArtifactRef(
                path=str(target), sha256=sha256_file(target), size_bytes=target.stat().st_size
            )
        inputs = dict(outcome.inputs)
        if mode is not None:
            inputs["media_mode"] = mode
        episode.stage_manifests[name] = StageManifest(
            stage=name,
            status="completed",
            inputs=inputs,
            outputs=outputs,
            config_sha256=self.config_sha256,
        )
        if name not in episode.completed_stages:
            episode.completed_stages.append(name)
        if name in episode.stale_stages:
            episode.stale_stages.remove(name)
        self.store.save_episode(episode)

    def _manifest_current(
        self,
        episode: EpisodeState,
        name: str,
        *,
        mode: Literal["technical_sample", "full"] | None,
    ) -> bool:
        manifest = episode.stage_manifests.get(name)
        if manifest is None or manifest.status != "completed" or name in episode.stale_stages:
            return False
        if mode is not None and manifest.inputs.get("media_mode") != mode:
            return False
        for artifact in manifest.outputs.values():
            path = Path(artifact.path)
            if (
                not path.is_file() or path.is_symlink()
                or path.stat().st_size != artifact.size_bytes
                or sha256_file(path) != artifact.sha256
            ):
                return False
        return True

    def _load(self, book_id: str, episode_id: str) -> EpisodeState:
        try:
            episode = self.store.load_episode(book_id, episode_id)
        except Exception:
            raise MediaWorkflowError("episode_state_unavailable") from None
        if episode.book_id != book_id or episode.episode_id != episode_id:
            raise MediaWorkflowError("episode_state_invalid")
        return episode

    def _context(self, episode: EpisodeState) -> StageContext:
        return StageContext(
            book_id=episode.book_id,
            episode_id=episode.episode_id,
            episode_root=self._episode_root(episode.book_id, episode.episode_id),
            episode_state=episode,
        )

    def _episode_root(self, book_id: str, episode_id: str) -> Path:
        return self.store.root / "books" / book_id / "episodes" / episode_id

    def _view(
        self,
        episode: EpisodeState,
        *,
        present: tuple[str, ...] = (),
        missing: tuple[str, ...] = (),
        batch_unlocked: bool = False,
    ) -> MediaProductionView:
        actions = {
            "awaiting_voice_audition_review": "向用户播放同一文稿片段的候选音色并等待批准",
            "voice_profile_approved": "使用已批准音色生成完整旁白",
            "representative_generation_running": "在当前对话生成并导入缺少的代表插画",
            "awaiting_representative_review": "向用户展示开头、中段、结尾三张代表图并等待批准",
            "representatives_approved": "准备剩余插画任务",
            "illustration_batch_running": "生成并导入剩余插画",
            "illustrations_ready": "渲染静音手绘画面和最终视频",
            "awaiting_final_review": "向用户展示最终视频并等待批准",
            "final_approved": "本 Episode 已完成",
        }
        delivery_manifest = episode.stage_manifests.get("delivery")
        approval_manifest = episode.stage_manifests.get("final_approval")
        delivery_root: Path | None = None
        delivery_version: int | None = None
        approval_path: Path | None = None
        if delivery_manifest is not None and delivery_manifest.status == "completed":
            try:
                delivery_root = Path(delivery_manifest.inputs["delivery_root"])
                delivery_version = int(delivery_manifest.inputs["delivery_version"])
            except (KeyError, TypeError, ValueError):
                delivery_root = None
                delivery_version = None
        if approval_manifest is not None and approval_manifest.status == "completed":
            artifact = approval_manifest.outputs.get("approval_record")
            if artifact is not None:
                approval_path = Path(artifact.path)
        return MediaProductionView(
            book_id=episode.book_id,
            episode_id=episode.episode_id,
            status=episode.status,
            next_action=actions.get(episode.status, "继续当前媒体阶段"),
            present_scene_ids=present,
            missing_scene_ids=missing,
            batch_unlocked=batch_unlocked,
            cover_ready=(
                self.cover.ready(self._context(episode)) if self.cover is not None else False
            ),
            social_cover_ready=(
                self.social_cover.ready(self._context(episode))
                if self.social_cover is not None
                else False
            ),
            delivery_root=delivery_root,
            delivery_version=delivery_version,
            approval_record_path=approval_path,
        )


class LocalIllustrationGateway:
    def __init__(self, *, ffmpeg_command: str | Path) -> None:
        self.ffmpeg_command = ffmpeg_command

    def status(self, context: StageContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
        manifest = self._manifest(context)
        representatives = tuple(job for job in manifest.jobs if job.representative)
        present = tuple(
            job.scene_id for job in representatives if job.status in {"generated", "approved"}
        )
        missing = tuple(job.scene_id for job in representatives if job.scene_id not in present)
        return present, missing

    def batch_status(self, context: StageContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
        manifest = self._manifest(context)
        present = tuple(
            job.scene_id for job in manifest.jobs if job.status in {"generated", "approved"}
        )
        missing = tuple(job.scene_id for job in manifest.jobs if job.scene_id not in present)
        return present, missing

    def import_master(
        self,
        context: StageContext,
        scene_id: str,
        source: Path,
        *,
        representative_only: bool,
    ) -> None:
        manifest = self._manifest(context)
        matches = [
            job for job in manifest.jobs
            if job.scene_id == scene_id and (job.representative or not representative_only)
        ]
        if len(matches) != 1:
            raise MediaWorkflowError("scene_not_in_representatives")
        try:
            imported = import_image_master(
                matches[0], source, ffmpeg_command=self.ffmpeg_command
            )
            updated = record_imported_image(manifest, imported)
        except Exception:
            raise MediaWorkflowError("illustration_import_failed") from None
        atomic_write_json(self._manifest_path(context), updated.model_dump(mode="json"))

    def approve(self, context: StageContext) -> None:
        manifest = self._manifest(context)
        storyboard = self._storyboard(context)
        try:
            review = build_representative_review(storyboard, manifest)
            approval = approve_representatives(
                review, approved_image_sha256s=review.image_sha256s, reviewer="user"
            )
        except Exception:
            raise MediaWorkflowError("representative_review_failed") from None
        root = context.episode_root / "media" / "illustration"
        atomic_write_json(root / "representative_review.json", review.model_dump(mode="json"))
        atomic_write_json(root / "representative_approval.json", approval.model_dump(mode="json"))

    def unlock(self, context: StageContext) -> tuple[object, ...]:
        manifest = self._manifest(context)
        path = context.episode_root / "media" / "illustration" / "representative_approval.json"
        try:
            approval = RepresentativeApproval.model_validate_json(path.read_text(encoding="utf-8"))
            jobs = unlock_batch_jobs(manifest, approval)
            unlocked = {job.scene_id: job for job in jobs}
            updated = manifest.model_copy(
                update={
                    "jobs": tuple(unlocked.get(job.scene_id, job) for job in manifest.jobs)
                }
            )
            atomic_write_json(self._manifest_path(context), updated.model_dump(mode="json"))
            return jobs
        except Exception as error:
            code = getattr(error, "error_code", "representative_approval_stale")
            raise MediaWorkflowError(code) from None

    def _manifest(self, context: StageContext) -> IllustrationManifest:
        try:
            return IllustrationManifest.model_validate_json(
                self._manifest_path(context).read_text(encoding="utf-8")
            )
        except Exception:
            raise MediaWorkflowError("illustration_manifest_invalid") from None

    def _storyboard(self, context: StageContext) -> IllustrationStoryboard:
        path = context.episode_root / "media" / "illustration" / "illustration_storyboard.json"
        try:
            return IllustrationStoryboard.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            raise MediaWorkflowError("illustration_storyboard_invalid") from None

    @staticmethod
    def _manifest_path(context: StageContext) -> Path:
        return context.episode_root / "media" / "illustration" / "illustration_manifest.json"


class LocalCoverGateway:
    def ready(self, context: StageContext) -> bool:
        root = context.episode_root / "cover"
        try:
            manifest = root / "manifest.json"
            if not manifest.is_file() or manifest.is_symlink():
                return False
            payload = CoverManifest.model_validate_json(
                manifest.read_text(encoding="utf-8")
            )
            if (
                payload.book_id != context.book_id
                or payload.episode_id != context.episode_id
                or payload.source_sha256 != payload.copied_sha256
                or payload.matches_product_version is not True
            ):
                return False
            digest = payload.copied_sha256
            candidates = tuple(path for path in root.iterdir() if path.name != "manifest.json")
            return (
                len(candidates) == 1
                and candidates[0].is_file()
                and not candidates[0].is_symlink()
                and sha256_file(candidates[0]) == digest
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def import_cover(
        self,
        context: StageContext,
        source: Path,
        *,
        source_type: str,
        matches_product_version: bool,
    ) -> None:
        try:
            approve_cover(
                book_id=context.book_id,
                episode_id=context.episode_id,
                source_path=source,
                episode_cover_root=context.episode_root / "cover",
                source_type=source_type,
                matches_product_version=matches_product_version,
            )
        except Exception:
            raise MediaWorkflowError("cover_import_failed") from None


class LocalSocialCoverGateway:
    def __init__(self, *, ffmpeg_command: str | Path) -> None:
        self.renderer = SocialCoverRenderer(ffmpeg_command=ffmpeg_command)

    def ready(self, context: StageContext) -> bool:
        output = context.episode_root / "media" / "final" / "social_cover.png"
        manifest_path = output.with_suffix(".json")
        try:
            if (
                not output.is_file()
                or output.is_symlink()
                or not manifest_path.is_file()
                or manifest_path.is_symlink()
            ):
                return False
            manifest = SocialCoverManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            validate_social_cover_manifest_current(
                manifest,
                episode_root=context.episode_root,
            )
            return True
        except Exception:
            return False

    def import_current_run(
        self,
        context: StageContext,
        *,
        cover_brief_path: Path,
        source: Path,
        source_sha256: str,
    ) -> StageOutcome:
        manifest = self.renderer.normalize(
            SocialCoverRequest(
                episode_root=context.episode_root,
                cover_brief_path=cover_brief_path,
                source_image_path=source,
                source_image_sha256=source_sha256,
            )
        )
        return StageOutcome(
            outputs={
                "social_cover": manifest.output_path,
                "social_cover_manifest": manifest.manifest_path,
            },
            inputs={
                "approved_script_sha256": manifest.approved_script_sha256,
                "semantic_lock_sha256": manifest.semantic_lock_sha256,
                "production_profile_sha256": manifest.production_profile_sha256,
                "style_meta_sha256": manifest.style_meta_sha256,
                "style_atom_sha256": manifest.style_atom_sha256,
                "blueprint_sha256": manifest.blueprint_sha256,
                "prompt_sha256": manifest.prompt_sha256,
                "source_image_sha256": manifest.source_image_sha256,
            },
        )


class LocalDeliveryGateway:
    def __init__(self, *, store: StateStore, exporter: DeliveryExporter | None = None) -> None:
        self.store = store
        self.exporter = exporter or DeliveryExporter()

    def export_candidate(self, context: StageContext) -> DeliveryBundle:
        try:
            book = self.store.load_book(context.book_id)
            profile_path = self._profile_snapshot(context)
            media = context.episode_root / "media"
            return self.exporter.export_candidate(
                DeliveryRequest(
                    episode_root=context.episode_root,
                    book_id=context.book_id,
                    episode_id=context.episode_id,
                    title=book.title,
                    author="、".join(book.authors) if book.authors else "作者未标注",
                    script_path=context.episode_root / "script" / "approved.txt",
                    voice_path=media / "voice" / "voice_master.wav",
                    subtitle_path=media / "subtitles" / "subtitles.ass",
                    video_path=media / "final" / "final.mp4",
                    cover_path=media / "final" / "social_cover.png",
                    production_profile_path=profile_path,
                    voice_manifest_path=media / "voice" / "voice_master.json",
                    render_manifest_path=media / "final" / "final.render.json",
                    qc_manifest_path=media / "final" / "final.qc.json",
                    social_cover_manifest_path=media / "final" / "social_cover.json",
                )
            )
        except Exception as error:
            code = getattr(error, "error_code", "delivery_export_failed")
            raise MediaWorkflowError(code) from None

    def record_final_approval(
        self,
        context: StageContext,
        *,
        manifest_path: Path,
        version: int,
    ) -> ApprovalRecord:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = payload["artifacts"]
            root = manifest_path.parent
            files = tuple(
                root / artifacts[key]["path"]
                for key in ("video", "cover", "script", "voice", "subtitles")
            ) + (manifest_path,)
            bundle = DeliveryBundle(
                root=root,
                version=version,
                manifest_path=manifest_path,
                manifest_sha256=sha256_file(manifest_path),
                files=files,
                video_path=files[0],
                cover_path=files[1],
            )
            return self.exporter.record_final_approval(
                bundle,
                approved_at=self.exporter.clock(),
            )
        except Exception as error:
            code = getattr(error, "error_code", "final_approval_failed")
            raise MediaWorkflowError(code) from None

    @staticmethod
    def _profile_snapshot(context: StageContext) -> Path:
        path = context.episode_root / "production_profile.json"
        if path.is_file() and not path.is_symlink():
            return path
        snapshot = (
            context.episode_root
            / ".private"
            / "media"
            / "delivery"
            / "legacy-production-profile.json"
        )
        profile = load_production_profile(context.episode_root)
        if snapshot.exists():
            try:
                current = type(profile).model_validate_json(
                    snapshot.read_text(encoding="utf-8")
                )
            except Exception:
                raise MediaWorkflowError("legacy_profile_snapshot_invalid") from None
            if current != profile:
                raise MediaWorkflowError("legacy_profile_snapshot_stale")
            return snapshot
        atomic_write_json(snapshot, profile.model_dump(mode="json"))
        return snapshot
