from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from bv.core.atomic import atomic_write_json
from bv.core.hashing import sha256_file
from bv.delivery.exporter import ApprovalRecord, DeliveryBundle, DeliveryExporter, DeliveryRequest
from bv.illustration.assets import (
    IllustrationManifest,
    import_image_asset,
    record_imported_image,
)
from bv.illustration.contracts import IllustrationStoryboard
from bv.illustration.prompts import canonical_model_sha256
from bv.illustration.replacement import (
    IllustrationReplacementError,
    ReplacementImportResult,
    ReplacementPreparation,
    ensure_ordinary_import_allowed,
    import_authorized_replacement,
    prepare_illustration_replacement,
)
from bv.illustration.restyle import (
    IllustrationRestyleError,
    RestyleProvenance,
    RestyleResult,
    restyle_illustrations,
    rollback_restyle_install,
    validate_current_illustration_provenance,
)
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
from bv.subtitles.inputs import (
    SubtitleBreakInputError,
    load_subtitle_approved_text,
    load_subtitle_break_input,
)
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


def _catalog_identity(path: Path) -> tuple[Path, str]:
    candidate = Path(os.path.abspath(path))
    current = candidate
    try:
        while True:
            if os.path.lexists(current):
                attributes = getattr(
                    current.stat(follow_symlinks=False),
                    "st_file_attributes",
                    0,
                )
                if current.is_symlink() or attributes & getattr(
                    os, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
                ):
                    raise ValueError
            parent = current.parent
            if parent == current:
                break
            current = parent
        if not candidate.is_file():
            raise ValueError
        resolved = candidate.resolve(strict=True)
        return resolved, sha256_file(resolved)
    except (OSError, ValueError):
        raise MediaWorkflowError("media_catalog_configuration_invalid") from None


def _trusted_catalog_path(
    stages: Mapping[str, StageRunner],
    explicit: Path | None,
) -> Path | None:
    from .media_stages import PrepareIllustrationsStage, StyleSelectionStage

    select = stages.get("select_style")
    prepare = stages.get("prepare_representatives")
    select_is_provider = type(select) is StyleSelectionStage
    prepare_is_provider = type(prepare) is PrepareIllustrationsStage
    if select_is_provider != prepare_is_provider:
        raise MediaWorkflowError("media_catalog_configuration_invalid")
    candidates: list[Path] = []
    if select_is_provider:
        assert type(select) is StyleSelectionStage
        assert type(prepare) is PrepareIllustrationsStage
        candidates.extend((select.catalog_path, prepare.catalog_path))
    if explicit is not None:
        candidates.append(Path(explicit))
    if not candidates:
        return None
    identities = tuple(_catalog_identity(candidate) for candidate in candidates)
    if len(set(identities)) != 1:
        raise MediaWorkflowError("media_catalog_configuration_invalid")
    return identities[0][0]


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
        legacy_config_sha256: str | None = None,
        stage_config_sha256s: Mapping[str, str] | None = None,
        stage_config_sha256_resolver: Callable[[str, Path], str] | None = None,
        configured_mode: Literal["technical_sample", "full"] | None = None,
        cover: CoverGateway | None = None,
        social_cover: SocialCoverGateway | None = None,
        delivery: DeliveryGateway | None = None,
        voice_auditions: VoiceAuditionGateway | None = None,
        restyle_catalog_path: Path | None = None,
        book_menu_refresher: Callable[[], object] | None = None,
    ) -> None:
        self.store = store
        self.stages = dict(stages)
        self.images = images
        self.config_sha256 = config_sha256
        self.legacy_config_sha256 = legacy_config_sha256
        self.stage_config_sha256s = (
            None if stage_config_sha256s is None else dict(stage_config_sha256s)
        )
        self.stage_config_sha256_resolver = stage_config_sha256_resolver
        if (
            self.stage_config_sha256s is not None
            and stage_config_sha256_resolver is not None
        ):
            raise MediaWorkflowError("media_stage_config_invalid")
        if self.stage_config_sha256s is not None and any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.stage_config_sha256s.values()
        ):
            raise MediaWorkflowError("media_stage_config_invalid")
        if legacy_config_sha256 is not None and (
            len(legacy_config_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in legacy_config_sha256
            )
        ):
            raise MediaWorkflowError("media_stage_config_invalid")
        self.configured_mode = configured_mode
        self.cover = cover
        self.social_cover = social_cover
        self.delivery = delivery
        self.voice_auditions = voice_auditions
        self.restyle_catalog_path = _trusted_catalog_path(
            self.stages,
            None if restyle_catalog_path is None else Path(restyle_catalog_path),
        )
        self.book_menu_refresher = book_menu_refresher

    def reject_illustration(
        self,
        book_id: str,
        episode_id: str,
        *,
        asset_id: str,
        expected_asset_sha256: str,
        reason: str,
        expected_representative_review_sha256: str | None = None,
    ) -> ReplacementPreparation:
        try:
            return prepare_illustration_replacement(
                store=self.store,
                book_id=book_id,
                episode_id=episode_id,
                asset_id=asset_id,
                expected_asset_sha256=expected_asset_sha256,
                catalog_path=self.restyle_catalog_path,
                expected_representative_review_sha256=(
                    expected_representative_review_sha256
                ),
                reason=reason,
            )
        except IllustrationReplacementError as error:
            raise MediaWorkflowError(error.error_code) from None

    def import_replacement(
        self,
        book_id: str,
        episode_id: str,
        *,
        asset_id: str,
        nonce: str,
        source: Path,
    ) -> ReplacementImportResult:
        ffmpeg_command = getattr(self.images, "ffmpeg_command", None)
        if not isinstance(ffmpeg_command, (str, Path)):
            raise MediaWorkflowError("replacement_import_not_configured")
        try:
            return import_authorized_replacement(
                store=self.store,
                book_id=book_id,
                episode_id=episode_id,
                asset_id=asset_id,
                nonce=nonce,
                source_path=Path(source),
                ffmpeg_command=ffmpeg_command,
            )
        except IllustrationReplacementError as error:
            raise MediaWorkflowError(error.error_code) from None

    def restyle(
        self,
        book_id: str,
        episode_id: str,
        *,
        style_id: str,
    ) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        if self.restyle_catalog_path is None:
            raise MediaWorkflowError("restyle_not_configured")
        mode = self._restyle_media_mode(episode)
        state_path = self._episode_root(book_id, episode_id) / "episode.json"
        try:
            old_state_bytes = state_path.read_bytes()
        except OSError:
            raise MediaWorkflowError("episode_state_unavailable") from None
        try:
            result = restyle_illustrations(
                episode=episode,
                episode_root=self._episode_root(book_id, episode_id),
                catalog_path=self.restyle_catalog_path,
                requested_style_id=style_id,
            )
        except IllustrationRestyleError as error:
            raise MediaWorkflowError(error.error_code) from None
        try:
            updated = self._restyled_state(episode, result=result, mode=mode)
            updated.status = "representative_generation_running"
            view = self._view(updated, present=(), missing=result.missing_asset_ids)
            self.store.save_episode(updated)
        except Exception:
            try:
                rollback_restyle_install(
                    episode_root=self._episode_root(book_id, episode_id), result=result
                )
            except IllustrationRestyleError:
                pass
            self._restore_episode_bytes(state_path, old_state_bytes)
            raise MediaWorkflowError("restyle_state_commit_failed") from None
        return view

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
        provenance = self._restyle_provenance(
            episode,
            allow_original_incomplete=True,
        )
        if provenance is not None and any(
            not self._manifest_current(episode, name, mode=mode)
            for name in ("select_style", "plan_illustrations", "prepare_representatives")
        ):
            return self._resume_restyle_override(
                episode,
                mode=mode,
                requested_style_id=provenance.override.requested_style_id,
            )
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
        view = self._view(episode, present=present, missing=missing)
        self.store.save_episode(episode)
        return view

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
        self._restyle_provenance(episode)
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
        self._restyle_provenance(episode)
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
        self._restyle_provenance(episode)
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
        self._require_current_subtitles_for_downstream(episode)
        self._restyle_provenance(episode)
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
        self._require_current_subtitles_for_downstream(episode)
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
        if self.book_menu_refresher is not None:
            self.book_menu_refresher()
        return self._view(episode)

    def status(self, book_id: str, episode_id: str) -> MediaProductionView:
        episode = self._load(book_id, episode_id)
        self._restyle_provenance(episode)
        representative_status = episode.status in {
            "representative_generation_running",
            "awaiting_representative_review",
        }
        try:
            if episode.status in {"illustration_batch_running", "illustrations_ready"}:
                present, missing = self.images.batch_status(self._context(episode))
            else:
                present, missing = self.images.status(self._context(episode))
        except Exception:
            if representative_status:
                raise MediaWorkflowError("representative_surface_invalid") from None
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
        self._apply_outcome_in_memory(episode, name, outcome, mode=mode)
        self.store.save_episode(episode)

    def _apply_outcome_in_memory(
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
            config_sha256=self._recorded_config_identity(episode, name),
        )
        if name not in episode.completed_stages:
            episode.completed_stages.append(name)
        if name in episode.stale_stages:
            episode.stale_stages.remove(name)

    def _manifest_current(
        self,
        episode: EpisodeState,
        name: str,
        *,
        mode: Literal["technical_sample", "full"] | None,
        require_outputs: bool = False,
    ) -> bool:
        manifest = episode.stage_manifests.get(name)
        expected_config_sha256 = self._expected_config_sha256(episode, name)
        if (
            manifest is None
            or manifest.stage != name
            or manifest.status != "completed"
            or name in episode.stale_stages
            or (require_outputs and not manifest.outputs)
            or not self._config_identity_current(
                manifest,
                expected_sha256=expected_config_sha256,
            )
        ):
            if name == "subtitles":
                invalidate_from(episode, "subtitles")
            return False
        if mode is not None and manifest.inputs.get("media_mode") != mode:
            if name == "subtitles":
                invalidate_from(episode, "subtitles")
            return False
        if name == "subtitles" and not self._subtitle_inputs_current(
            episode,
            manifest,
        ):
            invalidate_from(episode, "subtitles")
            return False
        for artifact in manifest.outputs.values():
            path = Path(artifact.path)
            mutable_restyle_ledger = (
                name == "prepare_representatives"
                and "restyle_plan_sha256" in manifest.inputs
            )
            try:
                invalid = (
                    not self._artifact_path_current(episode, path)
                    or (
                        not mutable_restyle_ledger
                        and (
                            path.stat().st_size != artifact.size_bytes
                            or sha256_file(path) != artifact.sha256
                        )
                    )
                )
            except (OSError, ValueError):
                invalid = True
            if invalid:
                if name == "subtitles":
                    invalidate_from(episode, "subtitles")
                return False
        return True

    def _artifact_path_current(self, episode: EpisodeState, path: Path) -> bool:
        root = self._episode_root(episode.book_id, episode.episode_id)
        try:
            canonical_root = Path(os.path.abspath(root))
            canonical_path = Path(os.path.abspath(path))
            if (
                canonical_path == canonical_root
                or not canonical_path.is_relative_to(canonical_root)
            ):
                return False
            current = canonical_path
            while True:
                if os.path.lexists(current):
                    attributes = getattr(
                        current.stat(follow_symlinks=False),
                        "st_file_attributes",
                        0,
                    )
                    if current.is_symlink() or attributes & getattr(
                        os,
                        "FILE_ATTRIBUTE_REPARSE_POINT",
                        0x400,
                    ):
                        return False
                if current == canonical_root:
                    break
                parent = current.parent
                if parent == current:
                    return False
                current = parent
            if not canonical_path.is_file():
                return False
            resolved_root = canonical_root.resolve(strict=True)
            resolved_path = canonical_path.resolve(strict=True)
            return resolved_path.is_relative_to(resolved_root)
        except (OSError, RuntimeError, ValueError):
            return False

    def _subtitle_inputs_current(
        self,
        episode: EpisodeState,
        manifest: StageManifest,
    ) -> bool:
        root = self._episode_root(episode.book_id, episode.episode_id)
        try:
            break_input = load_subtitle_break_input(root)
            approved_text = (
                load_subtitle_approved_text(root)
                if break_input.present
                else ""
            )
        except (OSError, UnicodeError, SubtitleBreakInputError):
            return False
        if break_input.present and not break_input.reconstructs(approved_text):
            return False
        expected = {
            "subtitle_sequence_mode": break_input.sequence_mode,
            "subtitle_breaks_presence": (
                "present" if break_input.present else "absent"
            ),
        }
        if any(manifest.inputs.get(key) != value for key, value in expected.items()):
            return False
        recorded_sha256 = manifest.inputs.get("subtitle_breaks_sha256")
        if break_input.sha256 is None:
            return recorded_sha256 is None
        return recorded_sha256 == break_input.sha256

    def _require_current_subtitles_for_downstream(
        self,
        episode: EpisodeState,
    ) -> None:
        mode = self._trusted_downstream_media_mode(episode)
        if mode is not None and self._manifest_current(
            episode,
            "subtitles",
            mode=mode,
        ):
            return
        invalidate_from(episode, "subtitles")
        if episode.status != "final_approved":
            episode.status = "illustration_planning"
            self.store.save_episode(episode)
        raise MediaWorkflowError("subtitles_stale")

    def _trusted_downstream_media_mode(
        self,
        episode: EpisodeState,
    ) -> Literal["technical_sample", "full"] | None:
        if self.configured_mode in {"technical_sample", "full"}:
            return self.configured_mode
        evidence: set[str] = set()
        for name in ("tts", "asr"):
            manifest = episode.stage_manifests.get(name)
            try:
                current = self._manifest_current(
                    episode,
                    name,
                    mode=None,
                    require_outputs=True,
                )
            except MediaWorkflowError:
                return None
            if manifest is None or not current:
                return None
            recorded = manifest.inputs.get("media_mode")
            if recorded not in {"technical_sample", "full"}:
                return None
            evidence.add(recorded)
        if len(evidence) != 1:
            return None
        mode = next(iter(evidence))
        return "full" if mode == "full" else "technical_sample"

    def _config_identity_current(
        self,
        manifest: StageManifest,
        *,
        expected_sha256: str,
    ) -> bool:
        scoped = (
            self.stage_config_sha256s is not None
            or self.stage_config_sha256_resolver is not None
        )
        expected_schema = "stage-v1" if scoped else "global-v1"
        versioned = f"{expected_schema}:{expected_sha256}"
        if manifest.config_sha256 == versioned:
            return True
        if manifest.config_sha256.startswith(("stage-v1:", "global-v1:")):
            return False
        if not scoped:
            return manifest.config_sha256 == expected_sha256
        return (
            self.legacy_config_sha256 is not None
            and manifest.config_sha256 == self.legacy_config_sha256
        )

    def _recorded_config_identity(self, episode: EpisodeState, name: str) -> str:
        schema = (
            "stage-v1"
            if self.stage_config_sha256s is not None
            or self.stage_config_sha256_resolver is not None
            else "global-v1"
        )
        return f"{schema}:{self._expected_config_sha256(episode, name)}"

    def _expected_config_sha256(self, episode: EpisodeState, name: str) -> str:
        if self.stage_config_sha256_resolver is not None:
            try:
                value = self.stage_config_sha256_resolver(
                    name, self._episode_root(episode.book_id, episode.episode_id)
                )
            except Exception:
                raise MediaWorkflowError("media_stage_config_invalid") from None
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise MediaWorkflowError("media_stage_config_invalid")
            return value
        if self.stage_config_sha256s is None:
            return self.config_sha256
        base = self.stage_config_sha256s.get(name)
        if base is None:
            raise MediaWorkflowError("media_stage_config_missing")
        return base

    def _restyle_provenance(
        self,
        episode: EpisodeState,
        *,
        allow_original_incomplete: bool = False,
    ) -> RestyleProvenance | None:
        root = self._episode_root(episode.book_id, episode.episode_id)
        try:
            return validate_current_illustration_provenance(
                episode=episode,
                episode_root=root,
                catalog_path=self.restyle_catalog_path,
                allow_original_incomplete=allow_original_incomplete,
            )
        except IllustrationRestyleError as error:
            raise MediaWorkflowError(error.error_code) from None

    def _resume_restyle_override(
        self,
        episode: EpisodeState,
        *,
        mode: Literal["technical_sample", "full"],
        requested_style_id: str,
    ) -> MediaProductionView:
        if self.restyle_catalog_path is None:
            raise MediaWorkflowError("restyle_not_configured")
        root = self._episode_root(episode.book_id, episode.episode_id)
        state_path = root / "episode.json"
        try:
            old_state_bytes = state_path.read_bytes()
            result = restyle_illustrations(
                episode=episode,
                episode_root=root,
                catalog_path=self.restyle_catalog_path,
                requested_style_id=requested_style_id,
                allow_selected_style=True,
            )
        except IllustrationRestyleError as error:
            raise MediaWorkflowError(error.error_code) from None
        except OSError:
            raise MediaWorkflowError("episode_state_unavailable") from None
        try:
            updated = self._restyled_state(episode, result=result, mode=mode)
            if any(
                not self._manifest_current(updated, name, mode=mode)
                for name in HANDDRAWN_MEDIA_PREPARE_ORDER
            ):
                raise MediaWorkflowError("restyle_rebase_requires_upstream")
            present, missing = self.images.status(self._context(updated))
            updated.status = (
                "representative_generation_running" if missing
                else "awaiting_representative_review"
            )
            view = self._view(updated, present=present, missing=missing)
            self.store.save_episode(updated)
        except Exception:
            try:
                rollback_restyle_install(episode_root=root, result=result)
            except IllustrationRestyleError:
                pass
            self._restore_episode_bytes(state_path, old_state_bytes)
            raise MediaWorkflowError("restyle_state_commit_failed") from None
        return view

    def _restyled_state(
        self,
        episode: EpisodeState,
        *,
        result: RestyleResult,
        mode: Literal["technical_sample", "full"],
    ) -> EpisodeState:
        updated = episode.model_copy(deep=True)
        invalidate_from(updated, "style_decision")
        illustration = self._episode_root(
            episode.book_id, episode.episode_id
        ) / "media" / "illustration"
        provenance_inputs = {
            "requested_style_id": result.selected_style,
            "style_fingerprint": result.style_fingerprint,
            "catalog_sha256": result.catalog_sha256,
            "restyle_override_sha256": result.override_sha256,
            "restyle_plan_sha256": result.plan_sha256,
        }
        self._apply_outcome_in_memory(
            updated,
            "select_style",
            StageOutcome(
                outputs={"style_decision": illustration / "style_decision.json"},
                inputs=provenance_inputs,
            ),
            mode=mode,
        )
        self._apply_outcome_in_memory(
            updated,
            "plan_illustrations",
            StageOutcome(
                outputs={
                    "character_bible": illustration / "character_bible.json",
                    "illustration_storyboard": illustration / "illustration_storyboard.json",
                },
                inputs=provenance_inputs,
            ),
            mode=mode,
        )
        self._apply_outcome_in_memory(
            updated,
            "prepare_representatives",
            StageOutcome(
                outputs={
                    "illustration_manifest": illustration / "illustration_manifest.json"
                },
                inputs=provenance_inputs,
            ),
            mode=mode,
        )
        return updated

    @staticmethod
    def _restore_episode_bytes(path: Path, payload: bytes) -> None:
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".restore", dir=path.parent
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
        except OSError:
            return
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def _restyle_media_mode(
        self, episode: EpisodeState
    ) -> Literal["technical_sample", "full"]:
        names = ("select_style", "plan_illustrations", "prepare_representatives")
        modes = {episode.stage_manifests.get(name).inputs.get("media_mode") if episode.stage_manifests.get(name) else None for name in names}
        if len(modes) != 1 or next(iter(modes)) not in {"technical_sample", "full"}:
            raise MediaWorkflowError("restyle_media_mode_invalid")
        mode = next(iter(modes))
        if self.configured_mode is not None and mode != self.configured_mode:
            raise MediaWorkflowError("restyle_media_mode_invalid")
        return mode

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
        sequence_mode = None
        if episode.status in {
            "representative_generation_running",
            "awaiting_representative_review",
        }:
            sequence_mode = self._validated_representative_mode(
                episode, present=present, missing=missing
            )
        representative_action = "向用户展示开头、中段、结尾三张代表图并等待批准"
        if sequence_mode == "color-story-pair":
            representative_action = (
                "向用户展示开头、中段、结尾三组完整 A/B 代表图，共六张，并逐对检查"
                "人物/服装/场景/镜头/动作连续性后等待批准"
            )
        actions = {
            "awaiting_voice_audition_review": "向用户播放同一文稿片段的候选音色并等待批准",
            "voice_profile_approved": "使用已批准音色生成完整旁白",
            "representative_generation_running": "在当前对话生成并导入缺少的代表插画",
            "awaiting_representative_review": representative_action,
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

    def _validated_representative_mode(
        self,
        episode: EpisodeState,
        *,
        present: tuple[str, ...],
        missing: tuple[str, ...],
    ) -> str:
        root = self._episode_root(episode.book_id, episode.episode_id)
        illustration = root / "media" / "illustration"
        try:
            storyboard = IllustrationStoryboard.model_validate_json(
                (illustration / "illustration_storyboard.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = IllustrationManifest.model_validate_json(
                (illustration / "illustration_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                manifest.storyboard_sha256 != canonical_model_sha256(storyboard)
                or manifest.book_id != episode.book_id
                or manifest.episode_id != episode.episode_id
                or manifest.episode_root != root.absolute()
            ):
                raise ValueError
            representatives = tuple(job for job in manifest.jobs if job.representative)
            storyboard_ids = tuple(
                scene.scene_id
                for scene in storyboard.scenes
                if scene.representative_frame
            )
            if storyboard.sequence_mode == "color-story-pair":
                all_scene_ids = tuple(scene.scene_id for scene in storyboard.scenes)
                if len(all_scene_ids) == 3:
                    expected_storyboard_ids = all_scene_ids
                elif len(all_scene_ids) == 4:
                    expected_storyboard_ids = (
                        all_scene_ids[0],
                        all_scene_ids[2],
                        all_scene_ids[3],
                    )
                else:
                    raise ValueError
                expected = tuple(
                    asset_id
                    for scene_id in expected_storyboard_ids
                    for asset_id in (f"{scene_id}-A", f"{scene_id}-B")
                )
                manifest_ids = tuple(job.asset_id for job in representatives)
                expected_identities = tuple(
                    (scene_id, phase, f"{scene_id}-{suffix}")
                    for scene_id in expected_storyboard_ids
                    for phase, suffix in (("anchor", "A"), ("continuation", "B"))
                )
                identities = tuple(
                    (job.scene_id, job.phase, job.asset_id)
                    for job in representatives
                )
                if (
                    storyboard_ids != expected_storyboard_ids
                    or manifest_ids != expected
                    or len(set(manifest_ids)) != 6
                    or identities != expected_identities
                    or any(
                        job.phase == "continuation"
                        and (
                            not job.reference_scene_ids
                            or job.reference_scene_ids[0] != job.scene_id
                            or job.reference_scene_ids.count(job.scene_id) != 1
                        )
                        for job in representatives
                    )
                ):
                    raise ValueError
            else:
                expected = tuple(job.scene_id for job in representatives)
                if (
                    len(expected) != 3
                    or len(set(expected)) != 3
                    or expected != storyboard_ids
                    or any(
                        job.phase != "legacy" or job.asset_id is not None
                        for job in representatives
                    )
                ):
                    raise ValueError
            if (
                len(present) + len(missing) != len(expected)
                or len(set((*present, *missing))) != len(expected)
                or set(present).intersection(missing)
                or set((*present, *missing)) != set(expected)
                or present != tuple(item for item in expected if item in present)
                or missing != tuple(item for item in expected if item in missing)
            ):
                raise ValueError
            return storyboard.sequence_mode
        except Exception:
            raise MediaWorkflowError("representative_surface_invalid") from None


class LocalIllustrationGateway:
    def __init__(self, *, ffmpeg_command: str | Path) -> None:
        self.ffmpeg_command = ffmpeg_command

    def status(self, context: StageContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
        manifest = self._manifest(context)
        representatives = tuple(job for job in manifest.jobs if job.representative)
        present = tuple(
            self._job_id(job) for job in representatives if job.status in {"generated", "approved"}
        )
        missing = tuple(
            self._job_id(job) for job in representatives if self._job_id(job) not in present
        )
        return present, missing

    def batch_status(self, context: StageContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
        manifest = self._manifest(context)
        present = tuple(
            self._job_id(job) for job in manifest.jobs if job.status in {"generated", "approved"}
        )
        missing = tuple(
            self._job_id(job) for job in manifest.jobs if self._job_id(job) not in present
        )
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
            if self._job_id(job) == scene_id and (job.representative or not representative_only)
        ]
        if len(matches) != 1:
            raise MediaWorkflowError("scene_not_in_representatives")
        try:
            ensure_ordinary_import_allowed(
                episode_root=context.episode_root,
                state=getattr(context, "episode_state", None),
                manifest=manifest,
                asset_id=scene_id,
            )
        except IllustrationReplacementError as error:
            raise MediaWorkflowError(error.error_code) from None
        pending = next(
            (
                job
                for job in manifest.jobs
                if job.status == "planned"
                and (job.representative or not representative_only)
            ),
            None,
        )
        if pending is None or self._job_id(matches[0]) != self._job_id(pending):
            raise MediaWorkflowError("illustration_import_order_invalid")
        try:
            job = matches[0]
            imported = import_image_asset(
                job, source, manifest=manifest, ffmpeg_command=self.ffmpeg_command
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
        storyboard = self._storyboard(context)
        path = context.episode_root / "media" / "illustration" / "representative_approval.json"
        try:
            approval = RepresentativeApproval.model_validate_json(path.read_text(encoding="utf-8"))
            jobs = unlock_batch_jobs(storyboard, manifest, approval)
            unlocked = {self._job_id(job): job for job in jobs}
            updated = manifest.model_copy(
                update={
                    "jobs": tuple(
                        unlocked.get(self._job_id(job), job) for job in manifest.jobs
                    )
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

    @staticmethod
    def _job_id(job: object) -> str:
        asset_id = getattr(job, "asset_id", None)
        return asset_id if isinstance(asset_id, str) else getattr(job, "scene_id")


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
