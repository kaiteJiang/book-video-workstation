from __future__ import annotations

import json
import hashlib
import os
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from bv.core.atomic import atomic_write_json
from bv.core.hashing import sha256_file
from bv.illustration.assets import (
    IllustrationManifest,
    image_prompt_identity_sha256,
    inspect_image,
    prepare_image_jobs,
    validate_generated_image_job,
)
from bv.illustration.contracts import (
    CharacterBible,
    IllustrationStoryboard,
    StyleDecision,
    load_character_bible,
)
from bv.illustration.prompts import canonical_model_sha256
from bv.illustration.reviews import (
    RepresentativeApproval,
    RepresentativeReview,
    build_representative_review,
)
from bv.illustration.style_selector import (
    StyleSelectionError,
    load_style_catalog,
    manual_style_decision,
)
from bv.production.profile import ProductionProfileError, load_production_profile
from bv.state.models import EpisodeState, StageManifest


_PLANNING_STAGE_OUTPUTS = {
    "select_style": {"style_decision": "style_decision.json"},
    "plan_illustrations": {
        "character_bible": "character_bible.json",
        "illustration_storyboard": "illustration_storyboard.json",
    },
    "prepare_representatives": {
        "illustration_manifest": "illustration_manifest.json"
    },
}


class IllustrationRestyleError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


@dataclass(frozen=True, slots=True)
class RestyleResult:
    selected_style: str
    style_fingerprint: str
    catalog_sha256: str
    override_sha256: str
    plan_sha256: str
    recovery_path: Path
    missing_asset_ids: tuple[str, ...]
    protected_final_relpaths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class RestyleOverride:
    requested_style_id: str
    style_fingerprint: str
    catalog_sha256: str


@dataclass(frozen=True, slots=True)
class RestyleProvenance:
    override: RestyleOverride
    override_sha256: str
    decision: StyleDecision
    storyboard: IllustrationStoryboard


def restyle_illustrations(
    *,
    episode: EpisodeState,
    episode_root: Path,
    catalog_path: Path,
    requested_style_id: str,
    allow_selected_style: bool = False,
) -> RestyleResult:
    """Replace only nonlegacy illustration style dependencies and assets."""
    root = Path(episode_root)
    if episode.status == "final_approved":
        raise IllustrationRestyleError("restyle_final_approved")
    illustration_root = root / "media" / "illustration"
    _validate_child(root, illustration_root)
    try:
        old_decision = StyleDecision.model_validate_json(
            (illustration_root / "style_decision.json").read_text(encoding="utf-8")
        )
        old_storyboard = IllustrationStoryboard.model_validate_json(
            (illustration_root / "illustration_storyboard.json").read_text(encoding="utf-8")
        )
        old_bible = load_character_bible(
            json.loads((illustration_root / "character_bible.json").read_text(encoding="utf-8"))
        )
    except Exception:
        raise IllustrationRestyleError("restyle_source_invalid") from None
    if old_storyboard.sequence_mode != "color-story-pair":
        raise IllustrationRestyleError("restyle_legacy_episode")
    if (
        old_decision.selected_style == requested_style_id
        and not allow_selected_style
    ):
        raise IllustrationRestyleError("restyle_style_unchanged")
    try:
        catalog = load_style_catalog(Path(catalog_path))
        decision = manual_style_decision(
            catalog,
            style_id=requested_style_id,
            source_script_sha256=old_decision.source_script_sha256,
        )
        style = next(item for item in catalog if item.style_id == decision.selected_style)
    except (StopIteration, StyleSelectionError):
        raise IllustrationRestyleError("restyle_style_not_in_catalog") from None
    if decision.selected_style == "legacy-monochrome-reveal":
        raise IllustrationRestyleError("restyle_style_not_in_catalog")

    bible = _restyled_bible(old_bible, decision)
    storyboard = old_storyboard.model_copy(
        update={
            "style_decision_sha256": canonical_model_sha256(decision),
            "character_lock_sha256": canonical_model_sha256(bible),
            "character_bible_sha256": canonical_model_sha256(bible),
        }
    )
    _assert_semantic_equivalence(old_storyboard, storyboard)
    try:
        manifest = prepare_image_jobs(storyboard, bible, style, decision, root)
    except Exception:
        raise IllustrationRestyleError("restyle_manifest_invalid") from None
    if not 6 <= len(manifest.jobs) <= 96 or any(
        job.status != "planned"
        or job.master_sha256 is not None
        or job.anchor_sha256 is not None
        or job.reference_image_sha256s
        for job in manifest.jobs
    ):
        raise IllustrationRestyleError("restyle_manifest_invalid")

    try:
        protected_paths = _social_cover_paths(episode, root)
        protected_final_relpaths = tuple(
            path.relative_to(root / "media" / "final") for path in protected_paths
        )
    except IllustrationRestyleError:
        raise
    except ValueError:
        raise IllustrationRestyleError("social_cover_recovery_invalid") from None
    result_selected_style = decision.selected_style
    result_style_fingerprint = decision.style_fingerprint
    result_missing_asset_ids = tuple(
        job.asset_id
        for job in manifest.jobs
        if job.representative and job.asset_id is not None
    )
    try:
        catalog_sha256 = sha256_file(Path(catalog_path))
    except OSError:
        raise IllustrationRestyleError("restyle_style_not_in_catalog") from None
    override = RestyleOverride(
        requested_style_id=result_selected_style,
        style_fingerprint=result_style_fingerprint,
        catalog_sha256=catalog_sha256,
    )

    staging = root / ".private" / f"style-restyle-staging-{uuid.uuid4().hex}"
    staged_illustration = staging / "illustration"
    recovery = root / ".recovery" / "style-restyle" / f"{time.time_ns()}-{uuid.uuid4().hex}"
    _validate_child(root, staging)
    _validate_child(root, recovery)
    try:
        staged_illustration.mkdir(parents=True, exist_ok=False)
        atomic_write_json(staged_illustration / "style_decision.json", decision.model_dump(mode="json"))
        atomic_write_json(staged_illustration / "character_bible.json", bible.model_dump(mode="json"))
        atomic_write_json(staged_illustration / "illustration_storyboard.json", storyboard.model_dump(mode="json"))
        atomic_write_json(staged_illustration / "illustration_manifest.json", manifest.model_dump(mode="json"))
        atomic_write_json(staging / "illustration_style_override.json", {
            "requested_style_id": override.requested_style_id,
            "style_fingerprint": override.style_fingerprint,
            "catalog_sha256": override.catalog_sha256,
        })
        override_sha256 = sha256_file(staging / "illustration_style_override.json")
        _validate_staged(staged_illustration, old_storyboard)
        _install_with_recovery(
            root,
            staging,
            recovery,
            protected_paths=protected_paths,
            staged_override=staging / "illustration_style_override.json",
            override_path=_restyle_override_path(root),
        )
    except IllustrationRestyleError:
        raise
    except Exception:
        raise IllustrationRestyleError("restyle_recovery_failed") from None
    return RestyleResult(
        selected_style=result_selected_style,
        style_fingerprint=result_style_fingerprint,
        catalog_sha256=catalog_sha256,
        override_sha256=override_sha256,
        plan_sha256=restyle_plan_sha256(storyboard, manifest),
        recovery_path=recovery,
        missing_asset_ids=result_missing_asset_ids,
        protected_final_relpaths=protected_final_relpaths,
    )


def rollback_restyle_install(*, episode_root: Path, result: RestyleResult) -> None:
    """Recover the pre-restyle media tree after the sole state commit fails."""
    root = Path(episode_root)
    recovery = result.recovery_path
    media = root / "media"
    override_path = _restyle_override_path(root)
    _validate_child(root, recovery)
    failed = recovery.parent / f"failed-new-{uuid.uuid4().hex}"
    _validate_child(root, failed)
    try:
        failed.mkdir(parents=True, exist_ok=False)
        for label in ("illustration", "render", "final", "qc"):
            current = media / label
            if current.exists():
                _validate_tree(root, current)
                os.replace(current, failed / label)
        if override_path.exists():
            _validate_child(root, override_path)
            os.replace(override_path, failed / "illustration_style_override.json")
        for label in ("illustration", "render", "final", "qc"):
            archived = recovery / label
            if archived.exists():
                _validate_tree(root, archived)
                _validate_child(root, media / label)
                os.replace(archived, media / label)
        archived_override = recovery / "illustration_style_override.json"
        if archived_override.exists():
            _validate_child(root, archived_override)
            os.replace(archived_override, override_path)
        for relative in result.protected_final_relpaths:
            source = failed / "final" / relative
            destination = media / "final" / relative
            if source.is_file():
                _validate_child(root, destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
    except Exception:
        raise IllustrationRestyleError("restyle_recovery_failed") from None


def _restyled_bible(bible: CharacterBible, decision: StyleDecision) -> CharacterBible:
    return bible.model_copy(
        update={
            "style_fingerprint": decision.style_fingerprint,
            "characters": tuple(
                character.model_copy(update={"style_fingerprint": decision.style_fingerprint})
                for character in bible.characters
            ),
        }
    )


def _semantic_payload(storyboard: IllustrationStoryboard) -> dict[str, object]:
    payload = storyboard.model_dump(mode="json")
    for field in ("style_decision_sha256", "character_lock_sha256", "character_bible_sha256"):
        payload.pop(field, None)
    return payload


def normalized_semantic_fingerprint(storyboard: IllustrationStoryboard) -> str:
    payload = json.dumps(
        _semantic_payload(storyboard),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_semantic_equivalence(
    old: IllustrationStoryboard,
    new: IllustrationStoryboard,
) -> None:
    if (
        normalized_semantic_fingerprint(old) != normalized_semantic_fingerprint(new)
        or old.scenes != new.scenes
    ):
        raise IllustrationRestyleError("restyle_semantic_change_detected")


def _validate_staged(staged: Path, old_storyboard: IllustrationStoryboard) -> None:
    try:
        decision = StyleDecision.model_validate_json((staged / "style_decision.json").read_text(encoding="utf-8"))
        bible = load_character_bible(json.loads((staged / "character_bible.json").read_text(encoding="utf-8")))
        storyboard = IllustrationStoryboard.model_validate_json(
            (staged / "illustration_storyboard.json").read_text(encoding="utf-8")
        )
        manifest = IllustrationManifest.model_validate_json(
            (staged / "illustration_manifest.json").read_text(encoding="utf-8")
        )
    except Exception:
        raise IllustrationRestyleError("restyle_staging_invalid") from None
    _assert_semantic_equivalence(old_storyboard, storyboard)
    if (
        bible.style_fingerprint != decision.style_fingerprint
        or manifest.style_fingerprint != decision.style_fingerprint
        or manifest.storyboard_sha256 != canonical_model_sha256(storyboard)
        or any(
            job.master_sha256 is not None
            or job.anchor_sha256 is not None
            or job.reference_image_sha256s
            or job.status != "planned"
            for job in manifest.jobs
        )
    ):
        raise IllustrationRestyleError("restyle_staging_invalid")


def _install_with_recovery(
    root: Path,
    staging: Path,
    recovery: Path,
    *,
    protected_paths: tuple[Path, ...],
    staged_override: Path,
    override_path: Path,
) -> None:
    media = root / "media"
    targets = (
        (media / "illustration", "illustration"),
        (media / "render", "render"),
        (media / "final", "final"),
        (media / "qc", "qc"),
    )
    journal: list[tuple[Path, Path]] = []
    created_directories: list[Path] = []
    try:
        recovery.mkdir(parents=True, exist_ok=False)
        _validate_child(root, recovery)
        for target, label in targets:
            _validate_tree(root, target)
            if target.exists():
                if target.is_symlink() or not target.is_dir():
                    raise IllustrationRestyleError("unsafe_restyle_recovery")
                destination = recovery / label
                os.replace(target, destination)
                journal.append((target, destination))
        for original in protected_paths:
            archived = recovery / "final" / original.relative_to(media / "final")
            _validate_child(root, archived)
            if not archived.is_file() or archived.is_symlink():
                raise IllustrationRestyleError("social_cover_recovery_invalid")
            _mkdir_transaction_parents(root, original.parent, created_directories)
            os.replace(archived, original)
            journal.append((archived, original))
        _mkdir_transaction_parents(root, media, created_directories)
        _validate_tree(root, staging)
        os.replace(staging / "illustration", media / "illustration")
        journal.append((staging / "illustration", media / "illustration"))
        _validate_child(root, override_path)
        _mkdir_transaction_parents(root, override_path.parent, created_directories)
        if override_path.exists():
            archived_override = recovery / "illustration_style_override.json"
            os.replace(override_path, archived_override)
            journal.append((override_path, archived_override))
        os.replace(staged_override, override_path)
        journal.append((staged_override, override_path))
    except Exception:
        if not _rollback_install_journal(
            root=root,
            journal=journal,
            created_directories=created_directories,
        ):
            raise IllustrationRestyleError("restyle_recovery_failed") from None
        raise


def _social_cover_paths(episode: EpisodeState, root: Path) -> tuple[Path, ...]:
    manifest = episode.stage_manifests.get("social_cover")
    if manifest is None or manifest.status != "completed":
        return ()
    paths: list[Path] = []
    for artifact in manifest.outputs.values():
        path = Path(artifact.path)
        try:
            path.relative_to(root)
        except ValueError:
            raise IllustrationRestyleError("social_cover_recovery_invalid") from None
        _validate_child(root, path)
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    return tuple(paths)


def load_restyle_override(
    episode_root: Path,
    *,
    required: bool = False,
    expected_sha256: str | None = None,
) -> RestyleOverride | None:
    root = Path(episode_root)
    path = _restyle_override_path(root)
    _validate_child(root, path)
    if not path.exists():
        if required:
            raise IllustrationRestyleError("restyle_override_missing")
        return None
    try:
        if not path.is_file() or path.is_symlink():
            raise ValueError
        actual_sha256 = sha256_file(path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise IllustrationRestyleError("restyle_override_digest_mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        requested_style_id = payload["requested_style_id"]
        style_fingerprint = payload["style_fingerprint"]
        catalog_sha256 = payload["catalog_sha256"]
        if (
            not all(isinstance(value, str) and value for value in (
                requested_style_id, style_fingerprint, catalog_sha256
            ))
            or requested_style_id == "legacy-monochrome-reveal"
            or len(style_fingerprint) != 64
            or len(catalog_sha256) != 64
        ):
            raise ValueError
    except IllustrationRestyleError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise IllustrationRestyleError("restyle_override_invalid") from None
    return RestyleOverride(requested_style_id, style_fingerprint, catalog_sha256)


def restyle_plan_sha256(
    storyboard: IllustrationStoryboard,
    manifest: IllustrationManifest,
) -> str:
    """Hash only the immutable pair-plan projection, never mutable image results."""
    payload = {
        "schema": "restyle-plan-v1",
        "sequence_mode": storyboard.sequence_mode,
        "book_id": manifest.book_id,
        "episode_id": manifest.episode_id,
        "episode_root": str(manifest.episode_root),
        "storyboard_sha256": manifest.storyboard_sha256,
        "style_fingerprint": manifest.style_fingerprint,
        "character_lock_sha256": manifest.character_lock_sha256,
        "jobs": [
            {
                "scene_id": job.scene_id,
                "representative": job.representative,
                "prompt": job.prompt,
                "style_fingerprint": job.style_fingerprint,
                "character_lock_sha256": job.character_lock_sha256,
                "reference_scene_ids": list(job.reference_scene_ids),
                "phase": job.phase,
                "asset_id": job.asset_id,
                "output_master": str(job.output_master),
                "output_bw": None if job.output_bw is None else str(job.output_bw),
            }
            for job in manifest.jobs
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_current_illustration_provenance(
    *,
    episode: EpisodeState,
    episode_root: Path,
    catalog_path: Path | None,
    allow_original_incomplete: bool = False,
) -> RestyleProvenance | None:
    """Validate the current immutable illustration plan before mutable work."""
    root = Path(episode_root)
    names = ("select_style", "plan_illustrations", "prepare_representatives")
    manifests = tuple(episode.stage_manifests.get(name) for name in names)
    tracked = any(
        manifest is not None
        and any(
            key in manifest.inputs
            for key in ("restyle_override_sha256", "requested_style_id")
        )
        for manifest in manifests
    )
    if not tracked:
        override = load_restyle_override(root)
        if override is not None:
            raise IllustrationRestyleError("restyle_override_untracked")
        if allow_original_incomplete and _clean_original_planning_suffix_missing(
            episode=episode,
            root=root,
            manifests=manifests,
        ):
            return None
        sequence_mode = _validated_current_sequence_mode(
            episode=episode,
            root=root,
        )
        if sequence_mode is None or sequence_mode == "legacy-monochrome-reveal":
            return None
        _validate_original_plan_provenance(
            episode=episode,
            root=root,
            catalog_path=catalog_path,
            manifests=manifests,
        )
        return None
    if catalog_path is None or any(manifest is None for manifest in manifests):
        raise IllustrationRestyleError("restyle_override_provenance_invalid")

    keys = (
        "restyle_override_sha256",
        "requested_style_id",
        "style_fingerprint",
        "catalog_sha256",
        "restyle_plan_sha256",
    )
    provenance_records: set[tuple[str, str, str, str, str]] = set()
    _validate_planning_stage_bindings(
        episode=episode,
        root=root,
        manifests=manifests,
        error_code="restyle_override_provenance_invalid",
    )
    for manifest in manifests:
        assert manifest is not None
        try:
            record = tuple(manifest.inputs[key] for key in keys)
        except KeyError:
            raise IllustrationRestyleError(
                "restyle_override_provenance_invalid"
            ) from None
        provenance_records.add(record)  # type: ignore[arg-type]
    if len(provenance_records) != 1:
        raise IllustrationRestyleError("restyle_override_provenance_invalid")
    (
        override_sha256,
        requested_style_id,
        style_fingerprint,
        catalog_sha256,
        plan_sha256,
    ) = next(iter(provenance_records))
    return validate_restyle_provenance(
        episode=episode,
        episode_root=root,
        catalog_path=catalog_path,
        expected_override_sha256=override_sha256,
        expected_requested_style_id=requested_style_id,
        expected_style_fingerprint=style_fingerprint,
        expected_catalog_sha256=catalog_sha256,
        expected_plan_sha256=plan_sha256,
    )


def _validate_planning_stage_bindings(
    *,
    episode: EpisodeState,
    root: Path,
    manifests: tuple[StageManifest | None, ...],
    error_code: str,
) -> None:
    illustration = root / "media" / "illustration"
    try:
        _validate_tree(root, illustration)
        for name, raw_manifest in zip(
            _PLANNING_STAGE_OUTPUTS, manifests, strict=True
        ):
            if raw_manifest is None:
                raise ValueError
            _validate_planning_stage_binding(
                episode=episode,
                illustration=illustration,
                name=name,
                manifest=raw_manifest,
                expected_outputs=_PLANNING_STAGE_OUTPUTS[name],
            )
    except IllustrationRestyleError:
        raise
    except (OSError, TypeError, ValueError):
        raise IllustrationRestyleError(error_code) from None


def _validate_planning_stage_binding(
    *,
    episode: EpisodeState,
    illustration: Path,
    name: str,
    manifest: StageManifest,
    expected_outputs: dict[str, str],
) -> None:
    if (
        manifest.stage != name
        or manifest.status != "completed"
        or name not in episode.completed_stages
        or name in episode.stale_stages
        or set(manifest.outputs) != set(expected_outputs)
    ):
        raise ValueError
    for label, filename in expected_outputs.items():
        artifact = manifest.outputs[label]
        path = illustration / filename
        if Path(artifact.path) != path or not path.is_file():
            raise ValueError
        if name != "prepare_representatives" and (
            path.stat().st_size != artifact.size_bytes
            or sha256_file(path) != artifact.sha256
        ):
            raise ValueError


def _clean_original_planning_suffix_missing(
    *,
    episode: EpisodeState,
    root: Path,
    manifests: tuple[StageManifest | None, ...],
) -> bool:
    if episode.status not in {
        "script_approved",
        "voice_profile_approved",
        "illustration_planning",
    }:
        return False
    illustration = root / "media" / "illustration"
    missing_started = False
    try:
        _validate_tree(root, illustration)
        for name, manifest in zip(
            _PLANNING_STAGE_OUTPUTS, manifests, strict=True
        ):
            if manifest is None:
                if name in episode.completed_stages or name in episode.stale_stages:
                    raise ValueError
                missing_started = True
                continue
            if missing_started:
                raise ValueError
            _validate_planning_stage_binding(
                episode=episode,
                illustration=illustration,
                name=name,
                manifest=manifest,
                expected_outputs=_PLANNING_STAGE_OUTPUTS[name],
            )
        return missing_started
    except IllustrationRestyleError:
        raise
    except (OSError, TypeError, ValueError):
        raise IllustrationRestyleError("illustration_provenance_invalid") from None


def _validated_current_sequence_mode(
    *,
    episode: EpisodeState,
    root: Path,
) -> str | None:
    error_code = "illustration_provenance_invalid"
    illustration = root / "media" / "illustration"
    storyboard_path = illustration / "illustration_storyboard.json"
    manifest_path = illustration / "illustration_manifest.json"
    storyboard_present = storyboard_path.exists() or os.path.lexists(storyboard_path)
    manifest_present = manifest_path.exists() or os.path.lexists(manifest_path)
    if not storyboard_present and not manifest_present:
        return None
    if not storyboard_present or not manifest_present:
        raise IllustrationRestyleError(error_code)
    try:
        _validate_tree(root, illustration)
        storyboard = IllustrationStoryboard.model_validate_json(
            storyboard_path.read_text(encoding="utf-8")
        )
        manifest = IllustrationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            storyboard.book_id != episode.book_id
            or storyboard.episode_id != episode.episode_id
            or manifest.book_id != episode.book_id
            or manifest.episode_id != episode.episode_id
            or manifest.episode_root != root.absolute()
            or manifest.storyboard_sha256 != canonical_model_sha256(storyboard)
        ):
            raise ValueError
        phases = {job.phase for job in manifest.jobs}
        if storyboard.sequence_mode == "color-story-pair":
            if phases != {"anchor", "continuation"}:
                raise ValueError
        elif phases != {"legacy"}:
            raise ValueError
        profile_path = root / "production_profile.json"
        if profile_path.exists() or os.path.lexists(profile_path):
            profile = load_production_profile(root)
            if profile.visual.sequence_mode != storyboard.sequence_mode:
                raise ValueError
        return storyboard.sequence_mode
    except IllustrationRestyleError:
        raise
    except (OSError, TypeError, ValueError, ProductionProfileError):
        raise IllustrationRestyleError(error_code) from None


def _validate_original_plan_provenance(
    *,
    episode: EpisodeState,
    root: Path,
    catalog_path: Path | None,
    manifests: tuple[StageManifest | None, ...],
) -> None:
    error_code = "illustration_provenance_invalid"
    if catalog_path is None:
        raise IllustrationRestyleError("illustration_catalog_not_configured")
    if any(manifest is None for manifest in manifests):
        raise IllustrationRestyleError(error_code)
    _validate_planning_stage_bindings(
        episode=episode,
        root=root,
        manifests=manifests,
        error_code=error_code,
    )
    illustration = root / "media" / "illustration"
    try:
        decision = StyleDecision.model_validate_json(
            (illustration / "style_decision.json").read_text(encoding="utf-8")
        )
        bible = load_character_bible(
            json.loads(
                (illustration / "character_bible.json").read_text(encoding="utf-8")
            )
        )
        storyboard = IllustrationStoryboard.model_validate_json(
            (illustration / "illustration_storyboard.json").read_text(encoding="utf-8")
        )
        manifest = IllustrationManifest.model_validate_json(
            (illustration / "illustration_manifest.json").read_text(encoding="utf-8")
        )
        catalog = load_style_catalog(Path(catalog_path))
        style = next(item for item in catalog if item.style_id == decision.selected_style)
        expected_manifest = prepare_image_jobs(storyboard, bible, style, decision, root)
        prepare_stage = manifests[2]
        assert prepare_stage is not None
        replacement_authorized_asset_ids: frozenset[str] = frozenset()
        if "illustration_replacement" in episode.stage_manifests:
            from .replacement import (
                IllustrationReplacementError,
                validated_replacement_authorized_asset_ids,
            )

            try:
                replacement_authorized_asset_ids = (
                    validated_replacement_authorized_asset_ids(
                        episode_root=root,
                        state=episode,
                        manifest=manifest,
                    )
                )
            except IllustrationReplacementError as error:
                raise IllustrationRestyleError(error.error_code) from None
        if (
            storyboard.sequence_mode != "color-story-pair"
            or bible.source_script_sha256 != decision.source_script_sha256
            or bible.style_fingerprint != decision.style_fingerprint
            or any(
                character.style_fingerprint != decision.style_fingerprint
                or character.source_script_sha256 != decision.source_script_sha256
                for character in bible.characters
            )
            or storyboard.script_sha256 != decision.source_script_sha256
            or storyboard.style_decision_sha256 != canonical_model_sha256(decision)
            or storyboard.character_lock_sha256 != canonical_model_sha256(bible)
            or storyboard.character_bible_sha256 != canonical_model_sha256(bible)
            or manifest.book_id != episode.book_id
            or manifest.episode_id != episode.episode_id
            or manifest.episode_root != root.absolute()
            or manifest.storyboard_sha256 != canonical_model_sha256(storyboard)
            or manifest.style_fingerprint != decision.style_fingerprint
            or manifest.character_lock_sha256 != canonical_model_sha256(bible)
            or len(manifest.jobs) != len(storyboard.scenes) * 2
            or not 6 <= len(manifest.jobs) <= 96
            or prepare_stage.inputs.get("storyboard_sha256")
            != manifest.storyboard_sha256
            or prepare_stage.inputs.get("style_fingerprint")
            != manifest.style_fingerprint
            or prepare_stage.inputs.get("character_lock_sha256")
            != manifest.character_lock_sha256
            or restyle_plan_sha256(storyboard, manifest)
            != restyle_plan_sha256(storyboard, expected_manifest)
        ):
            raise ValueError
        _validate_pair_manifest_lifecycle(
            episode,
            root,
            storyboard,
            manifest,
            replacement_authorized_asset_ids=replacement_authorized_asset_ids,
        )
    except IllustrationRestyleError:
        raise
    except Exception:
        raise IllustrationRestyleError(error_code) from None


def validate_restyle_provenance(
    *,
    episode: EpisodeState,
    episode_root: Path,
    catalog_path: Path,
    expected_override_sha256: str,
    expected_requested_style_id: str,
    expected_style_fingerprint: str,
    expected_catalog_sha256: str,
    expected_plan_sha256: str,
) -> RestyleProvenance:
    """Validate the durable explicit-style chain without trusting override claims."""
    root = Path(episode_root)
    override = load_restyle_override(
        root,
        required=True,
        expected_sha256=expected_override_sha256,
    )
    assert override is not None
    illustration = root / "media" / "illustration"
    try:
        _validate_tree(root, illustration)
        decision = StyleDecision.model_validate_json(
            (illustration / "style_decision.json").read_text(encoding="utf-8")
        )
        bible = load_character_bible(
            json.loads(
                (illustration / "character_bible.json").read_text(encoding="utf-8")
            )
        )
        storyboard = IllustrationStoryboard.model_validate_json(
            (illustration / "illustration_storyboard.json").read_text(encoding="utf-8")
        )
        manifest = IllustrationManifest.model_validate_json(
            (illustration / "illustration_manifest.json").read_text(encoding="utf-8")
        )
    except IllustrationRestyleError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        raise IllustrationRestyleError("restyle_provenance_invalid") from None
    try:
        catalog = load_style_catalog(Path(catalog_path))
        current_decision = manual_style_decision(
            catalog,
            style_id=override.requested_style_id,
            source_script_sha256=decision.source_script_sha256,
        )
        current_style = next(
            item for item in catalog if item.style_id == override.requested_style_id
        )
        live_catalog_sha256 = sha256_file(Path(catalog_path))
    except (OSError, ValueError, TypeError, StopIteration, StyleSelectionError):
        raise IllustrationRestyleError("restyle_style_not_in_catalog") from None
    if (
        override.requested_style_id != expected_requested_style_id
        or override.style_fingerprint != expected_style_fingerprint
        or override.catalog_sha256 != expected_catalog_sha256
        or decision.selected_style != override.requested_style_id
        or decision.style_fingerprint != override.style_fingerprint
        or not decision.manual_override
        or bible.source_script_sha256 != decision.source_script_sha256
        or bible.style_fingerprint != decision.style_fingerprint
        or any(
            character.style_fingerprint != decision.style_fingerprint
            or character.source_script_sha256 != decision.source_script_sha256
            for character in bible.characters
        )
        or storyboard.sequence_mode != "color-story-pair"
        or storyboard.script_sha256 != decision.source_script_sha256
        or storyboard.style_decision_sha256 != canonical_model_sha256(decision)
        or storyboard.character_lock_sha256 != canonical_model_sha256(bible)
        or storyboard.character_bible_sha256 != canonical_model_sha256(bible)
        or manifest.storyboard_sha256 != canonical_model_sha256(storyboard)
        or manifest.style_fingerprint != decision.style_fingerprint
        or manifest.character_lock_sha256 != canonical_model_sha256(bible)
        or manifest.book_id != episode.book_id
        or manifest.episode_id != episode.episode_id
        or manifest.episode_root != root.absolute()
        or len(manifest.jobs) != len(storyboard.scenes) * 2
        or not 6 <= len(manifest.jobs) <= 96
        or restyle_plan_sha256(storyboard, manifest) != expected_plan_sha256
    ):
        raise IllustrationRestyleError("restyle_provenance_invalid")
    try:
        replacement_authorized_asset_ids: frozenset[str] = frozenset()
        if "illustration_replacement" in episode.stage_manifests:
            from .replacement import validated_replacement_authorized_asset_ids

            replacement_authorized_asset_ids = (
                validated_replacement_authorized_asset_ids(
                    episode_root=root,
                    state=episode,
                    manifest=manifest,
                )
            )
        _validate_pair_manifest_lifecycle(
            episode,
            root,
            storyboard,
            manifest,
            replacement_authorized_asset_ids=replacement_authorized_asset_ids,
        )
    except Exception:
        raise IllustrationRestyleError("restyle_provenance_invalid") from None
    if override.catalog_sha256 == live_catalog_sha256:
        try:
            expected_manifest = prepare_image_jobs(
                storyboard,
                bible,
                current_style,
                current_decision,
                root,
            )
        except Exception:
            raise IllustrationRestyleError("restyle_provenance_invalid") from None
        if (
            decision != current_decision
            or restyle_plan_sha256(storyboard, expected_manifest)
            != expected_plan_sha256
        ):
            raise IllustrationRestyleError("restyle_provenance_invalid")
    return RestyleProvenance(
        override=override,
        override_sha256=expected_override_sha256,
        decision=decision,
        storyboard=storyboard,
    )


def _validate_pair_manifest_lifecycle(
    episode: EpisodeState,
    root: Path,
    storyboard: IllustrationStoryboard,
    manifest: IllustrationManifest,
    *,
    replacement_authorized_asset_ids: frozenset[str] = frozenset(),
) -> None:
    scene_ids = tuple(scene.scene_id for scene in storyboard.scenes)
    representative_scene_ids = tuple(
        scene.scene_id for scene in storyboard.scenes if scene.representative_frame
    )
    if not 3 <= len(scene_ids) <= 48:
        raise ValueError
    expected_representatives = tuple(
        scene.scene_id for scene in storyboard.scenes if scene.representative_frame
    )
    if len(expected_representatives) != 3:
        raise ValueError
    if representative_scene_ids != expected_representatives:
        raise ValueError
    ordered_scenes = (
        *expected_representatives,
        *(
            scene_id
            for scene_id in scene_ids
            if scene_id not in expected_representatives
        ),
    )
    expected_identities = tuple(
        (scene_id, phase, f"{scene_id}-{suffix}", scene_id in expected_representatives)
        for scene_id in ordered_scenes
        for phase, suffix in (("anchor", "A"), ("continuation", "B"))
    )
    identities = tuple(
        (job.scene_id, job.phase, job.asset_id, job.representative)
        for job in manifest.jobs
    )
    if identities != expected_identities or len(set(identities)) != len(identities):
        raise ValueError

    jobs_by_identity = {(job.scene_id, job.phase): job for job in manifest.jobs}
    batch_bound = episode.status in {
        "illustration_batch_running",
        "illustrations_ready",
        "awaiting_final_review",
        "final_approved",
    }
    for job in manifest.jobs:
        if job.anchor_sha256 is not None and job.phase != "continuation":
            raise ValueError
        expected_prompt_sha256 = image_prompt_identity_sha256(
            prompt=job.prompt,
            style_fingerprint=job.style_fingerprint,
            character_lock_sha256=job.character_lock_sha256,
            reference_image_sha256s=job.reference_image_sha256s,
        )
        if job.prompt_sha256 != expected_prompt_sha256:
            raise ValueError
        expected_references: list[str] = []
        references_ready = True
        for reference_scene_id in job.reference_scene_ids:
            phase = (
                "anchor"
                if reference_scene_id != job.scene_id or job.phase == "continuation"
                else job.phase
            )
            reference = jobs_by_identity.get((reference_scene_id, phase))
            if (
                reference is None
                or reference.status != "generated"
                or reference.master_sha256 is None
            ):
                references_ready = False
                break
            expected_references.append(reference.master_sha256)
        if job.phase == "continuation":
            anchor = jobs_by_identity.get((job.scene_id, "anchor"))
            anchor_ready = (
                anchor is not None
                and anchor.status == "generated"
                and anchor.master_sha256 is not None
            )
            if anchor_ready:
                replacement_pending = (
                    job.status == "planned"
                    and job.asset_id in replacement_authorized_asset_ids
                    and job.anchor_sha256 is None
                    and not job.reference_image_sha256s
                )
                if not replacement_pending and (
                    not references_ready
                    or tuple(expected_references) != job.reference_image_sha256s
                    or job.anchor_sha256 != anchor.master_sha256
                ):
                    raise ValueError
            elif job.anchor_sha256 is not None or job.reference_image_sha256s:
                raise ValueError
        elif not job.representative and references_ready and batch_bound:
            if tuple(expected_references) != job.reference_image_sha256s:
                raise ValueError
        elif job.reference_image_sha256s and (
            not references_ready
            or tuple(expected_references) != job.reference_image_sha256s
        ):
            raise ValueError

        if job.status == "planned":
            if (
                job.master_sha256 is not None
                or job.bw_sha256 is not None
                or (
                    job.asset_id not in replacement_authorized_asset_ids
                    and job.attempts != 0
                )
                or job.output_master.exists()
                or (job.output_bw is not None and job.output_bw.exists())
            ):
                raise ValueError
            continue
        if job.status != "generated":
            raise ValueError
        if job.attempts < 1:
            raise ValueError
        validate_generated_image_job(job, manifest=manifest)
        master = inspect_image(job.output_master)
        if (master.width, master.height) != (1080, 1920):
            raise ValueError
        if job.output_master.stat().st_size <= 0:
            raise ValueError
        if job.output_bw is not None:
            bw = inspect_image(job.output_bw)
            if (bw.width, bw.height) != (1080, 1920):
                raise ValueError

    illustration = root / "media" / "illustration"
    review_path = illustration / "representative_review.json"
    approval_path = illustration / "representative_approval.json"
    approval_required = episode.status in {
        "representatives_approved",
        "illustration_batch_running",
        "illustrations_ready",
        "awaiting_final_review",
        "final_approved",
    }
    if not approval_required:
        if review_path.exists() or approval_path.exists():
            raise ValueError
        return
    if (
        not review_path.is_file()
        or review_path.is_symlink()
        or not approval_path.is_file()
        or approval_path.is_symlink()
    ):
        raise ValueError
    review = RepresentativeReview.model_validate_json(
        review_path.read_text(encoding="utf-8")
    )
    approval = RepresentativeApproval.model_validate_json(
        approval_path.read_text(encoding="utf-8")
    )
    current_review = build_representative_review(storyboard, manifest)
    if review != current_review or (
        approval.review_sha256 != canonical_model_sha256(review)
        or approval.image_sha256s != review.image_sha256s
        or approval.storyboard_sha256 != review.storyboard_sha256
        or approval.style_fingerprint != review.style_fingerprint
        or approval.character_lock_sha256 != review.character_lock_sha256
        or approval.asset_ids != review.asset_ids
        or approval.all_scene_ids != review.all_scene_ids
        or approval.reviewer != "user"
    ):
        raise ValueError


def _restyle_override_path(root: Path) -> Path:
    return root / "script" / "illustration_style_override.json"


def _validate_child(root: Path, candidate: Path) -> None:
    root = Path(root).absolute()
    candidate = Path(candidate).absolute()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise IllustrationRestyleError("unsafe_restyle_recovery") from None
    for current in (*_path_chain(root, candidate), *root.parents):
        _validate_existing_path(current)
    try:
        canonical_root = root.resolve(strict=True)
        canonical_candidate = candidate.resolve(strict=False)
    except OSError:
        raise IllustrationRestyleError("unsafe_restyle_recovery") from None
    if canonical_candidate == canonical_root or not canonical_candidate.is_relative_to(canonical_root):
        raise IllustrationRestyleError("unsafe_restyle_recovery")


def _path_chain(root: Path, candidate: Path) -> tuple[Path, ...]:
    chain: list[Path] = []
    current = candidate
    while True:
        chain.append(current)
        if current == root:
            return tuple(chain)
        current = current.parent


def _validate_existing_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    try:
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
        if path.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise IllustrationRestyleError("unsafe_restyle_recovery")
    except OSError:
        raise IllustrationRestyleError("unsafe_restyle_recovery") from None


def _validate_tree(root: Path, candidate: Path) -> None:
    _validate_child(root, candidate)
    if not candidate.exists():
        return
    pending = [candidate]
    while pending:
        current = pending.pop()
        _validate_existing_path(current)
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    _validate_existing_path(path)
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
        except NotADirectoryError:
            continue
        except OSError:
            raise IllustrationRestyleError("unsafe_restyle_recovery") from None


def _mkdir_transaction_parents(root: Path, destination: Path, created: list[Path]) -> None:
    _validate_child(root, destination)
    missing: list[Path] = []
    current = destination
    while not current.exists():
        missing.append(current)
        if current == root:
            raise IllustrationRestyleError("unsafe_restyle_recovery")
        current = current.parent
    for path in reversed(missing):
        if path.parent != root:
            _validate_child(root, path.parent)
        path.mkdir()
        created.append(path)


def _rollback_install_journal(
    *,
    root: Path,
    journal: list[tuple[Path, Path]],
    created_directories: list[Path],
) -> bool:
    errors: list[Exception] = []
    for source, destination in reversed(journal):
        if source.parent == root / "media":
            continue
        try:
            if destination.exists() or destination.is_symlink():
                _validate_tree(root, destination)
                _mkdir_transaction_parents(root, source.parent, created_directories)
                os.replace(destination, source)
        except Exception as error:
            errors.append(error)
    for path in reversed(created_directories):
        try:
            _validate_child(root, path)
            if path.exists():
                path.rmdir()
        except OSError:
            # A directory with data is never removed: it can belong to a concurrent writer.
            continue
        except Exception as error:
            errors.append(error)
    # An archive restore can only start after transaction-created empty targets are gone.
    for source, destination in reversed(journal):
        if source.parent != root / "media":
            continue
        try:
            if destination.exists() or destination.is_symlink():
                _validate_tree(root, destination)
                _mkdir_transaction_parents(root, source.parent, created_directories)
                os.replace(destination, source)
        except Exception as error:
            errors.append(error)
    return not errors
