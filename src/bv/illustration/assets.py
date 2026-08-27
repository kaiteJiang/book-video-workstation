from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

try:
    import pymupdf
except ModuleNotFoundError:  # pragma: no cover
    import fitz as pymupdf
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bv.core.hashing import sha256_file
from bv.core.process import CommandResult, run_command

from .contracts import (
    CharacterBible,
    CharacterLock,
    IllustrationStoryboard,
    StyleDecision,
    load_character_bible,
)
from .prompts import (
    canonical_model_sha256,
    compile_image_prompt,
    image_prompt_identity_sha256,
)
from .style_selector import StyleRecipe


_MAX_IMAGE_BYTES = 20 * 1024 * 1024


class IllustrationAssetError(RuntimeError):
    """Stable, privacy-safe illustration asset failure."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _AssetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class ImageFacts(_AssetModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: Literal["png", "jpeg"]


class ImageJob(_AssetModel):
    episode_root: Path
    scene_id: str
    representative: bool
    prompt: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    style_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    character_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_scene_ids: tuple[str, ...]
    reference_image_sha256s: tuple[str, ...]
    output_master: Path
    output_bw: Path
    master_sha256: str | None = None
    bw_sha256: str | None = None
    attempts: int = Field(default=0, ge=0)
    status: Literal["planned", "generated", "approved"] = "planned"


class IllustrationManifest(_AssetModel):
    episode_root: Path
    book_id: str
    episode_id: str
    storyboard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    style_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    character_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    jobs: tuple[ImageJob, ...]


class ImportedImage(_AssetModel):
    scene_id: str
    master_path: Path
    master_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bw_path: Path
    bw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: Literal[1080]
    height: Literal[1920]


def prepare_image_jobs(
    storyboard: IllustrationStoryboard,
    character_lock: CharacterLock | CharacterBible,
    style: StyleRecipe,
    style_decision: StyleDecision,
    episode_root: Path,
) -> IllustrationManifest:
    root = _canonical_safe_root(Path(episode_root))
    character_bible = load_character_bible(character_lock)
    character_sha = canonical_model_sha256(character_lock)
    style_sha = canonical_model_sha256(style_decision)
    if (
        storyboard.character_lock_sha256 != character_sha
        or storyboard.character_bible_sha256 not in {None, character_sha}
        or storyboard.style_decision_sha256 != style_sha
        or character_bible.style_fingerprint != style_decision.style_fingerprint
        or style.style_id != style_decision.selected_style
        or storyboard.script_sha256 != character_bible.source_script_sha256
        or storyboard.script_sha256 != style_decision.source_script_sha256
    ):
        raise IllustrationAssetError("illustration_dependency_mismatch")

    opening_id = storyboard.scenes[0].scene_id
    representative_ids = {
        scene.scene_id for scene in storyboard.scenes if scene.representative_frame
    }
    character_anchor: dict[str, str] = {}
    for character in character_bible.characters:
        matching = tuple(
            scene
            for scene in storyboard.scenes
            if character.character_id in scene.character_refs
        )
        representative_matching = tuple(
            scene for scene in matching if scene.scene_id in representative_ids
        )
        candidates = representative_matching or matching
        if candidates:
            character_anchor[character.character_id] = candidates[0].scene_id
    compiled: list[ImageJob] = []
    for scene in storyboard.scenes:
        references = tuple(
            dict.fromkeys(
                character_anchor[character_id]
                for character_id in scene.character_refs
                if character_id in character_anchor
                and character_anchor[character_id] != scene.scene_id
            )
        )
        reference_scene_ids = references
        if (
            not reference_scene_ids
            and not scene.character_refs
            and scene.representative_frame
            and scene.scene_id != opening_id
        ):
            reference_scene_ids = (opening_id,)
        prompt = compile_image_prompt(
            scene,
            style=style,
            character_lock=character_lock,
            reference_scene_ids=reference_scene_ids,
        )
        image_root = root / "media" / "illustration" / "images"
        compiled.append(
            ImageJob(
                episode_root=root,
                scene_id=scene.scene_id,
                representative=scene.representative_frame,
                prompt=prompt.prompt,
                prompt_sha256=prompt.prompt_sha256,
                style_fingerprint=prompt.style_fingerprint,
                character_lock_sha256=prompt.character_lock_sha256,
                reference_scene_ids=reference_scene_ids,
                reference_image_sha256s=(),
                output_master=image_root / f"{scene.scene_id}_master.png",
                output_bw=image_root / f"{scene.scene_id}_bw.png",
            )
        )
    ordered = tuple(
        sorted(
            compiled,
            key=lambda job: (not job.representative, _scene_number(job.scene_id)),
        )
    )
    return IllustrationManifest(
        episode_root=root,
        book_id=storyboard.book_id,
        episode_id=storyboard.episode_id,
        storyboard_sha256=canonical_model_sha256(storyboard),
        style_fingerprint=style_decision.style_fingerprint,
        character_lock_sha256=character_sha,
        jobs=ordered,
    )


def import_image_master(
    job: ImageJob,
    source_path: Path,
    *,
    inspector: Callable[[Path], ImageFacts] | None = None,
    ffmpeg_command: str | Path,
    runner: Callable[..., CommandResult] = run_command,
) -> ImportedImage:
    try:
        job = ImageJob.model_validate(job)
    except ValidationError:
        raise IllustrationAssetError("invalid_image_job") from None
    _validate_job_paths(job)
    source = Path(source_path)
    _require_safe_source(source)
    if job.output_master.exists() or job.output_bw.exists():
        raise IllustrationAssetError("image_output_exists")
    _ensure_safe_directory(job.output_master.parent)
    snapshot: Path | None = None
    master_temp = job.output_master.parent / f".{job.scene_id}.{uuid.uuid4().hex}.master.png"
    bw_temp = job.output_bw.parent / f".{job.scene_id}.{uuid.uuid4().hex}.bw.png"
    published: list[tuple[Path, str]] = []
    inspect = inspector or inspect_image
    try:
        snapshot, source_hash = _snapshot_source(source, job.output_master.parent)
        try:
            source_facts = inspect(snapshot)
        except Exception:
            raise IllustrationAssetError("image_contract_invalid") from None
        if (
            not isinstance(source_facts, ImageFacts)
            or (source_facts.width, source_facts.height) != (1080, 1920)
            or source_facts.format not in {"png", "jpeg"}
        ):
            raise IllustrationAssetError("image_contract_invalid")
        if sha256_file(snapshot) != source_hash or sha256_file(source) != source_hash:
            raise IllustrationAssetError("image_source_changed")

        _run_ffmpeg(
            runner,
            [
                str(ffmpeg_command), "-hide_banner", "-loglevel", "error", "-nostdin",
                "-i", str(snapshot), "-map_metadata", "-1", "-frames:v", "1",
                "-vf", "format=rgb24", "-compression_level", "6", "-y", str(master_temp),
            ],
        )
        _validate_derived(master_temp, inspect, expected_format="png")
        _run_ffmpeg(
            runner,
            [
                str(ffmpeg_command), "-hide_banner", "-loglevel", "error", "-nostdin",
                "-i", str(master_temp), "-map_metadata", "-1", "-frames:v", "1",
                "-vf", "format=gray", "-compression_level", "6", "-y", str(bw_temp),
            ],
        )
        _validate_derived(bw_temp, inspect, expected_format="png")
        if sha256_file(source) != source_hash or _redirect_in_existing_chain(source):
            raise IllustrationAssetError("image_source_changed")

        master_hash = sha256_file(master_temp)
        bw_hash = sha256_file(bw_temp)
        if master_hash == bw_hash:
            raise IllustrationAssetError("grayscale_derivation_failed")
        _publish_new(master_temp, job.output_master, master_hash)
        published.append((job.output_master, master_hash))
        _publish_new(bw_temp, job.output_bw, bw_hash)
        published.append((job.output_bw, bw_hash))
        return ImportedImage(
            scene_id=job.scene_id,
            master_path=job.output_master,
            master_sha256=master_hash,
            bw_path=job.output_bw,
            bw_sha256=bw_hash,
            width=1080,
            height=1920,
        )
    except IllustrationAssetError:
        for path, expected_hash in reversed(published):
            _remove_if_hash(path, expected_hash)
        raise
    except Exception:
        for path, expected_hash in reversed(published):
            _remove_if_hash(path, expected_hash)
        raise IllustrationAssetError("image_import_failed") from None
    finally:
        _remove_owned(snapshot)
        _remove_owned(master_temp)
        _remove_owned(bw_temp)


def record_imported_image(
    manifest: IllustrationManifest,
    imported: ImportedImage,
) -> IllustrationManifest:
    try:
        manifest = IllustrationManifest.model_validate(manifest)
        imported = ImportedImage.model_validate(imported)
    except ValidationError:
        raise IllustrationAssetError("invalid_imported_image") from None
    matches = [job for job in manifest.jobs if job.scene_id == imported.scene_id]
    if len(matches) != 1:
        raise IllustrationAssetError("image_job_not_found")
    target = matches[0]
    _validate_job_paths(target)
    if (
        imported.master_path != target.output_master
        or imported.bw_path != target.output_bw
        or not target.output_master.is_file()
        or not target.output_bw.is_file()
        or _redirect_in_existing_chain(target.output_master)
        or _redirect_in_existing_chain(target.output_bw)
        or sha256_file(target.output_master) != imported.master_sha256
        or sha256_file(target.output_bw) != imported.bw_sha256
    ):
        raise IllustrationAssetError("imported_image_hash_mismatch")
    jobs = tuple(
        job.model_copy(
            update={
                "master_sha256": imported.master_sha256,
                "bw_sha256": imported.bw_sha256,
                "status": "generated",
                "attempts": job.attempts + 1,
            }
        )
        if job.scene_id == imported.scene_id
        else job
        for job in manifest.jobs
    )
    return manifest.model_copy(update={"jobs": jobs})


def bind_job_references(
    job: ImageJob,
    manifest: IllustrationManifest,
) -> ImageJob:
    try:
        job = ImageJob.model_validate(job)
        manifest = IllustrationManifest.model_validate(manifest)
    except ValidationError:
        raise IllustrationAssetError("invalid_image_job") from None
    if (
        job.episode_root != manifest.episode_root
        or job.style_fingerprint != manifest.style_fingerprint
        or job.character_lock_sha256 != manifest.character_lock_sha256
    ):
        raise IllustrationAssetError("illustration_dependency_mismatch")
    _validate_job_paths(job)
    hashes: list[str] = []
    for scene_id in job.reference_scene_ids:
        matches = [item for item in manifest.jobs if item.scene_id == scene_id]
        if len(matches) != 1:
            raise IllustrationAssetError("reference_image_not_ready")
        reference = matches[0]
        try:
            validate_generated_image_job(reference)
        except IllustrationAssetError:
            raise IllustrationAssetError("reference_image_not_ready") from None
        hashes.append(reference.master_sha256)
    bound_hashes = tuple(hashes)
    return job.model_copy(
        update={
            "reference_image_sha256s": bound_hashes,
            "prompt_sha256": image_prompt_identity_sha256(
                prompt=job.prompt,
                style_fingerprint=job.style_fingerprint,
                character_lock_sha256=job.character_lock_sha256,
                reference_image_sha256s=bound_hashes,
            ),
        }
    )


def validate_generated_image_job(job: ImageJob) -> None:
    try:
        job = ImageJob.model_validate(job)
    except ValidationError:
        raise IllustrationAssetError("invalid_image_job") from None
    _validate_job_paths(job)
    if (
        job.status not in {"generated", "approved"}
        or job.master_sha256 is None
        or job.bw_sha256 is None
        or not job.output_master.is_file()
        or not job.output_bw.is_file()
        or _redirect_in_existing_chain(job.output_master)
        or _redirect_in_existing_chain(job.output_bw)
        or sha256_file(job.output_master) != job.master_sha256
        or sha256_file(job.output_bw) != job.bw_sha256
    ):
        raise IllustrationAssetError("recorded_image_invalid")


def inspect_image(path: Path) -> ImageFacts:
    source = Path(path)
    _require_safe_source(source, allow_temporary=True)
    try:
        if source.stat().st_size > _MAX_IMAGE_BYTES:
            raise ValueError
        raw = source.read_bytes()
        image_format: Literal["png", "jpeg"]
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            image_format = "png"
        elif raw.startswith(b"\xff\xd8"):
            image_format = "jpeg"
        else:
            raise ValueError
        document = pymupdf.open(stream=raw, filetype=image_format)
        try:
            if document.page_count != 1:
                raise ValueError
            images = document[0].get_image_info(xrefs=True)
            if len(images) != 1:
                raise ValueError
            width = images[0].get("width")
            height = images[0].get("height")
            if not isinstance(width, int) or not isinstance(height, int):
                raise ValueError
        finally:
            document.close()
        return ImageFacts(width=width, height=height, format=image_format)
    except (OSError, ValueError, ValidationError):
        raise IllustrationAssetError("image_contract_invalid") from None


def _validate_job_paths(job: ImageJob) -> None:
    root = _canonical_safe_root(job.episode_root)
    expected_root = root / "media" / "illustration" / "images"
    expected_master = expected_root / f"{job.scene_id}_master.png"
    expected_bw = expected_root / f"{job.scene_id}_bw.png"
    if (
        Path(os.path.abspath(job.output_master)) != expected_master
        or Path(os.path.abspath(job.output_bw)) != expected_bw
        or not expected_master.is_relative_to(root)
        or not expected_bw.is_relative_to(root)
        or _redirect_in_existing_chain(expected_master)
        or _redirect_in_existing_chain(expected_bw)
    ):
        raise IllustrationAssetError("unsafe_image_path")


def _snapshot_source(source: Path, output_dir: Path) -> tuple[Path, str]:
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=".image-source-",
            suffix=source.suffix.lower(),
            dir=output_dir,
        )
        temporary = Path(name)
        digest = hashlib.sha256()
        total = 0
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            for chunk in iter(lambda: input_stream.read(1_048_576), b""):
                total += len(chunk)
                if total > _MAX_IMAGE_BYTES:
                    raise IllustrationAssetError("image_source_too_large")
                digest.update(chunk)
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        value = digest.hexdigest()
        if sha256_file(temporary) != value or sha256_file(source) != value:
            raise IllustrationAssetError("image_source_changed")
        return temporary, value
    except IllustrationAssetError:
        _remove_owned(temporary)
        raise
    except OSError:
        _remove_owned(temporary)
        raise IllustrationAssetError("image_snapshot_failed") from None


def _run_ffmpeg(runner: Callable[..., CommandResult], argv: list[str]) -> None:
    try:
        result = runner(argv, timeout=120)
    except Exception:
        raise IllustrationAssetError("image_conversion_failed") from None
    if not isinstance(result, CommandResult) or result.returncode != 0:
        raise IllustrationAssetError("image_conversion_failed")


def _validate_derived(
    path: Path,
    inspector: Callable[[Path], ImageFacts],
    *,
    expected_format: Literal["png", "jpeg"],
) -> None:
    if not path.is_file() or _redirect_in_existing_chain(path):
        raise IllustrationAssetError("image_conversion_failed")
    try:
        facts = inspector(path)
    except Exception:
        raise IllustrationAssetError("image_conversion_failed") from None
    if (
        not isinstance(facts, ImageFacts)
        or (facts.width, facts.height) != (1080, 1920)
        or facts.format != expected_format
    ):
        raise IllustrationAssetError("image_conversion_failed")


def _publish_new(source: Path, target: Path, expected_hash: str) -> None:
    if target.exists() or _redirect_in_existing_chain(target):
        raise IllustrationAssetError("image_output_exists")
    linked = False
    try:
        os.link(source, target)
        linked = True
        _fsync_file(target)
        if sha256_file(target) != expected_hash:
            raise OSError
    except FileExistsError:
        raise IllustrationAssetError("image_output_exists") from None
    except OSError:
        if linked:
            _remove_if_hash(target, expected_hash)
        raise IllustrationAssetError("image_publish_failed") from None


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _require_safe_source(path: Path, *, allow_temporary: bool = False) -> None:
    allowed = {".png", ".jpg", ".jpeg"}
    if path.suffix.lower() not in allowed or not path.is_file() or _redirect_in_existing_chain(path):
        raise IllustrationAssetError("unsafe_image_source")
    if not allow_temporary and path.stat().st_size > _MAX_IMAGE_BYTES:
        raise IllustrationAssetError("image_source_too_large")


def _canonical_safe_root(path: Path) -> Path:
    root = Path(os.path.abspath(path))
    if _redirect_in_existing_chain(root):
        raise IllustrationAssetError("unsafe_image_path")
    return root


def _ensure_safe_directory(path: Path) -> None:
    if _redirect_in_existing_chain(path):
        raise IllustrationAssetError("unsafe_image_path")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise IllustrationAssetError("unsafe_image_path") from None
    if not path.is_dir() or _redirect_in_existing_chain(path):
        raise IllustrationAssetError("unsafe_image_path")


def _redirect_in_existing_chain(path: Path) -> bool:
    current = Path(path)
    while True:
        try:
            if current.is_symlink():
                return True
            if current.exists():
                attributes = getattr(current.stat(follow_symlinks=False), "st_file_attributes", 0)
                if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                    return True
        except OSError:
            return True
        if current.parent == current:
            return False
        current = current.parent


def _remove_owned(path: Path | None) -> None:
    if path is None or _redirect_in_existing_chain(path):
        return
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _remove_if_hash(path: Path, expected_hash: str) -> None:
    try:
        if path.is_file() and not _redirect_in_existing_chain(path) and sha256_file(path) == expected_hash:
            path.unlink()
    except OSError:
        pass


def _scene_number(scene_id: str) -> int:
    try:
        return int(scene_id.removeprefix("S"))
    except ValueError:
        return 999_999
