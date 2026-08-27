from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bv.core.hashing import sha256_file
from bv.core.process import CommandResult, run_command
from bv.illustration.assets import ImageFacts, inspect_image
from bv.video.cover_brief import CoverBrief


class SocialCoverError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _CoverModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SocialCoverRequest(_CoverModel):
    episode_root: Path
    cover_brief_path: Path
    source_image_path: Path
    source_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SocialCoverManifest(_CoverModel):
    schema_version: int = 1
    title: str
    style_id: str
    width: Literal[1080] = 1080
    height: Literal[1440] = 1440
    approved_script_sha256: str
    semantic_lock_sha256: str
    production_profile_sha256: str
    style_meta_sha256: str
    style_atom_sha256: str
    blueprint_sha256: str
    prompt_sha256: str
    cover_brief_path: Path
    cover_brief_sha256: str
    source_image_path: Path
    source_image_sha256: str
    source_width: int
    source_height: int
    normalization_args: tuple[str, ...]
    output_path: Path
    output_sha256: str
    manifest_path: Path
    title_accuracy_review: str = "pending"
    visual_meaning_review: str = "pending"

class SocialCoverRenderer:
    def __init__(
        self,
        *,
        ffmpeg_command: str | Path,
        runner: Callable[..., CommandResult] = run_command,
        inspector: Callable[[Path], ImageFacts] = inspect_image,
    ) -> None:
        self.ffmpeg_command = str(ffmpeg_command)
        self.runner = runner
        self.inspector = inspector

    def normalize(self, request: SocialCoverRequest) -> SocialCoverManifest:
        try:
            request = SocialCoverRequest.model_validate(request)
        except ValidationError:
            raise SocialCoverError("cover_request_invalid") from None
        root = Path(request.episode_root).absolute()
        brief_path = Path(request.cover_brief_path).absolute()
        source = Path(request.source_image_path).absolute()
        output = root / "media" / "final" / "social_cover.png"
        manifest_path = output.with_suffix(".json")
        if not brief_path.is_relative_to(root):
            raise SocialCoverError("unsafe_cover_path")
        _require_safe_regular(brief_path)
        _require_safe_regular(source)
        _require_safe_destination(root, output)
        _require_safe_destination(root, manifest_path)
        if output.exists() or output.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
            raise SocialCoverError("cover_output_exists")

        try:
            brief_bytes = brief_path.read_bytes()
            brief_sha256 = hashlib.sha256(brief_bytes).hexdigest()
            brief = CoverBrief.model_validate_json(brief_bytes)
        except (OSError, ValueError, ValidationError):
            raise SocialCoverError("cover_brief_invalid") from None
        prompt_path = Path(brief.prompt_path).absolute()
        if (
            Path(brief.manifest_path).absolute() != brief_path
            or not prompt_path.is_relative_to(root)
        ):
            raise SocialCoverError("cover_brief_invalid")
        _require_safe_regular(prompt_path)
        if sha256_file(prompt_path) != brief.prompt_sha256:
            raise SocialCoverError("cover_prompt_stale")
        dependency_paths = (
            (Path(brief.approved_script_path).absolute(), brief.approved_script_sha256, True),
            (Path(brief.semantic_lock_path).absolute(), brief.semantic_lock_sha256, True),
            (Path(brief.style_meta_path).absolute(), brief.style_meta_sha256, False),
            (Path(brief.style_atom_path).absolute(), brief.style_atom_sha256, False),
            (Path(brief.blueprint_path).absolute(), brief.blueprint_sha256, False),
        )
        for dependency, expected_hash, must_be_internal in dependency_paths:
            if must_be_internal and not dependency.is_relative_to(root):
                raise SocialCoverError("cover_brief_invalid")
            _require_safe_regular(dependency)
            if sha256_file(dependency) != expected_hash:
                raise SocialCoverError("cover_dependency_stale")
        if sha256_file(source) != request.source_image_sha256:
            raise SocialCoverError("cover_source_changed")
        try:
            source_facts = self.inspector(source)
        except Exception:
            raise SocialCoverError("cover_source_invalid") from None
        if (
            not isinstance(source_facts, ImageFacts)
            or source_facts.format not in {"png", "jpeg"}
        ):
            raise SocialCoverError("cover_source_invalid")
        if source_facts.width * 4 != source_facts.height * 3:
            raise SocialCoverError("cover_aspect_ratio_invalid")

        output.parent.mkdir(parents=True, exist_ok=True)
        _require_safe_destination(root, output)
        _require_safe_destination(root, manifest_path)
        snapshot = output.parent / f".social-cover-source-{uuid.uuid4().hex}{source.suffix.lower()}"
        rendered = output.parent / f".social-cover-{uuid.uuid4().hex}.png"
        published_hash: str | None = None
        args = (
            "scale=1080:1440:flags=lanczos",
            "format=rgb24",
            "strip_metadata",
        )
        try:
            _snapshot_exact(source, snapshot, request.source_image_sha256)
            result = self.runner(
                [
                    self.ffmpeg_command,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-i",
                    str(snapshot),
                    "-map_metadata",
                    "-1",
                    "-frames:v",
                    "1",
                    "-vf",
                    ",".join(args[:2]),
                    "-compression_level",
                    "6",
                    "-y",
                    str(rendered),
                ],
                timeout=120,
            )
            if not isinstance(result, CommandResult) or result.returncode != 0:
                raise SocialCoverError("cover_normalization_failed")
            _validate_rendered(rendered, self.inspector)
            if sha256_file(source) != request.source_image_sha256:
                raise SocialCoverError("cover_source_changed")
            if sha256_file(brief_path) != brief_sha256:
                raise SocialCoverError("cover_brief_changed")
            published_hash = sha256_file(rendered)
            _publish_new(rendered, output, published_hash)
            manifest = SocialCoverManifest(
                title=brief.title,
                style_id=brief.style_id,
                approved_script_sha256=brief.approved_script_sha256,
                semantic_lock_sha256=brief.semantic_lock_sha256,
                production_profile_sha256=brief.production_profile_sha256,
                style_meta_sha256=brief.style_meta_sha256,
                style_atom_sha256=brief.style_atom_sha256,
                blueprint_sha256=brief.blueprint_sha256,
                prompt_sha256=brief.prompt_sha256,
                cover_brief_path=brief_path,
                cover_brief_sha256=brief_sha256,
                source_image_path=source,
                source_image_sha256=request.source_image_sha256,
                source_width=source_facts.width,
                source_height=source_facts.height,
                normalization_args=args,
                output_path=output,
                output_sha256=published_hash,
                manifest_path=manifest_path,
            )
            _publish_json_new(manifest_path, manifest.model_dump(mode="json"))
            return manifest
        except SocialCoverError:
            if published_hash is not None:
                _remove_if_hash(output, published_hash)
            raise
        except Exception:
            if published_hash is not None:
                _remove_if_hash(output, published_hash)
            raise SocialCoverError("cover_normalization_failed") from None
        finally:
            _remove_owned(snapshot)
            _remove_owned(rendered)


def validate_social_cover_manifest_current(
    manifest: SocialCoverManifest,
    *,
    episode_root: Path,
) -> None:
    try:
        manifest = SocialCoverManifest.model_validate(manifest)
    except ValidationError:
        raise SocialCoverError("cover_manifest_invalid") from None
    root = Path(episode_root).absolute()
    expected_output = root / "media" / "final" / "social_cover.png"
    expected_manifest = expected_output.with_suffix(".json")
    if (
        Path(manifest.output_path).absolute() != expected_output
        or Path(manifest.manifest_path).absolute() != expected_manifest
    ):
        raise SocialCoverError("cover_manifest_invalid")
    _require_safe_regular(expected_output)
    if sha256_file(expected_output) != manifest.output_sha256:
        raise SocialCoverError("cover_output_stale")

    brief_path = Path(manifest.cover_brief_path).absolute()
    if not brief_path.is_relative_to(root):
        raise SocialCoverError("cover_manifest_invalid")
    _require_safe_regular(brief_path)
    if sha256_file(brief_path) != manifest.cover_brief_sha256:
        raise SocialCoverError("cover_brief_changed")
    try:
        brief = CoverBrief.model_validate_json(brief_path.read_bytes())
    except (OSError, ValueError, ValidationError):
        raise SocialCoverError("cover_brief_invalid") from None
    shared = (
        (manifest.title, brief.title),
        (manifest.style_id, brief.style_id),
        (manifest.approved_script_sha256, brief.approved_script_sha256),
        (manifest.semantic_lock_sha256, brief.semantic_lock_sha256),
        (manifest.production_profile_sha256, brief.production_profile_sha256),
        (manifest.style_meta_sha256, brief.style_meta_sha256),
        (manifest.style_atom_sha256, brief.style_atom_sha256),
        (manifest.blueprint_sha256, brief.blueprint_sha256),
        (manifest.prompt_sha256, brief.prompt_sha256),
    )
    if any(recorded != current for recorded, current in shared):
        raise SocialCoverError("cover_manifest_invalid")
    prompt_path = Path(brief.prompt_path).absolute()
    _require_safe_regular(prompt_path)
    if not prompt_path.is_relative_to(root) or sha256_file(prompt_path) != brief.prompt_sha256:
        raise SocialCoverError("cover_prompt_stale")
    for dependency, expected_hash in (
        (Path(brief.approved_script_path).absolute(), brief.approved_script_sha256),
        (Path(brief.semantic_lock_path).absolute(), brief.semantic_lock_sha256),
        (Path(brief.style_meta_path).absolute(), brief.style_meta_sha256),
        (Path(brief.style_atom_path).absolute(), brief.style_atom_sha256),
        (Path(brief.blueprint_path).absolute(), brief.blueprint_sha256),
        (Path(manifest.source_image_path).absolute(), manifest.source_image_sha256),
    ):
        _require_safe_regular(dependency)
        if sha256_file(dependency) != expected_hash:
            raise SocialCoverError("cover_dependency_stale")


def _require_safe_regular(path: Path) -> None:
    try:
        if _redirect_in_existing_chain(path):
            raise SocialCoverError("unsafe_cover_path")
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise SocialCoverError("unsafe_cover_path")
    except SocialCoverError:
        raise
    except OSError:
        raise SocialCoverError("unsafe_cover_path") from None


def _require_safe_destination(root: Path, path: Path) -> None:
    if not path.absolute().is_relative_to(root) or _redirect_in_existing_chain(path):
        raise SocialCoverError("unsafe_cover_path")


def _redirect_in_existing_chain(path: Path) -> bool:
    candidate = Path(path)
    while True:
        if candidate.exists() or candidate.is_symlink():
            attributes = getattr(candidate.stat(follow_symlinks=False), "st_file_attributes", 0)
            if candidate.is_symlink() or attributes & 0x400:
                return True
        if candidate.parent == candidate:
            return False
        candidate = candidate.parent


def _snapshot_exact(source: Path, target: Path, expected_hash: str) -> None:
    try:
        digest = hashlib.sha256()
        with source.open("rb") as input_stream, target.open("xb") as output_stream:
            for chunk in iter(lambda: input_stream.read(1_048_576), b""):
                digest.update(chunk)
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if digest.hexdigest() != expected_hash or sha256_file(target) != expected_hash:
            raise SocialCoverError("cover_source_changed")
    except FileExistsError:
        raise SocialCoverError("cover_normalization_failed") from None
    except SocialCoverError:
        raise
    except OSError:
        raise SocialCoverError("cover_source_changed") from None


def _validate_rendered(path: Path, inspector: Callable[[Path], ImageFacts]) -> None:
    _require_safe_regular(path)
    try:
        facts = inspector(path)
    except Exception:
        raise SocialCoverError("cover_normalization_failed") from None
    if not isinstance(facts, ImageFacts) or (facts.width, facts.height, facts.format) != (1080, 1440, "png"):
        raise SocialCoverError("cover_normalization_failed")


def _publish_new(source: Path, target: Path, expected_hash: str) -> None:
    try:
        os.link(source, target)
        if sha256_file(target) != expected_hash:
            raise OSError
    except FileExistsError:
        raise SocialCoverError("cover_output_exists") from None
    except OSError:
        _remove_if_hash(target, expected_hash)
        raise SocialCoverError("cover_publish_failed") from None


def _publish_json_new(path: Path, payload: dict[str, object]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise SocialCoverError("cover_output_exists") from None
        except OSError:
            raise SocialCoverError("cover_publish_failed") from None
    finally:
        _remove_owned(temporary)


def _remove_if_hash(path: Path, expected_hash: str) -> None:
    try:
        if path.is_file() and not path.is_symlink() and sha256_file(path) == expected_hash:
            path.unlink()
    except OSError:
        pass


def _remove_owned(path: Path) -> None:
    try:
        if path.is_file() and not path.is_symlink():
            path.unlink()
    except OSError:
        pass
