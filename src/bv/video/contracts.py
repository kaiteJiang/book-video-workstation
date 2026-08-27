"""Provider-neutral video generation contracts."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from bv.core.hashing import sha256_file


VideoProviderName = Literal["grok_cli", "grok_manual", "h3_manual"]

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_REFERENCES = 7
_REPARSE_POINT = getattr(os, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class VideoGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    book_id: str
    episode_id: str
    segment_id: str
    provider: VideoProviderName
    prompt: str
    prompt_sha256: str
    expected_duration_ms: int = Field(ge=1, le=15_000)
    aspect_ratio: Literal["9:16"] = "9:16"
    resolution: Literal["480p", "720p"] = "480p"
    episode_root: Path
    reference_images: tuple[Path, ...] = ()
    reference_image_sha256s: tuple[str, ...]
    reference_image_specs: tuple[str, ...] = ()

    @field_validator("book_id", "episode_id", "segment_id")
    @classmethod
    def _require_identifier(cls, value: object) -> str:
        if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("unsafe_identifier")
        return value

    @model_validator(mode="after")
    def _validate_prompt_and_references(self) -> VideoGenerationRequest:
        if not self.prompt.strip():
            raise ValueError("prompt must be nonblank")
        expected = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        if self.prompt_sha256 != expected:
            raise ValueError("prompt_sha256 mismatch")

        images = tuple(Path(path) for path in self.reference_images)
        specs = self.reference_image_specs
        if any(not spec.strip() for spec in specs):
            raise ValueError("reference spec must be nonblank")
        if len(set(specs)) != len(specs):
            raise ValueError("duplicate reference specs")
        if len(images) + len(specs) > _MAX_REFERENCES:
            raise ValueError("too many references")
        if not images and not specs:
            raise ValueError("at least one reference image or spec is required")
        if self.segment_id != "S01" and (not images or not specs):
            raise ValueError("non-S01 segment requires an episode-local image and a scene spec")

        self.verify_reference_images()
        return self

    def verify_reference_images(self) -> None:
        try:
            root = Path(self.episode_root)
            _require_episode_directory(root)
            if root.parts[-4:] != ("books", self.book_id, "episodes", self.episode_id):
                raise ValueError("episode_root must end with books/<book_id>/episodes/<episode_id>")

            images = tuple(Path(path) for path in self.reference_images)
            hashes = self.reference_image_sha256s
            if len(images) != len(hashes):
                raise ValueError("reference image hashes must match images one-to-one")
            if any(_SHA256.fullmatch(item) is None for item in hashes):
                raise ValueError("reference image hash must be lowercase sha256")
            if len(set(images)) != len(images):
                raise ValueError("duplicate reference images")
            if len(set(hashes)) != len(hashes):
                raise ValueError("duplicate reference image hashes")

            for image, expected in zip(images, hashes, strict=True):
                _require_regular_file(image)
                if not _is_contained_under(image, root):
                    raise ValueError("reference image is outside the episode root")
                if sha256_file(image) != expected:
                    raise ValueError("reference_image_sha256 mismatch")
        except (OSError, ValueError) as error:
            raise ValidationError.from_exception_data(
                self.__class__.__name__,
                [
                    {
                        "type": "value_error",
                        "loc": ("reference_images",),
                        "input": self.reference_images,
                        "ctx": {"error": error},
                    }
                ],
            ) from error


class VideoGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: VideoProviderName
    segment_id: str
    source_path: Path
    source_sha256: str
    reference_images: tuple[Path, ...] = ()
    reference_image_sha256s: tuple[str, ...] = ()

    @field_validator("segment_id")
    @classmethod
    def _require_identifier(cls, value: object) -> str:
        if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("unsafe_identifier")
        return value

    @model_validator(mode="after")
    def _validate_source(self) -> VideoGenerationResult:
        source = Path(self.source_path)
        _require_regular_file(source)
        digest = sha256_file(source)
        if self.source_sha256 != digest:
            raise ValueError("source_sha256 mismatch")
        _validate_hashed_files(self.reference_images, self.reference_image_sha256s)
        return self


@runtime_checkable
class VideoGateway(Protocol):
    def generate(self, book_id: str, episode_id: str, shot_id: str) -> object: ...

    def import_video(
        self, book_id: str, episode_id: str, shot_id: str, source: Path
    ) -> object: ...

    def status(self, book_id: str, episode_id: str) -> object: ...


def _is_redirected(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return path.is_symlink()
    return path.is_symlink() or bool(attributes & _REPARSE_POINT)


def _redirect_in_chain(path: Path) -> bool:
    current = Path(path)
    while True:
        if current.exists() or current.is_symlink():
            if _is_redirected(current):
                return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _is_canonical_absolute(path: Path) -> bool:
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        return False
    return Path(os.path.normpath(path)) == path


def _require_regular_file(path: Path) -> None:
    if not _is_canonical_absolute(path):
        raise ValueError("path must be a canonical absolute path")
    if _redirect_in_chain(path):
        raise ValueError("path must be a regular non-reparse file")
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("path is missing") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("path must be a regular file")


def _require_episode_directory(path: Path) -> None:
    if not _is_canonical_absolute(path):
        raise ValueError("path must be a canonical absolute path")
    if _redirect_in_chain(path):
        raise ValueError("path must be a regular non-reparse directory")
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("episode_root is missing") from error
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("episode_root must be a directory")


def _validate_hashed_files(paths: tuple[Path, ...], hashes: tuple[str, ...]) -> None:
    images = tuple(Path(path) for path in paths)
    if len(images) != len(hashes):
        raise ValueError("reference image hashes must match images one-to-one")
    if any(_SHA256.fullmatch(item) is None for item in hashes):
        raise ValueError("reference image hash must be lowercase sha256")
    if len(set(images)) != len(images):
        raise ValueError("duplicate reference images")
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate reference image hashes")
    for image, expected in zip(images, hashes, strict=True):
        _require_regular_file(image)
        if sha256_file(image) != expected:
            raise ValueError("reference_image_sha256 mismatch")


def _is_contained_under(path: Path, root: Path) -> bool:
    try:
        Path(os.path.normpath(path)).relative_to(Path(os.path.normpath(root)))
    except ValueError:
        return False
    return True
