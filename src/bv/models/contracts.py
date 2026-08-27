from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel


ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredModel(Protocol):
    """A local CLI provider consuming a precomposed, source-bounded prompt."""

    def complete(
        self, prompt: str, schema_type: type[ModelT], request_dir: Path
    ) -> ModelT | ReviewSkipped: ...


@dataclass(frozen=True, slots=True)
class PromptAsset:
    """Versioned prompt text and its identity over exact source bytes."""

    name: str
    text: str
    sha256: str


class ReviewSkipped(BaseModel):
    """A non-blocking Grok review did not yield trustworthy structured output."""

    skipped: Literal[True] = True
    error_code: str
    user_message: str


class ModelCompletionError(RuntimeError):
    """A safe, stable failure from the mandatory Codex content provider."""

    def __init__(self, error_code: str, user_message: str) -> None:
        self.error_code = error_code
        self.user_message = user_message
        super().__init__(user_message)


class ModelInvalidResponseError(ModelCompletionError):
    """A private invalid response is available for one bounded local repair."""

    def __init__(
        self,
        error_code: str,
        user_message: str,
        private_response: str,
    ) -> None:
        self.private_response = private_response
        super().__init__(error_code, user_message)


def is_redirected(path: Path) -> bool:
    """Return whether a request artifact is a link or Windows reparse point."""
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    reparse_point = getattr(os, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_point)


def empty_request_directory(request_dir: Path) -> bool:
    """Accept only an existing empty real directory for one model invocation."""
    try:
        return request_dir.is_dir() and not is_redirected(request_dir) and not any(
            request_dir.iterdir()
        )
    except OSError:
        return False


def only_expected_artifacts(request_dir: Path, expected: set[Path]) -> bool:
    """Reject extra files and redirected paths instead of following them."""
    try:
        artifacts = set(request_dir.iterdir())
    except OSError:
        return False
    return artifacts == expected and not any(is_redirected(path) for path in artifacts)
