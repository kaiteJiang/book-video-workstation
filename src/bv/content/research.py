"""Whole-book research schemas and fail-closed research orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from bv.books.identity import normalize_book_key
from bv.core.atomic import atomic_write_json
from bv.models.contracts import is_redirected as _is_redirected
from bv.models.prompts import compose_source_prompt
from bv.state.models import BookState

_ModelT = TypeVar("_ModelT", bound=BaseModel)

_SOURCE_TYPES = Literal[
    "publisher",
    "author_interview",
    "official_sample",
    "table_of_contents",
    "academic",
    "professional_review",
    "reader_reception",
    "other",
]

_ACCEPTED_CONFIDENCE = frozenset({"high", "medium"})

_TRUSTED_REQUIREMENTS = (
    "Focus on whole-book reader value rather than isolated concepts.\n"
    "Verify material identity against publisher and catalog evidence.\n"
    "Cite source URLs for important judgments.\n"
    "Disclose uncertainty when evidence is incomplete.\n"
    "Do not invent quotations.\n"
)


class BookResearchError(RuntimeError):
    """A safe, stable failure from whole-book research orchestration."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class ResearchModel(Protocol):
    def complete(
        self, prompt: str, schema_type: type[_ModelT], request_dir: Path
    ) -> Any: ...


def _strip_nonblank(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be blank")
    return stripped


def _strip_nonblank_items(
    values: object, *, field_name: str, min_items: int = 1
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence")
    cleaned = tuple(_strip_nonblank(item, field_name=field_name) for item in values)
    if len(cleaned) < min_items:
        raise ValueError(f"{field_name} requires at least {min_items} item(s)")
    return cleaned


def _normalize_source_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            parts.path,
            parts.query,
            parts.fragment,
        )
    )


def _require_http_url(value: object) -> str:
    text = _strip_nonblank(value, field_name="url")
    parts = urlsplit(text)
    if parts.scheme.casefold() not in {"http", "https"} or not parts.netloc:
        raise ValueError("url must be an absolute HTTP(S) URL")
    return text


class ResearchSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    source_type: _SOURCE_TYPES
    supports: tuple[str, ...]

    @field_validator("url", mode="before")
    @classmethod
    def _validate_url(cls, value: object) -> str:
        return _require_http_url(value)

    @field_validator("source_type", mode="before")
    @classmethod
    def _validate_source_type(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("supports", mode="before")
    @classmethod
    def _validate_supports(cls, value: object) -> tuple[str, ...]:
        return _strip_nonblank_items(value, field_name="supports", min_items=1)


class BookResearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    authors: tuple[str, ...]
    identity_confidence: Literal["high", "medium"]
    central_question: str
    argument_or_narrative_arc: tuple[str, ...]
    key_ideas: tuple[str, ...]
    life_connections: tuple[str, ...]
    reader_value: tuple[str, ...]
    misunderstandings_and_boundaries: tuple[str, ...]
    sources: tuple[ResearchSource, ...]
    uncertainties: tuple[str, ...] = ()

    @field_validator("title", "central_question", mode="before")
    @classmethod
    def _validate_required_text(cls, value: object, info: Any) -> str:
        return _strip_nonblank(value, field_name=info.field_name)

    @field_validator("authors", mode="before")
    @classmethod
    def _validate_authors(cls, value: object) -> tuple[str, ...]:
        return _strip_nonblank_items(value, field_name="authors", min_items=1)

    @field_validator("argument_or_narrative_arc", mode="before")
    @classmethod
    def _validate_arc(cls, value: object) -> tuple[str, ...]:
        return _strip_nonblank_items(
            value, field_name="argument_or_narrative_arc", min_items=1
        )

    @field_validator("key_ideas", mode="before")
    @classmethod
    def _validate_key_ideas(cls, value: object) -> tuple[str, ...]:
        return _strip_nonblank_items(value, field_name="key_ideas", min_items=2)

    @field_validator("life_connections", mode="before")
    @classmethod
    def _validate_life_connections(cls, value: object) -> tuple[str, ...]:
        return _strip_nonblank_items(value, field_name="life_connections", min_items=1)

    @field_validator("reader_value", mode="before")
    @classmethod
    def _validate_reader_value(cls, value: object) -> tuple[str, ...]:
        return _strip_nonblank_items(value, field_name="reader_value", min_items=1)

    @field_validator("misunderstandings_and_boundaries", mode="before")
    @classmethod
    def _validate_boundaries(cls, value: object) -> tuple[str, ...]:
        return _strip_nonblank_items(
            value, field_name="misunderstandings_and_boundaries", min_items=1
        )

    @field_validator("sources", mode="before")
    @classmethod
    def _validate_sources_present(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)) or len(value) < 1:
            raise ValueError("sources requires at least one entry")
        return value

    @field_validator("uncertainties", mode="before")
    @classmethod
    def _validate_uncertainties(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("uncertainties must be a sequence")
        if len(value) == 0:
            return ()
        return _strip_nonblank_items(value, field_name="uncertainties", min_items=1)

    @model_validator(mode="after")
    def _reject_duplicate_source_urls(self) -> BookResearch:
        seen: set[str] = set()
        for source in self.sources:
            key = _normalize_source_url(source.url)
            if key in seen:
                raise ValueError("source URLs must be unique after normalization")
            seen.add(key)
        return self


class BookResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["accepted"]
    research: BookResearch
    json_path: Path
    markdown_path: Path
    prompt_sha256: str

    @field_validator("prompt_sha256")
    @classmethod
    def _validate_prompt_hash(cls, value: object) -> str:
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("prompt_sha256 must be 64 lowercase hex characters")
        if value != value.lower() or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("prompt_sha256 must be 64 lowercase hex characters")
        return value


def _redirect_in_existing_chain(path: Path) -> bool:
    candidate = Path(path)
    while True:
        if candidate.exists() or candidate.is_symlink():
            if _is_redirected(candidate):
                return True
        parent = candidate.parent
        if parent == candidate:
            return False
        candidate = parent


def _assert_output_paths_safe(output_root: Path, *artifact_paths: Path) -> None:
    if output_root.is_symlink() or (
        output_root.exists() and _is_redirected(output_root)
    ):
        raise BookResearchError("book_research_directory_unsafe")
    if _redirect_in_existing_chain(output_root):
        raise BookResearchError("book_research_directory_unsafe")
    for artifact in artifact_paths:
        if artifact.is_symlink() or (artifact.exists() and _is_redirected(artifact)):
            raise BookResearchError("book_research_directory_unsafe")
        if _redirect_in_existing_chain(artifact):
            raise BookResearchError("book_research_directory_unsafe")


def _compose_research_prompt(prompt_text: str, book: BookState) -> str:
    trusted = f"{prompt_text.rstrip()}\n\n{_TRUSTED_REQUIREMENTS}"
    source_payload = {
        "book_id": book.book_id,
        "title": book.title,
        "authors": list(book.authors),
    }
    return compose_source_prompt(
        trusted,
        json.dumps(source_payload, ensure_ascii=False, sort_keys=True),
    )


def _coerce_research(response: object) -> BookResearch:
    if isinstance(response, BookResearch):
        return response
    return BookResearch.model_validate(response)


def _identity_matches(book: BookState, research: BookResearch) -> bool:
    if research.identity_confidence not in _ACCEPTED_CONFIDENCE:
        return False
    return normalize_book_key(research.title, research.authors) == normalize_book_key(
        book.title, book.authors
    )


def _render_markdown(research: BookResearch) -> str:
    lines: list[str] = [
        f"# {research.title}",
        "",
        f"- Authors: {', '.join(research.authors)}",
        f"- Identity confidence: {research.identity_confidence}",
        "",
        "## Central Question",
        "",
        research.central_question,
        "",
        "## Argument or Narrative Arc",
        "",
    ]
    for item in research.argument_or_narrative_arc:
        lines.append(f"- {item}")
    lines.extend(("", "## Key Ideas", ""))
    for item in research.key_ideas:
        lines.append(f"- {item}")
    lines.extend(("", "## Life Connections", ""))
    for item in research.life_connections:
        lines.append(f"- {item}")
    lines.extend(("", "## Reader Value", ""))
    for item in research.reader_value:
        lines.append(f"- {item}")
    lines.extend(("", "## Misunderstandings and Boundaries", ""))
    for item in research.misunderstandings_and_boundaries:
        lines.append(f"- {item}")
    lines.extend(("", "## Sources", ""))
    for source in research.sources:
        support_text = "; ".join(source.supports)
        lines.append(
            f"- {source.url} ({source.source_type}): {support_text}"
        )
    lines.extend(("", "## Uncertainties", ""))
    if research.uncertainties:
        for item in research.uncertainties:
            lines.append(f"- {item}")
    else:
        lines.append("- None disclosed.")
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _is_safe_regular_file(path: Path) -> bool:
    path = Path(path)
    try:
        if path.is_symlink() or _is_redirected(path):
            return False
        return path.is_file()
    except OSError:
        return False


def _snapshot_final_artifacts(*paths: Path) -> dict[Path, bytes | None]:
    """Capture whether each final artifact exists and its exact prior bytes."""
    snapshots: dict[Path, bytes | None] = {}
    for path in paths:
        path = Path(path)
        if _is_safe_regular_file(path):
            snapshots[path] = path.read_bytes()
        else:
            snapshots[path] = None
    return snapshots


def _atomic_restore_bytes(path: Path, content: bytes) -> None:
    """Atomically restore exact prior bytes without using ``_atomic_write_text``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            if temporary.exists() and not temporary.is_symlink() and not _is_redirected(
                temporary
            ):
                temporary.unlink()
        except OSError:
            pass


def _rollback_final_artifacts(priors: dict[Path, bytes | None]) -> None:
    """Restore pre-existing finals; remove only newly created ones."""
    for path, prior_bytes in priors.items():
        try:
            if prior_bytes is not None:
                _atomic_restore_bytes(path, prior_bytes)
            elif _is_safe_regular_file(path):
                path.unlink()
        except OSError:
            pass


def _publish_artifacts(
    research: BookResearch,
    json_path: Path,
    markdown_path: Path,
) -> None:
    markdown = _render_markdown(research)
    priors = _snapshot_final_artifacts(json_path, markdown_path)
    try:
        atomic_write_json(json_path, research.model_dump(mode="json"))
        _atomic_write_text(markdown_path, markdown)
    except Exception:
        _rollback_final_artifacts(priors)
        raise


def _create_request_directory(output_root: Path) -> Path:
    requests_root = output_root / ".private" / "requests"
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        requests_root.mkdir(parents=True, exist_ok=True)
        request_dir = requests_root / uuid.uuid4().hex
        request_dir.mkdir(parents=False, exist_ok=False)
    except OSError:
        raise BookResearchError("book_research_directory_unsafe") from None
    if _is_redirected(request_dir) or _redirect_in_existing_chain(request_dir):
        raise BookResearchError("book_research_directory_unsafe")
    return request_dir


def research_book(
    *,
    book: BookState,
    model: ResearchModel,
    output_root: Path,
    prompt_text: str,
) -> BookResearchResult:
    """Research whole-book reader value and publish accepted artifacts atomically."""
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise BookResearchError("book_research_prompt_invalid")

    output_root = Path(output_root)
    json_path = output_root / "book_research.json"
    markdown_path = output_root / "book_research.md"
    _assert_output_paths_safe(output_root, json_path, markdown_path)

    prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    request_dir = _create_request_directory(output_root)
    composed_prompt = _compose_research_prompt(prompt_text, book)

    response = model.complete(composed_prompt, BookResearch, request_dir)
    try:
        research = _coerce_research(response)
    except Exception:
        raise BookResearchError("book_research_invalid") from None

    if not _identity_matches(book, research):
        raise BookResearchError("book_research_identity_mismatch")

    _assert_output_paths_safe(output_root, json_path, markdown_path)
    _publish_artifacts(research, json_path, markdown_path)

    return BookResearchResult(
        status="accepted",
        research=research,
        json_path=json_path,
        markdown_path=markdown_path,
        prompt_sha256=prompt_sha256,
    )
