import hashlib
import json
import unicodedata
from collections.abc import Iterable

from pydantic import BaseModel


def _trim_surrounding(value: str) -> str:
    start = 0
    end = len(value)
    while start < end and (
        value[start].isspace()
        or unicodedata.category(value[start]).startswith("P")
    ):
        start += 1
    while end > start and (
        value[end - 1].isspace()
        or unicodedata.category(value[end - 1]).startswith("P")
    ):
        end -= 1
    return value[start:end]


def _display_text(value: str | None) -> str:
    if value is None:
        return ""
    return _trim_surrounding(unicodedata.normalize("NFKC", str(value)))


def _normalized_text(value: str | None) -> str:
    return _display_text(value).casefold()


def _author_values(authors: Iterable[str] | str | None) -> list[str]:
    if authors is None:
        return []
    if isinstance(authors, str):
        authors = [authors]
    return sorted(
        value
        for value in (_normalized_text(author) for author in authors)
        if value
    )


def normalize_book_key(
    title: str,
    authors: Iterable[str] | str | None = None,
    edition: str | None = None,
    isbn: str | None = None,
) -> str:
    """Return a deterministic key from title, authors, and edition only."""

    values = {
        "authors": _author_values(authors),
        "edition": _normalized_text(edition),
        "title": _normalized_text(title),
    }
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_slug(title: str) -> str:
    slug: list[str] = []
    pending_separator = False
    for character in _normalized_text(title):
        if character.isascii() and character.isalnum():
            if pending_separator and slug and slug[-1] != "-":
                slug.append("-")
            slug.append(character)
            pending_separator = False
        else:
            pending_separator = bool(slug)
    result = "".join(slug).strip("-")
    return result or "title"


class BookIdentity(BaseModel):
    book_id: str
    title: str
    authors: list[str]
    isbn: str | None = None
    normalized_key: str

    @classmethod
    def from_metadata(
        cls,
        title: str,
        authors: Iterable[str] | str | None = None,
        *,
        edition: str | None = None,
        isbn: str | None = None,
    ) -> "BookIdentity":
        display_title = _display_text(title) or "Untitled"
        display_authors = [
            value
            for value in (
                _display_text(author)
                for author in ([authors] if isinstance(authors, str) else authors or [])
            )
            if value
        ]
        display_isbn = _display_text(isbn) or None
        normalized_key = normalize_book_key(
            display_title,
            display_authors,
            edition=edition,
            isbn=display_isbn,
        )
        digest = hashlib.sha256(normalized_key.encode("utf-8")).hexdigest()[:12]
        return cls(
            book_id=f"book-{_safe_slug(display_title)}-{digest}",
            title=display_title,
            authors=display_authors,
            isbn=display_isbn,
            normalized_key=normalized_key,
        )
