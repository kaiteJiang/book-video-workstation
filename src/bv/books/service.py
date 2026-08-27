import errno
import json
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import ConfigDict

from bv.core.atomic import atomic_write_json
from bv.core.hashing import sha256_file
from bv.production.profile import ProductionProfile
from bv.state.events import Event
from bv.state.locks import EpisodeLock, LockHeldError
from bv.state.models import BookState, EpisodeState
from bv.state.store import StateStore

from .identity import BookIdentity, _display_text


ALLOWED_SUFFIXES = {".txt", ".epub", ".pdf"}
EPISODE_ID = "E001"
IMPORT_LOCK_TIMEOUT_SECONDS = 5.0
IMPORT_LOCK_RETRY_SECONDS = 0.01
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class BookImportReceipt(BookState):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_id: str
    source_copy: Path
    episode_ids: list[str]


class BookCreationReceipt(BookState):
    episode_ids: list[str]
    idempotent: bool


@dataclass(frozen=True)
class _ImportPaths:
    root: Path
    books_root: Path
    book_id: str
    book_dir: Path
    source_dir: Path
    book_path: Path
    episodes_dir: Path
    episode_dir: Path
    episode_path: Path
    ledger_dir: Path
    events_path: Path
    lock_path: Path
    source_id: str | None = None
    source_suffix: str | None = None
    destination: Path | None = None


def _within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _resolve(path: Path) -> Path:
    try:
        return Path(path).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve path: {path}") from exc


def _is_redirect(path: Path) -> bool:
    try:
        info = os.lstat(os.fspath(path))
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"cannot inspect output path: {path}") from exc

    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _validate_component_chain(path: Path, root: Path, label: str) -> Path:
    candidate = _absolute_lexical(path)
    resolved_root = _absolute_lexical(root)
    if not _within(candidate, resolved_root):
        raise ValueError(f"{label} is outside resolved store root")

    current = resolved_root
    for component in candidate.relative_to(resolved_root).parts:
        current = current / component
        if _is_redirect(current):
            raise ValueError(f"{label} contains an output redirect: {current}")
        if _lexists(current) and not _within(_resolve(current), resolved_root):
            raise ValueError(f"{label} resolves outside store root: {current}")

    return _resolve(candidate)


def _validate_books_root(root: Path) -> Path:
    resolved_root = _absolute_lexical(root)
    books_candidate = resolved_root / "books"
    books_root = _validate_component_chain(
        books_candidate,
        resolved_root,
        "store books",
    )
    if not _within(books_root, resolved_root):
        raise ValueError("store books resolves outside store root")
    return books_root


def _validate_product_path(
    path: Path,
    root: Path,
    books_root: Path,
    label: str,
) -> Path:
    resolved = _validate_component_chain(path, root, label)
    if not _within(resolved, books_root):
        raise ValueError(f"{label} resolves outside store books")
    return resolved


def _build_paths(
    root: Path,
    book_id: str,
    *,
    source_id: str | None = None,
    source_suffix: str | None = None,
) -> _ImportPaths:
    resolved_root = _absolute_lexical(root)
    books_root = _validate_books_root(resolved_root)
    book_dir = _validate_product_path(
        resolved_root / "books" / book_id,
        resolved_root,
        books_root,
        "book directory",
    )
    source_dir = _validate_product_path(
        book_dir / "source",
        resolved_root,
        books_root,
        "source directory",
    )
    book_path = _validate_product_path(
        book_dir / "book.json",
        resolved_root,
        books_root,
        "book state",
    )
    episodes_dir = _validate_product_path(
        book_dir / "episodes",
        resolved_root,
        books_root,
        "episodes directory",
    )
    episode_dir = _validate_product_path(
        episodes_dir / EPISODE_ID,
        resolved_root,
        books_root,
        "E001 directory",
    )
    episode_path = _validate_product_path(
        episode_dir / "episode.json",
        resolved_root,
        books_root,
        "E001 state",
    )
    ledger_dir = _validate_product_path(
        book_dir / "ledger",
        resolved_root,
        books_root,
        "ledger directory",
    )
    events_path = _validate_product_path(
        ledger_dir / "events.jsonl",
        resolved_root,
        books_root,
        "event ledger",
    )
    lock_path = _validate_product_path(
        book_dir / ".import.lock",
        resolved_root,
        books_root,
        "import lock",
    )
    destination: Path | None = None
    if source_id is not None and source_suffix is not None:
        destination = _validate_product_path(
            source_dir / f"{source_id}{source_suffix}",
            resolved_root,
            books_root,
            "source destination",
        )
    return _ImportPaths(
        root=resolved_root,
        books_root=books_root,
        book_id=book_id,
        book_dir=book_dir,
        source_dir=source_dir,
        book_path=book_path,
        episodes_dir=episodes_dir,
        episode_dir=episode_dir,
        episode_path=episode_path,
        ledger_dir=ledger_dir,
        events_path=events_path,
        lock_path=lock_path,
        source_id=source_id,
        source_suffix=source_suffix,
        destination=destination,
    )


def _validate_paths(paths: _ImportPaths) -> _ImportPaths:
    return _build_paths(
        paths.root,
        paths.book_id,
        source_id=paths.source_id,
        source_suffix=paths.source_suffix,
    )


def _authors_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _metadata_values(
    metadata: Mapping[str, Any] | None,
    *,
    title: str | None,
    authors: Iterable[str] | str | None,
    edition: str | None,
    isbn: str | None,
) -> tuple[str | None, list[str], str | None, str | None]:
    values = dict(metadata or {})
    resolved_title = title if title is not None else values.get("title")
    resolved_authors = (
        authors
        if authors is not None
        else values.get("authors", values.get("author"))
    )
    resolved_edition = edition if edition is not None else values.get("edition")
    resolved_isbn = isbn if isbn is not None else values.get("isbn")
    return (
        resolved_title,
        _authors_value(resolved_authors),
        resolved_edition,
        resolved_isbn,
    )


def _resolve_source(
    source_path: Path,
    configured_root: Path,
    books_root: Path,
) -> Path:
    lexical_source = _absolute_lexical(source_path)
    lexical_books = _absolute_lexical(configured_root) / "books"
    if _within(lexical_source, lexical_books):
        raise ValueError("source must not be inside store books")

    source = _resolve(lexical_source)
    try:
        source_info = source.stat()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("source file does not exist") from exc
    if not stat.S_ISREG(source_info.st_mode):
        raise ValueError("source must be a regular file")
    if _within(source, books_root):
        raise ValueError("source must not be inside store books")
    if source.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("unsupported source suffix")
    return source


def _remove_empty_directory(path: Path) -> None:
    try:
        if _lexists(path) and path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


def _verify_existing_destination(
    destination: Path,
    source_id: str,
    paths: _ImportPaths,
) -> None:
    checked = _validate_product_path(
        destination,
        paths.root,
        paths.books_root,
        "source destination",
    )
    if not _lexists(checked) or not checked.is_file():
        raise ValueError("destination hash mismatch")
    if sha256_file(checked) != source_id:
        raise ValueError("destination hash mismatch")


def _publish_temp_without_overwrite(
    temporary: Path,
    destination: Path,
    source_id: str,
    paths: _ImportPaths,
) -> None:
    try:
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
            os.unlink(temporary)
    except FileExistsError:
        _verify_existing_destination(destination, source_id, paths)
    except OSError as exc:
        if exc.errno == errno.EEXIST or _lexists(destination):
            _verify_existing_destination(destination, source_id, paths)
        else:
            raise


def _copy_verified(
    source: Path,
    destination: Path,
    source_id: str,
    paths: _ImportPaths,
) -> None:
    if _lexists(destination):
        _verify_existing_destination(destination, source_id, paths)
        return

    source_dir_was_missing = not _lexists(paths.source_dir)
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        paths.source_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{source_id}.",
            suffix=".tmp",
            dir=paths.source_dir,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as destination_stream:
            descriptor = None
            with source.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, destination_stream)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())

        if sha256_file(temporary) != source_id:
            raise ValueError("copied temporary hash mismatch")
        _publish_temp_without_overwrite(
            temporary,
            destination,
            source_id,
            paths,
        )
    except BaseException:
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None and _lexists(temporary):
            try:
                temporary.unlink()
            except OSError:
                pass
        if source_dir_was_missing:
            _remove_empty_directory(paths.source_dir)


@contextmanager
def _import_lock(path: Path) -> Iterator[None]:
    deadline = time.monotonic() + IMPORT_LOCK_TIMEOUT_SECONDS
    while True:
        lock = EpisodeLock(path)
        try:
            lock.__enter__()
            break
        except LockHeldError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(IMPORT_LOCK_RETRY_SECONDS)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def _has_valid_source_event(
    path: Path,
    book_id: str,
    source_id: str,
) -> bool:
    if not _lexists(path):
        return False
    if not path.is_file():
        raise ValueError("event ledger is not a regular file")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            event = Event.model_validate_json(line)
        except ValueError:
            continue
        if (
            event.event == "source_imported"
            and event.book_id == book_id
            and event.episode_id == EPISODE_ID
            and event.artifact_hash == source_id
        ):
            return True
    return False


def _has_metadata_created_event(path: Path, book_id: str) -> bool:
    if not _lexists(path):
        return False
    if not path.is_file():
        raise ValueError("event ledger is not a regular file")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            event = Event.model_validate_json(line)
        except ValueError:
            continue
        if (
            event.event == "book_created_from_metadata"
            and event.book_id == book_id
            and event.episode_id == EPISODE_ID
        ):
            return True
    return False


def _import_locked(
    source_path: Path,
    configured_root: Path,
    store: StateStore,
    identity: BookIdentity,
    paths: _ImportPaths,
) -> BookImportReceipt:
    source = _resolve_source(source_path, configured_root, paths.books_root)
    source_id = sha256_file(source)
    paths = _build_paths(
        paths.root,
        identity.book_id,
        source_id=source_id,
        source_suffix=source.suffix,
    )
    _validate_paths(paths)
    destination = paths.destination
    assert destination is not None
    _copy_verified(source, destination, source_id, paths)

    _validate_paths(paths)
    if paths.book_path.exists():
        current_book = store.load_book(identity.book_id)
    else:
        current_book = BookState(
            book_id=identity.book_id,
            title=identity.title,
            authors=list(identity.authors),
            source_mode="file",
            source_ids=[],
        )
    source_ids = list(current_book.source_ids)
    if source_id not in source_ids:
        source_ids.append(source_id)
    persisted_book = BookState(
        book_id=current_book.book_id,
        title=current_book.title,
        authors=list(current_book.authors) or list(identity.authors),
        source_mode="file",
        source_ids=source_ids,
    )

    _validate_paths(paths)
    store.save_book(persisted_book)

    _validate_paths(paths)
    if paths.episode_path.exists() and not paths.episode_path.is_file():
        raise ValueError("E001 state is not a regular file")
    if not paths.episode_path.exists():
        store.save_episode(
            EpisodeState(book_id=identity.book_id, episode_id=EPISODE_ID)
        )

    _validate_paths(paths)
    if not _has_valid_source_event(paths.events_path, identity.book_id, source_id):
        store.append_event(
            Event(
                event="source_imported",
                book_id=identity.book_id,
                episode_id=EPISODE_ID,
                timestamp=datetime.now(timezone.utc).isoformat(),
                artifact_hash=source_id,
                stage="source_imported",
                detail={
                    "size_bytes": str(source.stat().st_size),
                    "suffix": source.suffix.lower(),
                },
            )
        )
    return BookImportReceipt(
        book_id=persisted_book.book_id,
        title=persisted_book.title,
        authors=persisted_book.authors,
        source_mode=persisted_book.source_mode,
        source_ids=persisted_book.source_ids,
        source_id=source_id,
        source_copy=destination,
        episode_ids=[EPISODE_ID],
    )


def _create_book_from_metadata_locked(
    store: StateStore,
    identity: BookIdentity,
    paths: _ImportPaths,
) -> BookCreationReceipt:
    paths = _build_paths(paths.root, identity.book_id)
    _validate_paths(paths)

    already_complete = paths.book_path.exists() and _has_metadata_created_event(
        paths.events_path,
        identity.book_id,
    )

    if paths.book_path.exists():
        current_book = store.load_book(identity.book_id)
        # File-backed projects with source imports win over metadata-only create.
        if current_book.source_ids:
            return BookCreationReceipt(
                book_id=current_book.book_id,
                title=current_book.title,
                authors=list(current_book.authors),
                source_mode="file",
                source_ids=list(current_book.source_ids),
                episode_ids=[EPISODE_ID],
                idempotent=True,
            )
        persisted_book = BookState(
            book_id=current_book.book_id,
            title=current_book.title,
            authors=list(current_book.authors) or list(identity.authors),
            source_mode="title_author",
            source_ids=list(current_book.source_ids),
        )
    else:
        persisted_book = BookState(
            book_id=identity.book_id,
            title=identity.title,
            authors=list(identity.authors),
            source_mode="title_author",
            source_ids=[],
        )

    _validate_paths(paths)
    store.save_book(persisted_book)

    _validate_paths(paths)
    if paths.episode_path.exists() and not paths.episode_path.is_file():
        raise ValueError("E001 state is not a regular file")
    if not paths.episode_path.exists():
        store.save_episode(
            EpisodeState(book_id=identity.book_id, episode_id=EPISODE_ID)
        )
        atomic_write_json(
            paths.episode_dir / "production_profile.json",
            ProductionProfile.short_book_default().model_dump(
                mode="json",
                exclude_none=False,
            ),
        )

    _validate_paths(paths)
    if not _has_metadata_created_event(paths.events_path, identity.book_id):
        store.append_event(
            Event(
                event="book_created_from_metadata",
                book_id=identity.book_id,
                episode_id=EPISODE_ID,
                timestamp=datetime.now(timezone.utc).isoformat(),
                artifact_hash=None,
                stage="source_imported",
                detail={
                    "author_count": str(len(identity.authors)),
                    "source_mode": "title_author",
                },
            )
        )

    return BookCreationReceipt(
        book_id=persisted_book.book_id,
        title=persisted_book.title,
        authors=persisted_book.authors,
        source_mode="title_author",
        source_ids=list(persisted_book.source_ids),
        episode_ids=[EPISODE_ID],
        idempotent=already_complete,
    )


def create_book_from_metadata(
    title: str,
    authors: Iterable[str] | str,
    store: StateStore,
) -> BookCreationReceipt:
    """Create a title/author book without copying a source file."""

    if not _display_text(title):
        raise ValueError("title is required")
    # Materialize once: generators are exhausted after a single iteration.
    author_items = [authors] if isinstance(authors, str) else list(authors)
    if not any(_display_text(author) for author in author_items):
        raise ValueError("author is required")

    identity = BookIdentity.from_metadata(title, author_items)
    configured_root = _absolute_lexical(store.root)
    root = _resolve(configured_root)
    paths = _build_paths(root, identity.book_id)
    book_dir_was_missing = not _lexists(paths.book_dir)

    try:
        with _import_lock(paths.lock_path):
            paths = _build_paths(root, identity.book_id)
            return _create_book_from_metadata_locked(store, identity, paths)
    except BaseException:
        if book_dir_was_missing:
            _remove_empty_directory(paths.book_dir)
        raise


def import_book(
    source_path: Path,
    store: StateStore,
    metadata: Mapping[str, Any] | None = None,
    *,
    title: str | None = None,
    authors: Iterable[str] | str | None = None,
    edition: str | None = None,
    isbn: str | None = None,
) -> BookImportReceipt:
    """Safely copy a book source and converge its first-episode state."""

    configured_root = _absolute_lexical(store.root)
    root = _resolve(configured_root)
    lexical_source = _absolute_lexical(source_path)
    if _within(lexical_source, configured_root / "books"):
        raise ValueError("source must not be inside store books")

    resolved_title, resolved_authors, resolved_edition, resolved_isbn = _metadata_values(
        metadata,
        title=title,
        authors=authors,
        edition=edition,
        isbn=isbn,
    )
    identity = BookIdentity.from_metadata(
        resolved_title or lexical_source.stem,
        resolved_authors,
        edition=resolved_edition,
        isbn=resolved_isbn,
    )
    paths = _build_paths(root, identity.book_id)
    book_dir_was_missing = not _lexists(paths.book_dir)

    try:
        with _import_lock(paths.lock_path):
            paths = _build_paths(root, identity.book_id)
            return _import_locked(
                source_path,
                configured_root,
                store,
                identity,
                paths,
            )
    except BaseException:
        if book_dir_was_missing:
            _remove_empty_directory(paths.book_dir)
        raise
