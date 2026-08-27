import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bv.books import service as book_service
from bv.books.identity import BookIdentity, normalize_book_key
from bv.books.service import import_book
from bv.cli import app
from bv.config import AppConfig
from bv.state.models import BookState
from bv.state.store import StateStore


runner = CliRunner()


def _events(store: StateStore, book_id: str) -> list[dict[str, object]]:
    path = store.root / "books" / book_id / "ledger" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_identity_normalization_is_nfkc_casefold_and_deterministic() -> None:
    first = normalize_book_key(" Ｄｅｍｏ！ ", [" Author ", "第二作者"], edition=" １ｓｔ ")
    second = normalize_book_key("demo", ["author", "第二作者"], edition="1ST")

    assert first == second
    identity = BookIdentity.from_metadata(" Ｄｅｍｏ！ ", [" Author ", "第二作者"], edition=" １ｓｔ ")
    expected_digest = hashlib.sha256(first.encode("utf-8")).hexdigest()[:12]
    assert identity.normalized_key == first
    assert identity.book_id.startswith("book-demo-")
    assert identity.book_id.endswith(expected_digest)
    assert re.fullmatch(r"book-[A-Za-z0-9\u0080-\uffff_-]+-[0-9a-f]{12}", identity.book_id)


def test_isbn_is_metadata_only_and_does_not_change_identity() -> None:
    without_isbn = BookIdentity.from_metadata(
        "Demo",
        ["Author"],
        edition="1st",
    )
    with_isbn = BookIdentity.from_metadata(
        "Demo",
        ["Author"],
        edition="1st",
        isbn="978-1-4028-9462-6",
    )
    with_different_isbn = BookIdentity.from_metadata(
        "Demo",
        ["Author"],
        edition="1st",
        isbn="978-0-545-01022-1",
    )

    assert without_isbn.book_id == with_isbn.book_id == with_different_isbn.book_id
    assert without_isbn.normalized_key == with_isbn.normalized_key == with_different_isbn.normalized_key
    assert without_isbn.isbn is None
    assert with_isbn.isbn == "978-1-4028-9462-6"
    assert with_different_isbn.isbn == "978-0-545-01022-1"


def test_import_copies_source_and_creates_only_e001(tmp_path: Path) -> None:
    source = tmp_path / "input" / "demo.txt"
    source.parent.mkdir()
    source_bytes = "第一章\n正文".encode("utf-8")
    source.write_bytes(source_bytes)
    store = StateStore(tmp_path / "workspace")

    result = import_book(source, store, title="Demo", authors=["Author"])

    assert isinstance(result, BookState)
    assert result.source_id == hashlib.sha256(source_bytes).hexdigest()
    assert result.source_copy.exists()
    assert result.source_copy.read_bytes() == source_bytes
    assert result.source_copy.suffix == ".txt"
    assert source.read_bytes() == source_bytes
    assert result.episode_ids == ["E001"]
    assert (store.root / "books" / result.book_id / "episodes" / "E001" / "episode.json").exists()
    assert not (store.root / "books" / result.book_id / "episodes" / "E002").exists()

    persisted = store.load_book(result.book_id)
    assert type(persisted) is BookState
    assert persisted.source_ids == [result.source_id]
    assert set(json.loads((store.root / "books" / result.book_id / "book.json").read_text(encoding="utf-8"))) == {
        "book_id",
        "title",
        "authors",
        "source_mode",
        "source_ids",
    }
    assert store.load_episode(result.book_id, "E001").status == "source_imported"
    events = _events(store, result.book_id)
    assert len(events) == 1
    assert events[0]["event"] == "source_imported"
    assert events[0]["episode_id"] == "E001"
    assert str(source) not in json.dumps(events[0], ensure_ascii=False)


def test_same_title_author_reuses_logical_book_id_and_preserves_sources(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "workspace")
    txt = tmp_path / "a.txt"
    pdf = tmp_path / "a.PDF"
    txt.write_text("text", encoding="utf-8")
    pdf.write_bytes(b"%PDF-test")

    first = import_book(txt, store, title="Demo", authors=["Author"])
    second = import_book(pdf, store, title="Demo", authors=["Author"])

    assert first.book_id == second.book_id
    assert first.source_id != second.source_id
    assert first.source_copy.exists()
    assert second.source_copy.exists()
    assert store.load_book(first.book_id).source_ids == [
        first.source_id,
        second.source_id,
    ]
    assert second.episode_ids == ["E001"]
    assert len(_events(store, first.book_id)) == 2


def test_import_accepts_metadata_mapping_when_keywords_are_omitted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-name.txt"
    source.write_text("text", encoding="utf-8")

    result = import_book(
        source,
        StateStore(tmp_path / "workspace"),
        {"title": "Metadata Title", "authors": ["Metadata Author"]},
    )

    assert result.title == "Metadata Title"
    assert result.book_id.startswith("book-metadata-title-")


def test_reimport_rejects_a_different_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "demo.txt"
    source.write_text("original", encoding="utf-8")
    store = StateStore(tmp_path / "workspace")
    result = import_book(source, store, title="Demo", authors=[])
    result.source_copy.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="destination hash"):
        import_book(source, store, title="Demo", authors=[])

    assert result.source_copy.read_text(encoding="utf-8") == "tampered"
    assert len(_events(store, result.book_id)) == 1


def test_copy_failure_cleans_final_and_temporary_files_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "demo.txt"
    source.write_text("original", encoding="utf-8")
    store = StateStore(tmp_path / "workspace")
    book_id = BookIdentity.from_metadata("Demo", []).book_id

    def fail_copy(source_stream, destination_stream, length=16 * 1024) -> None:
        del source_stream, length
        destination_stream.write(b"partial")
        raise OSError("injected copy failure")

    monkeypatch.setattr(book_service.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        import_book(source, store, title="Demo", authors=[])

    book_dir = store.root / "books" / book_id
    assert not (book_dir / "source").exists()
    assert not list(book_dir.rglob("*.tmp")) if book_dir.exists() else True

    monkeypatch.undo()
    result = import_book(source, store, title="Demo", authors=[])
    assert result.source_copy.exists()
    assert result.source_copy.read_text(encoding="utf-8") == "original"


def _make_symlink_or_skip(
    link: Path, target: Path, *, target_is_directory: bool
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314 or exc.errno in {1, 13, 95}:
            pytest.skip("symlink creation requires additional privilege")
        raise


@pytest.mark.parametrize(
    "component",
    ["books", "book", "source", "episodes", "episode", "ledger"],
)
def test_import_rejects_existing_output_symlink_components(
    tmp_path: Path, component: str
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("source", encoding="utf-8")
    root = tmp_path / "workspace"
    root.mkdir()
    book_id = BookIdentity.from_metadata("Demo", []).book_id
    outside = tmp_path / f"outside-{component}"
    outside.mkdir()

    books = root / "books"
    if component == "books":
        _make_symlink_or_skip(books, outside, target_is_directory=True)
    else:
        books.mkdir()
        book_dir = books / book_id
        if component == "book":
            _make_symlink_or_skip(book_dir, outside, target_is_directory=True)
        else:
            book_dir.mkdir()
            if component == "source":
                link = book_dir / "source"
            elif component == "episodes":
                link = book_dir / "episodes"
            elif component == "episode":
                episodes = book_dir / "episodes"
                episodes.mkdir()
                link = episodes / "E001"
            else:
                link = book_dir / "ledger"
            _make_symlink_or_skip(link, outside, target_is_directory=True)

    with pytest.raises(ValueError, match="redirect|outside"):
        import_book(source, StateStore(root), title="Demo", authors=[])


def test_lexical_source_under_books_is_rejected_before_resolving_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    books = root / "books"
    books.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("source", encoding="utf-8")
    lexical_source = books / "linked.txt"
    _make_symlink_or_skip(lexical_source, outside, target_is_directory=False)

    with pytest.raises(ValueError, match="store books"):
        import_book(lexical_source, StateStore(root), title="Demo", authors=[])


def test_configured_store_root_symlink_uses_resolved_root(
    tmp_path: Path,
) -> None:
    actual_root = tmp_path / "actual-workspace"
    actual_root.mkdir()
    configured_root = tmp_path / "configured-workspace"
    _make_symlink_or_skip(configured_root, actual_root, target_is_directory=True)
    source = tmp_path / "input.txt"
    source.write_text("source", encoding="utf-8")

    result = import_book(source, StateStore(configured_root), title="Demo", authors=[])

    assert result.source_copy.is_relative_to(actual_root / "books")
    assert result.source_copy.exists()


class _DelayedFirstSaveStore(StateStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.first_save_started = threading.Event()
        self._delayed = False

    def save_book(self, state: BookState) -> None:
        if not self._delayed:
            self._delayed = True
            self.first_save_started.set()
            time.sleep(0.2)
        super().save_book(state)


def test_concurrent_formats_preserve_both_source_ids_and_one_e001(
    tmp_path: Path,
) -> None:
    store = _DelayedFirstSaveStore(tmp_path / "workspace")
    txt = tmp_path / "demo.txt"
    pdf = tmp_path / "demo.pdf"
    txt.write_text("txt source", encoding="utf-8")
    pdf.write_bytes(b"pdf source")

    first_result: list[object] = []
    first_errors: list[BaseException] = []

    def import_first() -> None:
        try:
            first_result.append(import_book(txt, store, title="Demo", authors=[]))
        except BaseException as exc:
            first_errors.append(exc)

    first_thread = threading.Thread(target=import_first)
    first_thread.start()
    assert store.first_save_started.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        second_future = executor.submit(
            import_book,
            pdf,
            store,
            title="Demo",
            authors=[],
        )
        first_thread.join(timeout=5)
        assert not first_thread.is_alive()
        second = second_future.result(timeout=5)

    assert not first_errors
    first = first_result[0]
    stored = store.load_book(first.book_id)
    assert set(stored.source_ids) == {first.source_id, second.source_id}
    assert store.load_episode(first.book_id, "E001").status == "source_imported"
    assert len(_events(store, first.book_id)) == 2


class _FailOnceBeforeEpisodeStore(StateStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.fail = True

    def save_episode(self, state) -> None:
        if self.fail:
            self.fail = False
            raise RuntimeError("injected episode failure")
        super().save_episode(state)


class _FailOnceAfterEventStore(StateStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.fail = True

    def append_event(self, event) -> None:
        super().append_event(event)
        if self.fail:
            self.fail = False
            raise RuntimeError("injected post-event failure")


def test_retry_converges_after_failure_between_book_and_episode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "demo.txt"
    source.write_text("source", encoding="utf-8")
    store = _FailOnceBeforeEpisodeStore(tmp_path / "workspace")

    with pytest.raises(RuntimeError, match="injected episode failure"):
        import_book(source, store, title="Demo", authors=[])

    book_id = BookIdentity.from_metadata("Demo", []).book_id
    assert store.load_book(book_id).source_ids
    assert not (store.root / "books" / book_id / "episodes" / "E001" / "episode.json").exists()

    result = import_book(source, store, title="Demo", authors=[])

    assert store.load_book(book_id).source_ids == [result.source_id]
    assert store.load_episode(book_id, "E001").status == "source_imported"
    assert len(_events(store, book_id)) == 1


def test_retry_after_post_event_failure_does_not_duplicate_valid_event(
    tmp_path: Path,
) -> None:
    source = tmp_path / "demo.txt"
    source.write_text("source", encoding="utf-8")
    store = _FailOnceAfterEventStore(tmp_path / "workspace")

    with pytest.raises(RuntimeError, match="injected post-event failure"):
        import_book(source, store, title="Demo", authors=[])

    result = import_book(source, store, title="Demo", authors=[])

    assert store.load_book(result.book_id).source_ids == [result.source_id]
    assert store.load_episode(result.book_id, "E001").status == "source_imported"
    events = _events(store, result.book_id)
    assert len(events) == 1
    assert events[0]["artifact_hash"] == result.source_id


def test_import_rejects_output_reimport_and_unsupported_sources(tmp_path: Path) -> None:
    source = tmp_path / "demo.txt"
    source.write_text("original", encoding="utf-8")
    store = StateStore(tmp_path / "workspace")
    result = import_book(source, store, title="Demo", authors=[])

    with pytest.raises(ValueError, match="store books"):
        import_book(result.source_copy, store, title="Demo", authors=[])

    unsupported = tmp_path / "demo.docx"
    unsupported.write_bytes(b"data")
    with pytest.raises(ValueError, match="suffix"):
        import_book(unsupported, store, title="Demo", authors=[])


def test_cli_new_uses_stem_and_renders_exact_success_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "My Book.TXT"
    source.write_text("chapter", encoding="utf-8")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(
        "bv.cli.load_config_default",
        lambda: AppConfig(workspace_dir=workspace),
    )

    result = runner.invoke(app, ["new", str(source)])

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    assert lines[0] == "Imported: My Book"
    assert lines[1].startswith("Book ID: book-")
    assert lines[2] == "Episode: E001"
    assert lines[3] == f"Next: bv next {lines[1].removeprefix('Book ID: ')}"


def test_cli_new_accepts_title_and_repeatable_authors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "My Book.txt"
    source.write_text("chapter", encoding="utf-8")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(
        "bv.cli.load_config_default",
        lambda: AppConfig(workspace_dir=workspace),
    )

    result = runner.invoke(
        app,
        [
            "new",
            str(source),
            "--title",
            "Override",
            "--author",
            "Alice",
            "--author",
            "Bob",
        ],
    )

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    assert lines[0] == "Imported: Override"
    assert lines[3] == f"Next: bv next {lines[1].removeprefix('Book ID: ')}"


def test_old_book_state_defaults_to_file_mode() -> None:
    state = BookState.model_validate({"book_id": "book-demo", "title": "Demo"})

    assert state.source_mode == "file"
    assert state.authors == []


def test_title_author_state_records_identity_without_source_file() -> None:
    state = BookState(
        book_id="book-demo",
        title="Demo",
        authors=["Writer"],
        source_mode="title_author",
    )

    assert state.authors == ["Writer"]
    assert state.source_mode == "title_author"
    assert state.source_ids == []


def test_create_book_from_metadata_is_idempotent(tmp_path: Path) -> None:
    from bv.books.service import create_book_from_metadata

    store = StateStore(tmp_path / "workspace")
    expected_key = json.dumps(
        {
            "authors": ["古贺史健", "岸见一郎"],
            "edition": "",
            "title": "被讨厌的勇气",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_book_id = (
        "book-title-"
        + hashlib.sha256(expected_key.encode("utf-8")).hexdigest()[:12]
    )

    first = create_book_from_metadata(
        "被讨厌的勇气",
        ["岸见一郎", "古贺史健"],
        store,
    )
    second = create_book_from_metadata(
        "被讨厌的勇气",
        ["岸见一郎", "古贺史健"],
        store,
    )

    assert first.book_id == expected_book_id
    assert first.book_id == second.book_id
    assert first.title == "被讨厌的勇气"
    assert first.authors == ["岸见一郎", "古贺史健"]
    assert first.source_mode == "title_author"
    assert first.source_ids == []
    assert first.episode_ids == ["E001"]
    assert first.idempotent is False
    assert second.source_mode == "title_author"
    assert second.source_ids == []
    assert second.episode_ids == ["E001"]
    assert second.idempotent is True

    book_dir = store.root / "books" / first.book_id
    assert (book_dir / "book.json").is_file()
    assert not (book_dir / "source").exists()
    assert store.load_episode(first.book_id, "E001").status == "source_imported"
    persisted = store.load_book(first.book_id)
    assert persisted.source_mode == "title_author"
    assert persisted.authors == ["岸见一郎", "古贺史健"]
    assert persisted.source_ids == []

    events = _events(store, first.book_id)
    assert len(events) == 1
    assert events[0]["event"] == "book_created_from_metadata"
    assert events[0]["book_id"] == first.book_id
    assert events[0]["episode_id"] == "E001"
    assert events[0]["stage"] == "source_imported"
    assert events[0]["artifact_hash"] is None
    assert events[0]["detail"] == {
        "author_count": "2",
        "source_mode": "title_author",
    }


def test_create_book_from_metadata_rejects_empty_title_and_authors_before_write(
    tmp_path: Path,
) -> None:
    from bv.books.service import create_book_from_metadata

    store = StateStore(tmp_path / "workspace")
    books_root = store.root / "books"

    with pytest.raises(ValueError):
        create_book_from_metadata("   ", ["岸见一郎"], store)
    assert not books_root.exists()

    with pytest.raises(ValueError):
        create_book_from_metadata("被讨厌的勇气", [], store)
    assert not books_root.exists()

    with pytest.raises(ValueError):
        create_book_from_metadata("被讨厌的勇气", ["  ", ""], store)
    assert not books_root.exists()


def test_import_book_persists_file_mode_and_normalized_authors(tmp_path: Path) -> None:
    source = tmp_path / "demo.txt"
    source.write_text("第一章\n正文", encoding="utf-8")
    store = StateStore(tmp_path / "workspace")

    result = import_book(
        source,
        store,
        title="  Demo  ",
        authors=["  Author  ", "第二作者"],
    )

    assert result.source_mode == "file"
    assert result.authors == ["Author", "第二作者"]
    assert result.source_ids
    persisted = store.load_book(result.book_id)
    assert persisted.source_mode == "file"
    assert persisted.authors == ["Author", "第二作者"]
    assert persisted.source_ids == result.source_ids


def test_create_book_from_metadata_preserves_authors_from_generator(
    tmp_path: Path,
) -> None:
    from bv.books.service import create_book_from_metadata

    store = StateStore(tmp_path / "workspace")
    authors = (name for name in ["岸见一郎", "古贺史健"])

    result = create_book_from_metadata("被讨厌的勇气", authors, store)

    assert result.authors == ["岸见一郎", "古贺史健"]
    persisted = store.load_book(result.book_id)
    assert persisted.authors == ["岸见一郎", "古贺史健"]


def test_create_book_from_metadata_defers_to_existing_file_project(
    tmp_path: Path,
) -> None:
    from bv.books.service import create_book_from_metadata

    source = tmp_path / "demo.txt"
    source.write_text("第一章\n正文", encoding="utf-8")
    store = StateStore(tmp_path / "workspace")

    imported = import_book(source, store, title="Demo", authors=["Writer"])
    original_source_ids = list(imported.source_ids)
    original_events = _events(store, imported.book_id)
    assert original_source_ids
    assert len(original_events) == 1
    assert original_events[0]["event"] == "source_imported"

    result = create_book_from_metadata("Demo", ["Writer"], store)

    assert result.book_id == imported.book_id
    assert result.source_mode == "file"
    assert result.source_ids == original_source_ids
    assert result.idempotent is True

    persisted = store.load_book(imported.book_id)
    assert persisted.source_mode == "file"
    assert persisted.source_ids == original_source_ids

    events = _events(store, imported.book_id)
    assert events == original_events
    assert [event["event"] for event in events] == ["source_imported"]
