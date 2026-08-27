"""CLI contract for title/author book creation (dual-mode `bv new`)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bv.cli import app
from bv.production.profile import load_production_profile
from bv.state.store import StateStore


runner = CliRunner()


def _prepare_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "workspace_dir: workspace\n",
        encoding="utf-8",
    )
    return tmp_path / "workspace"


def _books_root(workspace: Path) -> Path:
    return workspace / "books"


def _assert_no_book_project(workspace: Path) -> None:
    books = _books_root(workspace)
    assert not books.exists()


def test_cli_new_title_author_creates_project_and_prints_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _prepare_workspace(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "new",
            "--title",
            "被讨厌的勇气",
            "--author",
            "岸见一郎",
            "--author",
            "古贺史健",
        ],
    )

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    assert lines[0] == "Created: 被讨厌的勇气"
    assert lines[1].startswith("Book ID: ")
    book_id = lines[1].removeprefix("Book ID: ")
    assert book_id
    assert lines[2] == "Episode: E001"
    assert lines[3] == f"Next: bv next {book_id}"

    store = StateStore(workspace)
    book = store.load_book(book_id)
    assert book.source_mode == "title_author"
    assert book.authors == ["岸见一郎", "古贺史健"]
    assert book.source_ids == []
    episode = store.load_episode(book_id, "E001")
    assert episode.episode_id == "E001"
    assert episode.book_id == book_id
    profile = load_production_profile(
        workspace / "books" / book_id / "episodes" / "E001"
    )
    assert profile.duration.hard_min_seconds == 30.0
    assert profile.duration.hard_max_seconds == 45.0
    assert profile.visual.scene_count == 4
    assert profile.visual.style_id == "retro-gouache-concept"


def test_cli_new_without_source_exits_with_book_source_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _prepare_workspace(tmp_path, monkeypatch)

    result = runner.invoke(app, ["new"])

    assert result.exit_code == 2
    assert "Error: book_source_required" in result.stdout
    _assert_no_book_project(workspace)


def test_cli_new_positional_file_with_title_author_overrides_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _prepare_workspace(tmp_path, monkeypatch)
    source = tmp_path / "source.txt"
    original = "第一章\n原始正文不会被改写\n"
    source.write_text(original, encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "new",
            str(source),
            "--title",
            "Override",
            "--author",
            "Writer",
        ],
    )

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[0] == "Imported: Override"
    assert source.read_text(encoding="utf-8") == original

    book_id = lines[1].removeprefix("Book ID: ")
    store = StateStore(workspace)
    book = store.load_book(book_id)
    assert book.title == "Override"
    assert book.authors == ["Writer"]
    assert book.source_mode == "file"
    assert book.source_ids


def test_cli_new_title_without_author_exits_with_book_author_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _prepare_workspace(tmp_path, monkeypatch)

    result = runner.invoke(app, ["new", "--title", "Demo"])

    assert result.exit_code == 2
    assert "Error: book_author_required" in result.stdout
    _assert_no_book_project(workspace)


def test_cli_new_positional_file_mode_still_prints_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _prepare_workspace(tmp_path, monkeypatch)
    source = tmp_path / "demo.txt"
    source.write_text("第一章\n正文\n", encoding="utf-8")

    result = runner.invoke(app, ["new", str(source)])

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    assert lines[0] == "Imported: demo"
    assert "Created:" not in result.stdout
    assert lines[1].startswith("Book ID: ")
    book_id = lines[1].removeprefix("Book ID: ")
    assert lines[2] == "Episode: E001"
    assert lines[3] == f"Next: bv next {book_id}"

    store = StateStore(workspace)
    book = store.load_book(book_id)
    assert book.source_mode == "file"
    assert book.source_ids
    assert store.load_episode(book_id, "E001").episode_id == "E001"
