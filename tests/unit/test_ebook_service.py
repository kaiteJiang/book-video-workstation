import os
from pathlib import Path
import subprocess

import pytest

from bv.ebook.service import parse_book, persist_parsed_book


def test_parse_book_dispatches_only_supported_formats(fixtures_dir: Path) -> None:
    txt = parse_book(fixtures_dir / "books" / "utf8.txt")
    epub = parse_book(fixtures_dir / "books" / "demo.epub")
    pdf = parse_book(fixtures_dir / "books" / "text-layer.pdf")

    assert txt.chapters[0].anchor.source_type == "txt"
    assert epub.chapters[0].anchor.source_type == "epub"
    assert pdf.chapters[0].anchor.source_type == "pdf"


def test_parse_book_rejects_unsupported_suffix_explicitly(tmp_path: Path) -> None:
    source = tmp_path / "book.docx"
    source.write_bytes(b"not supported")

    with pytest.raises(ValueError, match="unsupported ebook suffix"):
        parse_book(source)


def test_persist_parsed_book_writes_deterministic_chapters_and_quality(
    fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    parsed = parse_book(fixtures_dir / "books" / "text-layer.pdf")

    status = persist_parsed_book(parsed, tmp_path / "episode")

    destination = tmp_path / "episode"
    assert status == "source_parsed"
    assert (destination / "analysis" / "source_quality.json").read_text(
        encoding="utf-8"
    ) == (
        "{\n"
        '  "blocking": false,\n'
        '  "codes": [],\n'
        '  "warnings": []\n'
        "}"
    )
    assert (destination / "chapters" / "index.json").exists()
    first_chapter = (destination / "chapters" / "001.md").read_text(
        encoding="utf-8"
    )
    assert first_chapter.startswith("# 第一章\n\n第一页正文：")
    status_again = persist_parsed_book(parsed, destination)
    assert status_again == "source_parsed"
    assert (destination / "chapters" / "001.md").read_text(
        encoding="utf-8"
    ) == first_chapter
    assert (destination / "chapters" / "002.md").exists()


def test_persist_blocking_book_writes_only_quality_report(
    fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    parsed = parse_book(fixtures_dir / "books" / "scanned-placeholder.pdf")

    status = persist_parsed_book(parsed, tmp_path / "episode")

    destination = tmp_path / "episode"
    assert status == "source_invalid"
    assert (destination / "analysis" / "source_quality.json").exists()
    assert not (destination / "chapters").exists()
    assert not (destination / "analysis" / "chapter_inputs").exists()


def test_persist_blocking_book_removes_previous_chapter_inputs(
    fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "episode"
    valid = parse_book(fixtures_dir / "books" / "text-layer.pdf")
    invalid = parse_book(fixtures_dir / "books" / "scanned-placeholder.pdf")
    persist_parsed_book(valid, destination)
    assert (destination / "chapters" / "001.md").exists()

    status = persist_parsed_book(invalid, destination)

    assert status == "source_invalid"
    assert not (destination / "chapters").exists()


def test_persist_fewer_chapters_removes_stale_numbered_markdown(
    fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "episode"
    parsed = parse_book(fixtures_dir / "books" / "text-layer.pdf")
    persist_parsed_book(parsed, destination)
    assert (destination / "chapters" / "002.md").exists()

    one_chapter = parsed.model_copy(update={"chapters": parsed.chapters[:1]})
    status = persist_parsed_book(one_chapter, destination)

    assert status == "source_parsed"
    assert (destination / "chapters" / "001.md").exists()
    assert not (destination / "chapters" / "002.md").exists()


@pytest.mark.parametrize("redirect_name", ["analysis", "chapters"])
def test_persist_rejects_redirected_output_directories_before_writing(
    fixtures_dir: Path,
    tmp_path: Path,
    redirect_name: str,
) -> None:
    parsed = parse_book(fixtures_dir / "books" / "text-layer.pdf")
    destination = tmp_path / "episode"
    destination.mkdir()
    outside = tmp_path / f"outside-{redirect_name}"
    outside.mkdir()
    redirect = destination / redirect_name
    _make_directory_redirect(redirect, outside)

    try:
        with pytest.raises(ValueError, match="redirect|regular directory"):
            persist_parsed_book(parsed, destination)
        assert list(outside.iterdir()) == []
    finally:
        _remove_directory_redirect(redirect)


def _make_directory_redirect(link: Path, target: Path) -> None:
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        return
    link.symlink_to(target, target_is_directory=True)


def _remove_directory_redirect(link: Path) -> None:
    if not link.exists() and not link.is_symlink():
        return
    if os.name == "nt":
        os.rmdir(link)
        return
    link.unlink()
