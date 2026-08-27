from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Literal

from bv.core.atomic import atomic_write_json

from .epub import parse_epub
from .models import ParsedBook
from .pdf import parse_pdf
from .txt import parse_txt


_NUMBERED_CHAPTER_RE = re.compile(r"\d{3}\.md")
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def parse_book(path: Path) -> ParsedBook:
    """Dispatch a supported local ebook source to its parser."""

    source = Path(path)
    parsers = {
        ".txt": parse_txt,
        ".epub": parse_epub,
        ".pdf": parse_pdf,
    }
    parser = parsers.get(source.suffix.lower())
    if parser is None:
        raise ValueError(f"unsupported ebook suffix: {source.suffix or '<none>'}")
    return parser(source)


def persist_parsed_book(
    parsed: ParsedBook,
    destination: Path,
) -> Literal["source_parsed", "source_invalid"]:
    """Persist parser artifacts without mutating EpisodeState."""

    target = Path(destination)
    analysis_directory = target / "analysis"
    chapters_directory = target / "chapters"
    _validate_output_directory(target, "destination")
    _validate_output_directory(analysis_directory, "analysis output")
    _validate_output_directory(chapters_directory, "chapters output")
    atomic_write_json(
        analysis_directory / "source_quality.json",
        parsed.quality.model_dump(mode="json"),
    )
    if parsed.quality.blocking:
        _clear_chapter_outputs(chapters_directory)
        return "source_invalid"

    atomic_write_json(
        chapters_directory / "index.json",
        {
            "chapters": [
                {
                    "chapter_id": chapter.chapter_id,
                    "title": chapter.title,
                    "sha256": chapter.sha256,
                    "anchor": chapter.anchor.model_dump(mode="json"),
                }
                for chapter in parsed.chapters
            ]
        },
    )
    for number, chapter in enumerate(parsed.chapters, 1):
        _atomic_write_text(
            chapters_directory / f"{number:03d}.md",
            f"# {chapter.title}\n\n{chapter.text.rstrip()}\n",
        )
    _remove_stale_chapter_markdown(chapters_directory, len(parsed.chapters))
    return "source_parsed"


def _clear_chapter_outputs(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if _is_redirect(path) or not path.is_dir():
        raise ValueError("chapters output is not a regular directory")
    entries = list(path.iterdir())
    if any(
        entry.name != "index.json" and not _NUMBERED_CHAPTER_RE.fullmatch(entry.name)
        for entry in entries
    ):
        raise ValueError("chapters output contains unknown artifacts")
    for entry in entries:
        if _is_redirect(entry) or not entry.is_file():
            raise ValueError("chapter output is not a regular file")
        entry.unlink()
    path.rmdir()


def _validate_output_directory(path: Path, label: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if _is_redirect(path):
        raise ValueError(f"{label} is an output redirect")
    if not path.is_dir():
        raise ValueError(f"{label} is not a regular directory")


def _remove_stale_chapter_markdown(path: Path, chapter_count: int) -> None:
    expected = {f"{number:03d}.md" for number in range(1, chapter_count + 1)}
    for entry in path.iterdir():
        if not _NUMBERED_CHAPTER_RE.fullmatch(entry.name) or entry.name in expected:
            continue
        if _is_redirect(entry) or not entry.is_file():
            raise ValueError("chapter output is not a regular file")
        entry.unlink()


def _is_redirect(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _atomic_write_text(path: Path, value: str) -> None:
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
