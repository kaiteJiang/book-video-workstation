from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .models import Chapter, ParsedBook, SourceAnchor, SourceQualityReport
from .quality import assess_source_quality


_HEADING_RE = re.compile(
    r"^(第[一二三四五六七八九十百千0-9]+[章节卷部篇]|"
    r"chapter\s+\d+|序言|前言|后记|尾声)\s*.*$",
    re.IGNORECASE,
)
_MIN_TEXT_LAYER_CHARS = 50


class PdfParseError(ValueError):
    """Raised when a PDF cannot be safely read as text."""


@dataclass(frozen=True)
class _TextBlock:
    page_number: int
    text: str


@dataclass(frozen=True)
class _PageText:
    page_number: int
    text: str
    has_two_columns: bool


def parse_pdf(path: Path) -> ParsedBook:
    """Extract a text-layer PDF in stable visual block order."""

    source = Path(path)
    try:
        document = pymupdf.open(source)
    except Exception as exc:
        raise PdfParseError(f"could not read PDF: {source.name}") from exc

    try:
        if not document.is_pdf:
            raise PdfParseError(f"could not read PDF: {source.name}")
        pages = [_extract_page(page, number) for number, page in enumerate(document, 1)]
    except PdfParseError:
        raise
    except Exception as exc:
        raise PdfParseError(f"could not read PDF: {source.name}") from exc
    finally:
        document.close()

    full_text = "\n".join(page.text for page in pages).strip()
    quality = _assess_pdf_quality(full_text, pages)
    return ParsedBook(
        source_path=str(source),
        encoding="utf-8",
        full_text=full_text,
        chapters=_chapters_from_pages(source, pages),
        quality=quality,
    )


def _extract_page(page: pymupdf.Page, page_number: int) -> _PageText:
    raw_blocks = [
        block
        for block in page.get_text("blocks", sort=False)
        if str(block[4]).strip()
    ]
    has_two_columns = _has_two_columns(raw_blocks, page.rect.width)
    ordered_lines = sorted(_page_text_lines(page), key=_visual_order_key)
    text = "\n".join(line[4] for line in ordered_lines)
    return _PageText(
        page_number=page_number,
        text=text,
        has_two_columns=has_two_columns,
    )


def _page_text_lines(page: pymupdf.Page) -> list[tuple[float, float, float, float, str]]:
    lines: list[tuple[float, float, float, float, str]] = []
    for block in page.get_text("dict", sort=False)["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            lines.append((x0, y0, x1, y1, text))
    return lines


def _visual_order_key(line: tuple[float, float, float, float, str]) -> tuple:
    x0, y0, x1, y1, text = line
    return (
        round(float(y0), 1),
        round(float(x0), 1),
        round(float(y1), 1),
        round(float(x1), 1),
        text,
    )


def _has_two_columns(blocks: list[tuple], page_width: float) -> bool:
    if page_width <= 0 or len(blocks) < 4:
        return False

    columns: list[str] = []
    for block in blocks:
        midpoint = (float(block[0]) + float(block[2])) / 2
        if midpoint < page_width * 0.45:
            columns.append("left")
        elif midpoint > page_width * 0.55:
            columns.append("right")
    if columns.count("left") < 2 or columns.count("right") < 2:
        return False
    return True


def _assess_pdf_quality(
    full_text: str,
    pages: list[_PageText],
) -> SourceQualityReport:
    shared = assess_source_quality(full_text)
    codes = list(shared.codes)
    if len("".join(full_text.split())) < _MIN_TEXT_LAYER_CHARS:
        codes.append("pdf_no_text_layer")
    if _has_repeated_pages(pages):
        codes.append("pdf_repeated_pages")
    if _has_persistent_two_columns(pages):
        codes.append("pdf_multicolumn_suspected")
    return SourceQualityReport(
        blocking=bool(codes),
        codes=codes,
        warnings=list(shared.warnings),
    )


def _has_repeated_pages(pages: list[_PageText]) -> bool:
    if not pages:
        return False
    normalized_pages = [
        " ".join(page.text.split()).casefold() for page in pages if page.text.strip()
    ]
    repeats_after_first = sum(
        count - 1 for count in Counter(normalized_pages).values() if count > 1
    )
    return repeats_after_first / len(pages) > 0.30


def _has_persistent_two_columns(pages: list[_PageText]) -> bool:
    two_column_pages = sum(page.has_two_columns for page in pages)
    return two_column_pages >= 2 and two_column_pages / max(len(pages), 1) > 0.5


def _chapters_from_pages(source: Path, pages: list[_PageText]) -> list[Chapter]:
    blocks = [
        _TextBlock(page.page_number, text)
        for page in pages
        for text in page.text.splitlines()
        if text.strip()
    ]
    if not blocks:
        return []

    chapters: list[Chapter] = []
    current_title = "Part 1"
    current_start_page = blocks[0].page_number
    current_blocks: list[_TextBlock] = []
    for block in blocks:
        if _HEADING_RE.fullmatch(block.text.strip()):
            _append_chapter(
                chapters,
                source,
                current_title,
                current_start_page,
                current_blocks,
            )
            current_title = block.text.strip()
            current_start_page = block.page_number
            current_blocks = []
            continue
        current_blocks.append(block)
    _append_chapter(
        chapters,
        source,
        current_title,
        current_start_page,
        current_blocks,
    )
    return chapters


def _append_chapter(
    chapters: list[Chapter],
    source: Path,
    title: str,
    start_page: int,
    blocks: list[_TextBlock],
) -> None:
    text = "\n".join(block.text for block in blocks).rstrip()
    if not text:
        return
    chapter_id = f"pdf-{len(chapters) + 1:04d}"
    chapters.append(
        Chapter(
            chapter_id=chapter_id,
            title=title,
            text=text,
            anchor=SourceAnchor(
                source_type="pdf",
                source_path=str(source),
                chapter_id=chapter_id,
                start_page=start_page,
                end_page=blocks[-1].page_number,
            ),
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
    )
