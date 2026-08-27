from pathlib import Path

import pymupdf
import pytest

from bv.ebook.pdf import PdfParseError, parse_pdf


def test_text_layer_pdf_preserves_page_anchors_and_heading_body(
    fixtures_dir: Path,
) -> None:
    parsed = parse_pdf(fixtures_dir / "books" / "text-layer.pdf")

    assert parsed.quality.blocking is False
    assert [chapter.title for chapter in parsed.chapters] == ["第一章", "第二章"]
    assert parsed.chapters[0].anchor.start_page == 1
    assert parsed.chapters[0].anchor.end_page == 1
    assert parsed.chapters[1].anchor.start_page == 2
    assert "第一页正文" in parsed.chapters[0].text
    assert "第二页正文" in parsed.full_text


def test_scanned_placeholder_pdf_is_rejected_for_missing_text_layer(
    fixtures_dir: Path,
) -> None:
    parsed = parse_pdf(fixtures_dir / "books" / "scanned-placeholder.pdf")

    assert parsed.quality.blocking is True
    assert "pdf_no_text_layer" in parsed.quality.codes


def test_pdf_block_extraction_is_vertical_then_horizontal(tmp_path: Path) -> None:
    source = tmp_path / "block-order.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((250, 80), "right top")
    page.insert_text((50, 80), "left top")
    page.insert_text((50, 160), "left lower")
    document.save(source)
    document.close()

    parsed = parse_pdf(source)

    assert parsed.full_text == "left top\nright top\nleft lower"


def test_repeated_normalized_pdf_pages_are_blocking(tmp_path: Path) -> None:
    source = tmp_path / "repeated-pages.pdf"
    document = pymupdf.open()
    for page_text in ("独有页", "重复页", "重复页"):
        page = document.new_page()
        page.insert_text((50, 80), page_text)
    document.save(source)
    document.close()

    parsed = parse_pdf(source)

    assert parsed.quality.blocking is True
    assert "pdf_repeated_pages" in parsed.quality.codes


def test_persistent_alternating_two_column_pdf_is_blocking(tmp_path: Path) -> None:
    source = tmp_path / "columns.pdf"
    document = pymupdf.open()
    for number in range(2):
        page = document.new_page(width=600, height=800)
        page.insert_text((50, 80), f"left {number} top")
        page.insert_text((350, 100), f"right {number} top")
        page.insert_text((50, 140), f"left {number} lower")
        page.insert_text((350, 160), f"right {number} lower")
    document.save(source)
    document.close()

    parsed = parse_pdf(source)

    assert parsed.quality.blocking is True
    assert "pdf_multicolumn_suspected" in parsed.quality.codes


def test_grouped_two_column_blocks_are_blocking_before_visual_sort(
    tmp_path: Path,
) -> None:
    source = tmp_path / "grouped-columns.pdf"
    document = pymupdf.open()
    for number in range(2):
        page = document.new_page(width=600, height=800)
        page.insert_text((50, 80), f"left {number} top")
        page.insert_text((50, 140), f"left {number} lower")
        page.insert_text((350, 100), f"right {number} top")
        page.insert_text((350, 160), f"right {number} lower")
    document.save(source)
    document.close()

    parsed = parse_pdf(source)

    assert parsed.full_text.splitlines()[:4] == [
        "left 0 top",
        "right 0 top",
        "left 0 lower",
        "right 0 lower",
    ]
    assert "pdf_multicolumn_suspected" in parsed.quality.codes


def test_two_columns_with_full_width_heading_are_still_blocking(
    tmp_path: Path,
) -> None:
    source = tmp_path / "headed-columns.pdf"
    document = pymupdf.open()
    for number in range(2):
        page = document.new_page(width=600, height=800)
        page.insert_text((250, 40), f"heading {number}")
        page.insert_text((50, 80), f"left {number} top")
        page.insert_text((50, 140), f"left {number} lower")
        page.insert_text((350, 100), f"right {number} top")
        page.insert_text((350, 160), f"right {number} lower")
    document.save(source)
    document.close()

    parsed = parse_pdf(source)

    assert "pdf_multicolumn_suspected" in parsed.quality.codes


def test_corrupt_pdf_fails_explicitly(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.pdf"
    source.write_bytes(b"not a PDF")

    with pytest.raises(PdfParseError, match="could not read PDF"):
        parse_pdf(source)
