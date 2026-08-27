from __future__ import annotations

from pathlib import Path

import pymupdf


BOOKS_DIRECTORY = Path(__file__).parent / "books"
_FIXED_METADATA = {
    "title": "BV PDF parser fixture",
    "author": "BV Workstation tests",
    "subject": "Private-safe parser fixture",
    "keywords": "",
    "creator": "",
    "producer": "",
    "creationDate": "D:20200101000000Z",
    "modDate": "D:20200101000000Z",
}


def build_pdf_fixtures(destination: Path = BOOKS_DIRECTORY) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    _build_text_layer_pdf(destination / "text-layer.pdf")
    _build_scanned_placeholder_pdf(destination / "scanned-placeholder.pdf")


def _save(document: pymupdf.Document, path: Path) -> None:
    document.set_metadata(_FIXED_METADATA)
    document.save(path, garbage=4, deflate=True, no_new_id=True)
    document.close()


def _build_text_layer_pdf(path: Path) -> None:
    document = pymupdf.open()
    page_one = document.new_page()
    page_one.insert_text((72, 72), "第一章", fontname="china-s", fontsize=16)
    page_one.insert_textbox(
        pymupdf.Rect(72, 110, 523, 760),
        _body_text("第一页正文：", 1),
        fontname="china-s",
        fontsize=11,
    )
    page_two = document.new_page()
    page_two.insert_text((72, 72), "第二章", fontname="china-s", fontsize=16)
    page_two.insert_textbox(
        pymupdf.Rect(72, 110, 523, 760),
        _body_text("第二页正文：", 101),
        fontname="china-s",
        fontsize=11,
    )
    _save(document, path)


def _body_text(prefix: str, first_number: int) -> str:
    return prefix + "".join(
        f"公开解析样本文本段落{number:03d}用于验证文本层提取。"
        for number in range(first_number, first_number + 32)
    )


def _build_scanned_placeholder_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.draw_rect(
        pymupdf.Rect(72, 72, 523, 770),
        color=(0.25, 0.25, 0.25),
        fill=(0.9, 0.9, 0.9),
    )
    _save(document, path)


if __name__ == "__main__":
    build_pdf_fixtures()
