from pathlib import Path
import tracemalloc

from bv.ebook.quality import assess_source_quality
from bv.ebook.txt import parse_txt


def test_utf8_txt_preserves_line_anchors(fixtures_dir: Path) -> None:
    parsed = parse_txt(fixtures_dir / "books" / "utf8.txt", min_complete_chars=1)

    assert parsed.encoding.lower() == "utf-8"
    assert parsed.full_text == "第一章\n第一段正文\n第二段正文\n第二章\n第二章正文\n"
    assert parsed.chapters[0].title == "第一章"
    assert parsed.chapters[0].anchor.start_line == 1
    assert parsed.chapters[0].anchor.end_line == 3
    assert "第一段正文" in parsed.chapters[0].text
    assert parsed.chapters[0].text == "第一段正文\n第二段正文"
    assert parsed.chapters[0].anchor.source_type == "txt"


def test_gb18030_txt_is_decoded(fixtures_dir: Path) -> None:
    parsed = parse_txt(fixtures_dir / "books" / "gb18030.txt", min_complete_chars=1)

    assert parsed.encoding.lower() in {"gb18030", "gbk"}
    assert parsed.full_text == "第一章\n中文正文"
    assert "中文正文" in parsed.full_text


def test_utf8_bom_is_removed_and_chapter_heading_is_recognized(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbf" + "第一章\r\n正文".encode("utf-8"))

    parsed = parse_txt(path, min_complete_chars=1)

    assert parsed.encoding == "utf-8-sig"
    assert parsed.full_text == "第一章\n正文"
    assert "\ufeff" not in parsed.full_text
    assert parsed.chapters[0].title == "第一章"
    assert parsed.chapters[0].text == "正文"


def test_replacement_character_rate_blocks_garbled_text(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("�" * 200 + "正文", encoding="utf-8")

    parsed = parse_txt(path, min_complete_chars=1)

    assert parsed.quality.blocking
    assert "replacement_character_rate" in parsed.quality.codes


def test_empty_normalized_text_is_blocking(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(" \r\n\t", encoding="utf-8")

    parsed = parse_txt(path)

    assert parsed.quality.blocking
    assert "empty_normalized_text" in parsed.quality.codes


def test_complete_book_threshold_defaults_to_500_and_is_configurable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "short.txt"
    path.write_text("有效正文", encoding="utf-8")

    default_report = parse_txt(path)
    fixture_report = parse_txt(path, min_complete_chars=4)

    assert "too_short_for_complete_book" in default_report.quality.codes
    assert not fixture_report.quality.blocking


def test_repeated_200_character_window_blocks_source(tmp_path: Path) -> None:
    path = tmp_path / "repeated.txt"
    window = "甲" * 200
    path.write_text(window * 10, encoding="utf-8")

    parsed = parse_txt(path, min_complete_chars=1)

    assert parsed.quality.blocking
    assert "repeated_window" in parsed.quality.codes


def test_repeated_window_detection_includes_windows_after_paragraph_breaks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "repeated-paragraphs.txt"
    path.write_text(("乙" * 200 + "\n\n") * 10, encoding="utf-8")

    parsed = parse_txt(path, min_complete_chars=1)

    assert parsed.quality.blocking
    assert "repeated_window" in parsed.quality.codes


def test_repeated_window_detection_counts_overlapping_occurrences(tmp_path: Path) -> None:
    path = tmp_path / "repeated-overlap.txt"
    path.write_text("丙" * 209, encoding="utf-8")

    parsed = parse_txt(path, min_complete_chars=1)

    assert parsed.quality.blocking
    assert "repeated_window" in parsed.quality.codes


def test_large_unique_text_uses_bounded_memory_for_repeated_window_gate() -> None:
    text = "".join(map(chr, range(1_000_000)))

    tracemalloc.start()
    try:
        report = assess_source_quality(text, min_complete_chars=1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert "repeated_window" not in report.codes
    assert peak < 64 * 1024 * 1024


def test_headingless_text_splits_at_paragraph_boundaries_with_traceable_offsets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "headingless.txt"
    path.write_text("甲乙丙丁\n\n戊己庚辛\n\n壬癸子丑", encoding="utf-8")

    parsed = parse_txt(path, min_complete_chars=1, chunk_char_ceiling=10)

    assert [chapter.title for chapter in parsed.chapters] == ["Part 1", "Part 2"]
    assert [chapter.text for chapter in parsed.chapters] == ["甲乙丙丁\n\n戊己庚辛", "壬癸子丑"]
    assert all(len(chapter.text) <= 10 for chapter in parsed.chapters)
    for chapter in parsed.chapters:
        anchor = chapter.anchor
        assert anchor.start_char is not None
        assert anchor.end_char is not None
        assert parsed.full_text[anchor.start_char : anchor.end_char] == chapter.text
        assert anchor.start_line is not None
        assert anchor.end_line is not None


def test_text_before_first_heading_is_preserved_as_a_traceable_chunk(
    tmp_path: Path,
) -> None:
    path = tmp_path / "front-matter.txt"
    path.write_text("出版说明\n\n第一章\n章节一正文\n第二章\n章节二正文", encoding="utf-8")

    parsed = parse_txt(path, min_complete_chars=1)

    assert [chapter.title for chapter in parsed.chapters] == ["Part 1", "第一章", "第二章"]
    assert [chapter.text for chapter in parsed.chapters] == ["出版说明", "章节一正文", "章节二正文"]
    assert parsed.chapters[1].anchor.start_line == 3
    assert parsed.chapters[2].anchor.start_line == 5


def test_whitespace_before_first_heading_does_not_create_empty_part(
    tmp_path: Path,
) -> None:
    path = tmp_path / "whitespace-before-preface.txt"
    path.write_text("\n \t\n前言\n导言正文", encoding="utf-8")

    parsed = parse_txt(path, min_complete_chars=1)

    assert [chapter.title for chapter in parsed.chapters] == ["前言"]
    assert parsed.chapters[0].text == "导言正文"
    assert parsed.chapters[0].anchor.start_line == 3
    assert all(chapter.text.strip() for chapter in parsed.chapters)


def test_consecutive_headings_skip_empty_chapter_and_keep_body_anchor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "consecutive-headings.txt"
    path.write_text("第一卷\n第一章\n正文", encoding="utf-8")

    parsed = parse_txt(path, min_complete_chars=1)

    assert [chapter.title for chapter in parsed.chapters] == ["第一章"]
    assert parsed.chapters[0].text == "正文"
    assert parsed.chapters[0].anchor.start_line == 2
    assert all(chapter.text.strip() for chapter in parsed.chapters)
