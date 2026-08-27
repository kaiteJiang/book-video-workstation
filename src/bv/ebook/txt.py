import hashlib
import re
from pathlib import Path

from charset_normalizer import from_bytes

from .models import Chapter, ParsedBook, SourceAnchor
from .quality import DEFAULT_MIN_COMPLETE_CHARS, assess_source_quality


CHAPTER_RE = re.compile(
    r"^(第[一二三四五六七八九十百千0-9]+[章节卷部篇]|"
    r"chapter\s+\d+|序言|前言|后记|尾声)\s*.*$",
    re.IGNORECASE,
)
DEFAULT_CHUNK_CHAR_CEILING = 20_000


def parse_txt(
    path: Path,
    *,
    min_complete_chars: int = DEFAULT_MIN_COMPLETE_CHARS,
    chunk_char_ceiling: int = DEFAULT_CHUNK_CHAR_CEILING,
) -> ParsedBook:
    """Decode a TXT source without rewriting its content and attach anchors."""

    if min_complete_chars < 0:
        raise ValueError("min_complete_chars must be non-negative")
    if chunk_char_ceiling < 1:
        raise ValueError("chunk_char_ceiling must be positive")

    text, encoding = _decode_txt(path.read_bytes())
    full_text = _normalize_newlines(text)
    chapters = _chapters_from_text(path, full_text, chunk_char_ceiling)
    return ParsedBook(
        source_path=str(path),
        encoding=encoding,
        full_text=full_text,
        chapters=chapters,
        quality=assess_source_quality(
            full_text,
            min_complete_chars=min_complete_chars,
        ),
    )


def _decode_txt(raw: bytes) -> tuple[str, str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="strict"), "utf-8-sig"
    try:
        return raw.decode("utf-8", errors="strict"), "utf-8"
    except UnicodeDecodeError:
        pass

    match = from_bytes(raw).best()
    if match is None or not match.encoding:
        return raw.decode("utf-8", errors="replace"), "utf-8"

    encoding = match.encoding
    try:
        text = raw.decode(encoding, errors="strict")
    except UnicodeDecodeError:
        text = raw.decode(encoding, errors="replace")

    gb_match = from_bytes(raw, cp_isolation=["gb18030", "gbk"]).best()
    if gb_match is not None and gb_match.encoding:
        gb_text = raw.decode(gb_match.encoding, errors="strict")
        if _cjk_character_count(gb_text) > _cjk_character_count(text):
            return gb_text, gb_match.encoding
    return text, encoding


def _cjk_character_count(text: str) -> int:
    return sum("\u3400" <= character <= "\u9fff" for character in text)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _chapters_from_text(
    path: Path,
    text: str,
    chunk_char_ceiling: int,
) -> list[Chapter]:
    lines = list(_lines_with_offsets(text))
    heading_indexes = [
        index for index, (_, line, _) in enumerate(lines) if CHAPTER_RE.fullmatch(line)
    ]
    if not heading_indexes:
        return _fallback_chunks(path, text, chunk_char_ceiling)

    chapter_specs: list[tuple[str, int, int, int | None]] = []
    first_heading = heading_indexes[0]
    if lines[first_heading][0] > 0:
        chapter_specs.append(("Part 1", 0, lines[first_heading][0], None))

    for heading_position, line_index in enumerate(heading_indexes):
        heading_start, heading, heading_end = lines[line_index]
        next_heading_start = (
            lines[heading_indexes[heading_position + 1]][0]
            if heading_position + 1 < len(heading_indexes)
            else len(text)
        )
        chapter_specs.append(
            (heading.strip(), heading_start, next_heading_start, heading_end)
        )

    chapters: list[Chapter] = []
    for title, start_char, end_char, text_start in chapter_specs:
        content_start = start_char if text_start is None else text_start
        if not text[content_start:end_char].strip():
            continue
        chapters.append(
            _chapter(
                path,
                text,
                title,
                start_char,
                end_char,
                text_start=text_start,
                chapter_number=len(chapters) + 1,
            )
        )
    return chapters


def _lines_with_offsets(text: str):
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line[:-1] if raw_line.endswith("\n") else raw_line
        end = offset + len(raw_line)
        yield offset, line, end
        offset = end
    if not text:
        return


def _fallback_chunks(path: Path, text: str, chunk_char_ceiling: int) -> list[Chapter]:
    paragraphs = [(match.start(), match.end()) for match in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text)]
    if not paragraphs:
        return [_chapter(path, text, "Part 1", 0, len(text))] if text else []

    spans: list[tuple[int, int]] = []
    start, end = paragraphs[0]
    for paragraph_start, paragraph_end in paragraphs[1:]:
        if paragraph_end - start <= chunk_char_ceiling:
            end = paragraph_end
            continue
        spans.extend(_split_span(start, end, chunk_char_ceiling))
        start, end = paragraph_start, paragraph_end
    spans.extend(_split_span(start, end, chunk_char_ceiling))

    chapters: list[Chapter] = []
    for start, end in spans:
        chapter = _chapter(
            path,
            text,
            f"Part {len(chapters) + 1}",
            start,
            end,
            chapter_number=len(chapters) + 1,
        )
        if chapter.text.strip():
            chapters.append(chapter)
    return chapters


def _split_span(start: int, end: int, chunk_char_ceiling: int) -> list[tuple[int, int]]:
    return [
        (chunk_start, min(chunk_start + chunk_char_ceiling, end))
        for chunk_start in range(start, end, chunk_char_ceiling)
    ]


def _chapter(
    path: Path,
    full_text: str,
    title: str,
    start_char: int,
    end_char: int,
    *,
    text_start: int | None = None,
    chapter_number: int = 1,
) -> Chapter:
    chapter_id = f"txt-{chapter_number:04d}"
    content_start = start_char if text_start is None else text_start
    text = full_text[content_start:end_char].rstrip("\n")
    return Chapter(
        chapter_id=chapter_id,
        title=title,
        text=text,
        anchor=SourceAnchor(
            source_type="txt",
            source_path=str(path),
            chapter_id=chapter_id,
            start_line=_line_number(full_text, start_char),
            end_line=_line_number(full_text, max(start_char, end_char - 1)),
            start_char=start_char,
            end_char=end_char,
        ),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1
