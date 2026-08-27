from __future__ import annotations

import hashlib
import posixpath
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from ebooklib import epub

from .models import Chapter, ParsedBook, SourceAnchor
from .quality import assess_source_quality


_IGNORED_TAGS = {"script", "style", "noscript"}
_NAVIGATION_MARKERS = ("nav", "toc", "目录", "contents")
_COVER_MARKERS = ("cover", "封面")
_COPYRIGHT_MARKERS = ("copyright", "版权", "版权页", "版权信息", "colophon")
_PROMOTIONAL_MARKERS = (
    "advert",
    "广告",
    "广告页",
    "推广",
    "推广页",
    "限时推广",
    "subscribe",
    "下载更多",
)
_TRANSLATOR_PREFACE_MARKERS = ("translator preface", "译者序", "译者前言")
_PREFACE_MARKERS = ("preface", "foreword", "前言", "序言")
_ENCODING_RE = re.compile(br"<\?xml[^>]+encoding=[\"']([^\"']+)[\"']", re.IGNORECASE)


class EpubParseError(ValueError):
    """Raised when an EPUB cannot be safely read as structured text."""


@dataclass(frozen=True)
class _ExtractedDocument:
    title: str
    text: str
    start_paragraph: int | None
    end_paragraph: int | None


@dataclass(frozen=True)
class _ArchiveIndex:
    members: dict[str, str]
    package_directory: str


def parse_epub(path: Path) -> ParsedBook:
    """Parse a DRM-free EPUB in declared spine order into anchored chapters."""

    source = Path(path)
    archive = _validate_archive(source)
    try:
        book = epub.read_epub(str(source), {"raise_exceptions": True})
    except Exception as exc:
        raise EpubParseError(f"could not read EPUB: {source.name}") from exc

    navigation_titles = _navigation_titles(source, book, archive)
    chapters: list[Chapter] = []
    seen_resource_paths: set[str] = set()
    for entry in book.spine:
        item_id, linear = _spine_entry(entry)
        item = book.get_item_with_id(item_id)
        if item is None:
            raise EpubParseError(f"missing EPUB spine resource: {item_id}")
        if str(linear).lower() == "no":
            continue

        resource_path = _resource_path(item)
        if not _archive_contains(archive, resource_path):
            raise EpubParseError(f"missing EPUB resource: {resource_path}")
        if resource_path in seen_resource_paths:
            continue
        seen_resource_paths.add(resource_path)

        if not _is_xhtml(item) or _is_excluded_resource(item, resource_path):
            continue
        document = _extract_document(
            _archive_content(source, archive, resource_path), resource_path
        )
        if not document.text:
            continue
        if _document_category(item, resource_path, document.title) in {
            "navigation",
            "cover",
            "copyright",
            "promotional",
        }:
            continue

        title = navigation_titles.get(resource_path) or document.title
        chapter_id = str(item.get_id())
        anchor = SourceAnchor(
            source_type="epub",
            source_path=resource_path,
            chapter_id=chapter_id,
            start_paragraph=document.start_paragraph,
            end_paragraph=document.end_paragraph,
        )
        chapters.append(
            Chapter(
                chapter_id=chapter_id,
                title=title,
                text=document.text,
                anchor=anchor,
                sha256=hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
            )
        )

    full_text = "\n\n".join(chapter.text for chapter in chapters)
    return ParsedBook(
        source_path=str(source),
        encoding="utf-8",
        full_text=full_text,
        chapters=chapters,
        quality=assess_source_quality(full_text),
    )


def _validate_archive(source: Path) -> _ArchiveIndex:
    try:
        with ZipFile(source) as archive:
            if archive.testzip() is not None:
                raise EpubParseError(f"could not read EPUB: {source.name}")
            members: dict[str, str] = {}
            for name in archive.namelist():
                normalized = _normalize_archive_name(name)
                if normalized in members:
                    raise EpubParseError(f"duplicate EPUB resource: {normalized}")
                members[normalized] = name
            container_name = next(
                (
                    name
                    for normalized, name in members.items()
                    if normalized.lower() == "meta-inf/container.xml"
                ),
                None,
            )
            if container_name is None:
                raise EpubParseError(f"could not read EPUB container: {source.name}")
            try:
                container = ElementTree.fromstring(archive.read(container_name))
            except (KeyError, ElementTree.ParseError) as exc:
                raise EpubParseError(
                    f"could not read EPUB container: {source.name}"
                ) from exc
    except EpubParseError:
        raise
    except (BadZipFile, OSError) as exc:
        raise EpubParseError(f"could not read EPUB: {source.name}") from exc

    if any(name.lower() == "meta-inf/encryption.xml" for name in members):
        raise EpubParseError(f"encrypted EPUB is not supported: {source.name}")
    package_path = next(
        (
            element.attrib.get("full-path", "")
            for element in container.iter()
            if _local_name(element.tag) == "rootfile"
            and element.attrib.get("full-path")
        ),
        "",
    )
    normalized_package = _normalize_archive_name(package_path)
    if not normalized_package or normalized_package not in members:
        raise EpubParseError(f"could not read EPUB package: {source.name}")
    return _ArchiveIndex(
        members=members,
        package_directory=posixpath.dirname(normalized_package),
    )


def _navigation_titles(
    source: Path,
    book: epub.EpubBook,
    archive: _ArchiveIndex,
) -> dict[str, str]:
    titles: dict[str, str] = {}
    for item in book.get_items():
        resource_path = _resource_path(item)
        if not _is_navigation_item(item) or not _archive_contains(archive, resource_path):
            continue
        try:
            root = ElementTree.fromstring(
                _decode_xhtml(
                    _archive_content(source, archive, resource_path), resource_path
                )
            )
        except EpubParseError:
            raise
        except ElementTree.ParseError as exc:
            raise EpubParseError(f"could not read EPUB navigation: {resource_path}") from exc
        for element in root.iter():
            if _local_name(element.tag) != "a":
                continue
            href = element.attrib.get("href")
            title = _normalized_text(_visible_text(element))
            if not href or not title:
                continue
            target = _resolve_href(resource_path, href)
            titles.setdefault(target, title)
    ncx_base = next(
        (
            _resource_path(item)
            for item in book.get_items()
            if isinstance(item, epub.EpubNcx)
        ),
        "",
    )
    for target, title in _toc_navigation_titles(book.toc, ncx_base):
        titles.setdefault(target, title)
    return titles


def _toc_navigation_titles(
    entries: object,
    base_path: str,
) -> list[tuple[str, str]]:
    titles: list[tuple[str, str]] = []

    def visit(values: object) -> None:
        if not isinstance(values, (list, tuple)):
            values = (values,)
        for value in values:
            if isinstance(value, tuple):
                if value:
                    visit(value[0])
                if len(value) > 1:
                    visit(value[1])
                continue
            href = getattr(value, "href", "")
            title = _normalized_text(str(getattr(value, "title", "")))
            if href and title:
                target = _resolve_href(base_path, str(href))
                if target:
                    titles.append((target, title))

    visit(entries)
    return titles


def _spine_entry(entry: object) -> tuple[str, str]:
    if isinstance(entry, tuple):
        if not entry:
            raise EpubParseError("could not read EPUB spine entry")
        item_id = entry[0]
        linear = entry[1] if len(entry) > 1 else "yes"
    else:
        item_id = entry
        linear = "yes"
    if not item_id:
        raise EpubParseError("could not read EPUB spine entry")
    return str(item_id), str(linear)


def _resource_path(item: object) -> str:
    file_name = getattr(item, "file_name", "") or getattr(item, "get_name", lambda: "")()
    return str(PurePosixPath(str(file_name).replace("\\", "/")))


def _archive_contains(archive: _ArchiveIndex, resource_path: str) -> bool:
    return _archive_member(archive, resource_path) in archive.members


def _archive_content(
    source: Path,
    archive: _ArchiveIndex,
    resource_path: str,
) -> bytes:
    normalized = _archive_member(archive, resource_path)
    archive_name = archive.members.get(normalized)
    if archive_name is None:
        raise EpubParseError(f"missing EPUB resource: {resource_path}")
    try:
        with ZipFile(source) as archive:
            return archive.read(archive_name)
    except (BadZipFile, KeyError, OSError) as exc:
        raise EpubParseError(f"could not read EPUB resource: {resource_path}") from exc


def _archive_member(archive: _ArchiveIndex, resource_path: str) -> str:
    relative = unquote(resource_path).replace("\\", "/")
    if not relative or relative.startswith("/"):
        raise EpubParseError(f"unsafe EPUB resource path: {resource_path}")
    normalized = posixpath.normpath(
        posixpath.join(archive.package_directory, relative)
    )
    if normalized == ".." or normalized.startswith("../"):
        raise EpubParseError(f"unsafe EPUB resource path: {resource_path}")
    return normalized


def _normalize_archive_name(value: str) -> str:
    normalized = posixpath.normpath(str(value).replace("\\", "/"))
    if normalized in {"", "."} or normalized.startswith("/"):
        return ""
    if normalized == ".." or normalized.startswith("../"):
        return ""
    return normalized


def _is_xhtml(item: object) -> bool:
    media_type = str(getattr(item, "media_type", "")).lower()
    return media_type in {"application/xhtml+xml", "text/html"}


def _is_navigation_item(item: object) -> bool:
    properties = getattr(item, "properties", ())
    return isinstance(item, epub.EpubNav) or "nav" in properties


def _is_excluded_resource(item: object, resource_path: str) -> bool:
    return _document_category(item, resource_path, "") in {
        "navigation",
        "cover",
        "copyright",
        "promotional",
    }


def _document_category(item: object, resource_path: str, title: str) -> str:
    marker_text = " ".join(
        (
            resource_path.lower(),
            str(getattr(item, "title", "")).lower(),
            title.lower(),
            " ".join(str(value).lower() for value in getattr(item, "properties", ())),
        )
    )
    if _is_navigation_item(item):
        return "navigation"
    if _has_marker(marker_text, _COVER_MARKERS):
        return "cover"
    if _has_marker(marker_text, _COPYRIGHT_MARKERS):
        return "copyright"
    if _has_marker(marker_text, _PROMOTIONAL_MARKERS):
        return "promotional"
    if _has_marker(marker_text, _TRANSLATOR_PREFACE_MARKERS):
        return "translator_preface"
    if _has_marker(marker_text, _PREFACE_MARKERS):
        return "preface"
    return "body"


def _has_marker(value: str, markers: tuple[str, ...]) -> bool:
    for marker in markers:
        if re.search(
            rf"(?<![^\W_]){re.escape(marker)}(?![^\W_])",
            value,
        ):
            return True
    return False


def _extract_document(content: bytes, resource_path: str) -> _ExtractedDocument:
    try:
        root = ElementTree.fromstring(_decode_xhtml(content, resource_path))
    except EpubParseError:
        raise
    except ElementTree.ParseError as exc:
        raise EpubParseError(f"could not read EPUB XHTML: {resource_path}") from exc

    h1 = ""
    html_title = ""
    paragraphs: list[tuple[int, str]] = []
    paragraph_number = 0
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag == "h1" and not h1:
            h1 = _normalized_text(_visible_text(element))
        elif tag == "title" and not html_title:
            html_title = _normalized_text(_visible_text(element))
        elif tag == "p":
            paragraph_number += 1
            text = _normalized_text(_visible_text(element))
            if text:
                paragraphs.append((paragraph_number, text))

    text = "\n".join(value for _, value in paragraphs)
    return _ExtractedDocument(
        title=h1 or html_title or Path(resource_path).stem,
        text=text,
        start_paragraph=paragraphs[0][0] if paragraphs else None,
        end_paragraph=paragraphs[-1][0] if paragraphs else None,
    )


def _decode_xhtml(content: bytes, resource_path: str) -> str:
    encoding_match = _ENCODING_RE.search(content[:200])
    encoding = encoding_match.group(1).decode("ascii") if encoding_match else "utf-8"
    try:
        return content.decode(encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise EpubParseError(f"could not read EPUB XHTML: {resource_path}") from exc


def _visible_text(element: ElementTree.Element[str]) -> str:
    parts: list[str] = []

    def visit(node: ElementTree.Element[str]) -> None:
        if _local_name(node.tag) in _IGNORED_TAGS:
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            visit(child)
            if child.tail:
                parts.append(child.tail)

    visit(element)
    return "".join(parts)


def _normalized_text(value: str) -> str:
    return " ".join(unescape(value).replace("\u00a0", " ").split())


def _local_name(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1].lower()


def _resolve_href(base_path: str, href: str) -> str:
    decoded_path = unquote(urlsplit(href).path).replace("\\", "/")
    if not decoded_path:
        return ""
    return posixpath.normpath(str(PurePosixPath(base_path).parent / decoded_path)).lstrip("./")
