import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_STORED, ZipFile

import pytest
from ebooklib import epub

from bv.ebook.epub import parse_epub


def test_epub_follows_spine_and_preserves_paragraph_anchor(
    fixtures_dir: Path,
) -> None:
    parsed = parse_epub(fixtures_dir / "books" / "demo.epub")

    assert [chapter.title for chapter in parsed.chapters] == ["第一章", "第二章"]
    assert parsed.chapters[0].anchor.source_path.endswith("chapter1.xhtml")
    assert parsed.chapters[0].anchor.start_paragraph == 1
    assert "正文一" in parsed.chapters[0].text


def test_epub_uses_tuple_spine_order_once_and_skips_non_linear_and_ads(
    tmp_path: Path,
) -> None:
    path = _write_epub(
        tmp_path / "spine.epub",
        resources={
            "nav.xhtml": _nav(
                ("chapter1.xhtml", "导航第一章"),
                ("chapter2.xhtml", "导航第二章"),
            ),
            "chapter1.xhtml": _document("文档第一章", "第一章", "正文一"),
            "chapter2.xhtml": _document("文档第二章", "第二章", "正文二"),
            "appendix.xhtml": _document("附录", "附录", "不应出现的附录"),
            "advertisement.xhtml": _document("推广", "限时推广", "不应出现的广告"),
            "empty.xhtml": _document("空白", "空白", "   "),
        },
        spine=(
            ("nav", "no"),
            ("chapter2", "yes"),
            ("chapter1", "yes"),
            ("chapter1", "yes"),
            ("appendix", "no"),
            ("advertisement", "yes"),
            ("empty", "yes"),
        ),
    )

    parsed = parse_epub(path)

    assert [chapter.title for chapter in parsed.chapters] == [
        "导航第二章",
        "导航第一章",
    ]
    assert [chapter.anchor.source_path for chapter in parsed.chapters] == [
        "chapter2.xhtml",
        "chapter1.xhtml",
    ]
    assert "不应出现的附录" not in parsed.full_text
    assert "不应出现的广告" not in parsed.full_text


def test_epub_strips_non_content_tags_and_keeps_paragraph_order(tmp_path: Path) -> None:
    path = _write_epub(
        tmp_path / "cleaning.epub",
        resources={
            "chapter.xhtml": """
                <html xmlns=\"http://www.w3.org/1999/xhtml\">
                  <head><title>文档标题</title><style>样式内容</style></head>
                  <body><h1>文档标题</h1><script>脚本内容</script>
                    <p>正文甲 <em>保留</em></p><noscript>后备内容</noscript>
                    <p>正文乙</p></body>
                </html>
            """,
        },
        spine=(("chapter", "yes"),),
    )

    chapter = parse_epub(path).chapters[0]

    assert chapter.title == "文档标题"
    assert chapter.text == "正文甲 保留\n正文乙"
    assert chapter.anchor.start_paragraph == 1
    assert chapter.anchor.end_paragraph == 2
    assert "脚本内容" not in chapter.text
    assert "样式内容" not in chapter.text
    assert "后备内容" not in chapter.text


def test_epub_falls_back_from_h1_to_document_title(tmp_path: Path) -> None:
    path = _write_epub(
        tmp_path / "title-fallback.epub",
        resources={
            "h1.xhtml": _document("HTML 标题一", "H1 标题一", "正文一"),
            "title.xhtml": _document("HTML 标题二", None, "正文二"),
        },
        spine=(("h1", "yes"), ("title", "yes")),
    )

    parsed = parse_epub(path)

    assert [chapter.title for chapter in parsed.chapters] == ["H1 标题一", "HTML 标题二"]


def test_epub_uses_ncx_navigation_title_when_epub3_nav_is_absent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ncx-only.epub"
    book = epub.EpubBook()
    book.set_identifier("ncx-only")
    book.set_title("NCX fixture")
    book.set_language("zh")
    chapter = epub.EpubHtml(
        uid="chapter1",
        title="清单标题",
        file_name="chapter.xhtml",
    )
    chapter.content = _document("HTML 标题", "H1 标题", "正文" * 200)
    book.add_item(chapter)
    book.toc = (epub.Link("chapter.xhtml", "NCX 导航标题", "chapter1"),)
    book.add_item(epub.EpubNcx())
    book.spine = [chapter]
    epub.write_epub(str(path), book, {"raise_exceptions": True})

    parsed = parse_epub(path)

    assert [item.title for item in parsed.chapters] == ["NCX 导航标题"]


def test_epub_does_not_treat_discover_chapter_as_cover(tmp_path: Path) -> None:
    path = _write_epub(
        tmp_path / "discover.epub",
        resources={
            "discover.xhtml": _document("发现自己", "发现自己", "保留的正文" * 200),
        },
        spine=(("discover", "yes"),),
    )

    parsed = parse_epub(path)

    assert [chapter.title for chapter in parsed.chapters] == ["发现自己"]
    assert "保留的正文" in parsed.full_text


def test_epub_does_not_treat_chinese_cover_story_as_cover(tmp_path: Path) -> None:
    path = _write_epub(
        tmp_path / "cover-story.epub",
        resources={
            "chapter.xhtml": _document("封面故事", "封面故事", "保留的正文" * 200),
        },
        spine=(("chapter", "yes"),),
    )

    parsed = parse_epub(path)

    assert [chapter.title for chapter in parsed.chapters] == ["封面故事"]
    assert "保留的正文" in parsed.full_text


def test_epub_resolves_manifest_href_against_exact_opf_directory(
    tmp_path: Path,
) -> None:
    path = _write_epub(
        tmp_path / "duplicate-basename.epub",
        resources={
            "chapter.xhtml": _document("正确章节", "正确章节", "正确正文" * 200),
        },
        spine=(("chapter", "yes"),),
        package_directory="OPS",
        extra_members={
            "WRONG/chapter.xhtml": _document(
                "错误章节",
                "错误章节",
                "错误正文" * 200,
            ),
        },
    )
    probe = (
        "from pathlib import Path; "
        "from bv.ebook.epub import parse_epub; "
        f"print(parse_epub(Path({str(path)!r})).chapters[0].title)"
    )
    titles = []
    for seed in range(1, 9):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(seed)
        titles.append(
            subprocess.check_output(
                [sys.executable, "-X", "utf8", "-c", probe],
                env=environment,
                text=True,
                encoding="utf-8",
            ).strip()
        )

    assert titles == ["正确章节"] * 8


def test_epub_excludes_labeled_non_body_pages_but_keeps_relevant_prefaces(
    tmp_path: Path,
) -> None:
    path = _write_epub(
        tmp_path / "classified-pages.epub",
        resources={
            "page1.xhtml": _document("书名", "封面", "不应出现的封面文字"),
            "page2.xhtml": _document("书名", "版权页", "不应出现的版权声明"),
            "page3.xhtml": _document("书名", "限时推广", "不应出现的推广文案"),
            "page4.xhtml": _document("书名", "前言", "保留的前言"),
            "page5.xhtml": _document("书名", "译者前言", "保留的译者前言"),
            "page6.xhtml": _document("书名", "第一章", "保留的正文"),
        },
        spine=tuple((f"page{number}", "yes") for number in range(1, 7)),
    )

    parsed = parse_epub(path)

    assert [chapter.text for chapter in parsed.chapters] == [
        "保留的前言",
        "保留的译者前言",
        "保留的正文",
    ]


@pytest.mark.parametrize(
    ("name", "contents", "message"),
    [
        ("encrypted.epub", b"", "encrypted"),
        ("corrupt.epub", b"not an epub archive", "could not read"),
    ],
)
def test_epub_rejects_encrypted_or_corrupt_files(
    tmp_path: Path,
    name: str,
    contents: bytes,
    message: str,
) -> None:
    path = tmp_path / name
    if name == "encrypted.epub":
        path = _write_epub(
            path,
            resources={"chapter.xhtml": _document("标题", "标题", "正文")},
            spine=(("chapter", "yes"),),
            encrypted=True,
        )
    else:
        path.write_bytes(contents)

    with pytest.raises(ValueError, match=message):
        parse_epub(path)


def test_epub_rejects_missing_spine_resource_instead_of_returning_empty_book(
    tmp_path: Path,
) -> None:
    path = _write_epub(
        tmp_path / "missing-resource.epub",
        resources={},
        spine=(("missing", "yes"),),
        manifest={"missing": ("missing.xhtml", "application/xhtml+xml", "")},
    )

    with pytest.raises(ValueError, match="could not read|missing"):
        parse_epub(path)


def _write_epub(
    path: Path,
    *,
    resources: dict[str, str],
    spine: tuple[tuple[str, str], ...],
    manifest: dict[str, tuple[str, str, str]] | None = None,
    encrypted: bool = False,
    package_directory: str = "OEBPS",
    extra_members: dict[str, str] | None = None,
) -> Path:
    item_manifest = manifest or {
        _resource_id(href): (
            href,
            "application/xhtml+xml",
            "nav" if href == "nav.xhtml" else "",
        )
        for href in resources
    }
    manifest_entries = []
    for item_id, (href, media_type, properties) in item_manifest.items():
        properties_attribute = f' properties="{properties}"' if properties else ""
        manifest_entries.append(
            f'<item id="{item_id}" href="{href}" media-type="{media_type}"'
            f"{properties_attribute}/>"
        )
    manifest_xml = "\n".join(manifest_entries)
    spine_xml = "\n".join(
        f'<itemref idref="{item_id}" linear="{linear}"/>'
        for item_id, linear in spine
    )
    opf = f"""<?xml version="1.0" encoding="utf-8"?>
        <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">
          <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="book-id">test</dc:identifier><dc:title>测试书</dc:title><dc:language>zh</dc:language></metadata>
          <manifest>{manifest_xml}</manifest><spine>{spine_xml}</spine>
        </package>"""
    package_path = f"{package_directory}/content.opf"
    container = f"""<?xml version="1.0"?>
        <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
          <rootfiles><rootfile full-path="{package_path}" media-type="application/oebps-package+xml"/></rootfiles>
        </container>"""
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        if encrypted:
            archive.writestr("META-INF/encryption.xml", "<encryption/>")
        archive.writestr(package_path, opf)
        for href, contents in resources.items():
            archive.writestr(f"{package_directory}/{href}", contents)
        for member, contents in (extra_members or {}).items():
            archive.writestr(member, contents)
    return path


def _resource_id(href: str) -> str:
    return href.removesuffix(".xhtml").replace("-", "_")


def _document(title: str, h1: str | None, *paragraphs: str) -> str:
    heading = f"<h1>{h1}</h1>" if h1 else ""
    body = "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
    return f"<html><head><title>{title}</title></head><body>{heading}{body}</body></html>"


def _nav(*entries: tuple[str, str]) -> str:
    links = "".join(
        f'<li><a href="{href}">{title}</a></li>' for href, title in entries
    )
    return (
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"><body>'
        f'<nav epub:type="toc"><ol>{links}</ol></nav></body></html>'
    )
