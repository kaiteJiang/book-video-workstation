from pathlib import Path

from ebooklib import epub


FIXED_COVER_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63f8cfc0f01f000500023f49c2f8190000000049454e44"
    "ae426082"
)


def build_demo_epub(destination: Path | None = None) -> Path:
    output = destination or Path(__file__).parent / "books" / "demo.epub"
    output.parent.mkdir(parents=True, exist_ok=True)

    book = epub.EpubBook()
    book.set_identifier("bv-workstation-demo-epub")
    book.set_title("EPUB parser fixture")
    book.set_language("zh-CN")
    book.add_author("BV Workstation")

    cover = epub.EpubItem(
        uid="cover-image",
        file_name="images/cover.png",
        media_type="image/png",
        content=FIXED_COVER_PNG,
    )
    chapter1 = epub.EpubHtml(
        uid="chapter1",
        title="第一章",
        file_name="chapter1.xhtml",
        lang="zh-CN",
    )
    chapter1.content = (
        "<html><head><title>错误的第一章标题</title></head><body>"
        "<h1>第一章</h1><p>正文一</p><p>正文二</p></body></html>"
    )
    chapter2 = epub.EpubHtml(
        uid="chapter2",
        title="第二章",
        file_name="chapter2.xhtml",
        lang="zh-CN",
    )
    chapter2.content = (
        "<html><head><title>错误的第二章标题</title></head><body>"
        "<h1>第二章</h1><p>正文三</p></body></html>"
    )

    book.add_item(cover)
    book.add_metadata(None, "meta", "", {"name": "cover", "content": "cover-image"})
    book.add_item(chapter1)
    book.add_item(chapter2)
    book.toc = (
        epub.Link("chapter1.xhtml", "第一章", "chapter1"),
        epub.Link("chapter2.xhtml", "第二章", "chapter2"),
    )
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", "chapter1", "chapter2"]
    epub.write_epub(str(output), book, {"raise_exceptions": True})
    return output


if __name__ == "__main__":
    build_demo_epub()
