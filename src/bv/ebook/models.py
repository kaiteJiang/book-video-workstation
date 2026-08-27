from typing import Literal

from pydantic import BaseModel, Field


class SourceAnchor(BaseModel):
    source_type: Literal["txt", "epub", "pdf"]
    source_path: str
    chapter_id: str
    start_line: int | None = None
    end_line: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    start_page: int | None = None
    end_page: int | None = None
    start_paragraph: int | None = None
    end_paragraph: int | None = None


class Chapter(BaseModel):
    chapter_id: str
    title: str
    text: str
    anchor: SourceAnchor
    sha256: str


class SourceQualityReport(BaseModel):
    blocking: bool
    codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ParsedBook(BaseModel):
    source_path: str
    encoding: str
    full_text: str
    chapters: list[Chapter]
    quality: SourceQualityReport
