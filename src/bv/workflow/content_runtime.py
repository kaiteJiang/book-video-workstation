from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from bv.books.identity import normalize_book_key
from bv.content.evidence import (
    ChapterAnalysis,
    ChapterAnalysisCacheManifest,
    EvidenceCard,
    analyze_chapters,
)
from bv.content.human_writing import (
    check_human_writing_candidate,
    combined_skill_sha256,
)
from bv.content.research import BookResearch, research_book
from bv.content.reviews import build_script_review, extract_review_voiceover
from bv.content.scripts import (
    FactDiffResult,
    ScriptApprovalError,
    ScriptDraft,
    ScriptPackage,
    ScriptPipelineError,
    ScriptService,
    approve_script,
    validate_value_script,
)
from bv.content.synthesis import (
    BookReport,
    ClaimCluster,
    WholeBookValue,
    synthesize_book,
    validate_claim_references,
    validate_value_coverage,
)
from bv.content.story import approve_story_candidate
from bv.content.topics import EpisodeBrief, TopicGenerationError, TopicService
from bv.core.atomic import atomic_write_json
from bv.core.hashing import sha256_file
from bv.ebook.service import parse_book, persist_parsed_book
from bv.models.contracts import empty_request_directory
from bv.models.prompts import compose_source_prompt, load_prompt
from bv.state.models import ArtifactRef, StageManifest
from bv.state.store import StateStore
from bv.workflow.runtime import RuntimeAuthorization, require_external_authorization
from bv.workflow.stages import StageContext, StageOutcome

_SOURCE_SUFFIXES = {".txt", ".pdf", ".epub"}
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_PUBLIC_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CHAPTER_REGIONS = ("early", "middle", "late")
_ALLOWED_REGIONS = frozenset({"whole", *_CHAPTER_REGIONS})
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RESEARCH_PROMPT = _REPO_ROOT / "prompts" / "book_research" / "v1.md"
_DEFAULT_CHAPTER_PROMPT = _REPO_ROOT / "prompts" / "chapter_analysis" / "v1.md"
_DEFAULT_FILE_VALUE_PROMPT = _REPO_ROOT / "prompts" / "book_synthesis" / "v2.md"
_DEFAULT_TITLE_VALUE_PROMPT = _REPO_ROOT / "prompts" / "title_book_synthesis" / "v1.md"
_DEFAULT_TOPIC_PROMPT = _REPO_ROOT / "prompts" / "topic_generation" / "v2.md"
_DEFAULT_TOPIC_DEDUP_PROMPT = _REPO_ROOT / "prompts" / "topic_dedup" / "v2.md"
_DEFAULT_GROK_TOPIC_PROMPT = _REPO_ROOT / "prompts" / "grok_topic_review" / "v2.md"
_DEFAULT_SCRIPT_DRAFT_PROMPT = _REPO_ROOT / "prompts" / "script_draft" / "v2.md"
_DEFAULT_FACT_DIFF_PROMPT = _REPO_ROOT / "prompts" / "script_fact_diff" / "v2.md"
_DEFAULT_GROK_SCRIPT_PROMPT = _REPO_ROOT / "prompts" / "grok_script_review" / "v2.md"
_TITLE_RESEARCH_REGIONS = (
    "central_question",
    "arc",
    "key_ideas",
    "life_connections",
    "reader_value",
    "boundaries",
)
_VALUE_FIELDS = (
    "value_thesis",
    "target_reader",
    "reader_before",
    "reader_after",
    "central_life_tension",
    "practical_value",
    "reading_reason",
)
_NARRATIVE_ORDER = {"problem": 0, "reframe": 1, "application": 2, "boundary": 3}
_SCRIPT_REVISION_STAGE = "prepare_script_revision"
_UNSUPPORTED_ATTRIBUTION = (
    "书中说",
    "书里说",
    "作者说",
    "作者认为",
    "这本书告诉我们",
    "这本书证明",
)
_SCOPE_EXPANSION = ("保证", "一定能", "彻底解决", "治愈", "解决所有")
_FABRICATED_TESTIMONY = ("我朋友", "读者小", "真实案例", "亲身经历", "有个读者", "我的朋友")
_QUOTE_CHARS = ("“", "”", '"', "‘", "’")
_ISOLATED_CONCEPTS = ("孤立概念", "isolated concept", "一个概念", "单一概念")
_INVENTED_LOCATOR = re.compile(
    r"(第\s*\d+\s*章)|(\bpage\s+\d+)|(\bp\.\s*\d+)|(\d+\s*页)",
    re.I,
)


class ContentRuntimeError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        if not isinstance(error_code, str) or _PUBLIC_ERROR_CODE.fullmatch(error_code) is None:
            raise ValueError("invalid_error_code")
        self.error_code = error_code
        super().__init__(error_code)

    def __str__(self) -> str:
        return self.error_code


class TitleMetadataSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    book_id: str
    title: str
    authors: tuple[str, ...]
    source_mode: Literal["title_author"]
    identity_sha256: str


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_sha256: str
    evidence: tuple[EvidenceCard, ...]
    chapter_region_by_id: dict[str, str]
    chapter_sha256_by_id: dict[str, str]
    prompt_sha256: str

    @field_validator("source_sha256", "prompt_sha256")
    @classmethod
    def _require_sha256(cls, value: object) -> str:
        if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("hash must be 64 lowercase hex characters")
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def _require_evidence(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, list):
            value = tuple(value)
        if not isinstance(value, tuple) or not value:
            raise ValueError("evidence must be a non-empty sequence")
        return value

    @field_validator("chapter_region_by_id")
    @classmethod
    def _require_regions(cls, value: object) -> dict[str, str]:
        if not isinstance(value, dict) or not value:
            raise ValueError("chapter_region_by_id must be a non-empty object")
        cleaned: dict[str, str] = {}
        for chapter_id, region in value.items():
            if not isinstance(chapter_id, str) or not chapter_id or chapter_id.strip() != chapter_id:
                raise ValueError("chapter id is invalid")
            if not isinstance(region, str) or region not in _ALLOWED_REGIONS:
                raise ValueError("chapter region is invalid")
            cleaned[chapter_id] = region
        return cleaned

    @field_validator("chapter_sha256_by_id")
    @classmethod
    def _require_chapter_hashes(cls, value: object) -> dict[str, str]:
        if not isinstance(value, dict) or not value:
            raise ValueError("chapter_sha256_by_id must be a non-empty object")
        cleaned: dict[str, str] = {}
        for chapter_id, digest in value.items():
            if not isinstance(chapter_id, str) or not chapter_id or chapter_id.strip() != chapter_id:
                raise ValueError("chapter id is invalid")
            if not isinstance(digest, str) or _SHA256_HEX.fullmatch(digest) is None:
                raise ValueError("hash must be 64 lowercase hex characters")
            cleaned[chapter_id] = digest
        return cleaned

    @model_validator(mode="after")
    def _require_known_unique_claims(self) -> EvidenceBundle:
        if set(self.chapter_region_by_id) != set(self.chapter_sha256_by_id):
            raise ValueError("chapter maps must cover the same identifiers")
        seen: set[str] = set()
        for card in self.evidence:
            claim_id = card.claim_id
            if not isinstance(claim_id, str) or not claim_id or claim_id.strip() != claim_id:
                raise ValueError("claim id is invalid")
            if claim_id in seen:
                raise ValueError("claim ids must be unique")
            seen.add(claim_id)
            if card.chapter_id not in self.chapter_region_by_id:
                raise ValueError("chapter reference is unknown")
        return self


def _raise(error_code: str) -> None:
    raise ContentRuntimeError(error_code)


def _lexical(path: Path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True
    try:
        info = path.lstat()
    except OSError:
        return False
    attrs = int(getattr(info, "st_file_attributes", 0) or 0)
    return bool(attrs & _REPARSE_POINT)


def _assert_safe_segment(value: str) -> None:
    if not value or value.strip() != value:
        _raise("parse_context_unsafe")
    if value in {".", ".."} or "/" in value or "\\" in value or ".." in value:
        _raise("parse_context_unsafe")


def _assert_no_reparse(*paths: Path) -> None:
    seen: set[str] = set()
    for path in paths:
        current = Path(path)
        for item in (current, *current.parents):
            key = _lexical(item)
            if key in seen:
                continue
            seen.add(key)
            if item.exists() and _is_reparse_or_symlink(item):
                _raise("parse_context_unsafe")


def _identity_sha256(book_id: str, title: str, authors: list[str], source_mode: str) -> str:
    payload = {
        "authors": authors,
        "book_id": book_id,
        "source_mode": source_mode,
        "title": title,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return sha256_file(path).lower()


def _require_regular_file(path: Path, error_code: str) -> None:
    if not path.exists() or not path.is_file() or _is_reparse_or_symlink(path):
        _raise(error_code)


def _require_nonempty_file(path: Path, error_code: str = "parsed_artifact_invalid") -> None:
    _require_regular_file(path, error_code)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        _raise(error_code)
    if not text.strip():
        _raise(error_code)


def _require_exact_text(value: object, error_code: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _raise(error_code)
    return value


def _chapter_index_entry(entry: object) -> tuple[str, str]:
    if not isinstance(entry, Mapping):
        _raise("parsed_artifact_invalid")
    chapter_id = entry.get("chapter_id")
    title = entry.get("title")
    digest = entry.get("sha256")
    if not isinstance(chapter_id, str) or not chapter_id or chapter_id.strip() != chapter_id:
        _raise("parsed_artifact_invalid")
    if not isinstance(title, str) or not title or title.strip() != title:
        _raise("parsed_artifact_invalid")
    if not isinstance(digest, str) or _SHA256_HEX.fullmatch(digest) is None:
        _raise("parsed_artifact_invalid")
    return chapter_id, digest


def _file_mode_outputs(episode_root: Path) -> dict[str, Path]:
    quality = episode_root / "analysis" / "source_quality.json"
    index_path = episode_root / "chapters" / "index.json"
    _require_nonempty_file(quality)
    _require_nonempty_file(index_path)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _raise("parsed_artifact_invalid")
    if not isinstance(payload, dict):
        _raise("parsed_artifact_invalid")
    declared = payload.get("chapters")
    if not isinstance(declared, list) or not declared:
        _raise("parsed_artifact_invalid")
    names: list[str] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for index, item in enumerate(declared, start=1):
        chapter_id, digest = _chapter_index_entry(item)
        if chapter_id in seen_ids or digest in seen_hashes:
            _raise("parsed_artifact_invalid")
        seen_ids.add(chapter_id)
        seen_hashes.add(digest)
        names.append(f"{index:03d}.md")
    chapter_dir = episode_root / "chapters"
    try:
        actual = {
            item.name
            for item in chapter_dir.iterdir()
            if item.is_file() and item.suffix.lower() == ".md"
        }
    except OSError:
        _raise("parsed_artifact_invalid")
    if actual != set(names):
        _raise("parsed_artifact_invalid")
    outputs: dict[str, Path] = {
        "source_quality": quality,
        "chapters_index": index_path,
    }
    for name in names:
        path = chapter_dir / name
        _require_nonempty_file(path)
        outputs[f"chapter_{path.stem}"] = path
    return outputs


def _load_current_book(store: StateStore, context: StageContext) -> tuple[Any, Path]:
    _assert_safe_segment(context.book_id)
    _assert_safe_segment(context.episode_id)
    expected_root = Path(store.root) / "books" / context.book_id / "episodes" / context.episode_id
    if _lexical(Path(context.episode_root)) != _lexical(expected_root):
        _raise("parse_context_unsafe")
    book_dir = Path(store.root) / "books" / context.book_id
    book_json = book_dir / "book.json"
    _assert_no_reparse(Path(store.root), book_dir, book_json, Path(context.episode_root))
    _require_regular_file(book_json, "parse_context_unsafe")
    book = store.load_book(context.book_id)
    if book.book_id != context.book_id:
        _raise("book_identity_mismatch")
    return book, book_json


def _resolve_imported_source(store: StateStore, book: Any, book_id: str) -> Path:
    source_ids = list(book.source_ids)
    if len(source_ids) == 0:
        _raise("source_missing")
    if len(source_ids) > 1:
        _raise("source_ambiguous")
    source_id = source_ids[0]
    if not isinstance(source_id, str) or _SHA256_HEX.fullmatch(source_id) is None:
        _raise("source_invalid")
    source_dir = Path(store.root) / "books" / book_id / "source"
    if source_dir.exists() and _is_reparse_or_symlink(source_dir):
        _raise("parse_context_unsafe")
    if not source_dir.is_dir():
        _raise("source_missing")
    entries: list[Path] = []
    try:
        children = list(source_dir.iterdir())
    except OSError:
        _raise("source_missing")
    for item in children:
        if _is_reparse_or_symlink(item):
            _raise("parse_context_unsafe")
        if item.is_file():
            entries.append(item)
    if len(entries) == 0:
        _raise("source_missing")
    if len(entries) > 1:
        _raise("source_ambiguous")
    source = entries[0]
    if source.suffix.lower() not in _SOURCE_SUFFIXES:
        _raise("source_invalid")
    if source.stem != source_id or _sha256(source) != source_id:
        _raise("source_integrity_mismatch")
    return source


def _load_title_metadata(episode_root: Path, book: Any) -> Path:
    destination = Path(episode_root) / "analysis" / "title_metadata.json"
    _require_regular_file(destination, "title_metadata_invalid")
    try:
        loaded = TitleMetadataSnapshot.model_validate_json(
            destination.read_text(encoding="utf-8")
        )
    except Exception:
        _raise("title_metadata_invalid")
    recomputed = _identity_sha256(
        loaded.book_id,
        loaded.title,
        list(loaded.authors),
        loaded.source_mode,
    )
    if loaded.identity_sha256 != recomputed or loaded.source_mode != "title_author":
        _raise("title_metadata_invalid")
    title = _require_exact_text(book.title, "title_metadata_invalid")
    raw_authors = book.authors
    if not isinstance(raw_authors, list) or not raw_authors:
        _raise("title_metadata_invalid")
    authors = [_require_exact_text(author, "title_metadata_invalid") for author in raw_authors]
    current = _identity_sha256(book.book_id, title, authors, book.source_mode)
    if (
        recomputed != current
        or loaded.book_id != book.book_id
        or loaded.title != title
        or list(loaded.authors) != authors
        or book.source_mode != "title_author"
    ):
        _raise("book_identity_mismatch")
    return destination


def _current_prompt(path: Path, expected_sha256: str) -> Any:
    try:
        asset = load_prompt(path)
    except Exception as exc:
        if isinstance(exc, (ContentRuntimeError, SystemExit, KeyboardInterrupt)):
            raise
        _raise("content_prompt_changed")
    if asset.sha256 != expected_sha256:
        _raise("content_prompt_changed")
    if not isinstance(asset.text, str) or not asset.text.strip():
        _raise("content_prompt_invalid")
    return asset


def _require_unchanged(path: Path, expected: str, error_code: str) -> None:
    try:
        current = _sha256(path)
    except OSError:
        _raise(error_code)
    if current != expected:
        _raise(error_code)


def _human_writing_skill_sha256(skill_path: Path) -> str:
    return combined_skill_sha256(Path(skill_path).parent)


def _require_human_writing_skill_unchanged(
    skill_path: Path, expected: str
) -> None:
    try:
        current = _human_writing_skill_sha256(skill_path)
    except OSError:
        _raise("human_writing_skill_changed")
    if current != expected:
        _raise("human_writing_skill_changed")


def _chapter_identity_pairs(chapters: list[Any]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for chapter in chapters:
        chapter_id = getattr(chapter, "chapter_id", None)
        digest = getattr(chapter, "sha256", None)
        if not isinstance(chapter_id, str) or not chapter_id or chapter_id.strip() != chapter_id:
            _raise("parsed_artifact_invalid")
        if not isinstance(digest, str) or _SHA256_HEX.fullmatch(digest.lower()) is None:
            _raise("parsed_artifact_invalid")
        pairs.append((chapter_id, digest.lower()))
    return tuple(pairs)


def _index_pairs(index_path: Path) -> list[tuple[str, str]]:
    _require_nonempty_file(index_path)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _raise("parsed_artifact_invalid")
    if not isinstance(payload, dict):
        _raise("parsed_artifact_invalid")
    declared = payload.get("chapters")
    if not isinstance(declared, list) or not declared:
        _raise("parsed_artifact_invalid")
    return [_chapter_index_entry(item) for item in declared]


def _parsed_chapters(parsed: Any) -> list[Any]:
    chapters = list(getattr(parsed, "chapters", []) or [])
    if not chapters:
        _raise("parsed_artifact_invalid")
    return chapters


def _assert_index_matches_parsed(index_path: Path, chapters: list[Any]) -> None:
    indexed = _index_pairs(index_path)
    parsed: list[tuple[str, str]] = []
    for chapter in chapters:
        chapter_id = getattr(chapter, "chapter_id", None)
        digest = getattr(chapter, "sha256", None)
        if not isinstance(chapter_id, str) or not chapter_id or chapter_id.strip() != chapter_id:
            _raise("parsed_artifact_invalid")
        if not isinstance(digest, str) or _SHA256_HEX.fullmatch(digest.lower()) is None:
            _raise("parsed_artifact_invalid")
        parsed.append((chapter_id, digest.lower()))
    if indexed != parsed:
        _raise("chapter_index_stale")


def _assign_chapter_regions(chapter_ids: list[str]) -> dict[str, str]:
    count = len(chapter_ids)
    if count == 1:
        return {chapter_ids[0]: "whole"}
    if count == 2:
        return {chapter_ids[0]: "early", chapter_ids[1]: "late"}
    assigned = {
        chapter_ids[index]: _CHAPTER_REGIONS[(index * 3) // count] for index in range(count)
    }
    if set(assigned.values()) != set(_CHAPTER_REGIONS):
        _raise("evidence_bundle_invalid")
    return assigned


def _wrap_stable_error(exc: BaseException, fallback: str) -> None:
    if isinstance(exc, (ContentRuntimeError, SystemExit, KeyboardInterrupt)):
        raise exc
    code = getattr(exc, "error_code", None)
    if isinstance(code, str) and _PUBLIC_ERROR_CODE.fullmatch(code) is not None:
        _raise(code)
    _raise(fallback)


class TitleOrFileParseStage:
    def __init__(
        self,
        store: StateStore,
        *,
        parser: Callable[[Path], Any] = parse_book,
        persister: Callable[..., Any] = persist_parsed_book,
    ) -> None:
        self.store = store
        self.parser = parser
        self.persister = persister

    def run(self, context: StageContext) -> StageOutcome:
        book, book_json = _load_current_book(self.store, context)
        if book.source_mode == "title_author":
            return self._run_title(book, context, book_json)
        if book.source_mode == "file":
            return self._run_file(book, context, book_json)
        _raise("source_invalid")
        raise AssertionError("unreachable")

    def _run_title(self, book: Any, context: StageContext, book_json: Path) -> StageOutcome:
        title = _require_exact_text(book.title, "title_metadata_invalid")
        raw_authors = book.authors
        if not isinstance(raw_authors, list) or not raw_authors:
            _raise("title_metadata_invalid")
        authors = [_require_exact_text(author, "title_metadata_invalid") for author in raw_authors]
        if list(book.source_ids):
            _raise("title_metadata_invalid")
        digest = _identity_sha256(book.book_id, title, authors, "title_author")
        destination = Path(context.episode_root) / "analysis" / "title_metadata.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            destination,
            {
                "book_id": book.book_id,
                "title": title,
                "authors": authors,
                "source_mode": "title_author",
                "identity_sha256": digest,
            },
        )
        _require_regular_file(destination, "title_metadata_invalid")
        try:
            loaded = TitleMetadataSnapshot.model_validate_json(
                destination.read_text(encoding="utf-8")
            )
        except Exception:
            _raise("title_metadata_invalid")
        recomputed = _identity_sha256(
            loaded.book_id,
            loaded.title,
            list(loaded.authors),
            loaded.source_mode,
        )
        if (
            loaded.identity_sha256 != digest
            or recomputed != digest
            or loaded.book_id != book.book_id
            or loaded.title != book.title
            or list(loaded.authors) != list(book.authors)
            or loaded.source_mode != "title_author"
        ):
            _raise("title_metadata_invalid")
        return StageOutcome(
            outputs={"title_metadata": destination},
            inputs={
                "book_state": _sha256(book_json),
                "title_metadata": _sha256(destination),
            },
        )

    def _run_file(self, book: Any, context: StageContext, book_json: Path) -> StageOutcome:
        source = _resolve_imported_source(self.store, book, context.book_id)
        parsed = self.parser(source)
        result = self.persister(parsed, Path(context.episode_root))
        if result != "source_parsed":
            _raise("source_invalid")
        return StageOutcome(
            outputs=_file_mode_outputs(Path(context.episode_root)),
            inputs={
                "book_state": _sha256(book_json),
                "source": _sha256(source),
            },
        )


class BookAnalysisStage:
    def __init__(
        self,
        store: StateStore,
        *,
        authorization: RuntimeAuthorization = RuntimeAuthorization(),
        research_model: Any = None,
        chapter_model: Any = None,
        research_prompt_path: Path | None = None,
        chapter_prompt_path: Path | None = None,
        parser: Callable[[Path], Any] = parse_book,
        researcher: Callable[..., Any] = research_book,
        analyzer: Callable[..., Any] = analyze_chapters,
        model_identifier: str = "chapter-analysis",
    ) -> None:
        self.store = store
        self.authorization = authorization
        self.research_model = research_model
        self.chapter_model = chapter_model
        self.parser = parser
        self.researcher = researcher
        self.analyzer = analyzer
        self.model_identifier = model_identifier
        self._research_prompt_path = Path(research_prompt_path or _DEFAULT_RESEARCH_PROMPT)
        self._chapter_prompt_path = Path(chapter_prompt_path or _DEFAULT_CHAPTER_PROMPT)
        self._research_prompt_sha256 = load_prompt(self._research_prompt_path).sha256
        self._chapter_prompt_sha256 = load_prompt(self._chapter_prompt_path).sha256

    def run(self, context: StageContext) -> StageOutcome:
        book, book_json = _load_current_book(self.store, context)
        if book.source_mode == "title_author":
            return self._run_title(book, context, book_json)
        if book.source_mode == "file":
            return self._run_file(book, context, book_json)
        _raise("source_invalid")
        raise AssertionError("unreachable")

    def _run_title(self, book: Any, context: StageContext, book_json: Path) -> StageOutcome:
        metadata_path = _load_title_metadata(Path(context.episode_root), book)
        prompt = _current_prompt(self._research_prompt_path, self._research_prompt_sha256)
        require_external_authorization(self.authorization)
        book_hash = _sha256(book_json)
        metadata_hash = _sha256(metadata_path)
        prompt_hash = prompt.sha256
        output_root = Path(context.episode_root) / "analysis"
        try:
            result = self.researcher(
                book=book,
                model=self.research_model,
                output_root=output_root,
                prompt_text=prompt.text,
            )
        except Exception as exc:
            _wrap_stable_error(exc, "book_research_invalid")
        _require_unchanged(book_json, book_hash, "content_input_changed")
        _require_unchanged(metadata_path, metadata_hash, "content_input_changed")
        _current_prompt(self._research_prompt_path, prompt_hash)
        if getattr(result, "prompt_sha256", None) != prompt.sha256:
            _raise("content_prompt_changed")
        book, book_json = _load_current_book(self.store, context)
        metadata_path = _load_title_metadata(Path(context.episode_root), book)
        if getattr(result, "status", None) != "accepted":
            _raise("book_research_invalid")
        json_path = output_root / "book_research.json"
        markdown_path = output_root / "book_research.md"
        _require_nonempty_file(json_path, "book_research_invalid")
        _require_nonempty_file(markdown_path, "book_research_invalid")
        try:
            research = BookResearch.model_validate_json(json_path.read_text(encoding="utf-8"))
        except Exception:
            _raise("book_research_invalid")
        if normalize_book_key(research.title, research.authors) != normalize_book_key(
            book.title, book.authors
        ):
            _raise("book_research_identity_mismatch")
        return StageOutcome(
            outputs={
                "book_research": json_path,
                "book_research_md": markdown_path,
            },
            inputs={
                "book_state": _sha256(book_json),
                "title_metadata": _sha256(metadata_path),
                "book_research_prompt": prompt.sha256,
            },
        )

    def _run_file(self, book: Any, context: StageContext, book_json: Path) -> StageOutcome:
        source = _resolve_imported_source(self.store, book, context.book_id)
        episode_root = Path(context.episode_root)
        _file_mode_outputs(episode_root)
        parsed = self.parser(source)
        chapters = _parsed_chapters(parsed)
        index_path = episode_root / "chapters" / "index.json"
        _assert_index_matches_parsed(index_path, chapters)
        prompt = _current_prompt(self._chapter_prompt_path, self._chapter_prompt_sha256)
        require_external_authorization(self.authorization)
        source_sha256 = _sha256(source)
        book_hash = _sha256(book_json)
        index_hash = _sha256(index_path)
        prompt_hash = prompt.sha256
        chapter_pairs = _chapter_identity_pairs(chapters)
        output_dir = episode_root / "analysis"
        try:
            result = self.analyzer(
                chapters,
                model=self.chapter_model,
                output_dir=output_dir,
                prompt_text=prompt.text,
                prompt_sha256=prompt.sha256,
                source_sha256=source_sha256,
                model_identifier=self.model_identifier,
            )
        except Exception as exc:
            _wrap_stable_error(exc, "book_analysis_invalid")
        _require_unchanged(book_json, book_hash, "content_input_changed")
        _require_unchanged(source, source_sha256, "content_input_changed")
        _require_unchanged(index_path, index_hash, "content_input_changed")
        prompt = _current_prompt(self._chapter_prompt_path, prompt_hash)
        try:
            chapters = _parsed_chapters(self.parser(source))
        except Exception as exc:
            _wrap_stable_error(exc, "content_input_changed")
        if _chapter_identity_pairs(chapters) != chapter_pairs:
            _raise("content_input_changed")
        _assert_index_matches_parsed(index_path, chapters)
        if getattr(result, "status", None) != "book_analysis_ready":
            _raise("book_analysis_failed")
        cards: list[EvidenceCard] = []
        outputs: dict[str, Path] = {}
        seen_claims: set[str] = set()
        for index, chapter in enumerate(chapters, start=1):
            analysis_path = output_dir / "chapters" / f"{chapter.chapter_id}.analysis.json"
            manifest_path = output_dir / "chapters" / f"{chapter.chapter_id}.manifest.json"
            _require_nonempty_file(analysis_path, "book_analysis_invalid")
            _require_nonempty_file(manifest_path, "book_analysis_invalid")
            try:
                manifest = ChapterAnalysisCacheManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                analysis = ChapterAnalysis.model_validate_json(
                    analysis_path.read_text(encoding="utf-8")
                )
            except Exception:
                _raise("book_analysis_invalid")
            if (
                manifest.cache_key.source_sha256 != source_sha256
                or manifest.cache_key.chapter_sha256 != str(chapter.sha256).lower()
                or manifest.cache_key.prompt_sha256 != prompt.sha256
                or manifest.cache_key.model_identifier != self.model_identifier
                or manifest.analysis_sha256 != sha256_file(analysis_path)
            ):
                _raise("chapter_manifest_stale")
            if analysis.chapter_id != chapter.chapter_id:
                _raise("book_analysis_invalid")
            for card in analysis.cards:
                if card.claim_id in seen_claims:
                    _raise("claim_id_duplicate")
                seen_claims.add(card.claim_id)
            cards.extend(analysis.cards)
            outputs[f"chapter_{index:03d}_analysis"] = analysis_path
            outputs[f"chapter_{index:03d}_manifest"] = manifest_path
        if not cards:
            _raise("book_analysis_invalid")
        chapter_ids = [chapter.chapter_id for chapter in chapters]
        try:
            bundle = EvidenceBundle(
                source_sha256=_sha256(source),
                evidence=tuple(cards),
                chapter_region_by_id=_assign_chapter_regions(chapter_ids),
                chapter_sha256_by_id={
                    chapter.chapter_id: str(chapter.sha256).lower() for chapter in chapters
                },
                prompt_sha256=prompt.sha256,
            )
        except Exception:
            _raise("evidence_bundle_invalid")
        bundle_path = output_dir / "evidence_bundle.json"
        try:
            atomic_write_json(bundle_path, bundle.model_dump(mode="json"))
            EvidenceBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
        except Exception:
            _raise("evidence_bundle_invalid")
        outputs["evidence_bundle"] = bundle_path
        return StageOutcome(
            outputs=outputs,
            inputs={
                "book_state": _sha256(book_json),
                "source": _sha256(source),
                "chapters_index": _sha256(episode_root / "chapters" / "index.json"),
                "chapter_analysis_prompt": prompt.sha256,
            },
        )


class TitleResearchClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str
    research_region: Literal[
        "central_question",
        "arc",
        "key_ideas",
        "life_connections",
        "reader_value",
        "boundaries",
    ]
    text: str

    @field_validator("claim_id", "text")
    @classmethod
    def _require_canonical(cls, value: object) -> str:
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError("catalog field is invalid")
        return value


class TitleValueProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    book_research_sha256: str
    prompt_sha256: str
    source_urls: tuple[str, ...]
    uncertainties: tuple[str, ...] = ()
    catalog: tuple[TitleResearchClaim, ...]

    @field_validator("book_research_sha256", "prompt_sha256")
    @classmethod
    def _require_sha256(cls, value: object) -> str:
        if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("hash must be 64 lowercase hex characters")
        return value

    @field_validator("source_urls")
    @classmethod
    def _require_urls(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("source_urls must be a non-empty sequence")
        cleaned: list[str] = []
        for url in value:
            if not isinstance(url, str) or not url or url.strip() != url:
                raise ValueError("source url is invalid")
            cleaned.append(url)
        return tuple(cleaned)

    @field_validator("uncertainties", mode="before")
    @classmethod
    def _require_uncertainties(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("uncertainties must be a sequence")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item or item.strip() != item:
                raise ValueError("uncertainty is invalid")
            cleaned.append(item)
        return tuple(cleaned)

    @field_validator("catalog", mode="before")
    @classmethod
    def _require_catalog(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, list):
            value = tuple(value)
        if not isinstance(value, tuple) or not value:
            raise ValueError("catalog must be a non-empty sequence")
        return value


def build_title_research_claim_catalog(
    research: BookResearch,
) -> tuple[TitleResearchClaim, ...]:
    items: list[TitleResearchClaim] = [
        TitleResearchClaim(
            claim_id="central_question_001",
            research_region="central_question",
            text=research.central_question,
        )
    ]
    sequences = (
        (research.argument_or_narrative_arc, "arc", "arc"),
        (research.key_ideas, "key_idea", "key_ideas"),
        (research.life_connections, "life_connection", "life_connections"),
        (research.reader_value, "reader_value", "reader_value"),
        (research.misunderstandings_and_boundaries, "boundary", "boundaries"),
    )
    for values, prefix, region in sequences:
        for index, text in enumerate(values, start=1):
            items.append(
                TitleResearchClaim(
                    claim_id=f"{prefix}_{index:03d}",
                    research_region=region,  # type: ignore[arg-type]
                    text=text,
                )
            )
    return tuple(items)


def synthesize_title_value(
    *,
    research: BookResearch,
    catalog: tuple[TitleResearchClaim, ...],
    model: Any,
    prompt_text: str,
    request_root: Path,
) -> WholeBookValue:
    request_root = Path(request_root)
    try:
        request_root.mkdir(parents=True, exist_ok=True)
        request_dir = Path(tempfile.mkdtemp(prefix="title-value-", dir=request_root))
    except OSError:
        _raise("unsafe_request_directory")
    if not empty_request_directory(request_dir):
        _raise("unsafe_request_directory")
    payload = {
        "catalog": [
            {
                "claim_id": item.claim_id,
                "research_region": item.research_region,
                "text": item.text,
            }
            for item in catalog
        ],
        "sources": [
            {
                "url": source.url,
                "source_type": source.source_type,
                "supports": list(source.supports),
            }
            for source in research.sources
        ],
        "uncertainties": list(research.uncertainties),
    }
    prompt = compose_source_prompt(
        prompt_text,
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )
    try:
        response = model.complete(prompt, WholeBookValue, request_dir)
    except Exception as exc:
        _wrap_stable_error(exc, "title_value_invalid")
        raise AssertionError("unreachable")
    if isinstance(response, WholeBookValue):
        return response
    try:
        return WholeBookValue.model_validate(response)
    except Exception:
        _raise("title_value_invalid")
        raise AssertionError("unreachable")


def _title_reader_text(value: WholeBookValue, extra: str | None) -> str:
    parts = [getattr(value, field) for field in _VALUE_FIELDS]
    parts.extend(cluster.summary for cluster in value.supporting_claim_clusters)
    if extra is not None:
        parts.append(extra)
    return "\n".join(part for part in parts if isinstance(part, str))


def _progressive_title_roles(roles: list[str]) -> bool:
    return (
        bool(roles)
        and roles[0] == "problem"
        and len(roles) == len(set(roles))
        and roles == sorted(roles, key=_NARRATIVE_ORDER.__getitem__)
        and any(role in {"reframe", "application"} for role in roles[1:])
    )


def _require_valid_title_value(
    value: object, catalog: tuple[TitleResearchClaim, ...]
) -> WholeBookValue:
    if not isinstance(value, WholeBookValue):
        try:
            value = WholeBookValue.model_validate(value)
        except Exception:
            _raise("title_value_invalid")
    central: dict[str, str] = {}
    for field in _VALUE_FIELDS:
        text = getattr(value, field)
        if not isinstance(text, str) or not text.strip():
            _raise("blank_value_field")
        central[field] = text.strip()
    coverage_exception = (
        value.coverage_exception.strip()
        if isinstance(value.coverage_exception, str)
        else None
    )
    if coverage_exception == "":
        coverage_exception = None
    combined = _title_reader_text(value, coverage_exception)
    if any(token in combined for token in _UNSUPPORTED_ATTRIBUTION):
        _raise("unsupported_attribution")
    if any(token in combined for token in _SCOPE_EXPANSION):
        _raise("scope_expansion")
    if any(token in combined for token in _FABRICATED_TESTIMONY):
        _raise("fabricated_testimony")
    if any(char in combined for char in _QUOTE_CHARS):
        _raise("invented_quote")
    if _INVENTED_LOCATOR.search(combined):
        _raise("invented_quote")
    if any(token in combined for token in _ISOLATED_CONCEPTS):
        _raise("isolated_concept_framing")
    clusters = list(value.supporting_claim_clusters)
    if len(clusters) < 2:
        _raise("isolated_concept_framing")
    if len(clusters) > 3:
        _raise("invalid_cluster_count")
    known = {item.claim_id: item.research_region for item in catalog}
    seen_claims: set[str] = set()
    canonical_clusters: list[ClaimCluster] = []
    roles: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster.cluster_id, str) or not cluster.cluster_id or cluster.cluster_id.strip() != cluster.cluster_id:
            _raise("invalid_cluster_id")
        if not isinstance(cluster.summary, str) or not cluster.summary.strip():
            _raise("blank_cluster_summary")
        if not cluster.claim_ids or any(
            not isinstance(claim_id, str) or not claim_id or claim_id.strip() != claim_id
            for claim_id in cluster.claim_ids
        ):
            _raise("invalid_cluster_claim_id")
        if len(cluster.claim_ids) != len(set(cluster.claim_ids)):
            _raise("duplicate_cluster_claim")
        if any(claim_id not in known for claim_id in cluster.claim_ids):
            _raise("unknown_claim_id")
        if any(claim_id in seen_claims for claim_id in cluster.claim_ids):
            _raise("duplicate_cluster_claim")
        seen_claims.update(cluster.claim_ids)
        expected_regions = {known[claim_id] for claim_id in cluster.claim_ids}
        provided_regions = list(cluster.chapter_regions)
        if set(provided_regions) != expected_regions or len(provided_regions) != len(
            set(provided_regions)
        ):
            _raise("cluster_regions_mismatch")
        if any(region not in _TITLE_RESEARCH_REGIONS for region in provided_regions):
            _raise("cluster_regions_mismatch")
        roles.append(cluster.narrative_role)
        canonical_clusters.append(
            ClaimCluster(
                cluster_id=cluster.cluster_id,
                summary=cluster.summary.strip(),
                narrative_role=cluster.narrative_role,
                claim_ids=list(cluster.claim_ids),
                chapter_regions=list(provided_regions),
            )
        )
    if len({item.cluster_id for item in canonical_clusters}) != len(canonical_clusters):
        _raise("duplicate_cluster_id")
    if not _progressive_title_roles(roles):
        _raise("non_progressive_narrative")
    candidate = WholeBookValue(
        **central,
        supporting_claim_clusters=canonical_clusters,
        coverage_exception=coverage_exception,
    )
    report = BookReport(value=candidate, concepts=[])
    reference = validate_claim_references(report, set(known))
    if not reference.valid:
        _raise(reference.error_codes[0] if reference.error_codes else "unknown_claim_id")
    coverage = validate_value_coverage(report, available_regions=set(known.values()))
    if not coverage.valid:
        code = coverage.error_codes[0] if coverage.error_codes else "title_value_invalid"
        if code == "isolated_single_cluster":
            _raise("isolated_concept_framing")
        _raise(code)
    return candidate


class BookValueStage:
    def __init__(
        self,
        store: StateStore,
        *,
        authorization: RuntimeAuthorization = RuntimeAuthorization(),
        file_model: Any = None,
        title_model: Any = None,
        file_prompt_path: Path | None = None,
        title_prompt_path: Path | None = None,
        parser: Callable[[Path], Any] = parse_book,
        file_synthesizer: Callable[..., Any] = synthesize_book,
        title_synthesizer: Callable[..., Any] | None = None,
    ) -> None:
        self.store = store
        self.authorization = authorization
        self.file_model = file_model
        self.title_model = title_model
        self.parser = parser
        self.file_synthesizer = file_synthesizer
        self.title_synthesizer = title_synthesizer or synthesize_title_value
        self._file_prompt_path = Path(file_prompt_path or _DEFAULT_FILE_VALUE_PROMPT)
        self._title_prompt_path = Path(title_prompt_path or _DEFAULT_TITLE_VALUE_PROMPT)
        self._file_prompt_sha256 = load_prompt(self._file_prompt_path).sha256
        self._title_prompt_sha256 = load_prompt(self._title_prompt_path).sha256

    def run(self, context: StageContext) -> StageOutcome:
        book, book_json = _load_current_book(self.store, context)
        if book.source_mode == "title_author":
            return self._run_title(book, context, book_json)
        if book.source_mode == "file":
            return self._run_file(book, context, book_json)
        _raise("source_invalid")
        raise AssertionError("unreachable")

    def _run_file(self, book: Any, context: StageContext, book_json: Path) -> StageOutcome:
        source = _resolve_imported_source(self.store, book, context.book_id)
        episode_root = Path(context.episode_root)
        _file_mode_outputs(episode_root)
        parsed = self.parser(source)
        chapters = _parsed_chapters(parsed)
        index_path = episode_root / "chapters" / "index.json"
        _assert_index_matches_parsed(index_path, chapters)
        bundle_path = episode_root / "analysis" / "evidence_bundle.json"
        _require_nonempty_file(bundle_path, "evidence_bundle_invalid")
        try:
            bundle = EvidenceBundle.model_validate_json(
                bundle_path.read_text(encoding="utf-8")
            )
        except Exception:
            _raise("evidence_bundle_invalid")
        source_sha256 = _sha256(source)
        if bundle.source_sha256 != source_sha256:
            _raise("evidence_bundle_stale")
        current_chapter_hashes = {
            chapter.chapter_id: str(chapter.sha256).lower() for chapter in chapters
        }
        if bundle.chapter_sha256_by_id != current_chapter_hashes:
            _raise("evidence_bundle_stale")
        prompt = _current_prompt(self._file_prompt_path, self._file_prompt_sha256)
        require_external_authorization(self.authorization)
        book_hash = _sha256(book_json)
        index_hash = _sha256(index_path)
        bundle_hash = _sha256(bundle_path)
        prompt_hash = prompt.sha256
        chapter_pairs = _chapter_identity_pairs(chapters)
        try:
            result = self.file_synthesizer(
                list(bundle.evidence),
                chapter_region_by_id=dict(bundle.chapter_region_by_id),
                model=self.file_model,
                output_root=episode_root,
                prompt_text=prompt.text,
            )
        except Exception as exc:
            _wrap_stable_error(exc, "book_synthesis_failed")
        _require_unchanged(book_json, book_hash, "content_input_changed")
        _require_unchanged(source, source_sha256, "content_input_changed")
        _require_unchanged(index_path, index_hash, "content_input_changed")
        _require_unchanged(bundle_path, bundle_hash, "content_input_changed")
        prompt = _current_prompt(self._file_prompt_path, prompt_hash)
        try:
            chapters = _parsed_chapters(self.parser(source))
        except Exception as exc:
            _wrap_stable_error(exc, "content_input_changed")
        if _chapter_identity_pairs(chapters) != chapter_pairs:
            _raise("content_input_changed")
        _assert_index_matches_parsed(index_path, chapters)
        if getattr(result, "status", None) != "book_value_ready":
            codes = list(getattr(result, "error_codes", None) or [])
            if (
                codes
                and isinstance(codes[0], str)
                and _PUBLIC_ERROR_CODE.fullmatch(codes[0]) is not None
            ):
                _raise(codes[0])
            _raise("book_synthesis_failed")
        report = getattr(result, "report", None)
        value = getattr(report, "value", None)
        if not isinstance(value, WholeBookValue):
            _raise("book_value_invalid")
        analysis = episode_root / "analysis"
        accepted = {
            "book_report": analysis / "book_report.json",
            "book_report_md": analysis / "book_report.md",
            "concept_map": analysis / "concept_map.json",
            "value_synthesis": analysis / "value_synthesis.json",
        }
        for path in accepted.values():
            _require_nonempty_file(path, "accepted_artifact_invalid")
        try:
            loaded = WholeBookValue.model_validate_json(
                accepted["value_synthesis"].read_text(encoding="utf-8")
            )
        except Exception:
            _raise("accepted_artifact_invalid")
        if loaded.model_dump(mode="json") != value.model_dump(mode="json"):
            _raise("accepted_artifact_invalid")
        card_path = analysis / "book_value_card.json"
        try:
            atomic_write_json(card_path, value.model_dump(mode="json"))
            card = WholeBookValue.model_validate_json(card_path.read_text(encoding="utf-8"))
        except Exception:
            _raise("book_value_card_invalid")
        if card.model_dump(mode="json") != value.model_dump(mode="json"):
            _raise("book_value_card_invalid")
        book, book_json = _load_current_book(self.store, context)
        return StageOutcome(
            outputs={"book_value_card": card_path, **accepted},
            inputs={
                "book_state": _sha256(book_json),
                "source": _sha256(source),
                "evidence_bundle": _sha256(bundle_path),
                "chapters_index": _sha256(index_path),
                "book_synthesis_prompt": prompt.sha256,
            },
        )

    def _run_title(self, book: Any, context: StageContext, book_json: Path) -> StageOutcome:
        episode_root = Path(context.episode_root)
        metadata_path = _load_title_metadata(episode_root, book)
        research_path = episode_root / "analysis" / "book_research.json"
        _require_nonempty_file(research_path, "book_research_invalid")
        try:
            research = BookResearch.model_validate_json(research_path.read_text(encoding="utf-8"))
        except Exception:
            _raise("book_research_invalid")
        if normalize_book_key(research.title, research.authors) != normalize_book_key(book.title, book.authors):
            _raise("book_research_identity_mismatch")
        catalog = build_title_research_claim_catalog(research)
        prompt = _current_prompt(self._title_prompt_path, self._title_prompt_sha256)
        require_external_authorization(self.authorization)
        book_hash = _sha256(book_json)
        metadata_hash = _sha256(metadata_path)
        research_hash = _sha256(research_path)
        prompt_hash = prompt.sha256
        try:
            candidate = self.title_synthesizer(
                research=research,
                catalog=catalog,
                model=self.title_model,
                prompt_text=prompt.text,
                request_root=episode_root / "analysis" / ".private" / "requests",
            )
        except Exception as exc:
            _wrap_stable_error(exc, "title_value_invalid")
        _require_unchanged(book_json, book_hash, "content_input_changed")
        _require_unchanged(metadata_path, metadata_hash, "content_input_changed")
        _require_unchanged(research_path, research_hash, "content_input_changed")
        prompt = _current_prompt(self._title_prompt_path, prompt_hash)
        book, book_json = _load_current_book(self.store, context)
        metadata_path = _load_title_metadata(episode_root, book)
        value = _require_valid_title_value(candidate, catalog)
        analysis = episode_root / "analysis"
        provenance = TitleValueProvenance(
            book_research_sha256=research_hash,
            prompt_sha256=prompt.sha256,
            source_urls=tuple(source.url for source in research.sources),
            uncertainties=tuple(research.uncertainties),
            catalog=catalog,
        )
        provenance_path = analysis / "title_value_provenance.json"
        card_path = analysis / "book_value_card.json"
        try:
            atomic_write_json(provenance_path, provenance.model_dump(mode="json"))
            TitleValueProvenance.model_validate_json(provenance_path.read_text(encoding="utf-8"))
            atomic_write_json(card_path, value.model_dump(mode="json"))
            card = WholeBookValue.model_validate_json(card_path.read_text(encoding="utf-8"))
        except Exception:
            _raise("book_value_card_invalid")
        if card.model_dump(mode="json") != value.model_dump(mode="json"):
            _raise("book_value_card_invalid")
        return StageOutcome(
            outputs={"book_value_card": card_path, "title_value_provenance": provenance_path},
            inputs={
                "book_state": _sha256(book_json),
                "title_metadata": _sha256(metadata_path),
                "book_research": _sha256(research_path),
                "title_book_synthesis_prompt": prompt.sha256,
            },
        )


def _book_root(store: StateStore, context: StageContext) -> Path:
    _assert_safe_segment(context.book_id)
    _assert_safe_segment(context.episode_id)
    root = store.root / "books" / context.book_id
    expected_episode_root = root / "episodes" / context.episode_id
    if os.path.abspath(os.fspath(context.episode_root)) != os.path.abspath(
        os.fspath(expected_episode_root)
    ):
        _raise("episode_root_invalid")
    return root


def _load_value_card(context: StageContext) -> tuple[WholeBookValue, Path]:
    path = Path(context.episode_root) / "analysis" / "book_value_card.json"
    _require_nonempty_file(path, "book_value_card_invalid")
    try:
        return WholeBookValue.model_validate_json(path.read_text(encoding="utf-8")), path
    except Exception:
        _raise("book_value_card_invalid")
        raise AssertionError("unreachable")


def _load_episode_brief(context: StageContext) -> tuple[EpisodeBrief, Path]:
    path = Path(context.episode_root) / "topic" / "episode_brief.json"
    _require_nonempty_file(path, "episode_brief_invalid")
    try:
        brief = EpisodeBrief.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        _raise("episode_brief_invalid")
        raise AssertionError("unreachable")
    if brief.episode_id != context.episode_id:
        _raise("episode_brief_identity_mismatch")
    return brief, path


class EpisodeBriefStage:
    """Create the one durable E001 whole-book-value topic."""

    def __init__(
        self,
        store: StateStore,
        *,
        authorization: RuntimeAuthorization = RuntimeAuthorization(),
        service_factory: Callable[[Path], TopicService],
        topic_prompt_path: Path | None = None,
        dedup_prompt_path: Path | None = None,
        grok_prompt_path: Path | None = None,
    ) -> None:
        self.store = store
        self.authorization = authorization
        self.service_factory = service_factory
        self._prompt_paths = {
            "topic": Path(topic_prompt_path or _DEFAULT_TOPIC_PROMPT),
            "dedup": Path(dedup_prompt_path or _DEFAULT_TOPIC_DEDUP_PROMPT),
            "grok": Path(grok_prompt_path or _DEFAULT_GROK_TOPIC_PROMPT),
        }
        self._prompt_hashes = {
            name: load_prompt(path).sha256 for name, path in self._prompt_paths.items()
        }

    def run(self, context: StageContext) -> StageOutcome:
        if context.episode_id != "E001":
            _raise("first_episode_required")
        value, value_path = _load_value_card(context)
        prompts = {
            name: _current_prompt(path, self._prompt_hashes[name])
            for name, path in self._prompt_paths.items()
        }
        require_external_authorization(self.authorization)
        value_hash = _sha256(value_path)
        try:
            brief = self.service_factory(_book_root(self.store, context)).generate_first_topic(value)
        except Exception as exc:
            _wrap_stable_error(exc, "topic_generation_failed")
        _require_unchanged(value_path, value_hash, "content_input_changed")
        for name, path in self._prompt_paths.items():
            _current_prompt(path, prompts[name].sha256)
        if not isinstance(brief, EpisodeBrief) or brief.episode_id != context.episode_id:
            _raise("episode_brief_invalid")
        output = Path(context.episode_root) / "topic" / "episode_brief.json"
        try:
            atomic_write_json(output, brief.model_dump(mode="json"))
            persisted = EpisodeBrief.model_validate_json(output.read_text(encoding="utf-8"))
        except Exception:
            _raise("episode_brief_invalid")
        if persisted != brief:
            _raise("episode_brief_invalid")
        return StageOutcome(
            outputs={"episode_brief": output},
            inputs={
                "book_value_card": _sha256(value_path),
                **{f"topic_{name}_prompt": prompt.sha256 for name, prompt in prompts.items()},
            },
        )


class ScriptDraftStage:
    """Generate one reviewed, humanized script package from a locked E001 brief."""

    def __init__(
        self,
        store: StateStore,
        *,
        authorization: RuntimeAuthorization = RuntimeAuthorization(),
        service_factory: Callable[[Path], ScriptService],
        human_writing_skill_path: Path,
        draft_prompt_path: Path | None = None,
        fact_diff_prompt_path: Path | None = None,
        grok_prompt_path: Path | None = None,
    ) -> None:
        self.store = store
        self.authorization = authorization
        self.service_factory = service_factory
        self.human_writing_skill_path = Path(human_writing_skill_path)
        self._prompt_paths = {
            "draft": Path(draft_prompt_path or _DEFAULT_SCRIPT_DRAFT_PROMPT),
            "fact_diff": Path(fact_diff_prompt_path or _DEFAULT_FACT_DIFF_PROMPT),
            "grok": Path(grok_prompt_path or _DEFAULT_GROK_SCRIPT_PROMPT),
        }
        self._prompt_hashes = {
            name: load_prompt(path).sha256 for name, path in self._prompt_paths.items()
        }

    def run(self, context: StageContext) -> StageOutcome:
        value, value_path = _load_value_card(context)
        brief, brief_path = _load_episode_brief(context)
        prompts = {
            name: _current_prompt(path, self._prompt_hashes[name])
            for name, path in self._prompt_paths.items()
        }
        _require_nonempty_file(self.human_writing_skill_path, "human_writing_skill_invalid")
        require_external_authorization(self.authorization)
        value_hash, brief_hash = _sha256(value_path), _sha256(brief_path)
        try:
            skill_hash = _human_writing_skill_sha256(
                self.human_writing_skill_path
            )
        except OSError:
            _raise("human_writing_skill_invalid")
        try:
            service = self.service_factory(_book_root(self.store, context))
            if getattr(service, "prompt_hashes", None) != {
                name: prompt.sha256 for name, prompt in prompts.items()
            }:
                _raise("content_prompt_changed")
            package = service.create(brief, value)
        except Exception as exc:
            _wrap_stable_error(exc, "script_draft_failed")
        _require_unchanged(value_path, value_hash, "content_input_changed")
        _require_unchanged(brief_path, brief_hash, "content_input_changed")
        _require_human_writing_skill_unchanged(
            self.human_writing_skill_path, skill_hash
        )
        for name, path in self._prompt_paths.items():
            _current_prompt(path, prompts[name].sha256)
        if not isinstance(package, ScriptPackage):
            _raise("script_package_invalid")
        if package.human_writing_sha256 != skill_hash:
            _raise("human_writing_skill_changed")
        script_root = Path(context.episode_root) / "script"
        package_path = script_root / "script_package.json"
        review_path = script_root / "review.md"
        try:
            atomic_write_json(package_path, package.model_dump(mode="json"))
            persisted = ScriptPackage.model_validate_json(package_path.read_text(encoding="utf-8"))
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review_path.write_text(build_script_review(package), encoding="utf-8")
            _require_nonempty_file(review_path, "script_review_invalid")
        except Exception:
            _raise("script_package_invalid")
        if persisted != package:
            _raise("script_package_invalid")
        return StageOutcome(
            outputs={"script_package": package_path, "script_review": review_path},
            inputs={
                "book_value_card": _sha256(value_path),
                "episode_brief": _sha256(brief_path),
                "human_writing_skill": skill_hash,
                **{f"script_{name}_prompt": prompt.sha256 for name, prompt in prompts.items()},
            },
        )


class ScriptRevisionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_package_sha256: str
    review_sha256: str
    reviewed_voiceover_sha256: str
    reviewed_draft_sha256: str
    fact_diff_sha256: str
    semantic_lock_sha256: str
    human_writing_sha256: str
    human_writing_checker_passed: Literal[True]
    prompt_sha256: dict[str, str]


def prepare_script_revision(
    *,
    store: StateStore,
    book_id: str,
    episode_id: str,
    reviewed_draft: ScriptDraft,
    fact_diff: FactDiffResult,
    human_writing_skill_path: Path,
    prompt_paths: Mapping[str, Path] | None = None,
) -> Path:
    """Prepare a reviewed script revision and anchor it in workflow state."""
    _assert_safe_segment(book_id)
    _assert_safe_segment(episode_id)
    episode = store.load_episode(book_id, episode_id)
    if episode.status != "awaiting_script_review":
        _raise("script_revision_not_ready")
    episode_root = store.root / "books" / book_id / "episodes" / episode_id
    script_root = episode_root / "script"
    package_path = script_root / "script_package.json"
    review_path = script_root / "review.md"
    _require_nonempty_file(package_path, "script_package_invalid")
    _require_nonempty_file(review_path, "script_review_invalid")
    try:
        package = ScriptPackage.model_validate_json(package_path.read_text(encoding="utf-8"))
        reviewed_voiceover = extract_review_voiceover(review_path.read_text(encoding="utf-8"))
    except Exception:
        _raise("script_review_invalid")
    if reviewed_voiceover == package.voiceover:
        _raise("script_revision_unchanged")
    if reviewed_draft.recommended_voiceover != reviewed_voiceover:
        _raise("script_revision_draft_mismatch")
    if not _base_script_package_is_anchored(episode, package_path):
        _raise("script_package_integrity_failed")
    if package.content_hashes.get("semantic_lock") != package.semantic_lock.sha256:
        _raise("script_package_integrity_failed")

    validation = validate_value_script(
        reviewed_voiceover,
        package.semantic_lock,
        reviewed_draft,
        human_revision=True,
    )
    if not validation.valid:
        _raise("script_revision_validation_failed")
    reviewed_hash = hashlib.sha256(reviewed_voiceover.encode("utf-8")).hexdigest()
    if (
        fact_diff.script_sha256 != reviewed_hash
        or not fact_diff.valid
        or any(
            (
                fact_diff.violations,
                fact_diff.unknown_claims,
                fact_diff.changed_numbers,
                fact_diff.changed_negations,
                fact_diff.dropped_clusters,
                fact_diff.scope_expansion,
                fact_diff.unsupported_attribution,
            )
        )
    ):
        _raise("script_revision_fact_diff_failed")

    paths = {
        "draft": Path(_DEFAULT_SCRIPT_DRAFT_PROMPT),
        "fact_diff": Path(_DEFAULT_FACT_DIFF_PROMPT),
        "grok": Path(_DEFAULT_GROK_SCRIPT_PROMPT),
    }
    if prompt_paths is not None:
        paths = {name: Path(path) for name, path in prompt_paths.items()}
    if set(paths) != {"draft", "fact_diff", "grok"}:
        _raise("content_prompt_changed")
    prompts = {name: load_prompt(path).sha256 for name, path in paths.items()}
    if any(package.content_hashes.get(f"prompt_{name}") != digest for name, digest in prompts.items()):
        _raise("content_prompt_changed")

    skill_path = Path(human_writing_skill_path)
    request_root = episode_root / ".private" / "requests"
    if _path_chain_redirected(request_root):
        _raise("unsafe_script_revision_path")
    try:
        request_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        _raise("unsafe_script_revision_path")
    if _path_chain_redirected(request_root):
        _raise("unsafe_script_revision_path")
    try:
        skill_hash = check_human_writing_candidate(
            voiceover=reviewed_voiceover,
            skill_path=skill_path,
            request_root=request_root,
        )
    except Exception:
        _raise("human_writing_checker_failed")
    if package.human_writing_sha256 != skill_hash:
        _raise("human_writing_skill_changed")

    revision_root = episode_root / ".private" / "script_revision"
    draft_path = revision_root / "reviewed_draft.json"
    fact_diff_path = revision_root / "fact_diff.json"
    manifest_path = revision_root / "revision_manifest.json"
    if any(_path_chain_redirected(path) for path in (episode_root, revision_root, draft_path, fact_diff_path, manifest_path)):
        _raise("unsafe_script_revision_path")
    draft_payload = reviewed_draft.model_dump(mode="json")
    fact_diff_payload = fact_diff.model_dump(mode="json")
    draft_hash = _hash_json_payload(draft_payload)
    fact_diff_hash = _hash_json_payload(fact_diff_payload)
    revision_manifest = ScriptRevisionManifest(
        base_package_sha256=sha256_file(package_path),
        review_sha256=sha256_file(review_path),
        reviewed_voiceover_sha256=reviewed_hash,
        reviewed_draft_sha256=draft_hash,
        fact_diff_sha256=fact_diff_hash,
        semantic_lock_sha256=package.semantic_lock.sha256,
        human_writing_sha256=skill_hash,
        human_writing_checker_passed=True,
        prompt_sha256=prompts,
    )
    try:
        atomic_write_json(draft_path, draft_payload)
        atomic_write_json(fact_diff_path, fact_diff_payload)
        atomic_write_json(manifest_path, revision_manifest.model_dump(mode="json"))
    except Exception:
        _raise("script_revision_write_failed")
    refs = {
        name: _artifact_ref(path)
        for name, path in {
            "reviewed_draft": draft_path,
            "fact_diff": fact_diff_path,
            "revision_manifest": manifest_path,
        }.items()
    }
    base_manifest = episode.stage_manifests.get("draft_script")
    config_sha256 = base_manifest.config_sha256 if base_manifest is not None else "0" * 64
    episode.stage_manifests[_SCRIPT_REVISION_STAGE] = StageManifest(
        stage=_SCRIPT_REVISION_STAGE,
        status="completed",
        inputs={
            "base_package": revision_manifest.base_package_sha256,
            "review": revision_manifest.review_sha256,
            "semantic_lock": revision_manifest.semantic_lock_sha256,
            "human_writing_skill": revision_manifest.human_writing_sha256,
            **{f"script_{name}_prompt": digest for name, digest in prompts.items()},
        },
        outputs=refs,
        config_sha256=config_sha256,
    )
    if _SCRIPT_REVISION_STAGE not in episode.completed_stages:
        episode.completed_stages.append(_SCRIPT_REVISION_STAGE)
    store.save_episode(episode)
    return manifest_path


class ScriptApprovalStage:
    """Validate the reviewed script and publish the approved local narration source."""

    def __init__(
        self,
        store: StateStore,
        *,
        topic_service_factory: Callable[[Path], TopicService],
        human_writing_skill_path: Path,
        draft_prompt_path: Path | None = None,
        fact_diff_prompt_path: Path | None = None,
        grok_prompt_path: Path | None = None,
    ) -> None:
        self.store = store
        self.topic_service_factory = topic_service_factory
        self.human_writing_skill_path = Path(human_writing_skill_path)
        self._prompt_paths = {
            "draft": Path(draft_prompt_path or _DEFAULT_SCRIPT_DRAFT_PROMPT),
            "fact_diff": Path(fact_diff_prompt_path or _DEFAULT_FACT_DIFF_PROMPT),
            "grok": Path(grok_prompt_path or _DEFAULT_GROK_SCRIPT_PROMPT),
        }
        self._prompt_hashes = {
            name: load_prompt(path).sha256 for name, path in self._prompt_paths.items()
        }

    def run(self, context: StageContext) -> StageOutcome:
        script_root = Path(context.episode_root) / "script"
        story_manifest_path = script_root / "story_candidate.json"
        episode = context.episode_state or self.store.load_episode(
            context.book_id, context.episode_id
        )
        draft_manifest = episode.stage_manifests.get("draft_script")
        if (
            story_manifest_path.is_file()
            and draft_manifest is not None
            and "story_candidate" in draft_manifest.outputs
        ):
            try:
                result = approve_story_candidate(
                    self.store, context.book_id, context.episode_id
                )
            except Exception as exc:
                _wrap_stable_error(exc, "script_approval_failed")
            if context.episode_state is not None:
                context.episode_state.script_hash = result.episode_state.script_hash
                context.episode_state.completed_stages = list(
                    result.episode_state.completed_stages
                )
                context.episode_state.stale_stages = list(
                    result.episode_state.stale_stages
                )
                context.episode_state.stage_manifests = dict(
                    result.episode_state.stage_manifests
                )
            return StageOutcome(
                outputs={
                    "approved_script": result.approved_path,
                    "script_manifest": result.manifest_path,
                },
                inputs={"story_candidate_manifest": _sha256(story_manifest_path)},
            )
        package_path = script_root / "script_package.json"
        review_path = script_root / "review.md"
        _require_nonempty_file(package_path, "script_package_invalid")
        _require_nonempty_file(review_path, "script_review_invalid")
        try:
            package = ScriptPackage.model_validate_json(package_path.read_text(encoding="utf-8"))
            reviewed_voiceover = extract_review_voiceover(review_path.read_text(encoding="utf-8"))
        except Exception:
            _raise("script_review_invalid")
        episode = context.episode_state or self.store.load_episode(
            context.book_id, context.episode_id
        )
        if not _base_script_package_is_anchored(episode, package_path):
            _raise("script_package_integrity_failed")
        human_revision = reviewed_voiceover != package.voiceover
        if human_revision:
            reviewed_draft, fact_diff, revision = _load_prepared_script_revision(
                episode=episode,
                episode_root=Path(context.episode_root),
                package_path=package_path,
                review_path=review_path,
                reviewed_voiceover=reviewed_voiceover,
            )
        else:
            reviewed_draft, fact_diff, revision = None, package.fact_diff, None
        prompts = {
            name: _current_prompt(path, self._prompt_hashes[name])
            for name, path in self._prompt_paths.items()
        }
        for name, prompt in prompts.items():
            if package.content_hashes.get(f"prompt_{name}") != prompt.sha256:
                _raise("content_prompt_changed")
        _require_nonempty_file(self.human_writing_skill_path, "human_writing_skill_invalid")
        try:
            skill_hash = _human_writing_skill_sha256(
                self.human_writing_skill_path
            )
        except OSError:
            _raise("human_writing_skill_invalid")
        if package.human_writing_sha256 != skill_hash:
            _raise("human_writing_skill_changed")
        if human_revision:
            if revision is None or revision.prompt_sha256 != {
                name: prompt.sha256 for name, prompt in prompts.items()
            }:
                _raise("content_prompt_changed")
            try:
                checked_hash = check_human_writing_candidate(
                    voiceover=reviewed_voiceover,
                    skill_path=self.human_writing_skill_path,
                    request_root=Path(context.episode_root) / ".private" / "requests",
                )
            except Exception:
                _raise("human_writing_checker_failed")
            if checked_hash != skill_hash or revision.human_writing_sha256 != skill_hash:
                _raise("human_writing_skill_changed")
        try:
            result = approve_script(
                reviewed_voiceover,
                package.semantic_lock,
                fact_diff,
                episode,
                self.topic_service_factory(_book_root(self.store, context)),
                Path(context.episode_root),
                prompt_hashes={name: prompt.sha256 for name, prompt in prompts.items()},
                human_writing_sha256=package.human_writing_sha256,
                human_writing_checker_passed=package.human_writing_checker_passed,
                human_revision=human_revision,
                revision_draft=reviewed_draft,
            )
        except Exception as exc:
            _wrap_stable_error(exc, "script_approval_failed")
        if context.episode_state is not None:
            context.episode_state.script_hash = result.episode_state.script_hash
        approved_path = script_root / "approved.txt"
        manifest_path = script_root / "script_manifest.json"
        _require_nonempty_file(approved_path, "approved_script_invalid")
        _require_nonempty_file(manifest_path, "approved_script_invalid")
        return StageOutcome(
            outputs={"approved_script": approved_path, "script_manifest": manifest_path},
            inputs={
                "script_package": _sha256(package_path),
                "script_review": _sha256(review_path),
                "human_writing_skill": skill_hash,
                **{f"script_{name}_prompt": prompt.sha256 for name, prompt in prompts.items()},
            },
        )


def _base_script_package_is_anchored(episode: object, package_path: Path) -> bool:
    try:
        manifest = episode.stage_manifests["draft_script"]
        ref = manifest.outputs["script_package"]
    except (AttributeError, KeyError, TypeError):
        return False
    return _artifact_matches(package_path, ref)


def _load_prepared_script_revision(
    *,
    episode: object,
    episode_root: Path,
    package_path: Path,
    review_path: Path,
    reviewed_voiceover: str,
) -> tuple[ScriptDraft, FactDiffResult, ScriptRevisionManifest]:
    try:
        stage = episode.stage_manifests[_SCRIPT_REVISION_STAGE]
        if stage.status != "completed":
            raise ValueError
        revision_root = Path(episode_root) / ".private" / "script_revision"
        paths = {
            "reviewed_draft": revision_root / "reviewed_draft.json",
            "fact_diff": revision_root / "fact_diff.json",
            "revision_manifest": revision_root / "revision_manifest.json",
        }
        if any(not _artifact_matches(path, stage.outputs[name]) for name, path in paths.items()):
            raise ValueError
        revision = ScriptRevisionManifest.model_validate_json(
            paths["revision_manifest"].read_text(encoding="utf-8")
        )
        draft = ScriptDraft.model_validate_json(
            paths["reviewed_draft"].read_text(encoding="utf-8")
        )
        fact_diff = FactDiffResult.model_validate_json(
            paths["fact_diff"].read_text(encoding="utf-8")
        )
        if (
            revision.base_package_sha256 != sha256_file(package_path)
            or revision.review_sha256 != sha256_file(review_path)
            or revision.reviewed_voiceover_sha256
            != hashlib.sha256(reviewed_voiceover.encode("utf-8")).hexdigest()
            or revision.reviewed_draft_sha256
            != _hash_json_payload(draft.model_dump(mode="json"))
            or revision.fact_diff_sha256
            != _hash_json_payload(fact_diff.model_dump(mode="json"))
            or draft.recommended_voiceover != reviewed_voiceover
            or fact_diff.script_sha256 != revision.reviewed_voiceover_sha256
            or stage.inputs.get("base_package") != revision.base_package_sha256
            or stage.inputs.get("review") != revision.review_sha256
            or stage.inputs.get("semantic_lock") != revision.semantic_lock_sha256
            or stage.inputs.get("human_writing_skill") != revision.human_writing_sha256
        ):
            raise ValueError
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        _raise("script_revision_evidence_invalid")
    return draft, fact_diff, revision


def _artifact_ref(path: Path) -> ArtifactRef:
    candidate = Path(path)
    return ArtifactRef(
        path=str(candidate),
        sha256=sha256_file(candidate),
        size_bytes=candidate.stat().st_size,
    )


def _artifact_matches(path: Path, ref: ArtifactRef) -> bool:
    candidate = Path(path)
    try:
        return (
            Path(ref.path) == candidate
            and candidate.is_file()
            and not _path_chain_redirected(candidate)
            and ref.sha256 == sha256_file(candidate)
            and ref.size_bytes == candidate.stat().st_size
        )
    except (OSError, TypeError, ValueError):
        return False


def _hash_json_payload(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _path_chain_redirected(path: Path) -> bool:
    return any(
        item.exists() and _is_reparse_or_symlink(item)
        for item in (Path(path), *Path(path).parents)
    )
