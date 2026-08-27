"""RED tests for whole-book research schema and research_book orchestration.

Production module under test: bv.content.research (absent in this pass).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bv.state.models import BookState


# ---------------------------------------------------------------------------
# Production import (lazy so collection succeeds while the module is missing)
# ---------------------------------------------------------------------------


def _load_research_api() -> tuple[Any, ...]:
    """Import production symbols only inside tests so collection stays green."""
    from bv.content.research import (
        BookResearch,
        BookResearchError,
        BookResearchResult,
        ResearchSource,
        research_book,
    )

    return (
        BookResearch,
        BookResearchError,
        BookResearchResult,
        ResearchSource,
        research_book,
    )


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------

_TITLE = "被讨厌的勇气"
_AUTHORS = ("岸见一郎", "古贺史健")

_TRUSTED_PROMPT = (
    "Research the whole-book reader value of the named work.\n"
    "Verify material identity against publisher and catalog evidence.\n"
    "Cite source URLs for important judgments.\n"
    "Disclose uncertainty when evidence is incomplete.\n"
    "Do not invent quotations.\n"
)


def _book(
    *,
    title: str = _TITLE,
    authors: list[str] | None = None,
    book_id: str = "book-courage-demo",
) -> BookState:
    return BookState(
        book_id=book_id,
        title=title,
        authors=list(authors if authors is not None else _AUTHORS),
        source_mode="title_author",
    )


def _source_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "url": "https://www.diamond.co.jp/book/9784478025819.html",
        "source_type": "publisher",
        "supports": ("书名与作者身份", "出版信息"),
    }
    payload.update(overrides)
    return payload


def _research_payload(**overrides: Any) -> dict[str, Any]:
    """A whole-book research record, not an isolated-concept summary."""
    payload: dict[str, Any] = {
        "title": _TITLE,
        "authors": list(_AUTHORS),
        "identity_confidence": "high",
        "central_question": "人能否摆脱对他人认可的依赖，按自己的选择生活？",
        "argument_or_narrative_arc": [
            "从人际关系的课题分离出发",
            "说明认可欲求如何制造自由的假象",
            "落到可承担的自我决定与共同体感觉",
        ],
        "key_ideas": [
            "课题分离：分清自己的课题与他人的课题",
            "认可欲求会把人生的决定权交给他人",
            "共同体感觉建立在贡献而非讨好之上",
        ],
        "life_connections": [
            "回复消息前反复猜测对方评价",
            "在工作选择里用他人眼光替代自己判断",
        ],
        "reader_value": [
            "能区分自己该负责的行动和他人的评价",
            "在重要关系里保留边界而不必切断联系",
        ],
        "misunderstandings_and_boundaries": [
            "不是鼓励自私或切断一切关系",
            "不保证消除所有人际痛苦",
        ],
        "sources": [
            _source_payload(),
            _source_payload(
                url="https://example.com/reviews/courage-to-be-disliked",
                source_type="professional_review",
                supports=("全书论证主线概述",),
            ),
        ],
        "uncertainties": [
            "部分二手解读是否准确反映原书措辞仍待核验",
        ],
    }
    payload.update(overrides)
    return payload


def _valid_research(**overrides: Any) -> Any:
    BookResearch, *_ = _load_research_api()
    return BookResearch.model_validate(_research_payload(**overrides))


def _valid_source(**overrides: Any) -> Any:
    _, _, _, ResearchSource, _ = _load_research_api()
    return ResearchSource.model_validate(_source_payload(**overrides))


class _FakeResearchModel:
    """Records complete() calls and returns a scripted BookResearch (never a process)."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(self, prompt: str, schema_type: type, request_dir: Path) -> object:
        request_path = Path(request_dir)
        self.calls.append(
            {
                "prompt": prompt,
                "schema_type": schema_type,
                "request_dir": request_path,
            }
        )
        assert request_path.is_dir()
        assert list(request_path.iterdir()) == []
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _make_symlink_or_skip(
    link: Path, target: Path, *, target_is_directory: bool
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError:
        pytest.skip("symlink creation requires additional privilege")
    if not link.exists() and not link.is_symlink():
        pytest.skip("symlink creation is unavailable on this platform")


def _prompt_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_research(
    tmp_path: Path,
    model: _FakeResearchModel,
    *,
    book: BookState | None = None,
    prompt_text: str = _TRUSTED_PROMPT,
    output_root: Path | None = None,
) -> Any:
    *_, research_book = _load_research_api()
    root = output_root if output_root is not None else tmp_path / "research-out"
    return research_book(
        book=book if book is not None else _book(),
        model=model,
        output_root=root,
        prompt_text=prompt_text,
    )


def _seed_existing_artifacts(output_root: Path) -> tuple[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "book_research.json"
    md_path = output_root / "book_research.md"
    json_path.write_text('{"marker":"KEEP_JSON"}', encoding="utf-8")
    md_path.write_text("KEEP_MARKDOWN\n", encoding="utf-8")
    return json_path.read_text(encoding="utf-8"), md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Strict models: ResearchSource
# ---------------------------------------------------------------------------


def test_research_source_accepts_valid_http_source() -> None:
    source = _valid_source()
    assert source.url.startswith("https://")
    assert source.source_type == "publisher"
    assert source.supports == ("书名与作者身份", "出版信息")


def test_research_source_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _valid_source(extra="forbidden")


@pytest.mark.parametrize(
    "overrides",
    [
        {"url": ""},
        {"url": "   "},
        {"url": "ftp://example.com/book"},
        {"url": "file:///tmp/book.html"},
        {"url": "example.com/no-scheme"},
        {"source_type": "blog"},
        {"source_type": ""},
        {"supports": ()},
        {"supports": ("",)},
        {"supports": ("valid", "  ")},
    ],
)
def test_research_source_rejects_blank_unsupported_or_empty_support(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        _valid_source(**overrides)


def test_research_source_allows_every_documented_source_type() -> None:
    _, _, _, ResearchSource, _ = _load_research_api()
    allowed = (
        "publisher",
        "author_interview",
        "official_sample",
        "table_of_contents",
        "academic",
        "professional_review",
        "reader_reception",
        "other",
    )
    for source_type in allowed:
        source = ResearchSource.model_validate(
            _source_payload(
                url=f"https://example.com/{source_type}",
                source_type=source_type,
            )
        )
        assert source.source_type == source_type


# ---------------------------------------------------------------------------
# Strict models: BookResearch
# ---------------------------------------------------------------------------


def test_book_research_accepts_whole_book_fixture() -> None:
    research = _valid_research()
    assert research.title == _TITLE
    assert research.authors == _AUTHORS
    assert research.identity_confidence == "high"
    assert len(research.key_ideas) >= 2
    assert len(research.sources) >= 1
    assert research.uncertainties  # fixture discloses uncertainty; empty is also allowed


def test_book_research_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _valid_research(extra_field="nope")


def test_book_research_allows_empty_uncertainties() -> None:
    research = _valid_research(uncertainties=[])
    assert research.uncertainties == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": ""},
        {"title": "  "},
        {"authors": []},
        {"authors": ("岸见一郎", "")},
        {"authors": ("  ",)},
        {"identity_confidence": "low"},
        {"identity_confidence": "certain"},
        {"central_question": ""},
        {"central_question": "   "},
        {"argument_or_narrative_arc": []},
        {"argument_or_narrative_arc": ("valid", "  ")},
        {"key_ideas": []},
        {"key_ideas": ("only-one-idea",)},
        {"key_ideas": ("idea-one", "")},
        {"life_connections": []},
        {"life_connections": ("ok", " ")},
        {"reader_value": []},
        {"reader_value": ("", "ok")},
        {"misunderstandings_and_boundaries": []},
        {"misunderstandings_and_boundaries": ("ok", "")},
        {"sources": []},
        {"uncertainties": ("valid", "")},
        {"uncertainties": ("  ",)},
    ],
)
def test_book_research_rejects_blank_sparse_or_low_confidence_fields(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        _valid_research(**overrides)


def test_book_research_rejects_duplicate_source_urls_after_safe_normalization() -> None:
    with pytest.raises(ValidationError):
        _valid_research(
            sources=[
                _source_payload(url=" https://Example.COM/path/a "),
                _source_payload(
                    url="HTTPS://example.com/path/a",
                    source_type="other",
                    supports=("同一来源的重复条目",),
                ),
            ]
        )


def test_book_research_accepts_distinct_urls_that_only_share_host() -> None:
    research = _valid_research(
        sources=[
            _source_payload(url="https://example.com/path/a"),
            _source_payload(
                url="https://example.com/path/b",
                source_type="academic",
                supports=("另一条不同路径",),
            ),
        ]
    )
    assert len(research.sources) == 2


# ---------------------------------------------------------------------------
# Strict models: BookResearchResult
# ---------------------------------------------------------------------------


def test_book_research_result_requires_accepted_paths_and_prompt_hash(
    tmp_path: Path,
) -> None:
    BookResearch, _, BookResearchResult, _, _ = _load_research_api()
    research = _valid_research()
    prompt_sha = _prompt_sha(_TRUSTED_PROMPT)
    result = BookResearchResult(
        status="accepted",
        research=research,
        json_path=tmp_path / "book_research.json",
        markdown_path=tmp_path / "book_research.md",
        prompt_sha256=prompt_sha,
    )
    assert result.status == "accepted"
    assert result.research == research
    assert result.json_path == tmp_path / "book_research.json"
    assert result.markdown_path == tmp_path / "book_research.md"
    assert result.prompt_sha256 == prompt_sha
    assert len(result.prompt_sha256) == 64
    assert result.prompt_sha256 == result.prompt_sha256.lower()
    assert set(result.prompt_sha256) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "pending"},
        {"status": "failed"},
        {"prompt_sha256": "abc"},
        {"prompt_sha256": "A" * 64},
        {"prompt_sha256": "g" * 64},
    ],
)
def test_book_research_result_rejects_non_accepted_or_invalid_hash(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    _, _, BookResearchResult, _, _ = _load_research_api()
    fields: dict[str, Any] = {
        "status": "accepted",
        "research": _valid_research(),
        "json_path": tmp_path / "book_research.json",
        "markdown_path": tmp_path / "book_research.md",
        "prompt_sha256": "a" * 64,
    }
    fields.update(overrides)
    with pytest.raises(ValidationError):
        BookResearchResult.model_validate(fields)


# ---------------------------------------------------------------------------
# research_book orchestration
# ---------------------------------------------------------------------------


def test_research_book_is_keyword_only(tmp_path: Path) -> None:
    *_, research_book = _load_research_api()
    model = _FakeResearchModel(_valid_research())
    with pytest.raises(TypeError):
        research_book(_book(), model, tmp_path / "out", _TRUSTED_PROMPT)  # type: ignore[misc]


def test_successful_research_publishes_json_and_markdown_atomically(
    tmp_path: Path,
) -> None:
    BookResearch, _, _, _, _ = _load_research_api()
    research = _valid_research()
    model = _FakeResearchModel(research)
    output_root = tmp_path / "out"

    result = _run_research(tmp_path, model, output_root=output_root)

    assert result.status == "accepted"
    assert result.research == research
    assert result.json_path == output_root / "book_research.json"
    assert result.markdown_path == output_root / "book_research.md"
    assert result.prompt_sha256 == _prompt_sha(_TRUSTED_PROMPT)
    assert len(result.prompt_sha256) == 64
    assert result.prompt_sha256 == result.prompt_sha256.lower()

    public_names = {
        path.name for path in output_root.iterdir() if path.name != ".private"
    }
    assert public_names == {"book_research.json", "book_research.md"}
    assert not list(output_root.rglob("*.tmp"))

    round_tripped = BookResearch.model_validate(
        json.loads(result.json_path.read_text(encoding="utf-8"))
    )
    assert round_tripped == research

    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert research.title in markdown
    assert research.central_question in markdown
    for item in research.argument_or_narrative_arc:
        assert item in markdown
    for item in research.key_ideas:
        assert item in markdown
    for item in research.life_connections:
        assert item in markdown
    for item in research.reader_value:
        assert item in markdown
    for item in research.misunderstandings_and_boundaries:
        assert item in markdown
    for source in research.sources:
        assert source.url in markdown
    for item in research.uncertainties:
        assert item in markdown


def test_success_cleanup_preserves_unrelated_staging_named_sibling(
    tmp_path: Path,
) -> None:
    """Success cleanup must not delete non-transaction siblings that only look like debris.

    A normal file such as ``book_research.json.staging-notes.txt`` shares a name
    prefix with the final artifact and contains ``staging``, but this run did not
    create it. Cleanup must only remove paths owned/tracked by this transaction.
    """
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    sibling = output_root / "book_research.json.staging-notes.txt"
    known_bytes = b"user notes about staging workflow - not transaction debris\n"
    sibling.write_bytes(known_bytes)

    model = _FakeResearchModel(_valid_research())
    result = _run_research(tmp_path, model, output_root=output_root)

    assert result.status == "accepted"
    assert sibling.is_file()
    assert sibling.read_bytes() == known_bytes


def test_composed_prompt_keeps_trusted_instructions_and_encodes_identity_only(
    tmp_path: Path,
) -> None:
    BookResearch, _, _, _, _ = _load_research_api()
    model = _FakeResearchModel(_valid_research())
    output_root = tmp_path / "out"
    book = _book()

    result = _run_research(tmp_path, model, book=book, output_root=output_root)

    assert result.status == "accepted"
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["schema_type"] is BookResearch

    prompt = call["prompt"]
    assert isinstance(prompt, str)
    assert prompt.count("BEGIN_SOURCE_DATA") == prompt.count("END_SOURCE_DATA") == 1
    trusted, encoded_source = prompt.split("BEGIN_SOURCE_DATA\n", maxsplit=1)
    source_raw, _ = encoded_source.split("END_SOURCE_DATA", maxsplit=1)

    assert _TRUSTED_PROMPT.strip() in trusted
    for phrase in (
        "whole-book reader value",
        "material identity",
        "source URLs",
        "uncertainty",
        "invent quotations",
    ):
        assert phrase.casefold() in trusted.casefold()

    # Untrusted block is JSON-encoded source data (possibly double-encoded string).
    decoded = json.loads(source_raw)
    if isinstance(decoded, str):
        payload = json.loads(decoded)
    else:
        payload = decoded
    assert isinstance(payload, dict)
    assert payload.get("title") == book.title
    authors = payload.get("authors")
    assert authors == list(book.authors) or authors == tuple(book.authors)
    # Title and authors are the only source identity input; no ebook body.
    allowed_keys = {"title", "authors", "book_id"}
    assert set(payload) <= allowed_keys
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "chapter" not in serialized.casefold()
    assert "paragraph" not in serialized.casefold()
    assert "epub" not in serialized.casefold()

    request_dir = call["request_dir"]
    assert isinstance(request_dir, Path)
    requests_root = output_root / ".private" / "requests"
    assert requests_root in request_dir.parents or request_dir.parent == requests_root
    assert request_dir.is_relative_to(requests_root)
    assert request_dir != requests_root
    assert request_dir.is_dir()


def test_equivalent_nfkc_casefold_punctuation_identity_is_accepted(
    tmp_path: Path,
) -> None:
    book = _book(title="Demo Book", authors=["Jane Doe", "Alex Writer"])
    # Fullwidth / case / surrounding punctuation that normalize to the book identity.
    research = _valid_research(
        title=" Ｄｅｍｏ Ｂｏｏｋ！ ",
        authors=[" JANE DOE ", "alex writer."],
        identity_confidence="medium",
    )
    model = _FakeResearchModel(research)

    result = _run_research(tmp_path, model, book=book)

    assert result.status == "accepted"
    assert result.research.identity_confidence == "medium"


@pytest.mark.parametrize(
    "mutation",
    [
        {"title": "另一个书名"},
        {"authors": ["岸见一郎"]},  # missing co-author
        {"authors": ["岸见一郎", "古贺史健", "第三人"]},  # extra author
        {"authors": ["岸见一郎", "别人"]},  # changed author
        {"identity_confidence": "low"},
    ],
)
def test_identity_mismatches_raise_stable_error_and_keep_artifacts(
    tmp_path: Path, mutation: dict[str, Any]
) -> None:
    _, BookResearchError, _, _, research_book = _load_research_api()
    output_root = tmp_path / "out"
    prior_json, prior_md = _seed_existing_artifacts(output_root)

    if "identity_confidence" in mutation and mutation["identity_confidence"] == "low":
        # Schema forbids "low"; mutate after construction to exercise orchestration.
        research = _valid_research()
        object.__setattr__(research, "identity_confidence", "low")
    else:
        research = _valid_research(**mutation)

    model = _FakeResearchModel(research)

    with pytest.raises(BookResearchError) as raised:
        research_book(
            book=_book(),
            model=model,
            output_root=output_root,
            prompt_text=_TRUSTED_PROMPT,
        )

    assert raised.value.error_code == "book_research_identity_mismatch"
    assert str(raised.value) == "book_research_identity_mismatch"
    assert (output_root / "book_research.json").read_text(encoding="utf-8") == prior_json
    assert (output_root / "book_research.md").read_text(encoding="utf-8") == prior_md
    assert not list(output_root.rglob("*.tmp"))


def test_blank_trusted_prompt_is_rejected_before_model_call(tmp_path: Path) -> None:
    _, BookResearchError, _, _, research_book = _load_research_api()
    model = _FakeResearchModel(_valid_research())
    output_root = tmp_path / "out"

    with pytest.raises(BookResearchError) as raised:
        research_book(
            book=_book(),
            model=model,
            output_root=output_root,
            prompt_text="   \n\t  ",
        )

    assert raised.value.error_code == "book_research_prompt_invalid"
    assert model.calls == []
    assert not (output_root / "book_research.json").exists()
    assert not (output_root / "book_research.md").exists()


def test_model_failure_publishes_neither_artifact_nor_overwrites(
    tmp_path: Path,
) -> None:
    from bv.models.grok_research import GrokResearchError

    _, _, _, _, research_book = _load_research_api()
    output_root = tmp_path / "out"
    prior_json, prior_md = _seed_existing_artifacts(output_root)
    model = _FakeResearchModel(GrokResearchError("book_research_process_failed"))

    with pytest.raises(Exception):
        research_book(
            book=_book(),
            model=model,
            output_root=output_root,
            prompt_text=_TRUSTED_PROMPT,
        )

    assert len(model.calls) == 1
    assert (output_root / "book_research.json").read_text(encoding="utf-8") == prior_json
    assert (output_root / "book_research.md").read_text(encoding="utf-8") == prior_md
    assert not list(output_root.rglob("*.tmp"))


def test_validation_failure_from_model_does_not_publish_artifacts(
    tmp_path: Path,
) -> None:
    _, _, _, _, research_book = _load_research_api()
    output_root = tmp_path / "out"
    prior_json, prior_md = _seed_existing_artifacts(output_root)
    # Wrong type: orchestration must not treat it as accepted research.
    model = _FakeResearchModel({"title": _TITLE, "not": "a BookResearch"})

    with pytest.raises(Exception):
        research_book(
            book=_book(),
            model=model,
            output_root=output_root,
            prompt_text=_TRUSTED_PROMPT,
        )

    assert (output_root / "book_research.json").read_text(encoding="utf-8") == prior_json
    assert (output_root / "book_research.md").read_text(encoding="utf-8") == prior_md
    assert not list(output_root.rglob("*.tmp"))


def test_symlink_output_root_is_rejected_as_directory_unsafe(tmp_path: Path) -> None:
    _, BookResearchError, _, _, research_book = _load_research_api()
    real = tmp_path / "real-out"
    real.mkdir()
    link = tmp_path / "linked-out"
    _make_symlink_or_skip(link, real, target_is_directory=True)
    model = _FakeResearchModel(_valid_research())

    with pytest.raises(BookResearchError) as raised:
        research_book(
            book=_book(),
            model=model,
            output_root=link,
            prompt_text=_TRUSTED_PROMPT,
        )

    assert raised.value.error_code == "book_research_directory_unsafe"
    assert model.calls == []
    assert not (real / "book_research.json").exists()
    assert not (real / "book_research.md").exists()


def test_symlink_final_artifact_path_is_rejected_as_directory_unsafe(
    tmp_path: Path,
) -> None:
    _, BookResearchError, _, _, research_book = _load_research_api()
    output_root = tmp_path / "out"
    output_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    target = output_root / "book_research.json"
    _make_symlink_or_skip(target, outside, target_is_directory=False)
    model = _FakeResearchModel(_valid_research())

    with pytest.raises(BookResearchError) as raised:
        research_book(
            book=_book(),
            model=model,
            output_root=output_root,
            prompt_text=_TRUSTED_PROMPT,
        )

    assert raised.value.error_code == "book_research_directory_unsafe"
    assert outside.read_text(encoding="utf-8") == "{}"
    assert not (output_root / "book_research.md").exists() or (
        output_root / "book_research.md"
    ).read_text(encoding="utf-8") != _valid_research().model_dump_json()


def test_atomic_write_failure_leaves_no_partial_final_or_tmp_debris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, _, research_book = _load_research_api()
    import bv.content.research as research_module

    def _fail_atomic_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated atomic write failure")

    # Production should publish via atomic helpers imported into the module.
    if hasattr(research_module, "atomic_write_json"):
        monkeypatch.setattr(research_module, "atomic_write_json", _fail_atomic_write)
    else:
        import bv.core.atomic as atomic_module

        monkeypatch.setattr(atomic_module, "atomic_write_json", _fail_atomic_write)

    if hasattr(research_module, "_atomic_write_text"):
        monkeypatch.setattr(research_module, "_atomic_write_text", _fail_atomic_write)
    if hasattr(research_module, "atomic_write_text"):
        monkeypatch.setattr(research_module, "atomic_write_text", _fail_atomic_write)

    output_root = tmp_path / "out"
    model = _FakeResearchModel(_valid_research())

    with pytest.raises(Exception):
        research_book(
            book=_book(),
            model=model,
            output_root=output_root,
            prompt_text=_TRUSTED_PROMPT,
        )

    json_path = output_root / "book_research.json"
    md_path = output_root / "book_research.md"
    assert not json_path.is_file() or json_path.stat().st_size == 0
    assert not md_path.is_file() or md_path.stat().st_size == 0
    # Prefer neither final artifact exists after a failed publish.
    assert not json_path.exists()
    assert not md_path.exists()
    assert not list(output_root.rglob("*.tmp"))


def test_second_file_publish_failure_preserves_prior_artifact_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed markdown publish must not destroy previously accepted pair.

    JSON-first publish that cleans up both finals on markdown failure would
    overwrite or delete prior accepted artifacts; this regression forbids that.
    """
    _, _, _, _, research_book = _load_research_api()
    import bv.content.research as research_module

    output_root = tmp_path / "out"
    prior_json, prior_md = _seed_existing_artifacts(output_root)
    model = _FakeResearchModel(_valid_research())

    def _fail_markdown_only(path: Path, value: str) -> None:
        raise OSError("simulated markdown publish failure")

    # Narrow seam: JSON atomic write proceeds; markdown text write fails.
    assert hasattr(research_module, "_atomic_write_text")
    monkeypatch.setattr(research_module, "_atomic_write_text", _fail_markdown_only)

    with pytest.raises(Exception):
        research_book(
            book=_book(),
            model=model,
            output_root=output_root,
            prompt_text=_TRUSTED_PROMPT,
        )

    json_path = output_root / "book_research.json"
    md_path = output_root / "book_research.md"
    assert json_path.is_file()
    assert md_path.is_file()
    assert json_path.read_text(encoding="utf-8") == prior_json
    assert md_path.read_text(encoding="utf-8") == prior_md

    debris = [
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".tmp", ".bak"}
            or ".tmp" in path.name
            or ".bak" in path.name
            or "staging" in path.name.casefold()
            or path.name.endswith("~")
        )
    ]
    assert debris == []


def test_artifacts_and_prompt_never_include_full_ebook_body(tmp_path: Path) -> None:
    model = _FakeResearchModel(_valid_research())
    output_root = tmp_path / "out"

    result = _run_research(tmp_path, model, output_root=output_root)

    prompt = model.calls[0]["prompt"]
    assert isinstance(prompt, str)
    for banned in (
        "BEGIN_EBOOK",
        "full text",
        "chapter_001",
        "paragraph body",
        "EPUB_BODY",
    ):
        assert banned not in prompt

    markdown = result.markdown_path.read_text(encoding="utf-8")
    json_text = result.json_path.read_text(encoding="utf-8")
    for banned in ("BEGIN_EBOOK", "EPUB_BODY", "chapter_001 paragraph"):
        assert banned not in markdown
        assert banned not in json_text


def test_prompt_sha256_is_hash_of_exact_trusted_prompt_text(tmp_path: Path) -> None:
    custom = _TRUSTED_PROMPT + "\nExtra trusted line that must affect the hash.\n"
    model = _FakeResearchModel(_valid_research())

    result = _run_research(tmp_path, model, prompt_text=custom)

    assert result.prompt_sha256 == _prompt_sha(custom)
    assert result.prompt_sha256 != _prompt_sha(_TRUSTED_PROMPT)
