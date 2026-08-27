import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

import bv.content.evidence as evidence_module
from bv.content.evidence import ChapterAnalysisService, analyze_chapters
from bv.ebook.models import Chapter, SourceAnchor
from bv.models.contracts import ModelCompletionError, ModelInvalidResponseError


class ScriptedStructuredModel:
    def __init__(
        self,
        responses: list[dict[str, Any]],
        fail_on_call: int | None = None,
    ) -> None:
        self.responses = list(responses)
        self.fail_on_call = fail_on_call
        self.calls = 0

    def complete(self, prompt: str, schema_type: type, request_dir: Path):
        del prompt, request_dir
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("scripted model failure")
        return schema_type.model_validate(self.responses.pop(0))


class RecordingStructuredModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.request_dirs: list[Path] = []
        self.request_dirs_were_empty: list[bool] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def complete(self, prompt: str, schema_type: type, request_dir: Path):
        self.prompts.append(prompt)
        self.request_dirs.append(request_dir)
        self.request_dirs_were_empty.append(not any(request_dir.iterdir()))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return schema_type.model_validate(response)


def valid_analysis(chapter_id: str, excerpt: str = "正文") -> dict[str, object]:
    return {
        "chapter_id": chapter_id,
        "cards": [
            {
                "claim_id": f"{chapter_id}-001",
                "claim_type": "author_argument",
                "paraphrase": "A supported claim",
                "evidence_excerpt": excerpt,
                "chapter_id": chapter_id,
                "start_paragraph": 1,
                "end_paragraph": 1,
                "reasoning": {
                    "author_premise": "p",
                    "mechanism": "m",
                    "conclusion": "c",
                },
                "limitations": [],
                "common_misreading": [],
                "life_signals": [],
                "confidence": "high",
            }
        ],
    }


def make_chapter(chapter_id: str, text: str = "正文") -> Chapter:
    return Chapter(
        chapter_id=chapter_id,
        title=chapter_id,
        text=text,
        anchor=SourceAnchor(
            source_type="txt",
            source_path="book.txt",
            chapter_id=chapter_id,
            start_line=1,
            end_line=1,
        ),
        sha256=chapter_id.ljust(64, "0")[:64],
    )


def make_service(
    tmp_path: Path,
    model: object,
    **changes: object,
) -> ChapterAnalysisService:
    values: dict[str, object] = {
        "model": model,
        "output_dir": tmp_path / "analysis",
        "prompt_text": "Analyze source data.",
        "prompt_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "model_identifier": "test-model",
    }
    values.update(changes)
    return ChapterAnalysisService(**values)


def cache_paths(tmp_path: Path, chapter_id: str) -> tuple[Path, Path]:
    root = tmp_path / "analysis" / "chapters"
    return (
        root / f"{chapter_id}.analysis.json",
        root / f"{chapter_id}.manifest.json",
    )


def test_resume_skips_completed_chapters(tmp_path: Path) -> None:
    model = ScriptedStructuredModel(
        [valid_analysis("chapter_001"), valid_analysis("chapter_002")],
        fail_on_call=2,
    )
    service = make_service(tmp_path, model)
    chapters = [make_chapter("chapter_001"), make_chapter("chapter_002")]

    first = service.analyze(chapters)
    assert first.completed_chapters == ["chapter_001"]

    model.fail_on_call = None
    second = service.analyze(chapters)

    assert second.newly_called_chapters == ["chapter_002"]
    assert second.completed_chapters == ["chapter_001", "chapter_002"]


def test_success_writes_hashed_analysis_and_exact_five_input_cache_key(
    tmp_path: Path,
) -> None:
    model = RecordingStructuredModel([valid_analysis("chapter_001")])
    chapter = make_chapter("chapter_001")

    result = make_service(tmp_path, model).analyze([chapter])

    analysis_path, manifest_path = cache_paths(tmp_path, chapter.chapter_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result.status == "book_analysis_ready"
    assert analysis_path.is_file()
    assert manifest_path.is_file()
    assert set(manifest) == {"cache_key", "analysis_sha256"}
    assert manifest["cache_key"] == {
        "source_sha256": "b" * 64,
        "chapter_sha256": chapter.sha256,
        "prompt_sha256": "a" * 64,
        "schema_version": "1",
        "model_identifier": "test-model",
    }
    assert manifest["analysis_sha256"] == hashlib.sha256(
        analysis_path.read_bytes()
    ).hexdigest()
    assert not list((tmp_path / "analysis").rglob("*.tmp"))


def test_prompt_keeps_trusted_context_outside_one_chapter_source_block(
    tmp_path: Path,
) -> None:
    chapter_text = "First paragraph.\n\nSecond paragraph with PRIVATE_SOURCE."
    model = RecordingStructuredModel(
        [valid_analysis("chapter_001", excerpt="First paragraph.")]
    )

    make_service(tmp_path, model).analyze(
        [make_chapter("chapter_001", text=chapter_text)]
    )

    prompt = model.prompts[0]
    trusted, source = prompt.split("\nBEGIN_SOURCE_DATA\n", 1)
    assert prompt.count("\nBEGIN_SOURCE_DATA\n") == 1
    assert prompt.count("\nEND_SOURCE_DATA\n") == 1
    assert "chapter_id=chapter_001" in trusted
    assert "paragraph_count=2" in trusted
    assert "PRIVATE_SOURCE" not in trusted
    assert json.dumps(chapter_text, ensure_ascii=False) in source


@pytest.mark.parametrize(
    "cache_key_field",
    [
        "source_sha256",
        "chapter_sha256",
        "prompt_sha256",
        "schema_version",
        "model_identifier",
    ],
)
def test_any_cache_key_input_mismatch_forces_a_model_call(
    tmp_path: Path,
    cache_key_field: str,
) -> None:
    chapter = make_chapter("chapter_001")
    make_service(
        tmp_path,
        RecordingStructuredModel([valid_analysis(chapter.chapter_id)]),
    ).analyze([chapter])
    _, manifest_path = cache_paths(tmp_path, chapter.chapter_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cache_key"][cache_key_field] = "stale-value"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    replacement = RecordingStructuredModel([valid_analysis(chapter.chapter_id)])

    result = make_service(tmp_path, replacement).analyze([chapter])

    assert result.newly_called_chapters == [chapter.chapter_id]
    assert replacement.calls == 1


def test_corrupt_manifest_is_a_cache_miss(tmp_path: Path) -> None:
    chapter = make_chapter("chapter_001")
    make_service(
        tmp_path,
        RecordingStructuredModel([valid_analysis(chapter.chapter_id)]),
    ).analyze([chapter])
    _, manifest_path = cache_paths(tmp_path, chapter.chapter_id)
    manifest_path.write_text("not json", encoding="utf-8")
    replacement = RecordingStructuredModel([valid_analysis(chapter.chapter_id)])

    result = make_service(tmp_path, replacement).analyze([chapter])

    assert result.newly_called_chapters == [chapter.chapter_id]
    assert replacement.calls == 1


def test_parseable_cache_with_invalid_evidence_is_reanalyzed(tmp_path: Path) -> None:
    chapter = make_chapter("chapter_001")
    make_service(
        tmp_path,
        RecordingStructuredModel([valid_analysis(chapter.chapter_id)]),
    ).analyze([chapter])
    analysis_path, manifest_path = cache_paths(tmp_path, chapter.chapter_id)
    invalid = valid_analysis(chapter.chapter_id, excerpt="FABRICATED_PRIVATE_EXCERPT")
    analysis_path.write_text(json.dumps(invalid), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["analysis_sha256"] = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    replacement = RecordingStructuredModel([valid_analysis(chapter.chapter_id)])

    result = make_service(tmp_path, replacement).analyze([chapter])

    assert result.newly_called_chapters == [chapter.chapter_id]
    assert replacement.calls == 1


def test_normal_model_failure_stops_without_repair_and_preserves_prior_cache(
    tmp_path: Path,
) -> None:
    private_text = "PRIVATE_BOOK_TEXT"
    model = RecordingStructuredModel(
        [
            valid_analysis("chapter_001"),
            ModelCompletionError("codex_process_failed", "Codex request failed."),
        ]
    )
    chapters = [
        make_chapter("chapter_001"),
        make_chapter("chapter_002", text=private_text),
    ]

    result = make_service(tmp_path, model).analyze(chapters)

    first_analysis, first_manifest = cache_paths(tmp_path, "chapter_001")
    assert result.status == "chapter_analysis_failed"
    assert result.completed_chapters == ["chapter_001"]
    assert result.error_code == "codex_process_failed"
    assert model.calls == 2
    assert first_analysis.is_file() and first_manifest.is_file()
    assert not list(
        (tmp_path / "analysis" / ".private").rglob("invalid-response*.txt")
    )
    assert private_text not in str(result)
    assert str(tmp_path) not in str(result)


def test_schema_invalid_output_gets_one_repair_with_private_response(
    tmp_path: Path,
) -> None:
    private_response = '{"bad":"PRIVATE_INVALID_RESPONSE"}'
    model = RecordingStructuredModel(
        [
            ModelInvalidResponseError(
                "codex_output_invalid",
                "Codex response was invalid.",
                private_response,
            ),
            valid_analysis("chapter_001"),
        ]
    )

    result = make_service(tmp_path, model).analyze([make_chapter("chapter_001")])

    private_logs = list(
        (tmp_path / "analysis" / ".private").rglob("invalid-response*.txt")
    )
    assert result.status == "book_analysis_ready"
    assert model.calls == 2
    assert model.request_dirs[0] != model.request_dirs[1]
    assert model.request_dirs_were_empty == [True, True]
    assert "codex_output_invalid" in model.prompts[1]
    trusted, untrusted = model.prompts[1].split("\nBEGIN_SOURCE_DATA\n", 1)
    encoded_source, _ = untrusted.split("\nEND_SOURCE_DATA\n", 1)
    repair_payload = json.loads(json.loads(encoded_source))
    assert "PRIVATE_INVALID_RESPONSE" not in trusted
    assert repair_payload["previous_invalid_response"] == private_response
    assert repair_payload["chapter_text"] == "正文"
    assert len(private_logs) == 1
    assert private_logs[0].read_text(encoding="utf-8") == private_response
    assert private_response not in str(result)


def test_evidence_invalid_output_gets_one_repair_with_serialized_response(
    tmp_path: Path,
) -> None:
    private_excerpt = "FABRICATED_PRIVATE_EXCERPT"
    model = RecordingStructuredModel(
        [
            valid_analysis("chapter_001", excerpt=private_excerpt),
            valid_analysis("chapter_001"),
        ]
    )

    result = make_service(tmp_path, model).analyze([make_chapter("chapter_001")])

    private_logs = list(
        (tmp_path / "analysis" / ".private").rglob("invalid-response*.txt")
    )
    assert result.status == "book_analysis_ready"
    assert model.calls == 2
    assert "evidence_excerpt_not_found" in model.prompts[1]
    assert private_excerpt in model.prompts[1]
    trusted, untrusted = model.prompts[1].split("\nBEGIN_SOURCE_DATA\n", 1)
    encoded_source, _ = untrusted.split("\nEND_SOURCE_DATA\n", 1)
    repair_payload = json.loads(json.loads(encoded_source))
    assert private_excerpt not in trusted
    assert private_excerpt in repair_payload["previous_invalid_response"]
    assert len(private_logs) == 1
    assert private_excerpt in private_logs[0].read_text(encoding="utf-8")
    assert private_excerpt not in str(result)


def test_second_invalid_output_stops_with_safe_failure(tmp_path: Path) -> None:
    chapter_text = "PRIVATE_CHAPTER_TEXT"
    first_private = '{"bad":"FIRST_PRIVATE_RESPONSE"}'
    second_private = '{"bad":"SECOND_PRIVATE_RESPONSE"}'
    model = RecordingStructuredModel(
        [
            ModelInvalidResponseError(
                "codex_output_invalid",
                "Codex response was invalid.",
                first_private,
            ),
            ModelInvalidResponseError(
                "codex_output_invalid",
                "Codex response was invalid.",
                second_private,
            ),
        ]
    )

    result = make_service(tmp_path, model).analyze(
        [make_chapter("chapter_001", text=chapter_text)]
    )

    private_logs = list(
        (tmp_path / "analysis" / ".private").rglob("invalid-response*.txt")
    )
    public_result = str(result)
    assert result.status == "chapter_analysis_failed"
    assert result.error_code == "chapter_analysis_failed"
    assert model.calls == 2
    assert len(private_logs) == 2
    assert chapter_text not in public_result
    assert first_private not in public_result
    assert second_private not in public_result
    assert str(tmp_path) not in public_result


@pytest.mark.parametrize(
    "unsafe_chapter_id",
    [
        "../chapter_001",
        "chapter/001",
        "chapter\\001",
        ".hidden",
        "C:secret",
        "CON",
        "chapter.",
        "a" * 129,
    ],
)
def test_unsafe_chapter_id_is_rejected_before_model_or_filesystem_use(
    tmp_path: Path,
    unsafe_chapter_id: str,
) -> None:
    model = RecordingStructuredModel([valid_analysis(unsafe_chapter_id)])

    result = make_service(tmp_path, model).analyze(
        [make_chapter(unsafe_chapter_id)]
    )

    assert result.status == "chapter_analysis_failed"
    assert result.error_code == "unsafe_chapter_id"
    assert result.failed_chapter is None
    assert model.calls == 0
    assert unsafe_chapter_id not in str(result)


def test_duplicate_chapter_ids_are_rejected_before_model_call(
    tmp_path: Path,
) -> None:
    model = RecordingStructuredModel([valid_analysis("chapter_001")])

    result = make_service(tmp_path, model).analyze(
        [make_chapter("chapter_001"), make_chapter("chapter_001")]
    )

    assert result.status == "chapter_analysis_failed"
    assert result.error_code == "duplicate_chapter_id"
    assert model.calls == 0


def test_duplicate_claim_ids_across_chapters_fail_after_one_repair(
    tmp_path: Path,
) -> None:
    first = valid_analysis("chapter_001")
    second = valid_analysis("chapter_002")
    repaired = valid_analysis("chapter_002")
    duplicate_claim_id = first["cards"][0]["claim_id"]
    second["cards"][0]["claim_id"] = duplicate_claim_id
    repaired["cards"][0]["claim_id"] = duplicate_claim_id
    model = RecordingStructuredModel([first, second, repaired])

    result = make_service(tmp_path, model).analyze(
        [make_chapter("chapter_001"), make_chapter("chapter_002")]
    )

    assert result.status == "chapter_analysis_failed"
    assert result.failed_chapter == "chapter_002"
    assert result.error_code == "chapter_analysis_failed"
    assert model.calls == 3
    assert "claim_id_duplicate" in model.prompts[2]


def test_empty_chapter_analysis_fails_after_one_repair(
    tmp_path: Path,
) -> None:
    empty = {"chapter_id": "chapter_001", "cards": []}
    model = RecordingStructuredModel([empty, empty])

    result = make_service(tmp_path, model).analyze(
        [make_chapter("chapter_001")]
    )

    assert result.status == "chapter_analysis_failed"
    assert result.failed_chapter == "chapter_001"
    assert result.error_code == "chapter_analysis_failed"
    assert model.calls == 2
    assert "analysis_cards_empty" in model.prompts[1]


def test_claim_id_with_surrounding_whitespace_fails_after_one_repair(
    tmp_path: Path,
) -> None:
    invalid = valid_analysis("chapter_001")
    invalid["cards"][0]["claim_id"] = " C01-001 "
    model = RecordingStructuredModel([invalid, invalid])

    result = make_service(tmp_path, model).analyze(
        [make_chapter("chapter_001")]
    )

    assert result.status == "chapter_analysis_failed"
    assert model.calls == 2
    assert "claim_id_invalid" in model.prompts[1]


def test_private_output_redirect_is_rejected_before_model_or_external_write(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    outside = tmp_path / "outside-private"
    outside.mkdir()
    redirect = output_dir / ".private"
    _make_directory_redirect(redirect, outside)
    model = RecordingStructuredModel([valid_analysis("chapter_001")])

    try:
        result = make_service(
            tmp_path,
            model,
            output_dir=output_dir,
        ).analyze([make_chapter("chapter_001")])

        assert result.status == "chapter_analysis_failed"
        assert result.error_code == "unsafe_output_directory"
        assert model.calls == 0
        assert list(outside.iterdir()) == []
    finally:
        _remove_directory_redirect(redirect)


def _make_directory_redirect(link: Path, target: Path) -> None:
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        return
    link.symlink_to(target, target_is_directory=True)


def _remove_directory_redirect(link: Path) -> None:
    if not link.exists() and not link.is_symlink():
        return
    if os.name == "nt":
        os.rmdir(link)
        return
    link.unlink()


def test_safe_chapter_id_is_used_verbatim_for_cache_names(tmp_path: Path) -> None:
    chapter_id = "chapter.001-A_b"
    model = RecordingStructuredModel([valid_analysis(chapter_id)])

    result = make_service(tmp_path, model).analyze([make_chapter(chapter_id)])

    analysis_path, manifest_path = cache_paths(tmp_path, chapter_id)
    assert result.status == "book_analysis_ready"
    assert analysis_path.is_file()
    assert manifest_path.is_file()


def test_unicode_safe_chapter_id_is_used_verbatim_for_cache_names(
    tmp_path: Path,
) -> None:
    chapter_id = "章节一_证据"
    model = RecordingStructuredModel([valid_analysis(chapter_id)])

    result = make_service(tmp_path, model).analyze([make_chapter(chapter_id)])

    analysis_path, manifest_path = cache_paths(tmp_path, chapter_id)
    assert result.status == "book_analysis_ready"
    assert analysis_path.is_file()
    assert manifest_path.is_file()


def test_redirected_cache_artifact_is_rejected_without_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chapter = make_chapter("chapter_001")
    make_service(
        tmp_path,
        RecordingStructuredModel([valid_analysis(chapter.chapter_id)]),
    ).analyze([chapter])
    replacement = RecordingStructuredModel([valid_analysis(chapter.chapter_id)])
    _, manifest_path = cache_paths(tmp_path, chapter.chapter_id)
    monkeypatch.setattr(
        evidence_module,
        "_is_redirected",
        lambda path: path == manifest_path,
    )

    result = make_service(tmp_path, replacement).analyze([chapter])

    assert result.status == "chapter_analysis_failed"
    assert result.error_code == "unsafe_cache_artifact"
    assert replacement.calls == 0


def test_analyze_chapters_convenience_function_uses_the_service_contract(
    tmp_path: Path,
) -> None:
    model = RecordingStructuredModel([valid_analysis("chapter_001")])

    result = analyze_chapters(
        [make_chapter("chapter_001")],
        model=model,
        output_dir=tmp_path / "analysis",
        prompt_text="Analyze source data.",
        prompt_sha256="a" * 64,
        source_sha256="b" * 64,
        model_identifier="test-model",
    )

    assert result.status == "book_analysis_ready"
    assert result.completed_chapters == ["chapter_001"]
