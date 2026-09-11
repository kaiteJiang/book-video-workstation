from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bv.content.story import (
    StoryEvidence,
    StoryFactualReview,
    StoryCharacter,
    StoryMetadata,
    StoryReviewFinding,
    StorySection,
    approve_story_candidate,
    import_story_candidate,
)
from bv.state.models import BookState, EpisodeState, StageManifest
from bv.state.store import StateStore
from bv.workflow.stages import StageContext


TEXT = "我在雨夜收到一封没有寄件人的信。第二天，我才知道这封信改变了三个人的选择。"


def _contract():
    metadata = StoryMetadata(
        title="雨夜来信",
        book_title="示例书",
        authors=["示例作者"],
        point_of_view="first",
        direct_writing=True,
        selected_passage_reason="用一封信串起事件、选择和回望。",
    )
    evidence = [
        StoryEvidence(
            evidence_id="E1",
            source_title="示例书正文",
            source_locator="第一章",
            source_type="primary_text",
            support="主人公在雨夜收到匿名信，次日得知信件影响多人。",
        )
    ]
    sections = [
        StorySection(
            section_id="S1",
            start=0,
            end=len(TEXT),
            beat="触发事件与后果",
            evidence_ids=["E1"],
        )
    ]
    review = StoryFactualReview(
        reviewer="author",
        findings=[
            StoryReviewFinding(
                finding_id="F1",
                start=0,
                end=len(TEXT),
                verdict="supported",
                evidence_ids=["E1"],
                note="叙述范围与材料一致。",
            )
        ],
    )
    return metadata, evidence, sections, review


def _store(tmp_path: Path, *, downstream: bool = False) -> StateStore:
    store = StateStore(tmp_path / "workspace")
    store.save_book(BookState(book_id="book-demo", title="示例书", authors=["示例作者"]))
    completed = ["tts", "render"] if downstream else []
    manifests = {}
    if downstream:
        manifests = {
            name: StageManifest(
                stage=name,
                status="completed",
                inputs={},
                outputs={},
                config_sha256="a" * 64,
            )
            for name in completed
        }
    store.save_episode(
        EpisodeState(
            book_id="book-demo",
            episode_id="E001",
            status="source_imported",
            completed_stages=completed,
            stage_manifests=manifests,
            script_hash="0" * 64 if downstream else None,
        )
    )
    return store


def test_import_story_anchors_candidate_and_review_without_approving(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()

    result = import_story_candidate(
        store=store,
        book_id="book-demo",
        episode_id="E001",
        text=TEXT,
        metadata=metadata,
        evidence=evidence,
        sections=sections,
        factual_review=review,
    )

    episode = store.load_episode("book-demo", "E001")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert episode.status == "awaiting_script_review"
    assert episode.script_hash is None
    assert result.candidate_path.read_text(encoding="utf-8") == TEXT
    assert manifest["candidate_sha256"] == hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    assert manifest["status"] == "candidate"
    assert manifest["writing_method"] == "direct"
    assert manifest["narrative_mode"] == "story"
    profile = json.loads((result.candidate_path.parents[1] / "production_profile.json").read_text(encoding="utf-8"))
    assert profile["narrative_mode"] == "story"
    assert not (result.candidate_path.parent / "approved.txt").exists()


def test_story_metadata_characters_are_evidence_bound_and_backward_compatible(tmp_path: Path) -> None:
    assert StoryMetadata(
        title="兼容旧稿", book_title="示例书", authors=["示例作者"],
        point_of_view="third", direct_writing=True, selected_passage_reason="旧合同",
    ).characters == []
    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    metadata = metadata.model_copy(
        update={
            "characters": [
                StoryCharacter(
                    character_id="source-letter-writer",
                    name="写信人",
                    description="推动三个人重新选择的匿名写信人。",
                    evidence_ids=["E1"],
                )
            ]
        }
    )

    result = import_story_candidate(
        store=store, book_id="book-demo", episode_id="E001", text=TEXT,
        metadata=metadata, evidence=evidence, sections=sections, factual_review=review,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"]["characters"][0]["character_id"] == "source-letter-writer"


def test_story_characters_reject_duplicate_ids_and_unknown_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    duplicate = StoryCharacter(
        character_id="source-person", name="同一人", description="不同年龄阶段可共用姓名。", evidence_ids=["E1"]
    )
    metadata = metadata.model_copy(update={"characters": [duplicate, duplicate]})
    with pytest.raises(ValueError, match="story_characters_invalid"):
        import_story_candidate(
            store=store, book_id="book-demo", episode_id="E001", text=TEXT,
            metadata=metadata, evidence=evidence, sections=sections, factual_review=review,
        )

    metadata = metadata.model_copy(
        update={
            "characters": [
                duplicate.model_copy(update={"character_id": "source-other", "evidence_ids": ["missing"]})
            ]
        }
    )
    with pytest.raises(ValueError, match="story_character_evidence_invalid"):
        import_story_candidate(
            store=store, book_id="book-demo", episode_id="E001", text=TEXT,
            metadata=metadata, evidence=evidence, sections=sections, factual_review=review,
        )


def test_story_duration_over_ten_minutes_is_a_warning_not_a_rejection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metadata, evidence, _sections, review = _contract()
    text = "甲" * 2501
    sections = [StorySection(section_id="S1", start=0, end=len(text), beat="完整事件", evidence_ids=["E1"])]
    review = review.model_copy(
        update={"findings": [review.findings[0].model_copy(update={"end": len(text)})]}
    )

    result = import_story_candidate(
        store=store,
        book_id="book-demo",
        episode_id="E001",
        text=text,
        metadata=metadata,
        evidence=evidence,
        sections=sections,
        factual_review=review,
    )

    assert "estimated_duration_over_10_minutes" in result.warnings
    assert store.load_episode("book-demo", "E001").status == "awaiting_script_review"


def test_story_approval_rejects_tampered_candidate_before_writing_approved(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    result = import_story_candidate(
        store=store,
        book_id="book-demo",
        episode_id="E001",
        text=TEXT,
        metadata=metadata,
        evidence=evidence,
        sections=sections,
        factual_review=review,
    )
    result.candidate_path.write_text(TEXT + "被改动", encoding="utf-8")

    with pytest.raises(ValueError, match="story_candidate_integrity_failed"):
        approve_story_candidate(store, "book-demo", "E001")

    assert not (result.candidate_path.parent / "approved.txt").exists()


def test_story_approval_rejects_redirected_output_before_writing(tmp_path: Path, monkeypatch) -> None:
    from bv.content import story

    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    imported = import_story_candidate(
        store=store, book_id="book-demo", episode_id="E001", text=TEXT,
        metadata=metadata, evidence=evidence, sections=sections, factual_review=review,
    )
    monkeypatch.setattr(story, "_path_chain_redirected", lambda _path: True)

    with pytest.raises(ValueError, match="unsafe_story_output_path"):
        approve_story_candidate(store, "book-demo", "E001")
    assert not (imported.candidate_path.parent / "approved.txt").exists()


def test_story_approval_uses_manifest_bound_custom_candidate_name(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    imported = import_story_candidate(
        store=store,
        book_id="book-demo",
        episode_id="E001",
        text=TEXT,
        metadata=metadata,
        evidence=evidence,
        sections=sections,
        factual_review=review,
        candidate_filename="candidate-v2-longform.txt",
    )

    approved = approve_story_candidate(store, "book-demo", "E001")

    assert imported.candidate_path.name == "candidate-v2-longform.txt"
    assert approved.approved_path.read_text(encoding="utf-8") == TEXT


def test_story_import_rejects_reserved_approved_filename(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()

    with pytest.raises(ValueError, match="reserved_story_candidate_filename"):
        import_story_candidate(
            store=store,
            book_id="book-demo",
            episode_id="E001",
            text=TEXT,
            metadata=metadata,
            evidence=evidence,
            sections=sections,
            factual_review=review,
            candidate_filename="approved.txt",
        )


def test_changed_default_candidate_is_archived_but_identical_reimport_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    kwargs = dict(
        store=store,
        book_id="book-demo",
        episode_id="E001",
        text=TEXT,
        metadata=metadata,
        evidence=evidence,
        sections=sections,
        factual_review=review,
    )
    first = import_story_candidate(**kwargs)
    import_story_candidate(**kwargs)
    assert not (first.candidate_path.parent / "archive").exists()

    changed = TEXT + "后来我才明白，选择也会反过来改变写信的人。"
    kwargs["text"] = changed
    kwargs["sections"] = [StorySection(section_id="S1", start=0, end=len(changed), beat="回望", evidence_ids=["E1"])]
    kwargs["factual_review"] = review.model_copy(
        update={"findings": [review.findings[0].model_copy(update={"end": len(changed), "verdict": "bounded_interpretation"})]}
    )
    import_story_candidate(**kwargs)

    archived_candidates = list((first.candidate_path.parent / "archive").glob("candidate-*.txt"))
    archived_manifests = list((first.candidate_path.parent / "archive").glob("candidate-manifest-*.json"))
    assert len(archived_candidates) == len(archived_manifests) == 1
    assert archived_candidates[0].read_text(encoding="utf-8") == TEXT


def test_story_import_clears_old_script_hash_and_replaces_stale_draft_marker(tmp_path: Path) -> None:
    store = _store(tmp_path, downstream=True)
    episode = store.load_episode("book-demo", "E001")
    episode.stale_stages.append("draft_script")
    store.save_episode(episode)
    metadata, evidence, sections, review = _contract()

    import_story_candidate(
        store=store, book_id="book-demo", episode_id="E001", text=TEXT,
        metadata=metadata, evidence=evidence, sections=sections, factual_review=review,
    )

    updated = store.load_episode("book-demo", "E001")
    assert updated.script_hash is None
    assert "draft_script" in updated.completed_stages
    assert "draft_script" not in updated.stale_stages
    assert updated.stage_manifests["draft_script"].status == "completed"


def test_story_import_rejects_book_metadata_mismatch(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    metadata = metadata.model_copy(update={"book_title": "另一本书"})

    with pytest.raises(ValueError, match="story_book_identity_mismatch"):
        import_story_candidate(
            store=store, book_id="book-demo", episode_id="E001", text=TEXT,
            metadata=metadata, evidence=evidence, sections=sections, factual_review=review,
        )


def test_story_import_rejects_redirected_output_before_mutation(tmp_path: Path, monkeypatch) -> None:
    from bv.content import story

    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    monkeypatch.setattr(story, "_path_chain_redirected", lambda _path: True)

    with pytest.raises(ValueError, match="unsafe_story_output_path"):
        import_story_candidate(
            store=store, book_id="book-demo", episode_id="E001", text=TEXT,
            metadata=metadata, evidence=evidence, sections=sections, factual_review=review,
        )
    assert not (store.root / "books" / "book-demo" / "episodes" / "E001" / "script" / "candidate.txt").exists()


def test_importing_new_candidate_archives_previous_approval(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    import_story_candidate(
        store=store,
        book_id="book-demo",
        episode_id="E001",
        text=TEXT,
        metadata=metadata,
        evidence=evidence,
        sections=sections,
        factual_review=review,
    )
    approve_story_candidate(store, "book-demo", "E001")
    changed = TEXT + "后来，我又读懂了信里没有写出的那句话。"
    changed_sections = [StorySection(section_id="S1", start=0, end=len(changed), beat="事件后的回望", evidence_ids=["E1"])]
    changed_review = review.model_copy(
        update={"findings": [review.findings[0].model_copy(update={"end": len(changed), "verdict": "bounded_interpretation"})]}
    )

    imported = import_story_candidate(
        store=store,
        book_id="book-demo",
        episode_id="E001",
        text=changed,
        metadata=metadata,
        evidence=evidence,
        sections=changed_sections,
        factual_review=changed_review,
        candidate_filename="candidate-v2.txt",
    )

    assert not (imported.candidate_path.parent / "approved.txt").exists()
    archived = list((imported.candidate_path.parent / "archive").glob("approved-*.txt"))
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == TEXT


def test_story_approval_writes_approved_and_invalidates_old_downstream(tmp_path: Path) -> None:
    store = _store(tmp_path, downstream=True)
    metadata, evidence, sections, review = _contract()
    import_story_candidate(
        store=store,
        book_id="book-demo",
        episode_id="E001",
        text=TEXT,
        metadata=metadata,
        evidence=evidence,
        sections=sections,
        factual_review=review,
    )

    approved = approve_story_candidate(store, "book-demo", "E001")

    episode = store.load_episode("book-demo", "E001")
    assert approved.approved_path.read_text(encoding="utf-8") == TEXT
    assert episode.status == "script_approved"
    assert episode.script_hash == hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    assert {"tts", "render"} <= set(episode.stale_stages)
    assert json.loads(approved.manifest_path.read_text(encoding="utf-8"))["status"] == "approved"


def test_script_approval_stage_routes_story_without_human_writing_checker(tmp_path: Path) -> None:
    from bv.workflow.content_runtime import ScriptApprovalStage

    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    imported = import_story_candidate(
        store=store,
        book_id="book-demo",
        episode_id="E001",
        text=TEXT,
        metadata=metadata,
        evidence=evidence,
        sections=sections,
        factual_review=review,
    )
    episode = store.load_episode("book-demo", "E001")
    stage = ScriptApprovalStage(
        store,
        topic_service_factory=lambda _root: (_ for _ in ()).throw(AssertionError("legacy topic service called")),
        human_writing_skill_path=tmp_path / "missing-skill.md",
    )

    outcome = stage.run(
        StageContext(
            book_id="book-demo",
            episode_id="E001",
            episode_root=imported.candidate_path.parents[1],
            episode_state=episode,
        )
    )

    assert outcome.outputs["approved_script"].read_text(encoding="utf-8") == TEXT
    assert episode.script_hash == hashlib.sha256(TEXT.encode("utf-8")).hexdigest()


def test_cli_import_story_loads_contract_and_stops_at_review(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from bv import cli

    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    story_file = tmp_path / "story.txt"
    contract_file = tmp_path / "contract.json"
    story_file.write_text(TEXT, encoding="utf-8")
    contract_file.write_text(
        json.dumps(
            {
                "metadata": metadata.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "sections": [item.model_dump(mode="json") for item in sections],
                "factual_review": review.model_dump(mode="json"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "load_config_default", lambda: SimpleNamespace(workspace_dir=store.root))

    result = CliRunner().invoke(
        cli.app,
        ["import-story", "book-demo", "E001", str(story_file), str(contract_file), "--candidate-name", "candidate-v2-longform.txt"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Status: awaiting_script_review" in result.stdout
    assert "Warnings: none" in result.stdout
    assert (store.root / "books" / "book-demo" / "episodes" / "E001" / "script" / "candidate-v2-longform.txt").is_file()
    assert not (store.root / "books" / "book-demo" / "episodes" / "E001" / "script" / "approved.txt").exists()


def test_cli_voice_pool_is_read_only_and_reports_pending_unknown() -> None:
    from typer.testing import CliRunner

    from bv.cli import app

    result = CliRunner().invoke(app, ["voice-pool", "--gender", "male", "--tag", "natural"])

    assert result.exit_code == 0, result.stdout
    assert 1 <= result.stdout.count("Voice ID:") <= 3
    assert "Audition: pending" in result.stdout
    assert "Live availability: unknown" in result.stdout


def test_explicit_unknown_authors_can_import_without_inventing_attribution(tmp_path: Path) -> None:
    store = _store(tmp_path)
    book = store.load_book("book-demo")
    book.authors = []
    store.save_book(book)
    metadata, evidence, sections, review = _contract()
    metadata = StoryMetadata.model_validate({**metadata.model_dump(), "authors": []})
    result = import_story_candidate(
        store=store, book_id="book-demo", episode_id="E001", text=TEXT,
        metadata=metadata, evidence=evidence, sections=sections, factual_review=review,
    )
    assert result.candidate_path.is_file()
    assert store.load_episode("book-demo", "E001").status == "awaiting_script_review"


def test_blank_author_is_not_an_explicit_unknown_author() -> None:
    metadata, *_ = _contract()
    with pytest.raises(ValueError, match="story_authors_blank"):
        StoryMetadata.model_validate({**metadata.model_dump(), "authors": [" "]})
