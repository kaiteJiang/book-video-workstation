from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bv.content.scripts import (
    FactDiffResult,
    ScriptApprovalError,
    _ScriptManifest,
    approve_script,
    build_semantic_lock,
)
from bv.content.synthesis import WholeBookValue
from bv.content.topics import TopicService
from bv.state.models import EpisodeState

from test_value_scripts import (
    _FakeModel,
    _SKILL_HASH,
    _brief,
    _diff,
    _human_revision_draft,
    _value,
    _voiceover,
)


def _topic_service(root: Path, value: WholeBookValue) -> TopicService:
    return TopicService(root, model=_FakeModel([_brief(value)]), topic_prompt_text="topic", dedup_prompt_text="dedup", grok_prompt_text="grok")


def _state(script_hash: str | None = None) -> EpisodeState:
    return EpisodeState(book_id="book", episode_id="E001", status="awaiting_script_review", script_hash=script_hash, completed_stages=["tts", "render"])


def _approved_diff(text: str) -> FactDiffResult:
    return _diff(script_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())


def _prompt_hashes() -> dict[str, str]:
    return {"draft": "a" * 64, "fact_diff": "c" * 64, "grok": "d" * 64}


def test_approval_writes_manifest_then_audits_one_topic_transition_and_is_idempotent(tmp_path: Path) -> None:
    value = _value()
    topic = _topic_service(tmp_path / "book", value)
    topic.generate_first_topic(value)
    text = _voiceover()
    result = approve_script(
        text,
        build_semantic_lock(_brief(value), value),
        _approved_diff(text),
        _state(),
        topic,
        tmp_path / "book" / "episodes" / "E001",
        prompt_hashes=_prompt_hashes(),
        human_writing_sha256=_SKILL_HASH,
        human_writing_checker_passed=True,
    )
    script_dir = tmp_path / "book" / "episodes" / "E001" / "script"
    assert result.episode_state.status == "script_approved"
    assert result.topic.status == "script_approved"
    assert (script_dir / "approved.txt").read_text(encoding="utf-8") == text
    manifest = json.loads((script_dir / "script_manifest.json").read_text(encoding="utf-8"))
    assert manifest["script_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert manifest["human_writing_sha256"] == _SKILL_HASH
    assert manifest["human_writing_checker_passed"] is True
    assert set(manifest["prompt_sha256"]) == {"draft", "fact_diff", "grok"}
    again = approve_script(
        text,
        build_semantic_lock(_brief(value), value),
        _approved_diff(text),
        result.episode_state,
        topic,
        tmp_path / "book" / "episodes" / "E001",
        prompt_hashes=_prompt_hashes(),
        human_writing_sha256=_SKILL_HASH,
        human_writing_checker_passed=True,
    )
    assert again.idempotent is True
    assert len((tmp_path / "book" / "ledger" / "topics.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_approval_records_an_explicit_human_revision_with_its_reviewed_draft(
    tmp_path: Path,
) -> None:
    value = _value()
    topic = _topic_service(tmp_path / "book", value)
    topic.generate_first_topic(value)
    revision = _human_revision_draft()
    root = tmp_path / "book" / "episodes" / "E001"

    try:
        result = approve_script(
            revision.recommended_voiceover,
            build_semantic_lock(_brief(value), value),
            _approved_diff(revision.recommended_voiceover),
            _state(),
            topic,
            root,
            prompt_hashes=_prompt_hashes(),
            human_writing_sha256=_SKILL_HASH,
            human_writing_checker_passed=True,
            human_revision=True,
            revision_draft=revision,
        )
    except TypeError:
        pytest.fail("approve_script lacks the explicit human revision contract")

    assert result.episode_state.status == "script_approved"
    assert (root / "script" / "approved.txt").read_text(encoding="utf-8") == revision.recommended_voiceover
    manifest = json.loads((root / "script" / "script_manifest.json").read_text(encoding="utf-8"))
    assert manifest["human_revision"] is True
    assert manifest["reviewed_draft_sha256"]

    changed_proof = revision.model_copy(
        update={"hooks": ["另一个开头", revision.hooks[1]]}
    )
    with pytest.raises(ScriptApprovalError, match="approved_artifact_integrity_failed"):
        approve_script(
            revision.recommended_voiceover,
            build_semantic_lock(_brief(value), value),
            _approved_diff(revision.recommended_voiceover),
            result.episode_state,
            topic,
            root,
            prompt_hashes=_prompt_hashes(),
            human_writing_sha256=_SKILL_HASH,
            human_writing_checker_passed=True,
            human_revision=True,
            revision_draft=changed_proof,
        )


def test_human_revision_fails_closed_without_the_reviewed_draft(
    tmp_path: Path,
) -> None:
    value = _value()
    topic = _topic_service(tmp_path / "book", value)
    topic.generate_first_topic(value)
    revision = _human_revision_draft()

    with pytest.raises(ScriptApprovalError, match="human_revision_invalid"):
        approve_script(
            revision.recommended_voiceover,
            build_semantic_lock(_brief(value), value),
            _approved_diff(revision.recommended_voiceover),
            _state(),
            topic,
            tmp_path / "book" / "episodes" / "E001",
            prompt_hashes=_prompt_hashes(),
            human_writing_sha256=_SKILL_HASH,
            human_writing_checker_passed=True,
            human_revision=True,
        )


def test_old_script_manifest_loads_without_revision_fields() -> None:
    manifest = _ScriptManifest.model_validate(
        {
            "script_sha256": "a" * 64,
            "semantic_lock_sha256": "b" * 64,
            "prompt_sha256": _prompt_hashes(),
            "fact_diff_sha256": "c" * 64,
            "fact_diff_valid": True,
            "human_writing_sha256": _SKILL_HASH,
            "human_writing_checker_passed": True,
            "character_count": 200,
            "warnings": [],
            "approved_at": "2026-08-22T00:00:00+00:00",
        }
    )

    assert manifest.human_revision is False
    assert manifest.reviewed_draft_sha256 is None


def test_approval_rejects_stale_fact_diff_and_missing_clusters_before_writing(tmp_path: Path) -> None:
    value = _value()
    topic = _topic_service(tmp_path / "book", value)
    topic.generate_first_topic(value)
    root = tmp_path / "book" / "episodes" / "E001"
    with pytest.raises(ScriptApprovalError, match="stale_fact_diff"):
        approve_script(
            _voiceover(),
            build_semantic_lock(_brief(value), value),
            _diff(script_sha256="0" * 64),
            _state(),
            topic,
            root,
            prompt_hashes=_prompt_hashes(),
            human_writing_sha256=_SKILL_HASH,
            human_writing_checker_passed=True,
        )
    missing = _voiceover().replace("把判断落到可承担的行动", "只谈一个概念")
    with pytest.raises(ScriptApprovalError, match="script_validation_failed"):
        approve_script(
            missing,
            build_semantic_lock(_brief(value), value),
            _approved_diff(missing),
            _state(),
            topic,
            root,
            prompt_hashes=_prompt_hashes(),
            human_writing_sha256=_SKILL_HASH,
            human_writing_checker_passed=True,
        )
    assert not (root / "script" / "approved.txt").exists()


def test_approval_requires_human_writing_proof_before_writing(tmp_path: Path) -> None:
    import inspect

    sig = inspect.signature(approve_script)
    for name in ("human_writing_sha256", "human_writing_checker_passed"):
        assert name in sig.parameters, f"approve_script must require keyword-only {name}"
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    value = _value()
    topic = _topic_service(tmp_path / "book", value)
    topic.generate_first_topic(value)
    root = tmp_path / "book" / "episodes" / "E001"
    text = _voiceover()
    lock = build_semantic_lock(_brief(value), value)

    for kwargs in (
        {"human_writing_sha256": "E" * 64, "human_writing_checker_passed": True},
        {"human_writing_sha256": "not-a-hash", "human_writing_checker_passed": True},
        {"human_writing_sha256": "a" * 63, "human_writing_checker_passed": True},
        {"human_writing_sha256": _SKILL_HASH, "human_writing_checker_passed": False},
    ):
        with pytest.raises(ScriptApprovalError, match="human_writing_"):
            approve_script(
                text,
                lock,
                _approved_diff(text),
                _state(),
                topic,
                root,
                prompt_hashes=_prompt_hashes(),
                **kwargs,
            )
        assert not (root / "script" / "approved.txt").exists()


def test_changed_skill_hash_on_identical_script_fails_idempotent_integrity(tmp_path: Path) -> None:
    value = _value()
    topic = _topic_service(tmp_path / "book", value)
    topic.generate_first_topic(value)
    root = tmp_path / "book" / "episodes" / "E001"
    text = _voiceover()
    first = approve_script(
        text,
        build_semantic_lock(_brief(value), value),
        _approved_diff(text),
        _state(),
        topic,
        root,
        prompt_hashes=_prompt_hashes(),
        human_writing_sha256=_SKILL_HASH,
        human_writing_checker_passed=True,
    )
    with pytest.raises(ScriptApprovalError, match="approved_artifact_integrity_failed"):
        approve_script(
            text,
            build_semantic_lock(_brief(value), value),
            _approved_diff(text),
            first.episode_state,
            topic,
            root,
            prompt_hashes=_prompt_hashes(),
            human_writing_sha256="f" * 64,
            human_writing_checker_passed=True,
        )
    manifest = json.loads((root / "script" / "script_manifest.json").read_text(encoding="utf-8"))
    assert manifest["human_writing_sha256"] == _SKILL_HASH


def test_changed_approved_hash_invalidates_downstream_but_preserves_audit(tmp_path: Path) -> None:
    value = _value()
    topic = _topic_service(tmp_path / "book", value)
    topic.generate_first_topic(value)
    root = tmp_path / "book" / "episodes" / "E001"
    old = _voiceover()
    first = approve_script(
        old,
        build_semantic_lock(_brief(value), value),
        _approved_diff(old),
        _state(),
        topic,
        root,
        prompt_hashes=_prompt_hashes(),
        human_writing_sha256=_SKILL_HASH,
        human_writing_checker_passed=True,
    )
    new = old + "请带着自己的问题回到原书。"
    changed = approve_script(
        new,
        build_semantic_lock(_brief(value), value),
        _approved_diff(new),
        first.episode_state,
        topic,
        root,
        prompt_hashes=_prompt_hashes(),
        human_writing_sha256=_SKILL_HASH,
        human_writing_checker_passed=True,
    )
    assert changed.idempotent is False
    assert changed.episode_state.script_hash == hashlib.sha256(new.encode("utf-8")).hexdigest()
    assert {"tts", "render"} <= set(changed.episode_state.stale_stages)
    assert (root / "script" / "approved.txt").read_text(encoding="utf-8") == new


def test_approval_rejects_redirected_output_before_any_external_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bv.content import scripts

    value = _value()
    topic = _topic_service(tmp_path / "book", value)
    topic.generate_first_topic(value)
    monkeypatch.setattr(scripts, "_is_redirected", lambda path: Path(path).name == "script")
    with pytest.raises(ScriptApprovalError, match="unsafe_approval_path"):
        approve_script(
            _voiceover(),
            build_semantic_lock(_brief(value), value),
            _approved_diff(_voiceover()),
            _state(),
            topic,
            tmp_path / "book" / "episodes" / "E001",
            prompt_hashes=_prompt_hashes(),
            human_writing_sha256=_SKILL_HASH,
            human_writing_checker_passed=True,
        )


def test_approval_never_invokes_tts_or_live_service(tmp_path: Path) -> None:
    value = _value()
    topic = _topic_service(tmp_path / "book", value)
    topic.generate_first_topic(value)
    result = approve_script(
        _voiceover(),
        build_semantic_lock(_brief(value), value),
        _approved_diff(_voiceover()),
        _state(),
        topic,
        tmp_path / "book" / "episodes" / "E001",
        prompt_hashes=_prompt_hashes(),
        human_writing_sha256=_SKILL_HASH,
        human_writing_checker_passed=True,
    )
    assert result.episode_state.status == "script_approved"
    assert len(topic.model.calls) == 1


def test_identical_reapproval_fails_closed_when_manifest_is_corrupted(tmp_path: Path) -> None:
    value = _value()
    topic = _topic_service(tmp_path / "book", value)
    topic.generate_first_topic(value)
    root = tmp_path / "book" / "episodes" / "E001"
    text = _voiceover()
    approved = approve_script(
        text,
        build_semantic_lock(_brief(value), value),
        _approved_diff(text),
        _state(),
        topic,
        root,
        prompt_hashes=_prompt_hashes(),
        human_writing_sha256=_SKILL_HASH,
        human_writing_checker_passed=True,
    )
    (root / "script" / "script_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ScriptApprovalError, match="approved_artifact_integrity_failed"):
        approve_script(
            text,
            build_semantic_lock(_brief(value), value),
            _approved_diff(text),
            approved.episode_state,
            topic,
            root,
            prompt_hashes=_prompt_hashes(),
            human_writing_sha256=_SKILL_HASH,
            human_writing_checker_passed=True,
        )


def test_approval_rejects_tampered_lock_legacy_humanize_key_and_reserved_prompt_hash(tmp_path: Path) -> None:
    value = _value()
    topic = _topic_service(tmp_path / "book", value)
    topic.generate_first_topic(value)
    root = tmp_path / "book" / "episodes" / "E001"
    lock = build_semantic_lock(_brief(value), value)
    lock.cluster_coverage_terms["problem"] = "被篡改"
    with pytest.raises(ScriptApprovalError, match="semantic_lock_integrity_failed"):
        approve_script(
            _voiceover(),
            lock,
            _approved_diff(_voiceover()),
            _state(),
            topic,
            root,
            prompt_hashes=_prompt_hashes(),
            human_writing_sha256=_SKILL_HASH,
            human_writing_checker_passed=True,
        )
    with pytest.raises(ScriptApprovalError, match="invalid_prompt_hashes"):
        approve_script(
            _voiceover(),
            build_semantic_lock(_brief(value), value),
            _approved_diff(_voiceover()),
            _state(),
            topic,
            root,
            prompt_hashes={"voiceover": "a" * 64},
            human_writing_sha256=_SKILL_HASH,
            human_writing_checker_passed=True,
        )
    with pytest.raises(ScriptApprovalError, match="invalid_prompt_hashes"):
        approve_script(
            _voiceover(),
            build_semantic_lock(_brief(value), value),
            _approved_diff(_voiceover()),
            _state(),
            topic,
            root,
            prompt_hashes={"draft": "a" * 64, "humanize": "b" * 64, "fact_diff": "c" * 64, "grok": "d" * 64},
            human_writing_sha256=_SKILL_HASH,
            human_writing_checker_passed=True,
        )
