import hashlib
import json
from pathlib import Path

import pytest

from bv.content.story import StoryCharacter, import_story_candidate, approve_story_candidate
from bv.content.story_visual import load_approved_story_visual_context, StoryVisualContextError
from test_story_content import _store, _contract, TEXT


def _prepared(tmp_path: Path, *, approve=True):
    store = _store(tmp_path)
    metadata, evidence, sections, review = _contract()
    metadata.characters = [StoryCharacter(character_id="source-narrator", name="叙述者", description="收到来信的人", evidence_ids=["E1"])]
    imported = import_story_candidate(store=store, book_id="book-demo", episode_id="E001", text=TEXT, metadata=metadata, evidence=evidence, sections=sections, factual_review=review)
    if approve:
        approve_story_candidate(store, "book-demo", "E001")
    return imported.candidate_path.parent.parent


def _load(root):
    return load_approved_story_visual_context(root, book_id="book-demo", episode_id="E001", approved_text=TEXT, approved_sha256=hashlib.sha256(TEXT.encode()).hexdigest())


def test_story_visual_context_uses_approved_story_without_legacy_files(tmp_path):
    root = _prepared(tmp_path)
    result = _load(root)
    assert result is not None
    assert not (root / "script/semantic-lock.json").exists()
    assert result.characters[0].character_id == "source-narrator"
    assert result.important_events == ("触发事件与后果",)
    assert result.semantic_lock.allowed_claim_ids_by_cluster == {"S1": ("E1",)}
    payload = result.semantic_lock.model_dump(mode="json", exclude={"sha256"})
    assert hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == result.semantic_lock.sha256


@pytest.mark.parametrize("filename", ["story_candidate.json", "story_manifest.json", "approved.txt", "candidate.txt"])
def test_story_visual_context_rejects_stale_sources(tmp_path, filename):
    root = _prepared(tmp_path)
    path = root / "script" / filename
    if filename == "story_manifest.json":
        payload = json.loads(path.read_text("utf-8"))
        payload["metadata_sha256"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(path.read_text("utf-8") + " ", encoding="utf-8")
    with pytest.raises(StoryVisualContextError, match="story_visual_source_invalid"):
        _load(root)


def test_story_visual_context_rejects_unapproved_candidate(tmp_path):
    root = _prepared(tmp_path, approve=False)
    with pytest.raises(StoryVisualContextError):
        _load(root)


def test_story_visual_context_legacy_profile_returns_none(tmp_path):
    assert _load(tmp_path) is None


def test_import_adopts_existing_identical_authored_candidate(tmp_path):
    store = _store(tmp_path)
    root = store.root / "books/book-demo/episodes/E001/script"
    root.mkdir(exist_ok=True)
    (root / "candidate-v2.txt").write_text(TEXT, encoding="utf-8")
    metadata, evidence, sections, review = _contract()
    result = import_story_candidate(store=store, book_id="book-demo", episode_id="E001", text=TEXT, metadata=metadata, evidence=evidence, sections=sections, factual_review=review, candidate_filename="candidate-v2.txt")
    assert result.candidate_path.read_text("utf-8") == TEXT
    assert store.load_episode("book-demo", "E001").status == "awaiting_script_review"
    assert not (root / "approved.txt").exists()
