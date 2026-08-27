import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bv.state.events import Event
from bv.state.models import (
    ArtifactRef,
    BookState,
    EpisodeState,
    StageManifest,
)
from bv.state.store import StateStore


def test_state_round_trip(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    book = BookState(book_id="book-demo", title="Demo", source_ids=["sha256-a"])
    episode = EpisodeState(book_id="book-demo", episode_id="E001")

    store.save_book(book)
    store.save_episode(episode)

    assert store.load_book("book-demo") == book
    assert store.load_episode("book-demo", "E001") == episode


def test_episode_state_round_trips_shared_manifests_and_fields(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path)
    manifest = StageManifest(
        stage="script",
        status="completed",
        inputs={"topic": "topic-hash"},
        outputs={
            "script": ArtifactRef(
                path="episodes/E001/script/script.md",
                sha256="script-hash",
                size_bytes=42,
            )
        },
        config_sha256="config-hash",
        prompt_sha256="prompt-hash",
    )
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        status="script_approved",
        stage_manifests={"script": manifest},
        completed_stages=["script"],
        stale_stages=[],
        script_hash="script-hash",
        voice_id="voice-default-v1",
        topic_id="topic-001",
        failure_summary=None,
    )

    store.save_episode(episode)

    assert store.load_episode("book-demo", "E001") == episode


def test_append_event_writes_one_utf8_jsonl_record_per_event(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path)
    store.append_event(
        Event(
            event="script_approved",
            book_id="book-demo",
            episode_id="E001",
            timestamp="2026-08-09T00:00:00+00:00",
            artifact_hash="script-hash",
            stage="script",
            detail={"note": "已批准"},
        )
    )
    store.append_event(
        Event(
            event="tts_started",
            book_id="book-demo",
            episode_id="E001",
            timestamp="2026-08-09T00:00:01+00:00",
        )
    )

    ledger = tmp_path / "books" / "book-demo" / "ledger" / "events.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["detail"] == {"note": "已批准"}
    assert "已批准" in lines[0]
    assert json.loads(lines[1])["event"] == "tts_started"


@pytest.mark.parametrize(
    "status",
    [
        "awaiting_video_generation",
        "video_import_invalid",
        "visual_sample_ready",
        "video_partial",
        "video_imported",
    ],
)
def test_episode_state_accepts_neutral_video_statuses(status: str) -> None:
    """Dropping a migrated status from EpisodeStatus would reject current episode JSON."""
    assert (
        EpisodeState(book_id="book-demo", episode_id="E001", status=status).status
        == status
    )


@pytest.mark.parametrize(
    "status",
    [
        "illustration_planning",
        "illustration_plan_ready",
        "representative_generation_running",
        "awaiting_representative_review",
        "representatives_approved",
        "illustration_batch_running",
        "illustrations_ready",
        "visual_render_running",
        "visual_render_ready",
    ],
)
def test_episode_state_accepts_handdrawn_media_statuses(status: str) -> None:
    assert (
        EpisodeState(book_id="book-demo", episode_id="E001", status=status).status
        == status
    )


@pytest.mark.parametrize(
    ("legacy", "neutral"),
    [
        ("awaiting_h3_generation", "awaiting_video_generation"),
        ("h3_import_invalid", "video_import_invalid"),
        ("h3_partial", "video_partial"),
        ("h3_imported", "video_imported"),
    ],
)
def test_legacy_episode_status_normalizes_before_validation(legacy: str, neutral: str) -> None:
    """Old H3 episode statuses must become the matching neutral name, not stay as literals."""
    episode = EpisodeState.model_validate(
        {"book_id": "book-demo", "episode_id": "E001", "status": legacy}
    )
    assert episode.status == neutral
    assert episode.model_dump()["status"] == neutral


def test_legacy_episode_json_round_trip_serializes_only_neutral_names(
    tmp_path: Path,
) -> None:
    """A loaded H3-era episode.json must persist only await_video_segments / video statuses."""
    store = StateStore(tmp_path)
    episode_dir = tmp_path / "books" / "book-demo" / "episodes" / "E001"
    episode_dir.mkdir(parents=True)
    payload = {
        "book_id": "book-demo",
        "episode_id": "E001",
        "status": "awaiting_h3_generation",
        "stage_manifests": {
            "await_h3_segments": {
                "stage": "await_h3_segments",
                "status": "pending",
                "inputs": {},
                "outputs": {},
                "config_sha256": "0" * 64,
            }
        },
        "completed_stages": ["storyboard", "await_h3_segments"],
        "stale_stages": ["await_h3_segments", "render"],
    }
    (episode_dir / "episode.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = store.load_episode("book-demo", "E001")
    assert loaded.status == "awaiting_video_generation"
    assert "await_video_segments" in loaded.stage_manifests
    assert "await_h3_segments" not in loaded.stage_manifests
    assert loaded.stage_manifests["await_video_segments"].stage == "await_video_segments"
    assert loaded.completed_stages == ["storyboard", "await_video_segments"]
    assert loaded.stale_stages == ["await_video_segments", "render"]

    store.save_episode(loaded)
    dumped = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    serialized = json.dumps(dumped)
    assert dumped["status"] == "awaiting_video_generation"
    assert "await_video_segments" in dumped["stage_manifests"]
    assert dumped["stage_manifests"]["await_video_segments"]["stage"] == "await_video_segments"
    assert dumped["completed_stages"] == ["storyboard", "await_video_segments"]
    assert dumped["stale_stages"] == ["await_video_segments", "render"]
    assert "awaiting_h3_generation" not in serialized
    assert "await_h3_segments" not in serialized


def test_conflicting_legacy_and_neutral_stage_manifest_keys_fail_validation() -> None:
    """Keeping both await_h3_segments and await_video_segments must not drop one silently."""
    payload = {
        "book_id": "book-demo",
        "episode_id": "E001",
        "stage_manifests": {
            "await_h3_segments": {
                "stage": "await_h3_segments",
                "status": "completed",
                "inputs": {},
                "outputs": {},
                "config_sha256": "0" * 64,
            },
            "await_video_segments": {
                "stage": "await_video_segments",
                "status": "failed",
                "inputs": {},
                "outputs": {},
                "config_sha256": "0" * 64,
            },
        },
    }
    with pytest.raises(ValidationError):
        EpisodeState.model_validate(payload)


def _legacy_manifest(*, stage: str, status: str) -> dict[str, object]:
    return {
        "stage": stage,
        "status": status,
        "inputs": {},
        "outputs": {},
        "config_sha256": "0" * 64,
    }


def test_episode_state_model_validate_does_not_mutate_legacy_input() -> None:
    """Migration must copy nested mappings/lists; the caller's payload stays H3-era."""
    nested_manifest = _legacy_manifest(stage="await_h3_segments", status="pending")
    completed = ["storyboard", "await_h3_segments"]
    stale = ["await_h3_segments", "render"]
    payload = {
        "book_id": "book-demo",
        "episode_id": "E001",
        "status": "awaiting_h3_generation",
        "stage_manifests": {"await_h3_segments": nested_manifest},
        "completed_stages": completed,
        "stale_stages": stale,
    }
    snapshot = copy.deepcopy(payload)

    episode = EpisodeState.model_validate(payload)

    assert payload == snapshot
    assert payload["completed_stages"] is completed
    assert payload["stale_stages"] is stale
    assert payload["stage_manifests"]["await_h3_segments"] is nested_manifest
    assert nested_manifest["stage"] == "await_h3_segments"
    assert completed == ["storyboard", "await_h3_segments"]
    assert stale == ["await_h3_segments", "render"]
    assert episode.status == "awaiting_video_generation"
    assert list(episode.stage_manifests) == ["await_video_segments"]
    assert episode.completed_stages == ["storyboard", "await_video_segments"]
    assert episode.stale_stages == ["await_video_segments", "render"]


def test_unhashable_stage_list_item_raises_validation_error() -> None:
    """Set-based dedupe would leak TypeError; invalid items must stay ValidationError."""
    payload = {
        "book_id": "book-demo",
        "episode_id": "E001",
        "completed_stages": ["storyboard", {"not": "a-stage"}],
        "stale_stages": [{"also": "unhashable"}, "render"],
    }
    with pytest.raises(ValidationError) as captured:
        EpisodeState.model_validate(payload)
    assert not isinstance(captured.value.__cause__, TypeError)
    assert not isinstance(captured.value.__context__, TypeError)


def test_equal_stage_manifests_and_legacy_manifests_alias_collapse() -> None:
    """Independent normalize of both keys must collapse when the mappings match."""
    payload = {
        "book_id": "book-demo",
        "episode_id": "E001",
        "stage_manifests": {
            "await_h3_segments": _legacy_manifest(
                stage="await_h3_segments", status="pending"
            )
        },
        "manifests": {
            "await_video_segments": _legacy_manifest(
                stage="await_video_segments", status="pending"
            )
        },
    }
    snapshot = copy.deepcopy(payload)

    episode = EpisodeState.model_validate(payload)

    assert payload == snapshot
    assert "manifests" in payload
    assert list(episode.stage_manifests) == ["await_video_segments"]
    assert episode.stage_manifests["await_video_segments"].stage == "await_video_segments"
    assert episode.stage_manifests["await_video_segments"].status == "pending"
    dumped = episode.model_dump()
    assert "manifests" not in dumped
    assert list(dumped["stage_manifests"]) == ["await_video_segments"]


def test_conflicting_stage_manifests_and_legacy_manifests_alias_fail_validation() -> None:
    """Differing normalized mappings under both keys must not pick one silently."""
    payload = {
        "book_id": "book-demo",
        "episode_id": "E001",
        "stage_manifests": {
            "await_video_segments": _legacy_manifest(
                stage="await_video_segments", status="completed"
            )
        },
        "manifests": {
            "await_h3_segments": _legacy_manifest(
                stage="await_h3_segments", status="failed"
            )
        },
    }
    snapshot = copy.deepcopy(payload)

    with pytest.raises(ValidationError, match="conflicting_video_stage_manifests"):
        EpisodeState.model_validate(payload)

    assert payload == snapshot
