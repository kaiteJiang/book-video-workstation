import sys
from pathlib import Path

import pytest

from bv.state.models import BookState, EpisodeState
from bv.state.store import StateStore
from bv.video.contracts import VideoGateway
from bv.workflow.orchestrator import H3Gateway, Orchestrator, WorkflowError
from bv.workflow.stages import GATES, STAGE_ORDER

sys.path.insert(0, str(Path(__file__).parents[1]))
from fakes import ArtifactStage, FakeH3Gateway, FakeStage, FakeVideoGateway


def _workflow(tmp_path: Path) -> tuple[Orchestrator, FakeVideoGateway, dict[str, FakeStage]]:
    store = StateStore(tmp_path / "workspace")
    episode_path = store.root / "books" / "book-demo" / "episodes" / "E001"
    episode_path.mkdir(parents=True)
    store.save_book(BookState(book_id="book-demo", title="Demo"))
    store.save_episode(EpisodeState(book_id="book-demo", episode_id="E001"))
    stages = {
        name: ArtifactStage(name, "completed")
        for name in STAGE_ORDER
        if name not in {"review_script", "await_video_segments"}
    }
    video = FakeVideoGateway(["S01", "S02", "S03", "S04"])
    return (
        Orchestrator(
            store=store,
            stages=stages,
            video_gateway=video,
            gate_approvers={"script": ArtifactStage("approve_script", "completed")},
        ),
        video,
        stages,
    )


def test_stage_order_and_gate_use_neutral_video_names() -> None:
    """Leaving await_h3_segments in STAGE_ORDER would keep the H3-only gate."""
    assert "await_video_segments" in STAGE_ORDER
    assert "await_h3_segments" not in STAGE_ORDER
    assert GATES["await_video_segments"] == "awaiting_video_generation"
    assert "await_h3_segments" not in GATES


def test_h3_gateway_public_alias_is_video_gateway() -> None:
    assert H3Gateway is VideoGateway
    assert FakeH3Gateway is FakeVideoGateway
    assert isinstance(FakeVideoGateway(["S01", "S02", "S03", "S04"]), VideoGateway)


def test_video_gateway_is_primary_and_conflicts_with_deprecated_h3_gateway(
    tmp_path: Path,
) -> None:
    """Supplying both constructor names would silently pick one gateway."""
    store = StateStore(tmp_path / "workspace")
    primary = FakeVideoGateway(["S01", "S02", "S03", "S04"])
    deprecated = FakeVideoGateway(["S01", "S02", "S03", "S04"])

    workflow = Orchestrator(store=store, video_gateway=primary)
    assert workflow.video_gateway is primary

    legacy = Orchestrator(store=store, h3_gateway=deprecated)
    assert legacy.video_gateway is deprecated

    with pytest.raises(WorkflowError, match="video_gateway_conflict"):
        Orchestrator(store=store, video_gateway=primary, h3_gateway=deprecated)


def test_s01_does_not_unlock_render_but_all_segments_do(tmp_path: Path) -> None:
    """Removing the video completeness check would incorrectly run render after S01."""
    workflow, video, stages = _workflow(tmp_path)

    assert workflow.next("book-demo", "E001").status == "awaiting_script_review"
    workflow.approve_gate("book-demo", "E001", "script")
    gate = workflow.next("book-demo", "E001")
    assert gate.status == "awaiting_video_generation"
    assert gate.next_command == "bv open book-demo E001 video"

    source = tmp_path / "generated.mp4"
    source.write_bytes(b"local fixture")
    workflow.import_video("book-demo", "E001", "S01", source)
    sample = workflow.status_view("book-demo", "E001")
    assert sample.status == "visual_sample_ready"
    assert sample.next_command == "bv open book-demo E001 video"
    assert sample.present_shot_ids == ("S01",)
    assert sample.missing_shot_ids == ("S02", "S03", "S04")

    for shot in ("S02", "S03", "S04"):
        imported = workflow.import_video("book-demo", "E001", shot, source)

    assert imported.next_command == "bv next book-demo"
    assert imported.status == "video_imported"
    assert workflow.next("book-demo", "E001").status == "awaiting_final_review"
    assert stages["render"].calls == 1
    assert stages["qc"].calls == 1
    assert video.status("book-demo", "E001")["missing_shot_ids"] == ()


def test_missing_stage_artifact_invalidates_later_completed_stages(tmp_path: Path) -> None:
    """Removing a TTS artifact must not let ASR/subtitles/storyboard stay current."""
    store = StateStore(tmp_path / "workspace")
    episode_path = store.root / "books" / "book-demo" / "episodes" / "E001"
    episode_path.mkdir(parents=True)
    store.save_book(BookState(book_id="book-demo", title="Demo"))
    store.save_episode(EpisodeState(book_id="book-demo", episode_id="E001"))
    stages = {
        name: ArtifactStage(name, "completed")
        for name in STAGE_ORDER
        if name not in {"review_script", "await_video_segments"}
    }
    workflow = Orchestrator(
        store=store,
        stages=stages,
        video_gateway=FakeVideoGateway(["S01", "S02", "S03", "S04"]),
        gate_approvers={"script": ArtifactStage("approve_script", "completed")},
    )

    workflow.next("book-demo", "E001")
    workflow.approve_gate("book-demo", "E001", "script")
    assert workflow.next("book-demo", "E001").status == "awaiting_video_generation"
    (episode_path / ".test-stage" / "tts").unlink()

    assert workflow.next("book-demo", "E001").status == "awaiting_video_generation"
    assert stages["tts"].calls == 2
    assert stages["asr"].calls == stages["subtitles"].calls == stages["storyboard"].calls == 2


def test_script_gate_requires_a_real_approval_runner_and_durable_artifact(tmp_path: Path) -> None:
    workflow, _, _ = _workflow(tmp_path)
    workflow.gate_approvers.clear()

    assert workflow.next("book-demo", "E001").status == "awaiting_script_review"
    with pytest.raises(WorkflowError, match="approval_not_configured"):
        workflow.approve_gate("book-demo", "E001", "script")

    state = workflow.store.load_episode("book-demo", "E001")
    assert state.status == "awaiting_script_review"
    assert "review_script" not in state.completed_stages


def test_empty_or_invalid_stage_result_fails_durably_and_can_retry(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "workspace")
    episode_path = store.root / "books" / "book-demo" / "episodes" / "E001"
    episode_path.mkdir(parents=True)
    store.save_book(BookState(book_id="book-demo", title="Demo"))
    store.save_episode(EpisodeState(book_id="book-demo", episode_id="E001"))
    workflow = Orchestrator(
        store=store,
        stages={"parse_source": FakeStage("parse_source", "completed")},
        video_gateway=FakeVideoGateway(["S01", "S02", "S03", "S04"]),
    )

    with pytest.raises(WorkflowError, match="stage_output_missing"):
        workflow.next("book-demo", "E001")
    failed = store.load_episode("book-demo", "E001")
    assert failed.stage_manifests["parse_source"].status == "failed"
    assert failed.failure_summary == "stage_output_missing"

    workflow.stages["parse_source"] = ArtifactStage("parse_source", "completed")
    with pytest.raises(WorkflowError, match="stage_not_configured"):
        workflow.retry("book-demo", "E001")
    retried = store.load_episode("book-demo", "E001")
    assert retried.stage_manifests["parse_source"].status == "completed"


def test_forged_empty_video_partition_and_unsafe_identifiers_fail_closed(
    tmp_path: Path,
) -> None:
    workflow, _, _ = _workflow(tmp_path)
    assert workflow.next("book-demo", "E001").status == "awaiting_script_review"
    workflow.approve_gate("book-demo", "E001", "script")

    class EmptyGateway:
        def generate(self, book_id, episode_id, shot_id):
            raise RuntimeError("empty_gateway_does_not_generate")

        def status(self, book_id, episode_id):
            return {"status": "video_imported", "present_shot_ids": (), "missing_shot_ids": ()}

        def import_video(self, book_id, episode_id, shot_id, source):
            return self.status(book_id, episode_id)

    workflow.video_gateway = EmptyGateway()
    with pytest.raises(WorkflowError, match="video_manifest_invalid"):
        workflow.next("book-demo", "E001")

    with pytest.raises(WorkflowError, match="unsafe_identifier"):
        workflow.status_view("..", "E001")


def test_legacy_consistent_h3_gateway_status_normalizes_to_neutral(tmp_path: Path) -> None:
    """Old H3 gateway strings stay accepted only when IDs match, then become neutral."""
    workflow, _, _ = _workflow(tmp_path)
    workflow.next("book-demo", "E001")
    workflow.approve_gate("book-demo", "E001", "script")
    assert workflow.next("book-demo", "E001").status == "awaiting_video_generation"

    class LegacyGateway:
        def __init__(self, status: str, present: tuple[str, ...], missing: tuple[str, ...]):
            self._status = status
            self._present = present
            self._missing = missing

        def generate(self, book_id, episode_id, shot_id):
            raise RuntimeError("legacy_gateway_does_not_generate")

        def import_video(self, book_id, episode_id, shot_id, source):
            return self.status(book_id, episode_id)

        def status(self, book_id, episode_id):
            return {
                "status": self._status,
                "present_shot_ids": self._present,
                "missing_shot_ids": self._missing,
            }

    workflow.video_gateway = LegacyGateway(
        "h3_partial", ("S01", "S02"), ("S03", "S04")
    )
    partial = workflow.status_view("book-demo", "E001")
    assert partial.status == "video_partial"
    assert partial.present_shot_ids == ("S01", "S02")
    assert partial.missing_shot_ids == ("S03", "S04")

    workflow.video_gateway = LegacyGateway(
        "awaiting_h3_segments", ("S01", "S02"), ("S03", "S04")
    )
    assert workflow.status_view("book-demo", "E001").status == "video_partial"

    workflow.video_gateway = LegacyGateway(
        "h3_imported", ("S01", "S02", "S03", "S04"), ()
    )
    imported = workflow.status_view("book-demo", "E001")
    assert imported.status == "video_imported"

    workflow.video_gateway = LegacyGateway(
        "awaiting_h3_generation", (), ("S01", "S02", "S03", "S04")
    )
    assert workflow.status_view("book-demo", "E001").status == "awaiting_video_generation"

    workflow.video_gateway = LegacyGateway("h3_imported", (), ())
    with pytest.raises(WorkflowError, match="video_manifest_invalid"):
        workflow.status_view("book-demo", "E001")

    workflow.video_gateway = LegacyGateway(
        "h3_imported", ("S01",), ("S02", "S03", "S04")
    )
    with pytest.raises(WorkflowError, match="video_manifest_invalid"):
        workflow.status_view("book-demo", "E001")


def test_lost_video_clip_or_script_approval_forces_downstream_rerun(tmp_path: Path) -> None:
    workflow, video, stages = _workflow(tmp_path)
    workflow.next("book-demo", "E001")
    workflow.approve_gate("book-demo", "E001", "script")
    workflow.next("book-demo", "E001")
    source = tmp_path / "local.mp4"
    source.write_bytes(b"fixture")
    for shot in ("S01", "S02", "S03", "S04"):
        workflow.import_video("book-demo", "E001", shot, source)
    assert workflow.next("book-demo", "E001").status == "awaiting_final_review"
    assert stages["render"].calls == 1

    video.present_shot_ids.remove("S04")
    assert workflow.next("book-demo", "E001").status == "video_partial"
    workflow.import_video("book-demo", "E001", "S04", source)
    assert workflow.next("book-demo", "E001").status == "awaiting_final_review"
    assert stages["render"].calls == stages["qc"].calls == 2

    approval_file = (
        workflow.store.root / "books" / "book-demo" / "episodes" / "E001"
        / ".test-stage" / "approve_script"
    )
    approval_file.unlink()
    assert workflow.next("book-demo", "E001").status == "awaiting_script_review"
    workflow.approve_gate("book-demo", "E001", "script")
    assert workflow.next("book-demo", "E001").status == "awaiting_final_review"
    assert stages["tts"].calls == stages["asr"].calls == 2
    assert stages["render"].calls == stages["qc"].calls == 3


def test_video_gateway_errors_and_import_gate_are_neutral(tmp_path: Path) -> None:
    """Active video errors must not keep the H3-prefixed codes."""
    store = StateStore(tmp_path / "workspace")
    episode_path = store.root / "books" / "book-demo" / "episodes" / "E001"
    episode_path.mkdir(parents=True)
    store.save_book(BookState(book_id="book-demo", title="Demo"))
    store.save_episode(EpisodeState(book_id="book-demo", episode_id="E001"))
    stages = {
        name: ArtifactStage(name, "completed")
        for name in STAGE_ORDER
        if name not in {"review_script", "await_video_segments"}
    }
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fixture")

    unconfigured = Orchestrator(
        store=store,
        stages=stages,
        gate_approvers={"script": ArtifactStage("approve_script", "completed")},
    )
    unconfigured.next("book-demo", "E001")
    unconfigured.approve_gate("book-demo", "E001", "script")
    with pytest.raises(WorkflowError, match="video_gateway_not_configured"):
        unconfigured.next("book-demo", "E001")
    with pytest.raises(WorkflowError, match="video_gateway_not_configured"):
        unconfigured.import_video("book-demo", "E001", "S01", source)

    workflow, _, _ = _workflow(tmp_path / "configured")
    with pytest.raises(WorkflowError, match="video_gate_not_ready"):
        workflow.import_video("book-demo", "E001", "S01", source)

    class FailingImportGateway:
        def generate(self, book_id, episode_id, shot_id):
            raise RuntimeError("failing_gateway_does_not_generate")

        def import_video(self, book_id, episode_id, shot_id, source_path):
            raise RuntimeError("copy exploded")

        def status(self, book_id, episode_id):
            return {
                "status": "awaiting_video_generation",
                "present_shot_ids": (),
                "missing_shot_ids": ("S01", "S02", "S03", "S04"),
            }

    workflow.next("book-demo", "E001")
    workflow.approve_gate("book-demo", "E001", "script")
    assert workflow.next("book-demo", "E001").status == "awaiting_video_generation"
    workflow.video_gateway = FailingImportGateway()
    with pytest.raises(WorkflowError, match="video_import_failed"):
        workflow.import_video("book-demo", "E001", "S01", source)


def test_open_target_video_is_primary_and_h3_is_neutral_alias(tmp_path: Path) -> None:
    """open_target('h3') must not keep an H3 target or drop video_prompt_*.md files."""
    workflow, _, _ = _workflow(tmp_path)
    workflow.next("book-demo", "E001")
    workflow.approve_gate("book-demo", "E001", "script")
    assert workflow.next("book-demo", "E001").status == "awaiting_video_generation"

    storyboard = (
        workflow.store.root / "books" / "book-demo" / "episodes" / "E001" / "storyboard"
    )
    storyboard.mkdir(parents=True)
    video_prompt = storyboard / "video_prompt_S01.md"
    legacy_prompt = storyboard / "h3_prompt_S02.md"
    duplicate_new = storyboard / "video_prompt_S03.md"
    duplicate_old = storyboard / "h3_prompt_S03.md"
    video_prompt.write_text("new", encoding="utf-8")
    legacy_prompt.write_text("legacy", encoding="utf-8")
    duplicate_new.write_text("both-new", encoding="utf-8")
    duplicate_old.write_text("both-old", encoding="utf-8")

    opened = workflow.open_target("book-demo", "E001", "video")
    aliased = workflow.open_target("book-demo", "E001", "h3")

    assert opened.target == "video"
    assert aliased.target == "video"
    assert opened.path == storyboard
    assert aliased.path == storyboard
    assert opened.listing == aliased.listing
    assert "Present shots: none" in opened.listing
    assert "Missing shots: S01, S02, S03, S04" in opened.listing
    assert str(video_prompt) in opened.listing
    assert str(legacy_prompt) in opened.listing
    assert opened.listing.count(str(video_prompt)) == 1
    assert opened.listing.count(str(legacy_prompt)) == 1
    assert opened.listing.count(str(duplicate_new)) == 1
    assert opened.listing.count(str(duplicate_old)) == 1


def test_stage_output_must_be_a_regular_file_inside_workspace(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "workspace")
    root = store.root / "books" / "book-demo" / "episodes" / "E001"
    root.mkdir(parents=True)
    store.save_book(BookState(book_id="book-demo", title="Demo"))
    store.save_episode(EpisodeState(book_id="book-demo", episode_id="E001"))
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")

    class OutsideStage:
        def run(self, context):
            return {"outputs": {"outside": outside}}

    workflow = Orchestrator(store=store, stages={"parse_source": OutsideStage()})
    with pytest.raises(WorkflowError, match="stage_output_unsafe"):
        workflow.next("book-demo", "E001")
    assert store.load_episode("book-demo", "E001").failure_summary == "stage_output_unsafe"
