import sys
from pathlib import Path

from typer.testing import CliRunner

from bv.cli import app
from bv.state.models import BookState, EpisodeState
from bv.state.store import StateStore
from bv.workflow.orchestrator import Orchestrator
from bv.workflow.stages import STAGE_ORDER

sys.path.insert(0, str(Path(__file__).parents[1]))
from fakes import ArtifactStage, FakeVideoGateway


def _workflow(tmp_path: Path) -> Orchestrator:
    store = StateStore(tmp_path / "workspace")
    episode_path = store.root / "books" / "book-demo" / "episodes" / "E001"
    episode_path.mkdir(parents=True)
    store.save_book(BookState(book_id="book-demo", title="Demo"))
    store.save_episode(EpisodeState(book_id="book-demo", episode_id="E001"))
    return Orchestrator(
        store=store,
        stages={
            name: ArtifactStage(name, "completed")
            for name in STAGE_ORDER
            if name not in {"review_script", "await_video_segments"}
        },
        video_gateway=FakeVideoGateway(["S01", "S02", "S03", "S04"]),
        gate_approvers={"script": ArtifactStage("approve_script", "completed")},
    )


def test_cli_status_lists_shot_ids_and_import_video_uses_shot_id(
    tmp_path: Path, monkeypatch
) -> None:
    """Dropping the shot-id positional argument would leave the import ambiguous."""
    from bv import cli

    workflow = _workflow(tmp_path)
    monkeypatch.setattr(
        cli, "build_orchestrator", lambda config, authorization=None: workflow
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("workspace_dir: workspace\n", encoding="utf-8")
    source = tmp_path / "S01.mp4"
    source.write_bytes(b"local fixture")
    runner = CliRunner()

    initial = runner.invoke(app, ["status", "--book-id", "book-demo", "--episode-id", "E001"])
    review = runner.invoke(app, ["next", "book-demo"])
    approved = runner.invoke(app, ["approve", "book-demo", "E001", "script"])
    video_ready = runner.invoke(app, ["next", "book-demo"])
    imported = runner.invoke(
        app, ["import-video", "book-demo", "E001", "S01", str(source)]
    )

    assert initial.exit_code == 0
    assert "Present shots: none" in initial.stdout
    assert "Missing shots: none" in initial.stdout
    assert "Next: bv next book-demo" in initial.stdout
    assert review.exit_code == approved.exit_code == video_ready.exit_code == 0
    assert "Status: awaiting_video_generation" in video_ready.stdout
    assert "Next: bv open book-demo E001 video" in video_ready.stdout
    assert imported.exit_code == 0
    assert "Status: visual_sample_ready" in imported.stdout
    assert "Next: bv open book-demo E001 video" in imported.stdout
