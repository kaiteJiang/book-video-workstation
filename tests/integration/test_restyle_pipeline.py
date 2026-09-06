from __future__ import annotations

from pathlib import Path

import pytest

from bv.core.hashing import sha256_file
from bv.illustration.contracts import IllustrationStoryboard
from bv.state.models import ArtifactRef
from bv.state.store import StateStore
from bv.workflow.media_runtime import MediaProductionService, MediaWorkflowError
import bv.workflow.media_runtime as media_runtime
from bv.workflow.runtime import _media_runtime_config_sha256
from bv.config import AppConfig

from tests.unit.test_illustration_restyle import _episode_fixture


class _Images:
    def status(self, _context):
        return (), ("S01-A", "S01-B", "S03-A", "S03-B", "S04-A", "S04-B")


def test_restyle_preserves_current_social_cover_while_archiving_old_final_output(
    tmp_path: Path,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    social = root / "media" / "final" / "social_cover.png"
    social.write_bytes(b"current social cover")
    episode.stage_manifests["social_cover"].outputs = {
        "social_cover": ArtifactRef(
            path=str(social), sha256=sha256_file(social), size_bytes=social.stat().st_size
        )
    }
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = MediaProductionService(
        store=store,
        stages={},
        images=_Images(),
        restyle_catalog_path=Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json"),
    )

    view = service.restyle("book-demo", "E001", style_id="warm-flat-storybook")

    updated = store.load_episode("book-demo", "E001")
    recovery = next((root / ".recovery" / "style-restyle").iterdir())
    assert view.status == "representative_generation_running"
    assert social.read_bytes() == b"current social cover"
    assert (recovery / "final" / "final.mp4").read_bytes() == b"old final"
    assert updated.stage_manifests["social_cover"].status == "completed"
    assert "social_cover" not in updated.stale_stages


def test_restyle_resume_preserves_media_mode_and_never_runs_stages(tmp_path: Path) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    first = MediaProductionService(
        store=store, stages={}, images=_Images(), configured_mode="full",
        restyle_catalog_path=Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json"),
    )

    first.restyle("book-demo", "E001", style_id="warm-flat-storybook")
    assert not (root / "production_profile.json").exists()

    calls: list[str] = []
    class _PlannerTrap:
        def __init__(self, name: str) -> None:
            self.name = name
        def run(self, _context):
            calls.append(self.name)
            raise AssertionError("restyle resume must not invoke a stage")
    restarted = MediaProductionService(
        store=store,
        stages={name: _PlannerTrap(name) for name in ("tts", "asr", "subtitles", "select_style", "plan_illustrations", "prepare_representatives")},
        images=_Images(), configured_mode="full",
        restyle_catalog_path=Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json"),
    )

    view = restarted.status("book-demo", "E001")
    resumed = restarted.prepare("book-demo", "E001", mode="full")

    state = store.load_episode("book-demo", "E001")
    assert view.status == "representative_generation_running"
    assert resumed.status == "representative_generation_running"
    assert calls == []
    assert {state.stage_manifests[name].inputs["media_mode"] for name in ("select_style", "plan_illustrations", "prepare_representatives")} == {"full"}


def test_restyle_override_survives_changed_config_without_planning_model(tmp_path: Path) -> None:
    root, episode, old_storyboard, _ = _episode_fixture(tmp_path)
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes(Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json").read_bytes())
    store = StateStore(tmp_path)
    store.save_episode(episode)
    first = MediaProductionService(
        store=store, stages={}, images=_Images(), configured_mode="full",
        restyle_catalog_path=catalog,
    )
    first.restyle("book-demo", "E001", style_id="warm-flat-storybook")
    catalog.write_bytes(catalog.read_bytes() + b"\n")
    state = store.load_episode("book-demo", "E001")
    for name in ("tts", "asr", "subtitles"):
        state.stage_manifests[name].config_sha256 = "c" * 64
    store.save_episode(state)
    calls: list[str] = []

    class _PlanningTrap:
        def __init__(self, name: str) -> None:
            self.name = name

        def run(self, _context):
            calls.append(self.name)
            raise AssertionError("durable restyle must not invoke selector or planner")

    restarted = MediaProductionService(
        store=store,
        stages={name: _PlanningTrap(name) for name in ("select_style", "plan_illustrations", "prepare_representatives")},
        images=_Images(), configured_mode="full", config_sha256="c" * 64,
        restyle_catalog_path=catalog,
    )

    view = restarted.prepare("book-demo", "E001", mode="full")
    decision = (root / "media" / "illustration" / "style_decision.json").read_text(encoding="utf-8")
    storyboard = IllustrationStoryboard.model_validate_json(
        (root / "media" / "illustration" / "illustration_storyboard.json").read_text(encoding="utf-8")
    )

    assert view.status == "representative_generation_running"
    assert "warm-flat-storybook" in decision
    assert storyboard.scenes == old_storyboard.scenes
    assert calls == []
    assert (root / "script" / "illustration_style_override.json").is_file()
    assert {
        state.config_sha256
        for state in store.load_episode("book-demo", "E001").stage_manifests.values()
        if state.stage in {
            "select_style",
            "plan_illustrations",
            "prepare_representatives",
        }
    } == {f"global-v1:{'c' * 64}"}


def test_representative_action_describes_pair_review_and_rejects_missing_evidence(
    tmp_path: Path,
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    service = MediaProductionService(store=StateStore(tmp_path), stages={}, images=_Images())
    pair = service._view(
        episode.model_copy(update={"status": "awaiting_representative_review"}),
        missing=("S01-A", "S01-B", "S03-A", "S03-B", "S04-A", "S04-B"),
    )
    (root / "media" / "illustration" / "illustration_storyboard.json").unlink()

    assert "六张" in pair.next_action
    assert "A/B" in pair.next_action
    assert "人物/服装/场景/镜头/动作连续性" in pair.next_action
    assert len(pair.missing_scene_ids) == 6
    with pytest.raises(MediaWorkflowError, match="representative_surface_invalid"):
        service._view(
            episode.model_copy(update={"status": "awaiting_representative_review"}),
            missing=("S01", "S05", "S10"),
        )


def test_restyle_rejects_missing_or_conflicting_media_mode_before_install(tmp_path: Path) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    episode.stage_manifests["plan_illustrations"].inputs["media_mode"] = "technical_sample"
    store = StateStore(tmp_path)
    store.save_episode(episode)
    service = MediaProductionService(
        store=store, stages={}, images=_Images(), configured_mode="full",
        restyle_catalog_path=Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json"),
    )
    before = (root / "media" / "illustration" / "style_decision.json").read_bytes()

    with pytest.raises(Exception, match="restyle_media_mode_invalid"):
        service.restyle("book-demo", "E001", style_id="warm-flat-storybook")

    assert (root / "media" / "illustration" / "style_decision.json").read_bytes() == before


def test_restyle_state_save_failure_restores_old_files_and_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    before_state = (root / "episode.json").read_bytes()
    service = MediaProductionService(
        store=store, stages={}, images=_Images(), configured_mode="full",
        restyle_catalog_path=Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json"),
    )
    monkeypatch.setattr(store, "save_episode", lambda _episode: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(Exception, match="restyle_state_commit_failed"):
        service.restyle("book-demo", "E001", style_id="warm-flat-storybook")

    assert (root / "episode.json").read_bytes() == before_state
    assert (root / "media" / "illustration" / "images" / "S01_anchor.png").read_bytes() == b"S01-A"
    assert (root / "media" / "render" / "picture_silent.mp4").read_bytes() == b"old visual"


def test_restyle_outcome_hash_failure_restores_old_files_and_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    store = StateStore(tmp_path)
    store.save_episode(episode)
    before_state = (root / "episode.json").read_bytes()
    before_illustration = (root / "media" / "illustration" / "style_decision.json").read_bytes()
    before_render = (root / "media" / "render" / "picture_silent.mp4").read_bytes()
    service = MediaProductionService(
        store=store, stages={}, images=_Images(), configured_mode="full",
        restyle_catalog_path=Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json"),
    )
    original_hash = media_runtime.sha256_file

    def fail_new_style(path: Path) -> str:
        if path.name == "style_decision.json":
            raise OSError("injected outcome hashing failure")
        return original_hash(path)

    monkeypatch.setattr(media_runtime, "sha256_file", fail_new_style)
    with pytest.raises(Exception, match="restyle_state_commit_failed"):
        service.restyle("book-demo", "E001", style_id="warm-flat-storybook")

    assert (root / "episode.json").read_bytes() == before_state
    assert (root / "media" / "illustration" / "style_decision.json").read_bytes() == before_illustration
    assert (root / "media" / "render" / "picture_silent.mp4").read_bytes() == before_render


def test_config_fingerprint_invalidates_old_stage_manifest(tmp_path: Path) -> None:
    root, episode, _, _ = _episode_fixture(tmp_path)
    current = MediaProductionService(store=StateStore(tmp_path), stages={}, images=_Images())
    changed = MediaProductionService(store=StateStore(tmp_path), stages={}, images=_Images(), config_sha256="c" * 64)

    assert current._manifest_current(episode, "select_style", mode="full")
    assert not changed._manifest_current(episode, "select_style", mode="full")

    catalog = Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json")
    prompts = (Path("prompts/illustration/style_select.md"), Path("prompts/illustration/storyboard.md"))
    first = AppConfig(workspace_dir=tmp_path / "one")
    second = first.model_copy(update={"workspace_dir": tmp_path / "two"})
    assert _media_runtime_config_sha256(first, catalog_path=catalog, prompt_paths=prompts) == _media_runtime_config_sha256(first, catalog_path=catalog, prompt_paths=prompts)
    assert _media_runtime_config_sha256(first, catalog_path=catalog, prompt_paths=prompts) != _media_runtime_config_sha256(second, catalog_path=catalog, prompt_paths=prompts)
