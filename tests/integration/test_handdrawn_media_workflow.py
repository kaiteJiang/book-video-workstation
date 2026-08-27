from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from bv.core.atomic import atomic_write_json
from bv.illustration.assets import prepare_image_jobs
from bv.illustration.contracts import CharacterLock, IllustrationScene, IllustrationStoryboard, StyleDecision
from bv.illustration.style_selector import load_style_catalog
from bv.state.models import EpisodeState
from bv.state.store import StateStore
from bv.workflow.media_runtime import LocalIllustrationGateway, MediaProductionService
from bv.workflow.stages import HANDDRAWN_MEDIA_PREPARE_ORDER, StageOutcome


def _canonical(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _manifest(episode_root: Path):
    catalog = load_style_catalog(
        Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json")
    )
    style = next(item for item in catalog if item.style_id == "emotional-watercolor-sketch")
    decision = StyleDecision(
        library_version="1", selected_style=style.style_id,
        candidate_styles=(style.style_id, "colored-pencil-diary", "warm-flat-storybook"),
        selection_reasons=("测试",),
        rejected_reasons={"colored-pencil-diary": "测试", "warm-flat-storybook": "测试"},
        confidence=1.0, manual_override=False,
        source_script_sha256="a" * 64, style_fingerprint="b" * 64,
    )
    character = CharacterLock(
        role="普通读者", age_range="30至39岁", face="椭圆脸", hair="黑色短发",
        body="中等身材", base_clothing="深蓝衬衫", allowed_variations=(),
        color_markers=("深蓝",), personal_objects=("笔记本",), forbidden_changes=("发型",),
        source_script_sha256="a" * 64, style_fingerprint="b" * 64,
    )
    representative = {0, 2, 4}
    scenes = tuple(
        IllustrationScene(
            scene_id=f"S{index + 1:02d}", start_ms=index * 3_000,
            end_ms=(index + 1) * 3_000, from_frame=index * 90, to_frame=(index + 1) * 90,
            narration="测试批准旁白", narration_span=(index * 6, (index + 1) * 6),
            key_line="把生活还给自己", visual_purpose="连接真实生活", setting="家中餐桌",
            character_action="读者翻开笔记本", metaphor=None, composition="主体居中偏下",
            character_refs=("reader-01",), image_prompt="成年读者在餐桌前思考",
            negative_constraints=("禁止文字",), representative_frame=index in representative,
            asset_status="planned",
        )
        for index in range(5)
    )
    storyboard = IllustrationStoryboard(
        book_id="book-demo", episode_id="E001", width=1080, height=1920, fps=30,
        master_duration_ms=15_000, total_frames=450, script_sha256="a" * 64,
        audio_sha256="c" * 64, subtitle_sha256="d" * 64,
        style_decision_sha256=_canonical(decision.model_dump(mode="json")),
        character_lock_sha256=_canonical(character.model_dump(mode="json")), scenes=scenes,
    )
    return storyboard, prepare_image_jobs(storyboard, character, style, decision, episode_root)


class _PreparedStage:
    def __init__(self, name: str, storyboard, manifest) -> None:
        self.name = name
        self.storyboard = storyboard
        self.manifest = manifest

    def run(self, context) -> StageOutcome:
        root = context.episode_root / "media" / "illustration"
        if self.name == "prepare_representatives":
            storyboard_path = root / "illustration_storyboard.json"
            manifest_path = root / "illustration_manifest.json"
            atomic_write_json(storyboard_path, self.storyboard.model_dump(mode="json"))
            atomic_write_json(manifest_path, self.manifest.model_dump(mode="json"))
            return StageOutcome(outputs={"illustration_manifest": manifest_path})
        output = context.episode_root / "media" / "fixtures" / f"{self.name}.json"
        atomic_write_json(output, {"stage": self.name})
        return StageOutcome(outputs={self.name: output})


def test_real_ffmpeg_imports_representatives_then_unlocks_and_completes_batch(
    tmp_path: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("local FFmpeg is required")
    workspace = tmp_path / "workspace"
    episode_root = workspace / "books" / "book-demo" / "episodes" / "E001"
    episode_root.mkdir(parents=True)
    storyboard, manifest = _manifest(episode_root)
    store = StateStore(workspace)
    store.save_episode(EpisodeState(
        book_id="book-demo", episode_id="E001", status="script_approved", script_hash="a" * 64,
    ))
    stages = {
        name: _PreparedStage(name, storyboard, manifest)
        for name in HANDDRAWN_MEDIA_PREPARE_ORDER
    }
    service = MediaProductionService(
        store=store, stages=stages,
        images=LocalIllustrationGateway(ffmpeg_command=ffmpeg),
    )
    source = tmp_path / "generated.png"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         "color=c=#cf8b6c:s=1080x1920:d=1", "-frames:v", "1", "-y", str(source)],
        check=True,
    )

    view = service.prepare("book-demo", "E001", mode="technical_sample")
    assert view.missing_scene_ids == ("S01", "S03", "S05")
    for scene_id in view.missing_scene_ids:
        view = service.import_image("book-demo", "E001", scene_id, source)
    assert view.status == "awaiting_representative_review"
    service.approve_representatives("book-demo", "E001")
    view = service.prepare_batch("book-demo", "E001")
    assert view.missing_scene_ids == ("S02", "S04")
    for scene_id in view.missing_scene_ids:
        view = service.import_image("book-demo", "E001", scene_id, source)

    assert view.status == "illustrations_ready"
    assert view.missing_scene_ids == ()

