from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import bv.illustration.assets as assets_module
from bv.core.process import CommandResult
from bv.illustration.assets import (
    IllustrationAssetError,
    ImageFacts,
    ImportedImage,
    bind_job_references,
    import_image_master,
    prepare_image_jobs,
    record_imported_image,
)
from bv.illustration.contracts import (
    CharacterLock,
    IllustrationScene,
    IllustrationStoryboard,
    StyleDecision,
)
from bv.illustration.prompts import compile_image_prompt
from bv.illustration.style_selector import StyleRecipe, load_style_catalog


CATALOG = Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _style() -> StyleRecipe:
    return next(
        item for item in load_style_catalog(CATALOG)
        if item.style_id == "emotional-watercolor-sketch"
    )


def _decision() -> StyleDecision:
    return StyleDecision(
        library_version="1",
        selected_style="emotional-watercolor-sketch",
        candidate_styles=(
            "emotional-watercolor-sketch",
            "colored-pencil-diary",
            "warm-flat-storybook",
        ),
        selection_reasons=("匹配克制关系叙事",),
        rejected_reasons={
            "colored-pencil-diary": "情绪层次较弱",
            "warm-flat-storybook": "扁平感较强",
        },
        confidence=0.9,
        manual_override=False,
        source_script_sha256="a" * 64,
        style_fingerprint="b" * 64,
    )


def _character() -> CharacterLock:
    return CharacterLock(
        role="普通读者主人公",
        age_range="30至39岁",
        face="椭圆脸，眉眼克制",
        hair="黑色短发",
        body="中等身材",
        base_clothing="深蓝衬衫和浅灰外套",
        allowed_variations=("室内脱外套",),
        color_markers=("深蓝衬衫",),
        personal_objects=("旧笔记本",),
        forbidden_changes=("年龄变化", "发型变化"),
        source_script_sha256="a" * 64,
        style_fingerprint="b" * 64,
    )


def _storyboard(scene_count: int = 9) -> IllustrationStoryboard:
    duration_ms = scene_count * 5_000
    representative = {0, scene_count // 2, scene_count - 1}
    character = _character()
    scenes = tuple(
        IllustrationScene(
            scene_id=f"S{index + 1:02d}",
            start_ms=index * 5_000,
            end_ms=(index + 1) * 5_000,
            from_frame=index * 150,
            to_frame=(index + 1) * 150,
            narration=f"第{index + 1}段批准旁白内容",
            narration_span=(index * 10, (index + 1) * 10),
            key_line="把生活还给自己",
            visual_purpose="把书的价值放回真实生活",
            setting="通勤车厢",
            character_action="主人公停下解释，打开笔记本",
            metaphor=None if index < round(scene_count * 0.6) else "人群中的门",
            composition="主体居中偏下，安全留白",
            character_refs=("reader-01",) if index < round(scene_count * 0.6) else (),
            image_prompt="普通成年读者停下解释的生活场景",
            negative_constraints=("禁止画内文字",),
            representative_frame=index in representative,
            asset_status="planned",
        )
        for index in range(scene_count)
    )
    return IllustrationStoryboard(
        book_id="book-demo",
        episode_id="E001",
        width=1080,
        height=1920,
        fps=30,
        master_duration_ms=duration_ms,
        total_frames=round(duration_ms * 30 / 1000),
        script_sha256="a" * 64,
        audio_sha256="c" * 64,
        subtitle_sha256="d" * 64,
        style_decision_sha256=_canonical(_decision().model_dump(mode="json")),
        character_lock_sha256=_canonical(character.model_dump(mode="json")),
        scenes=scenes,
    )


def test_compiled_prompt_binds_exact_style_character_and_references() -> None:
    scene = _storyboard().scenes[0]
    first = compile_image_prompt(
        scene,
        style=_style(),
        character_lock=_character(),
    )
    referenced = compile_image_prompt(
        scene,
        style=_style(),
        character_lock=_character(),
        reference_image_sha256s=("e" * 64,),
    )

    assert "1080×1920" in first.prompt
    assert "9:16" in first.prompt
    assert "1024×1024" not in first.prompt
    assert "emotional-watercolor-sketch" in first.prompt
    assert "黑色短发" in first.prompt
    assert "禁止画内文字" in first.prompt
    assert first.style_fingerprint == _character().style_fingerprint
    assert first.prompt_sha256 != referenced.prompt_sha256


def test_representative_jobs_generate_opening_before_middle_and_end(tmp_path: Path) -> None:
    manifest = prepare_image_jobs(
        _storyboard(), _character(), _style(), _decision(), tmp_path / "episode"
    )
    representatives = [job for job in manifest.jobs if job.representative]

    assert [job.scene_id for job in representatives] == ["S01", "S05", "S09"]
    assert representatives[0].reference_scene_ids == ()
    assert representatives[1].reference_scene_ids == ("S01",)
    assert representatives[2].reference_scene_ids == ("S01",)
    assert [job.scene_id for job in manifest.jobs[:3]] == ["S01", "S05", "S09"]
    assert all(job.output_master.is_relative_to(manifest.episode_root) for job in manifest.jobs)


def test_prepare_rejects_stale_character_or_style_binding(tmp_path: Path) -> None:
    stale_character = _character().model_copy(update={"style_fingerprint": "f" * 64})
    with pytest.raises(IllustrationAssetError, match="illustration_dependency_mismatch"):
        prepare_image_jobs(
            _storyboard(), stale_character, _style(), _decision(), tmp_path / "episode"
        )


def test_import_rejects_wrong_dimensions_without_running_ffmpeg(tmp_path: Path) -> None:
    manifest = prepare_image_jobs(
        _storyboard(3), _character(), _style(), _decision(), tmp_path / "episode"
    )
    source = tmp_path / "generated.png"
    source.write_bytes(b"not decoded by injected inspector")
    called = False

    def runner(*_args: object, **_kwargs: object) -> CommandResult:
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(IllustrationAssetError, match="image_contract_invalid"):
        import_image_master(
            manifest.jobs[0],
            source,
            inspector=lambda _path: ImageFacts(width=512, height=512, format="png"),
            ffmpeg_command="ffmpeg",
            runner=runner,
        )

    assert called is False


def test_import_normalizes_and_derives_two_new_episode_local_images(tmp_path: Path) -> None:
    manifest = prepare_image_jobs(
        _storyboard(3), _character(), _style(), _decision(), tmp_path / "episode"
    )
    job = manifest.jobs[0]
    source = tmp_path / "generated.jpg"
    source.write_bytes(b"source remains unchanged")
    before = source.read_bytes()

    def runner(argv: list[str], *, timeout: float) -> CommandResult:
        output = Path(argv[-1])
        output.write_bytes(b"gray output" if "format=gray" in " ".join(argv) else b"color output")
        return CommandResult(argv=argv, returncode=0)

    imported = import_image_master(
        job,
        source,
        inspector=lambda path: ImageFacts(
            width=1080,
            height=1920,
            format="png" if Path(path).suffix.lower() == ".png" else "jpeg",
        ),
        ffmpeg_command="ffmpeg",
        runner=runner,
    )

    assert source.read_bytes() == before
    assert imported.master_path == job.output_master
    assert imported.bw_path == job.output_bw
    assert imported.master_sha256 != imported.bw_sha256
    assert job.output_master.read_bytes() == b"color output"
    assert job.output_bw.read_bytes() == b"gray output"


def test_import_refuses_existing_destination(tmp_path: Path) -> None:
    manifest = prepare_image_jobs(
        _storyboard(3), _character(), _style(), _decision(), tmp_path / "episode"
    )
    job = manifest.jobs[0]
    job.output_master.parent.mkdir(parents=True)
    job.output_master.write_bytes(b"keep")
    source = tmp_path / "generated.png"
    source.write_bytes(b"source")

    with pytest.raises(IllustrationAssetError, match="image_output_exists"):
        import_image_master(
            job,
            source,
            inspector=lambda _path: ImageFacts(width=1080, height=1920, format="png"),
            ffmpeg_command="ffmpeg",
        )

    assert job.output_master.read_bytes() == b"keep"


def test_opening_import_binds_middle_job_to_exact_identity_hash(tmp_path: Path) -> None:
    manifest = prepare_image_jobs(
        _storyboard(), _character(), _style(), _decision(), tmp_path / "episode"
    )
    opening = manifest.jobs[0]
    opening.output_master.parent.mkdir(parents=True)
    opening.output_master.write_bytes(b"approved opening color")
    opening.output_bw.write_bytes(b"approved opening gray")
    imported = ImportedImage(
        scene_id="S01",
        master_path=opening.output_master,
        master_sha256=hashlib.sha256(b"approved opening color").hexdigest(),
        bw_path=opening.output_bw,
        bw_sha256=hashlib.sha256(b"approved opening gray").hexdigest(),
        width=1080,
        height=1920,
    )
    updated = record_imported_image(manifest, imported)
    middle = next(job for job in updated.jobs if job.scene_id == "S05")

    bound = bind_job_references(middle, updated)

    assert updated.jobs[0].status == "generated"
    assert bound.reference_image_sha256s == (imported.master_sha256,)
    assert bound.prompt_sha256 != middle.prompt_sha256


def test_publish_fsync_failure_rolls_back_its_new_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = prepare_image_jobs(
        _storyboard(3), _character(), _style(), _decision(), tmp_path / "episode"
    )
    job = manifest.jobs[0]
    source = tmp_path / "generated.png"
    source.write_bytes(b"source")

    def runner(argv: list[str], *, timeout: float) -> CommandResult:
        Path(argv[-1]).write_bytes(
            b"gray output" if "format=gray" in " ".join(argv) else b"color output"
        )
        return CommandResult(argv=argv, returncode=0)

    monkeypatch.setattr(
        assets_module,
        "_fsync_file",
        lambda _path: (_ for _ in ()).throw(OSError("disk error")),
        raising=False,
    )

    with pytest.raises(IllustrationAssetError, match="image_publish_failed"):
        import_image_master(
            job,
            source,
            inspector=lambda path: ImageFacts(
                width=1080,
                height=1920,
                format="png" if Path(path).suffix.lower() == ".png" else "jpeg",
            ),
            ffmpeg_command="ffmpeg",
            runner=runner,
        )

    assert not job.output_master.exists()
    assert not job.output_bw.exists()
