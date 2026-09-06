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
    ImageJob,
    IllustrationManifest,
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


def _story_pair_storyboard() -> IllustrationStoryboard:
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
            metaphor=None,
            composition="主体居中偏下，安全留白",
            character_refs=("reader-01",),
            image_prompt="普通成年读者停下解释的生活场景",
            negative_constraints=("禁止画内文字",),
            representative_frame=index in {0, 2, 3},
            asset_status="planned",
            semantic_turn_span=(index * 10, index * 10 + 2),
            semantic_turn_ms=index * 5_000 + 1_000,
            semantic_turn_frame=index * 150 + 30,
            continuation_action="主人公合上笔记本，抬头看向车窗",
            continuation_prompt="同一车厢同一机位，主人公合上笔记本并抬头",
            continuity_constraints=(
                "same character",
                "same clothing",
                "same setting",
                "same camera direction",
            ),
        )
        for index in range(4)
    )
    return IllustrationStoryboard(
        book_id="book-demo",
        episode_id="E001",
        width=1080,
        height=1920,
        fps=30,
        master_duration_ms=20_000,
        total_frames=600,
        script_sha256="a" * 64,
        audio_sha256="c" * 64,
        subtitle_sha256="d" * 64,
        style_decision_sha256=_canonical(_decision().model_dump(mode="json")),
        character_lock_sha256=_canonical(character.model_dump(mode="json")),
        sequence_mode="color-story-pair",
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


def test_story_pair_jobs_are_ordered_and_block_continuations_until_anchors_exist(
    tmp_path: Path,
) -> None:
    manifest = prepare_image_jobs(
        _story_pair_storyboard(),
        _character(),
        _style(),
        _decision(),
        tmp_path / "episode",
    )

    assert [(job.scene_id, job.phase) for job in manifest.jobs] == [
        ("S01", "anchor"),
        ("S01", "continuation"),
        ("S03", "anchor"),
        ("S03", "continuation"),
        ("S04", "anchor"),
        ("S04", "continuation"),
        ("S02", "anchor"),
        ("S02", "continuation"),
    ]
    anchor, continuation = manifest.jobs[:2]
    assert anchor.asset_id == "S01-A"
    assert continuation.asset_id == "S01-B"
    assert anchor.output_master.name == "S01_anchor.png"
    assert continuation.output_master.name == "S01_continuation.png"
    assert anchor.output_bw is None
    assert continuation.output_bw is None
    with pytest.raises(IllustrationAssetError, match="anchor_image_not_ready"):
        bind_job_references(continuation, manifest)


def test_story_pair_continuation_import_binds_the_current_anchor_hash(
    tmp_path: Path,
) -> None:
    manifest = prepare_image_jobs(
        _story_pair_storyboard(),
        _character(),
        _style(),
        _decision(),
        tmp_path / "episode",
    )
    source = tmp_path / "generated.png"
    source.write_bytes(b"source")
    commands: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: object) -> CommandResult:
        commands.append(argv)
        Path(argv[-1]).write_bytes(b"color output")
        return CommandResult(argv=argv, returncode=0)

    inspector = lambda _path: ImageFacts(width=1080, height=1920, format="png")
    anchor_import = import_image_master(
        manifest.jobs[0], source, manifest=manifest, inspector=inspector,
        ffmpeg_command="ffmpeg", runner=runner,
    )
    manifest = record_imported_image(manifest, anchor_import)
    continuation = manifest.jobs[1]

    assert continuation.anchor_sha256 == anchor_import.master_sha256
    assert continuation.reference_image_sha256s == (anchor_import.master_sha256,)

    continuation_import = import_image_master(
        continuation,
        source,
        manifest=manifest,
        inspector=inspector,
        ffmpeg_command="ffmpeg",
        runner=runner,
    )
    updated = record_imported_image(manifest, continuation_import)

    assert continuation.anchor_sha256 == anchor_import.master_sha256
    assert continuation.reference_image_sha256s == (anchor_import.master_sha256,)
    assert continuation_import.bw_path is None
    assert updated.jobs[0].master_sha256 == continuation_import.master_sha256
    assert len(commands) == 2


def test_story_pair_continuation_rejects_missing_or_stale_persisted_anchor_binding(
    tmp_path: Path,
) -> None:
    manifest = prepare_image_jobs(
        _story_pair_storyboard(), _character(), _style(), _decision(), tmp_path / "episode"
    )
    source = tmp_path / "generated.png"
    source.write_bytes(b"source")
    inspector = lambda _path: ImageFacts(width=1080, height=1920, format="png")

    with pytest.raises(IllustrationAssetError, match="continuation_anchor_binding_invalid"):
        import_image_master(
            manifest.jobs[1], source, manifest=manifest, inspector=inspector,
            ffmpeg_command="ffmpeg",
        )

    anchor = manifest.jobs[0]
    anchor.output_master.parent.mkdir(parents=True)
    anchor.output_master.write_bytes(b"anchor")
    imported_anchor = ImportedImage(
        scene_id=anchor.scene_id,
        asset_id=anchor.asset_id,
        master_path=anchor.output_master,
        master_sha256=hashlib.sha256(anchor.output_master.read_bytes()).hexdigest(),
        width=1080,
        height=1920,
    )
    manifest = record_imported_image(manifest, imported_anchor)
    stale = manifest.jobs[1].model_copy(update={"anchor_sha256": "0" * 64})
    stale_manifest = manifest.model_copy(update={"jobs": (manifest.jobs[0], stale, *manifest.jobs[2:])})

    with pytest.raises(IllustrationAssetError, match="continuation_anchor_binding_invalid"):
        import_image_master(
            stale, source, manifest=stale_manifest, inspector=inspector,
            ffmpeg_command="ffmpeg",
        )
    stale.output_master.write_bytes(b"forged continuation")
    with pytest.raises(IllustrationAssetError, match="continuation_anchor_binding_invalid"):
        record_imported_image(
            stale_manifest,
            ImportedImage(
                scene_id=stale.scene_id,
                asset_id=stale.asset_id,
                master_path=stale.output_master,
                master_sha256=hashlib.sha256(stale.output_master.read_bytes()).hexdigest(),
                width=1080,
                height=1920,
            ),
        )


def test_pair_anchor_preflight_rejects_out_of_order_import_without_publishing(
    tmp_path: Path,
) -> None:
    manifest = prepare_image_jobs(
        _story_pair_storyboard(), _character(), _style(), _decision(), tmp_path / "episode"
    )
    source = tmp_path / "generated.png"
    source.write_bytes(b"source")
    commands: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: object) -> CommandResult:
        commands.append(argv)
        Path(argv[-1]).write_bytes(b"color output")
        return CommandResult(argv=argv, returncode=0)

    inspector = lambda _path: ImageFacts(width=1080, height=1920, format="png")
    out_of_order = next(job for job in manifest.jobs if job.asset_id == "S03-A")
    with pytest.raises(IllustrationAssetError, match="anchor_record_preflight_invalid"):
        import_image_master(
            out_of_order, source, manifest=manifest, inspector=inspector,
            ffmpeg_command="ffmpeg", runner=runner,
        )
    assert not out_of_order.output_master.exists()
    assert commands == []

    first = manifest.jobs[0]
    imported = import_image_master(
        first, source, manifest=manifest, inspector=inspector,
        ffmpeg_command="ffmpeg", runner=runner,
    )
    manifest = record_imported_image(manifest, imported)
    retried = next(job for job in manifest.jobs if job.asset_id == "S03-A")
    imported_retry = import_image_master(
        retried, source, manifest=manifest, inspector=inspector,
        ffmpeg_command="ffmpeg", runner=runner,
    )

    assert imported_retry.master_path == retried.output_master
    assert retried.output_master.is_file()


@pytest.mark.parametrize(
    "reference_scene_ids",
    [(), ("S02", "S01"), ("S01", "S01")],
)
def test_pair_anchor_preflight_rejects_invalid_continuation_self_reference_before_publish(
    tmp_path: Path,
    reference_scene_ids: tuple[str, ...],
) -> None:
    manifest = prepare_image_jobs(
        _story_pair_storyboard(), _character(), _style(), _decision(), tmp_path / "episode"
    )
    anchor, continuation = manifest.jobs[:2]
    damaged = manifest.model_copy(
        update={
            "jobs": (
                anchor,
                continuation.model_copy(update={"reference_scene_ids": reference_scene_ids}),
                *manifest.jobs[2:],
            )
        }
    )
    source = tmp_path / "generated.png"
    source.write_bytes(b"source")
    commands: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: object) -> CommandResult:
        commands.append(argv)
        Path(argv[-1]).write_bytes(b"color output")
        return CommandResult(argv=argv, returncode=0)

    inspector = lambda _path: ImageFacts(width=1080, height=1920, format="png")
    with pytest.raises(IllustrationAssetError, match="anchor_record_preflight_invalid"):
        import_image_master(
            anchor, source, manifest=damaged, inspector=inspector,
            ffmpeg_command="ffmpeg", runner=runner,
        )
    assert not anchor.output_master.exists()
    assert commands == []

    imported = import_image_master(
        anchor, source, manifest=manifest, inspector=inspector,
        ffmpeg_command="ffmpeg", runner=runner,
    )
    assert imported.master_path.is_file()


def test_legacy_image_job_and_manifest_json_shape_and_hash_are_stable(tmp_path: Path) -> None:
    manifest = prepare_image_jobs(
        _storyboard(3), _character(), _style(), _decision(), tmp_path / "episode"
    )
    job = manifest.jobs[0]

    expected_job = {
        "episode_root": str(job.episode_root), "scene_id": job.scene_id,
        "representative": job.representative, "prompt": job.prompt,
        "prompt_sha256": job.prompt_sha256, "style_fingerprint": job.style_fingerprint,
        "character_lock_sha256": job.character_lock_sha256,
        "reference_scene_ids": [], "reference_image_sha256s": [],
        "output_master": str(job.output_master), "output_bw": str(job.output_bw),
        "master_sha256": None, "bw_sha256": None, "attempts": 0, "status": "planned",
    }

    assert job.model_dump(mode="json") == expected_job
    assert json.loads(job.model_dump_json()) == expected_job
    manifest_json = manifest.model_dump(mode="json")
    assert all(
        "phase" not in item and "asset_id" not in item and "anchor_sha256" not in item
        for item in manifest_json["jobs"]
    )
    assert json.loads(manifest.model_dump_json()) == manifest_json
    assert _canonical(manifest_json) == _canonical(json.loads(manifest.model_dump_json()))


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
