from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pymupdf

from bv.asr.volcengine import AsrResult, VolcCredentials, WordTiming
from bv.config import IndexTTS2Config
from bv.content.scripts import SemanticLock
from bv.core.hashing import sha256_file
from bv.illustration.assets import IllustrationManifest
from bv.illustration.contracts import IllustrationStoryboard, SafeArea
from bv.illustration.prompts import canonical_model_sha256
from bv.production.profile import (
    DeliveryProfile,
    DurationProfile,
    ProductionProfile,
    TtsProfile,
    VisualProfile,
    production_profile_sha256,
)
from bv.state.models import BookState, EpisodeState
from bv.state.store import StateStore
from bv.subtitles.generate import SubtitleCue, SubtitleManifest
from bv.video.cover import CoverManifest
from bv.video.cover_brief import CoverBriefBuilder, CoverDerivedFields
from bv.video.probe import probe_media
from bv.video.render import CoverOverlay, HanddrawnRenderInputs, render_handdrawn_final
from bv.voice.indextts2 import SynthesizedVoice
from bv.voice.processing import NarrationDuration, VoiceManifest
from bv.workflow.media_stages import (
    AsrStage,
    FinalQcStage,
    FinalRenderStage,
    IllustrationPlanningStage,
    PrepareIllustrationsStage,
    StyleSelectionStage,
    SubtitleStage,
    TtsStage,
)
from bv.workflow.media_runtime import (
    LocalCoverGateway,
    LocalDeliveryGateway,
    LocalIllustrationGateway,
    LocalSocialCoverGateway,
    MediaWorkflowError,
    MediaProductionService,
)
from bv.workflow.media_stages import VisualRenderStage
from bv.workflow.runtime import RuntimeAuthorization
from bv.workflow.stages import HANDDRAWN_MEDIA_PREPARE_ORDER, StageContext
from bv.delivery.exporter import ApprovalRecord, DeliveryExporter
from tests.integration.test_handdrawn_media_workflow import _PreparedStage, _manifest


def _run(argv: list[str]) -> None:
    result = subprocess.run(
        argv, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    assert result.returncode == 0, result.stderr[-1200:]


_PRECOMMIT_LEGACY_STORYBOARD_JSON = '{"book_id":"book-demo","episode_id":"E001","width":1080,"height":1920,"fps":30,"master_duration_ms":10000,"total_frames":300,"script_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","audio_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","subtitle_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","style_decision_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","character_lock_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","scenes":[{"scene_id":"S01","start_ms":0,"end_ms":5000,"from_frame":0,"to_frame":150,"narration":"这本书让人重新看见自己的生活。","narration_span":[0,16],"key_line":"把人生还给自己","visual_purpose":"建立真实生活困境","setting":"清晨的通勤地铁","character_action":"主人公放下反复查看的手机","metaphor":null,"composition":"人物居中，右侧交互区留空","character_refs":["reader-01"],"image_prompt":"同一位普通读者，手绘插画，无画内文字","negative_constraints":["no_text","no_logo","no_watermark"],"representative_frame":true,"asset_status":"planned"},{"scene_id":"S02","start_ms":5000,"end_ms":10000,"from_frame":150,"to_frame":300,"narration":"这本书让人重新看见自己的生活。","narration_span":[0,16],"key_line":"把人生还给自己","visual_purpose":"建立真实生活困境","setting":"清晨的通勤地铁","character_action":"主人公放下反复查看的手机","metaphor":null,"composition":"人物居中，右侧交互区留空","character_refs":["reader-01"],"image_prompt":"同一位普通读者，手绘插画，无画内文字","negative_constraints":["no_text","no_logo","no_watermark"],"representative_frame":true,"asset_status":"planned"}]}'
_PRECOMMIT_LEGACY_STORYBOARD_SHA256 = "aea16439571152ca9a24073accb93d4163affdeeafc79c96ebbddd3ac0fd0885"
_PRECOMMIT_APPROVAL_RECORD_JSON = '{"delivery_manifest_sha256":"1111111111111111111111111111111111111111111111111111111111111111","video_sha256":"2222222222222222222222222222222222222222222222222222222222222222","cover_sha256":"3333333333333333333333333333333333333333333333333333333333333333","approved_at":"2026-08-23T10:00:00+08:00","status":"final_approved","path":"C:/frozen/2026-08-23-legacy-最终批准.json"}'
_PRECOMMIT_APPROVAL_RECORD_SHA256 = "c16e477c91c00114115ec57845328a69b605b8b66ec47c4d65269bb8c085f78f"


def _write_wav(path: Path, *, duration_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x00\x10" * (48 * duration_ms))


class _DeterministicVisualModel:
    def complete(self, prompt: str, schema_type: type, request_dir: Path):
        assert request_dir.is_dir()
        source = prompt.split("BEGIN_SOURCE_DATA\n", 1)[1].split("\nEND_SOURCE_DATA", 1)[0]
        payload = json.loads(json.loads(source))
        if schema_type.__name__ == "_StyleSelectionDraft":
            candidates = [item["style_id"] for item in payload["candidates"]]
            return schema_type.model_validate({
                "selected_style": candidates[0],
                "selection_reasons": ["deterministic local test"],
                "rejected_reasons": {item: "test alternative" for item in candidates[1:]},
                "confidence": 1.0,
            })
        windows = payload["fixed_windows"]
        scenes = []
        for window in windows:
            narration = window["narration"]
            offset = max(1, min(len(narration) - 1, round(len(narration) * 3 / 8)))
            scenes.append({
                "scene_id": window["scene_id"],
                "key_line": "把生活还给自己",
                "visual_purpose": "把书的价值落到普通人的生活选择",
                "setting": "家中餐桌",
                "character_action": "读者翻开笔记本",
                "metaphor": None,
                "composition": "主体居中偏下，顶部和底部留白",
                "character_refs": ["reader-01"],
                "image_prompt": "普通成年人在餐桌前停下解释动作",
                "negative_constraints": ["禁止画内文字", "禁止Logo和水印"],
                "semantic_turn_offset": offset,
                "continuation_action": "读者收起手机后抬头看向窗外",
                "continuation_prompt": "同一餐桌同一机位，读者收起手机后抬头看向窗外",
                "continuity_constraints": ["same character", "same clothing", "same setting", "same camera direction", "same visual style"],
            })
        return schema_type.model_validate({
            "characters": [{
                "character_id": "reader-01", "role": "普通读者", "age_range": "30至39岁",
                "face": "椭圆脸", "hair": "黑色短发", "body": "中等身材",
                "base_clothing": "深蓝衬衫", "allowed_variations": [],
                "color_markers": ["深蓝"], "personal_objects": ["笔记本"],
                "forbidden_changes": ["发型"],
            }],
            "scenes": scenes,
            "narrative_deviation_reason": "deterministic local fixture uses one reader identity",
            "legacy_single_character": False,
        })


def _write_semantic_sources(episode_root: Path, script_sha256: str) -> None:
    fields = {
        "episode_id": "E001", "topic_identity_sha256": "1" * 64,
        "value_thesis": "继续把日子过下去", "target_reader": "承受现实压力的成年人",
        "reader_before": "把别人的评价当成选择正确的证明", "reader_after": "保留自己的判断",
        "central_life_tension": "想忠于自己又害怕不被理解", "life_connection": "工作家庭和饭桌",
        "required_cluster_ids": ["cluster-a"], "allowed_claim_ids_by_cluster": {"cluster-a": ["claim-a"]},
        "chapter_regions_by_cluster": {"cluster-a": ["whole-book"]}, "cluster_coverage_terms": {"cluster-a": "继续生活"},
        "practical_boundary": "不承诺解决困境", "reading_reason": "看见具体生活",
        "allowed_numbers": [], "allowed_negations": ["不"],
    }
    digest = hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    lock = SemanticLock.model_validate({**fields, "sha256": digest})
    root = episode_root / "script"
    (root / "semantic-lock.json").write_text(lock.model_dump_json(), encoding="utf-8")
    (root / "script_manifest.json").write_text(json.dumps({"script_sha256": script_sha256, "semantic_lock_sha256": digest}), encoding="utf-8")


def _profile_for_pair() -> ProductionProfile:
    return ProductionProfile(
        schema_version=1,
        duration=DurationProfile(hard_min_seconds=32.0, ideal_min_seconds=32.0, ideal_max_seconds=32.0, hard_max_seconds=32.0),
        tts=TtsProfile(provider="indextts2", resource_id="local.indextts2", voice_type="黑金3", speed=1.0),
        visual=VisualProfile(seconds_per_scene_min=8.0, seconds_per_scene_max=8.0, representative_count=3, transition="cross-dissolve", sequence_mode="color-story-pair", scene_count=4, style_id="emotional-watercolor-sketch"),
        delivery=DeliveryProfile(timezone="Asia/Shanghai", include=("video", "cover", "script", "voice", "subtitles", "manifest")),
    )


def _make_real_service(tmp_path: Path, *, sequence_mode: str) -> tuple[MediaProductionService, StateStore, Path, int]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    npm = shutil.which("npm")
    assert ffmpeg and ffprobe and npm
    duration_ms = 32_000 if sequence_mode == "color-story-pair" else 48_000
    chunks = tuple("生活很难时先把眼前饭吃完" for _ in range(8 if sequence_mode == "color-story-pair" else 16))
    text = "".join(chunks)
    workspace = tmp_path / "workspace"
    root = workspace / "books" / "book-demo" / "episodes" / "E001"
    approved = root / "script" / "approved.txt"
    approved.parent.mkdir(parents=True)
    approved.write_text(text, encoding="utf-8")
    (approved.parent / "subtitle_breaks.txt").write_text("\n".join(chunks) + "\n", encoding="utf-8")
    if sequence_mode == "color-story-pair":
        (root / "production_profile.json").write_text(_profile_for_pair().model_dump_json(), encoding="utf-8")
    script_sha256 = sha256_file(approved)
    _write_semantic_sources(root, script_sha256)
    store = StateStore(workspace)
    store.save_book(BookState(book_id="book-demo", title="测试图书", authors=["测试作者"], source_mode="title_author"))
    store.save_episode(EpisodeState(book_id="book-demo", episode_id="E001", status="script_approved", script_hash=script_sha256))
    reference = tmp_path / "reference.wav"
    _write_wav(reference, duration_ms=1_000)

    def synthesize(**kwargs: object) -> SynthesizedVoice:
        raw = Path(kwargs["raw_output_path"])
        _write_wav(raw, duration_ms=duration_ms)
        return SynthesizedVoice(raw_path=raw, raw_sha256=sha256_file(raw))

    def recognize(_credentials: VolcCredentials, request) -> AsrResult:
        return AsrResult(
            text=text,
            words=tuple(WordTiming(text=character, start_time=duration_ms * index // len(text), end_time=max(duration_ms * (index + 1) // len(text), duration_ms * index // len(text) + 1), confidence=1.0) for index, character in enumerate(text)),
            duration_ms=duration_ms, audio_sha256=request.audio_sha256, request_id=request.request_id, response_sha256="a" * 64,
        )

    authorization = RuntimeAuthorization(allow_external=True)
    model = _DeterministicVisualModel()
    catalog = Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json").resolve()
    stages = {
        "tts": TtsStage(config=IndexTTS2Config(install_dir=tmp_path, model_dir=tmp_path, voice_id="黑金3"), reference_voice_path=reference, ffmpeg_command=ffmpeg, mode="full", synthesizer=synthesize),
        "asr": AsrStage(credentials=VolcCredentials(api_key="placeholder"), endpoint="https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash", authorization=authorization, recognizer=recognize),
        "subtitles": SubtitleStage(font_path=Path(r"C:\Windows\Fonts\msyh.ttc"), font_family="Microsoft YaHei"),
        "select_style": StyleSelectionStage(model=model, authorization=authorization, catalog_path=catalog, prompt_path=Path("prompts/illustration/style_select.md").resolve()),
        "plan_illustrations": IllustrationPlanningStage(model=model, authorization=authorization, prompt_path=Path("prompts/illustration/storyboard.md").resolve()),
        "prepare_representatives": PrepareIllustrationsStage(catalog_path=catalog),
        "visual_render": VisualRenderStage(vendor_dir=Path("vendor/story_to_handdrawn_video").resolve(), npm_command=npm, ffprobe_command=ffprobe, safe_area=SafeArea(), transition="cross-dissolve"),
        "render": FinalRenderStage(ffmpeg_command=ffmpeg, ffprobe_command=ffprobe),
        "qc": FinalQcStage(ffprobe_command=ffprobe),
    }
    return MediaProductionService(store=store, stages=stages, images=LocalIllustrationGateway(ffmpeg_command=ffmpeg), social_cover=LocalSocialCoverGateway(ffmpeg_command=ffmpeg), delivery=LocalDeliveryGateway(store=store, exporter=DeliveryExporter(clock=lambda: datetime(2026, 9, 1, tzinfo=UTC)))), store, root, duration_ms


def _write_cover_brief(episode_root: Path) -> tuple[Path, Path]:
    style_root = episode_root / ".private" / "cover-style"
    style_root.mkdir(parents=True)
    meta, atom, blueprint = (style_root / "META.md", style_root / "STYLE.md", style_root / "cover-prompt-blueprint.md")
    meta.write_text("```yaml\nid: emotional-watercolor-sketch\noutputs: [cover]\n```", encoding="utf-8")
    atom.write_text("restrained watercolor", encoding="utf-8")
    blueprint.write_text("one vertical cover", encoding="utf-8")
    script = episode_root / "script" / "approved.txt"
    lock = episode_root / "script" / "semantic-lock.json"
    brief = CoverBriefBuilder(blueprint_path=blueprint).build(
        episode_root=episode_root, slug="task5-cover",
        fields=CoverDerivedFields(title="测试图书", summary="继续把日子过下去", visual_subject="餐桌前的读者", audience="成年人", mood="克制", visual_metaphor="一盏灯"),
        compiled_prompt="Create one 3:4 测试图书 cover in emotional-watercolor-sketch style", approved_script_path=script, approved_script_sha256=sha256_file(script),
        semantic_lock_path=lock, semantic_lock_sha256=sha256_file(lock), production_profile_sha256=production_profile_sha256(episode_root),
        style_id="emotional-watercolor-sketch", style_meta_path=meta, style_atom_path=atom,
    )
    source = episode_root / ".private" / "cover-source.png"
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg
    _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=#b6413c:s=540x720:d=1", "-frames:v", "1", "-y", str(source)])
    return brief.manifest_path, source


def _write_color_sources(root: Path, count: int) -> tuple[Path, ...]:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg
    colors = ("#b6413c", "#d47a3f", "#5b8f56", "#3e77a8", "#8c5ba5", "#c1547c", "#4d9a95", "#b58f3e", "#738a3d", "#835a4b")
    sources = []
    for index, color in enumerate(colors[:count]):
        source = root / ".private" / "imports" / f"asset-{index}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={color}:s=1080x1920:d=1", "-frames:v", "1", "-y", str(source)])
        sources.append(source)
    return tuple(sources)


def _frame(video: Path, frame_number: int, output: Path) -> pymupdf.Pixmap:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg
    _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", f"select=eq(n\\,{frame_number})", "-frames:v", "1", "-y", str(output)])
    return pymupdf.Pixmap(str(output))


def _frame_count(video: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    assert ffprobe
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _delivery_manifest_path(delivery_root: Path) -> Path:
    return next(path for path in delivery_root.iterdir() if path.name.endswith("交付清单.json"))


def _chroma(image: pymupdf.Pixmap, x: int, y: int) -> int:
    red, green, blue = image.pixel(x, y)[:3]
    return max(red, green, blue) - min(red, green, blue)


def test_precommit_legacy_json_and_approval_record_remain_loadable() -> None:
    storyboard = IllustrationStoryboard.model_validate_json(_PRECOMMIT_LEGACY_STORYBOARD_JSON)
    approval = ApprovalRecord.model_validate_json(_PRECOMMIT_APPROVAL_RECORD_JSON)

    assert storyboard.sequence_mode == "legacy-monochrome-reveal"
    assert canonical_model_sha256(storyboard) == _PRECOMMIT_LEGACY_STORYBOARD_SHA256
    assert hashlib.sha256(_PRECOMMIT_APPROVAL_RECORD_JSON.encode("utf-8")).hexdigest() == _PRECOMMIT_APPROVAL_RECORD_SHA256
    assert approval.delivery_manifest_sha256 == "1" * 64
    assert approval.video_sha256 == "2" * 64
    assert approval.cover_sha256 == "3" * 64


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None or shutil.which("npm") is None,
    reason="Task 5 real local chain requires FFmpeg/FFprobe/npm",
)
def test_story_pair_real_pipeline_reaches_immutable_candidate(tmp_path: Path) -> None:
    service, store, root, duration_ms = _make_real_service(tmp_path, sequence_mode="color-story-pair")

    view = service.prepare("book-demo", "E001", mode="full")
    assert view.status == "representative_generation_running"
    assert view.missing_scene_ids == ("S01-A", "S01-B", "S03-A", "S03-B", "S04-A", "S04-B")
    assert service.status("book-demo", "E001").missing_scene_ids == view.missing_scene_ids
    with pytest.raises(MediaWorkflowError, match="representatives_not_approved"):
        service.prepare_batch("book-demo", "E001")
    with pytest.raises(MediaWorkflowError, match="scene_not_in_representatives"):
        service.import_image("book-demo", "E001", "S02-A", _write_color_sources(root, 1)[0])

    sources = _write_color_sources(root, 8)
    for asset_id, source in zip(view.missing_scene_ids, sources[:6], strict=True):
        view = service.import_image("book-demo", "E001", asset_id, source)
    assert view.status == "awaiting_representative_review"
    approved = service.approve_representatives("book-demo", "E001")
    review = json.loads((root / "media" / "illustration" / "representative_review.json").read_text(encoding="utf-8"))
    approval = json.loads((root / "media" / "illustration" / "representative_approval.json").read_text(encoding="utf-8"))
    expected_assets = ["S01-A", "S01-B", "S03-A", "S03-B", "S04-A", "S04-B"]
    assert review["asset_ids"] == expected_assets and approval["asset_ids"] == expected_assets
    assert len(set(review["image_sha256s"])) == 6 and approval["image_sha256s"] == review["image_sha256s"]
    assert approved.status == "representatives_approved"
    view = service.prepare_batch("book-demo", "E001")
    assert view.missing_scene_ids == ("S02-A", "S02-B")
    for asset_id, source in zip(view.missing_scene_ids, sources[6:], strict=True):
        view = service.import_image("book-demo", "E001", asset_id, source)
    assert view.status == "illustrations_ready"
    manifest = json.loads((root / "media" / "illustration" / "illustration_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["jobs"]) == 8
    assert len({job["master_sha256"] for job in manifest["jobs"]}) == 8
    assert all("output_bw" not in job or job["output_bw"] is None for job in manifest["jobs"])
    assert not tuple((root / "media" / "illustration" / "images").glob("*_bw.png"))

    brief, cover = _write_cover_brief(root)
    service.import_social_cover("book-demo", "E001", cover_brief_path=brief, source=cover, source_sha256=sha256_file(cover))
    candidate = service.render("book-demo", "E001")
    assert candidate.status == "awaiting_final_review"
    payload = json.loads((root / "media" / "illustration" / "render_storyboard.json").read_text(encoding="utf-8"))
    assert payload["project"]["ink_reveal_frames"] == 45
    assert payload["project"]["transition_frames"] == 15
    final = root / "media" / "final" / "final.mp4"
    facts = probe_media(final, ffprobe_command=shutil.which("ffprobe") or "ffprobe")
    assert facts.duration_ms == duration_ms
    assert _frame_count(final) == duration_ms * 30 // 1000
    delivery = json.loads((candidate.delivery_root / next(path.name for path in candidate.delivery_root.iterdir() if path.name.endswith("交付清单.json"))).read_text(encoding="utf-8"))
    assert delivery["state"] == "awaiting_final_review"
    state = store.load_episode("book-demo", "E001")
    assert "final_approval" not in state.stage_manifests
    assert not tuple(candidate.delivery_root.glob("*最终批准*.json"))


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None or shutil.which("npm") is None,
    reason="Task 5 real local chain requires FFmpeg/FFprobe/npm",
)
def test_legacy_real_pipeline_and_frozen_artifacts_remain_compatible(tmp_path: Path) -> None:
    service, store, root, duration_ms = _make_real_service(tmp_path, sequence_mode="legacy-monochrome-reveal")

    view = service.prepare("book-demo", "E001", mode="full")
    storyboard_path = root / "media" / "illustration" / "illustration_storyboard.json"
    manifest_path = root / "media" / "illustration" / "illustration_manifest.json"
    storyboard = IllustrationStoryboard.model_validate_json(storyboard_path.read_text(encoding="utf-8"))
    manifest = IllustrationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert storyboard.sequence_mode == "legacy-monochrome-reveal"
    assert all(job.phase == "legacy" and job.output_bw is not None for job in manifest.jobs)
    sources = _write_color_sources(root, len(manifest.jobs))
    for scene_id, source in zip(view.missing_scene_ids, sources, strict=False):
        view = service.import_image("book-demo", "E001", scene_id, source)
    service.approve_representatives("book-demo", "E001")
    view = service.prepare_batch("book-demo", "E001")
    remaining = tuple(view.missing_scene_ids)
    for scene_id, source in zip(remaining, sources[len(view.present_scene_ids):], strict=True):
        view = service.import_image("book-demo", "E001", scene_id, source)
    assert view.status == "illustrations_ready"

    brief, cover = _write_cover_brief(root)
    service.import_social_cover("book-demo", "E001", cover_brief_path=brief, source=cover, source_sha256=sha256_file(cover))
    candidate = service.render("book-demo", "E001")
    assert candidate.status == "awaiting_final_review"
    candidate_manifest_path = _delivery_manifest_path(candidate.delivery_root)
    candidate_manifest_bytes = candidate_manifest_path.read_bytes()
    candidate_manifest_sha256 = sha256_file(candidate_manifest_path)
    candidate_delivery = json.loads(candidate_manifest_bytes)
    pre_approval_state = store.load_episode("book-demo", "E001")
    delivery_stage = pre_approval_state.stage_manifests["delivery"]
    delivery_stage_before = delivery_stage.model_dump(mode="json")
    assert delivery_stage.outputs["delivery_manifest"].sha256 == candidate_manifest_sha256
    assert delivery_stage.inputs["delivery_manifest_sha256"] == candidate_manifest_sha256
    silent = root / "media" / "render" / "picture_silent.mp4"
    before = _frame(silent, 59, root / ".private" / "frame-59.png")
    after = _frame(silent, 61, root / ".private" / "frame-61.png")
    sample_points = tuple((x, y) for x in range(40, 1040, 80) for y in range(380, 1420, 80))
    assert all(_chroma(before, x, y) < 8 for x, y in sample_points)
    visibly_changed_pixels = sum(
        1
        for y in range(340, 1460, 4)
        for x in range(0, 1080, 4)
        if max(abs(after.pixel(x, y)[channel] - before.pixel(x, y)[channel]) for channel in range(3)) >= 3
    )
    assert visibly_changed_pixels > 20
    reveal_points = ((488, 1268), (496, 1268), (488, 1276), (496, 1276), (504, 1276))
    assert all(_chroma(before, x, y) <= 4 for x, y in reveal_points)
    assert all(_chroma(after, x, y) > 5 for x, y in reveal_points)
    exterior_chroma = tuple(
        _chroma(after, x, y)
        for y in range(340, 1460, 8)
        for x in range(0, 1080, 8)
        if not (464 <= x <= 528 and 1248 <= y <= 1336)
    )
    assert sum(value <= 4 for value in exterior_chroma) * 100 >= len(exterior_chroma) * 99
    final = root / "media" / "final" / "final.mp4"
    assert probe_media(final, ffprobe_command=shutil.which("ffprobe") or "ffprobe").duration_ms == duration_ms
    assert _frame_count(final) == duration_ms * 30 // 1000

    storyboard_before = canonical_model_sha256(
        IllustrationStoryboard.model_validate_json(storyboard_path.read_text(encoding="utf-8"))
    )
    manifest_before = canonical_model_sha256(
        IllustrationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    )
    approved = service.approve_final("book-demo", "E001")
    assert approved.status == "final_approved" and approved.approval_record_path is not None
    post_approval_state = store.load_episode("book-demo", "E001")
    approval_stage = post_approval_state.stage_manifests["final_approval"]
    approval_record = ApprovalRecord.model_validate_json(approved.approval_record_path.read_text(encoding="utf-8"))
    assert candidate_manifest_path.read_bytes() == candidate_manifest_bytes
    assert sha256_file(candidate_manifest_path) == candidate_manifest_sha256
    assert approval_record.delivery_manifest_sha256 == candidate_manifest_sha256
    assert approval_record.video_sha256 == sha256_file(candidate.delivery_root / candidate_delivery["artifacts"]["video"]["path"])
    assert approval_record.cover_sha256 == sha256_file(candidate.delivery_root / candidate_delivery["artifacts"]["cover"]["path"])
    assert approval_stage.inputs == {
        "delivery_manifest_sha256": approval_record.delivery_manifest_sha256,
        "video_sha256": approval_record.video_sha256,
        "cover_sha256": approval_record.cover_sha256,
    }
    assert approval_stage.outputs["approval_record"].sha256 == sha256_file(approved.approval_record_path)
    frozen_paths = (storyboard_path, manifest_path, *(path for job in IllustrationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8")).jobs for path in (job.output_master, job.output_bw) if path is not None), candidate_manifest_path, approved.approval_record_path)
    frozen = {path: (path.read_bytes(), sha256_file(path)) for path in frozen_paths}
    assert service.status("book-demo", "E001").status == "final_approved"
    load_only_state = store.load_episode("book-demo", "E001")
    reloaded_storyboard = IllustrationStoryboard.model_validate_json(storyboard_path.read_text(encoding="utf-8"))
    reloaded_manifest = IllustrationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    reloaded_approval = ApprovalRecord.model_validate_json(approved.approval_record_path.read_text(encoding="utf-8"))
    assert canonical_model_sha256(reloaded_storyboard) == storyboard_before
    assert canonical_model_sha256(reloaded_manifest) == manifest_before
    assert reloaded_approval.model_dump(mode="json") == approval_record.model_dump(mode="json")
    assert load_only_state.stage_manifests["delivery"].model_dump(mode="json") == delivery_stage_before
    assert load_only_state.stage_manifests["final_approval"].model_dump(mode="json") == approval_stage.model_dump(mode="json")
    assert all(path.read_bytes() == contents and sha256_file(path) == digest for path, (contents, digest) in frozen.items())


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None
    or shutil.which("ffprobe") is None
    or shutil.which("npm") is None,
    reason="local FFmpeg/FFprobe/npm unavailable",
)
def test_fake_providers_reach_real_remotion_and_ffmpeg_delivery(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    npm = shutil.which("npm")
    assert ffmpeg and ffprobe and npm
    workspace = tmp_path / "workspace"
    episode_root = workspace / "books" / "book-demo" / "episodes" / "E001"
    episode_root.mkdir(parents=True)
    storyboard, illustration_manifest = _manifest(episode_root)
    store = StateStore(workspace)
    store.save_episode(EpisodeState(
        book_id="book-demo", episode_id="E001", status="script_approved",
        script_hash="a" * 64,
    ))
    stages = {
        name: _PreparedStage(name, storyboard, illustration_manifest)
        for name in HANDDRAWN_MEDIA_PREPARE_ORDER
    }
    service = MediaProductionService(
        store=store,
        stages=stages,
        images=LocalIllustrationGateway(ffmpeg_command=ffmpeg),
        cover=LocalCoverGateway(),
    )
    generated = tmp_path / "codex-fixture.png"
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=#cf8b6c:s=1080x1920:d=1", "-frames:v", "1", "-y", str(generated),
    ])

    view = service.prepare("book-demo", "E001", mode="technical_sample")
    assert view.missing_scene_ids == ("S01", "S03", "S05")
    for scene_id in view.missing_scene_ids:
        service.import_image("book-demo", "E001", scene_id, generated)
    service.approve_representatives("book-demo", "E001")
    view = service.prepare_batch("book-demo", "E001")
    assert view.missing_scene_ids == ("S02", "S04")
    for scene_id in view.missing_scene_ids:
        view = service.import_image("book-demo", "E001", scene_id, generated)
    assert view.status == "illustrations_ready"
    assert service.import_cover("book-demo", "E001", generated).cover_ready is True

    context = StageContext(
        book_id="book-demo", episode_id="E001", episode_root=episode_root,
        episode_state=store.load_episode("book-demo", "E001"),
    )
    visual_outcome = VisualRenderStage(
        vendor_dir=Path("vendor/story_to_handdrawn_video").resolve(),
        npm_command=npm,
        ffprobe_command=ffprobe,
        safe_area=SafeArea(),
        transition="cross-dissolve",
    ).run(context)
    visual = visual_outcome.outputs["picture_silent"]
    render_storyboard = json.loads(
        visual_outcome.outputs["render_storyboard"].read_text(encoding="utf-8")
    )
    assert render_storyboard["project"]["transition"] == "cross-dissolve"
    assert render_storyboard["project"]["transition_frames"] == 15

    master = episode_root / "media" / "voice" / "voice_master.wav"
    master.parent.mkdir(parents=True, exist_ok=True)
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=15", "-c:a", "pcm_s16le",
        str(master),
    ])
    ass = episode_root / "media" / "subtitles" / "subtitles.ass"
    ass.parent.mkdir(parents=True, exist_ok=True)
    ass.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,Arial,56,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,80,180,280,1\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        "Dialogue: 0,0:00:00.00,0:00:15.00,Default,,0,0,0,,Fake provider acceptance\n",
        encoding="utf-8",
    )
    voice_sha = sha256_file(master)
    cue = SubtitleCue(index=0, start_ms=0, end_ms=15_000, text="Fake provider acceptance")
    cue_hash = hashlib.sha256(json.dumps(
        [{"index": 0, "start_ms": 0, "end_ms": 15_000, "text": cue.text}],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    voice_manifest = VoiceManifest(
        voice_id="黑金3", script_sha256="a" * 64, reference_sha256="b" * 64,
        raw_sha256="c" * 64, master_sha256=voice_sha, processing_sha256="d" * 64,
        sample_rate=48_000, channels=1, sample_width=2, raw_duration_ms=15_000,
        master_duration_ms=15_000, qualifying_start_trim_ms=0, qualifying_end_trim_ms=0,
        duration=NarrationDuration(seconds=15.0, level="pass", band="ideal"),
        created_at=datetime.now(UTC),
    )
    subtitle_manifest = SubtitleManifest(
        approved_sha256="a" * 64, audio_sha256=voice_sha, asr_sha256="e" * 64,
        cue_content_sha256=cue_hash, style_template_sha256="f" * 64,
        font_family="Arial", font_sha256="0" * 64, width=1080, height=1920,
        cue_count=1, duration_ms=15_000,
    )
    cover_root = episode_root / "cover"
    cover_manifest = CoverManifest.model_validate_json(
        (cover_root / "manifest.json").read_text(encoding="utf-8")
    )
    cover_path = next(path for path in cover_root.iterdir() if path.name != "manifest.json")
    overlay = "final cover"
    inputs = HanddrawnRenderInputs(
        book_id="book-demo", episode_id="E001", visual_path=visual,
        visual_sha256=sha256_file(visual), storyboard_sha256=illustration_manifest.storyboard_sha256,
        semantic_lock_sha256="b" * 64, script_sha256="a" * 64,
        audio_sha256=voice_sha, subtitle_sha256=cue_hash,
        voice_master_path=master, voice_master_sha256=voice_sha,
        voice_manifest=voice_manifest, master_duration_ms=15_000,
        ass_path=ass, ass_sha256=sha256_file(ass), asr_sha256="e" * 64,
        subtitle_manifest=subtitle_manifest, subtitle_cues=(cue,),
        cover=CoverOverlay(
            cover_path=cover_path, cover_sha256=cover_manifest.copied_sha256,
            manifest=cover_manifest, start_ms=12_000, end_ms=15_000,
            overlay_zone_text=overlay,
            overlay_zone_sha256=hashlib.sha256(overlay.encode()).hexdigest(),
            x=760, y=100, width=240, height=360,
        ),
        render_root=episode_root / ".private" / "final-render",
        final_path=episode_root / "media" / "final" / "final.mp4",
    )

    result = render_handdrawn_final(
        inputs, ffmpeg_command=ffmpeg, ffprobe_command=ffprobe, timeout_seconds=120,
    )

    assert result.final_path.is_file()
    assert result.manifest.output_sha256 == sha256_file(result.final_path)
