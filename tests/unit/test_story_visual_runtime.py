from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bv.content.story import StoryCharacter, StorySection, import_story_candidate, approve_story_candidate
from bv.core.atomic import atomic_write_json
from bv.illustration.assets import prepare_image_jobs
from bv.illustration.contracts import CharacterBible, IllustrationStoryboard
from bv.illustration.style_selector import load_style_catalog
from bv.voice.processing import VoiceManifest, NarrationDuration
from bv.workflow.media_stages import IllustrationPlanningStage, MediaStageError, _story_scene_boundaries, _visual_sources, approved_narration_input
from bv.workflow.runtime import RuntimeAuthorization, _media_runtime_stage_config_resolver
from bv.workflow.stages import StageContext
from test_story_content import _contract, _store
from test_illustration_planner import _aligned, _cues, _pair_draft, _source_payload, _style, _sha


class StoryModel:
    def __init__(self):
        self.payload = None

    def complete(self, prompt, schema_type, request_dir):
        self.payload = payload = _source_payload(prompt)
        windows = payload["fixed_windows"]
        draft = _pair_draft(len(windows), semantic_turn_offset=3)
        template = draft.pop("character")
        draft["characters"] = [dict(copy.deepcopy(template), character_id=character["character_id"], role=character["name"]) for character in payload["story_characters"]]
        draft["legacy_single_character"] = False
        for index, scene in enumerate(draft["scenes"]):
            scene["character_refs"] = list(dict.fromkeys((payload["story_characters"][0]["character_id"], payload["story_characters"][index % len(payload["story_characters"])]["character_id"])))
            unit = json.loads(payload['important_events'][index])
            scene["semantic_turn_offset"] = unit['start'] + 10 - windows[index]['narration_span'][0]
        return schema_type.model_validate(draft)


def _story_fixture(tmp_path):
    store = _store(tmp_path)
    metadata, evidence, _, review = _contract()
    text = "皇帝在宫门前停下。大臣呈上了奏章。信使赶往远方。工匠修好了旧门。老人重新打开信箱。" * 8
    metadata.point_of_view = "third"
    metadata.characters = [StoryCharacter(character_id=f"source-person-{index}", name=name, description="原著明确出现的人物", evidence_ids=["E1"]) for index, name in enumerate(("皇帝", "大臣", "信使", "工匠", "老人"))]
    edges = [0, 48, 128, 176, 272, len(text)]
    sections = [StorySection(section_id=f"S{index}", start=start, end=end, beat=f"人物选择与后果{index}", evidence_ids=["E1"]) for index, (start, end) in enumerate(zip(edges, edges[1:]))]
    review.findings[0].end = len(text)
    import_story_candidate(store=store, book_id="book-demo", episode_id="E001", text=text, metadata=metadata, evidence=evidence, sections=sections, factual_review=review)
    approve_story_candidate(store, "book-demo", "E001")
    root = store.root / "books/book-demo/episodes/E001"
    units = []
    for section in sections:
        units.append(dict(id=section.section_id, start=section.start, end=section.end, beat=section.beat,
            before=dict(time='此前', place='宫门', state='问题未解决', image='初始处境'),
            after=dict(time='后来', place='殿内', state='选择带来后果', image='结果处境'),
            events=['正文中的事件发展'], evidence=['E1'], change_kind='consequence',
            state_change='问题经选择产生后果', story_elapsed='一段发展',
            b_entry_text=text[section.start+10:section.end], next_link='后果引出下一节',
            semantic_review='结构测试样本，不作为实际生产故事'))
    atomic_write_json(root / 'script/ab_units.json', dict(schema_version='narrative-unit-v1', source_sha256=_sha(text), units=units))
    context = StageContext(book_id="book-demo", episode_id="E001", episode_root=root, episode_state=store.load_episode("book-demo", "E001"))
    return context, text, sections


def _write_media(context, text, duration=160_000):
    root = context.episode_root
    aligned = _aligned(text, duration)
    cues = _cues(text, duration, 40)
    voice = VoiceManifest(voice_id="黑金3", script_sha256=_sha(text), reference_sha256="1" * 64, raw_sha256="2" * 64, master_sha256="a" * 64, processing_sha256="4" * 64, sample_rate=48000, channels=1, sample_width=2, raw_duration_ms=duration, master_duration_ms=duration, qualifying_start_trim_ms=0, qualifying_end_trim_ms=0, duration=NarrationDuration(seconds=duration / 1000, level="pass", band="ideal"), created_at=datetime.now(UTC))
    for relative, payload in (("alignment/alignment.json", aligned.model_dump(mode="json")), ("subtitles/cues.json", [cue.model_dump(mode="json") for cue in cues]), ("voice/voice_master.json", voice.model_dump(mode="json")), ("illustration/style_decision.json", _style(text).model_dump(mode="json"))):
        path = root / "media" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload)
    return aligned, cues


def test_real_story_planning_stage_uses_approved_beats_cast_and_adapter(tmp_path):
    context, text, sections = _story_fixture(tmp_path)
    aligned, cues = _write_media(context, text)
    # A stale legacy package must never win over the hash-bound story approval.
    (context.episode_root / "script/script_package.json").write_text("{}", encoding="utf-8")
    narration, lock = _visual_sources(context)
    assert narration.text == text
    assert lock.required_cluster_ids == tuple(section.section_id for section in sections)
    assert not (context.episode_root / "script/semantic-lock.json").exists()
    model = StoryModel()
    stage = IllustrationPlanningStage(model=model, authorization=RuntimeAuthorization(allow_external=True), prompt_path=Path("prompts/illustration/storyboard.md"))
    outcome = stage.run(context)
    storyboard = IllustrationStoryboard.model_validate_json(outcome.outputs["illustration_storyboard"].read_text(encoding="utf-8"))
    bible = CharacterBible.model_validate_json(outcome.outputs["character_bible"].read_text(encoding="utf-8"))
    assert len(storyboard.scenes) == 5
    assert tuple(scene.start_ms for scene in storyboard.scenes[1:]) == _story_scene_boundaries(sections, aligned, cues, 160_000)
    assert len({scene.end_ms - scene.start_ms for scene in storyboard.scenes}) > 1
    assert len(bible.characters) == 5
    assert model.payload["story_point_of_view"] == "third"
    assert [json.loads(item)['beat'] for item in model.payload["important_events"]] == [section.beat for section in sections]
    assert "reader-01" not in model.payload["allowed_character_ids"]
    assert len(outcome.inputs["source_contract_sha256"]) == 64
    assert storyboard.scenes[-1].to_frame == 4800
    style = next(item for item in load_style_catalog(Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json")) if item.style_id == _style(text).selected_style)
    manifest = prepare_image_jobs(storyboard, bible, style, _style(text), context.episode_root)
    later_a = next(job for job in manifest.jobs if job.asset_id == "S03-A")
    later_b = next(job for job in manifest.jobs if job.asset_id == "S03-B")
    assert "S01" in later_a.reference_scene_ids
    assert later_b.reference_scene_ids[0] == "S03"
    assert "S01" in later_b.reference_scene_ids


def test_story_sample_does_not_mechanically_expand_two_units_into_three(tmp_path):
    context, text, _ = _story_fixture(tmp_path)
    sample = approved_narration_input(context, mode="technical_sample", sample_span=(48, 176))
    _write_media(context, sample.text, 24_000)
    model = StoryModel()
    stage = IllustrationPlanningStage(model=model, authorization=RuntimeAuthorization(allow_external=True), prompt_path=Path("prompts/illustration/storyboard.md"))
    with pytest.raises(MediaStageError, match='illustration_plan_failed'):
        stage.run(context)
    assert model.payload is None


def test_story_boundaries_merge_short_beats_at_real_cue_starts(tmp_path):
    context, text, _ = _story_fixture(tmp_path)
    aligned, cues = _write_media(context, text)
    sections = [StorySection(section_id=f"S{i}", start=start, end=end, beat="事件", evidence_ids=["E1"]) for i, (start, end) in enumerate(zip([0, 1, 4, 48, 128], [1, 4, 48, 128, len(text)]))]
    boundaries = _story_scene_boundaries(sections, aligned, cues, 160_000)
    assert all(value in {cue.start_ms for cue in cues} for value in boundaries)
    edges = [0, *boundaries, 160_000]
    assert min(right - left for left, right in zip(edges, edges[1:])) >= 4_000


def test_ab_plan_changes_invalidate_visual_cache_without_repeating_tts(tmp_path):
    context, _, _ = _story_fixture(tmp_path)
    resolver = _media_runtime_stage_config_resolver({'plan_illustrations': 'a'*64, 'tts': 'b'*64})
    root = context.episode_root
    before = resolver('plan_illustrations', root)
    voice_before = resolver('tts', root)
    path = root / 'script/ab_units.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['units'][0]['after']['image'] = '更新的结果构图'
    atomic_write_json(path, payload)
    assert resolver('plan_illustrations', root) != before
    assert resolver('tts', root) == voice_before


def test_missing_ab_plan_blocks_model_call(tmp_path):
    context, text, _ = _story_fixture(tmp_path)
    _write_media(context, text)
    (context.episode_root / 'script/ab_units.json').unlink()
    model = StoryModel()
    stage = IllustrationPlanningStage(model=model, authorization=RuntimeAuthorization(allow_external=True), prompt_path=Path('prompts/illustration/storyboard.md'))
    with pytest.raises(MediaStageError, match='illustration_plan_failed'):
        stage.run(context)
    assert model.payload is None
