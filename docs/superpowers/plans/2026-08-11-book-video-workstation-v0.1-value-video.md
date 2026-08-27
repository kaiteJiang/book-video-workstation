# BV Workstation V0.1 Whole-Book Value Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Windows-local CMD workflow so one complete TXT, text-layer PDF, or EPUB becomes one evidence-backed, approved 45–60-second Douyin video about the value the whole book gives its target reader; a roughly 15-second H3 clip is only a visual acceptance unit, never the final E001 deliverable.

**Architecture:** Continue from commit `9fea179` after completing Tasks 6–10 in `2026-08-09-book-video-workstation-v0.1.md`. Codex builds a whole-book value synthesis from verified chapter claims, E001 locks one reader/value/life through-line, IndexTTS2 creates one complete narration, Volc ASR supplies real timings, the storyboard splits that timeline into four or five H3 segments, and FFmpeg joins the imported segments under the single narration timeline. Every artifact is schema-validated, hashed, resumable, and stopped at explicit human gates.

**Tech Stack:** Python 3.11, uv, Typer, Rich, Pydantic v2, PyYAML, httpx, pytest, Codex CLI, Grok CLI, IndexTTS2 CLI, Volcengine ASR, FFmpeg/FFprobe, manual MiniMax H3.

## Global Constraints

- Active design: `docs/superpowers/specs/2026-08-09-book-video-workstation-v0.1-design.md`, revision dated 2026-08-11.
- Preconditions: old-plan Tasks 1–10 are complete and their full test suite passes before Task 11 starts.
- Initial Episode: create E001 only; `episode_kind` must be `whole_book_value`.
- E001 content: one target reader, one central life tension, one whole-book value thesis, and two or three progressive supporting claim clusters.
- Evidence breadth: key support should span at least three chapter/structure regions when the parsed book has at least three valid regions; a documented exception is required for structurally shorter books.
- Content exclusion: E001 must not become a chapter-by-chapter summary, a list of quotations, or a video about one isolated concept.
- Later Episodes: create only through `bv add <book-id>`; use `episode_kind: value_angle`; never degrade to an isolated knowledge-point video.
- Models: Codex CLI is primary. Grok CLI is a non-blocking reader-response reviewer for value briefs and scripts only. Claude is not used.
- Script: draft about 180–240 Chinese characters, but actual IndexTTS2 duration is authoritative.
- Master timeline: approved `voice_master.wav` is the only narration and timing authority.
- Final duration: 45.0–60.0 seconds; ideal narration duration is 48.0–56.0 seconds.
- H3: four segments S01–S04, with S05 only when needed; each segment target is at most 15.0 seconds and accepted probe duration is at most 15.5 seconds.
- Visual sample: importing/approving S01 may set `visual_sample_ready`, but cannot set `h3_imported` or unlock final render.
- Continuity: every segment prompt consumes the same `continuity_bible.yaml`.
- H3 output: no final narration, dialogue, subtitles, generated book-title text, fake cover, logo, or watermark.
- Final media: 1080×1920, 9:16, 30 fps, H.264, yuv420p, AAC stereo, faststart.
- Voice: one user-owned or explicitly authorized IndexTTS2 reference recording; no silent fallback.
- ASR: Volc supplies timings and pronunciation evidence; approved script remains subtitle text.
- UI: CMD only; no GUI, web server, browser automation, database, Docker, automatic H3 API call, or automatic Douyin publishing.
- Privacy: no ebook, voice, credentials, logs containing secrets, H3 clips, or final media in Git.
- Files: imports are copy-only and no original user file is moved, overwritten, or deleted.
- Tests: ordinary automated tests make no paid/cloud/model calls. Real tests require explicit environment flags and user-controlled assets.
- Development: TDD, one focused commit and one review gate per task.

---

## Revised file map

~~~text
src/bv/
  config.py
  state/models.py
  state/invalidation.py
  content/
    synthesis.py
    topics.py
    dedup.py
    scripts.py
    reviews.py
    final_review.py
  voice/
    indextts2.py
    processing.py
  asr/
    volcengine.py
    alignment.py
  subtitles/generate.py
  storyboard/generate.py
  video/
    probe.py
    import_h3.py
    cover.py
    render.py
    qc.py
  workflow/
    stages.py
    orchestrator.py
prompts/
  book_synthesis/v2.md
  topic_generation/v2.md
  topic_dedup/v2.md
  script_draft/v2.md
  script_humanize/v2.md
  script_fact_diff/v2.md
  grok_topic_review/v2.md
  grok_script_review/v2.md
  storyboard/v2.md
tests/unit/
tests/integration/
docs/operations/
~~~

## Shared revised contracts

~~~python
EpisodeKind = Literal["whole_book_value", "value_angle"]

class ClaimCluster(BaseModel):
    cluster_id: str
    summary: str
    narrative_role: Literal["problem", "reframe", "application", "boundary"]
    claim_ids: list[str]
    chapter_regions: list[str]

class WholeBookValue(BaseModel):
    value_thesis: str
    target_reader: str
    reader_before: str
    reader_after: str
    central_life_tension: str
    supporting_claim_clusters: list[ClaimCluster]
    practical_value: str
    reading_reason: str
    coverage_exception: str | None = None

class EpisodeBrief(BaseModel):
    episode_id: str
    episode_kind: EpisodeKind
    topic_name: str
    book_value_thesis: str
    target_reader: str
    reader_before: str
    reader_after: str
    central_life_tension: str
    supporting_claim_cluster_ids: list[str]
    source_claim_ids: list[str]
    life_connection: str
    practical_value: str
    reading_reason: str
    constructed_scene: bool
    status: Literal["reserved", "script_approved", "produced", "published", "rejected", "released"]

class H3Segment(BaseModel):
    shot_id: str
    start_ms: int
    end_ms: int
    narration_text: str
    prompt_path: Path
    imported_path: Path | None = None
~~~

---

### Task 11: Migrate configuration, statuses, and invalidation to the long-form multi-clip contract

**Files:**
- Modify: `src/bv/config.py`
- Modify: `config.example.yaml`
- Modify: `src/bv/state/models.py`
- Modify: `src/bv/state/invalidation.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_state.py`

**Interfaces:**
- Produces: `VideoConfig.final_min_seconds`, `final_max_seconds`, `h3_segment_target_seconds`, `h3_segment_max_seconds`; `AudioConfig.hard_min_seconds`, `ideal_min_seconds`, `ideal_max_seconds`, `hard_max_seconds`; statuses `visual_sample_ready` and `h3_partial`.

- [ ] **Step 1: Write failing configuration migration test**

~~~python
def test_value_video_defaults_are_long_form() -> None:
    config = AppConfig()
    assert config.video.final_min_seconds == 45.0
    assert config.video.final_max_seconds == 60.0
    assert config.video.h3_segment_target_seconds == 15.0
    assert config.video.h3_segment_max_seconds == 15.5
    assert config.audio.hard_min_seconds == 45.0
    assert config.audio.ideal_min_seconds == 48.0
    assert config.audio.ideal_max_seconds == 56.0
    assert config.audio.hard_max_seconds == 60.0
~~~

- [ ] **Step 2: Write failing invalidation test**

~~~python
def test_script_change_invalidates_all_multiclip_artifacts() -> None:
    stages = ["tts", "asr", "subtitles", "storyboard", "continuity", "h3_prompts", "h3_clips", "render", "qc"]
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        completed_stages=stages.copy(),
    )
    result = invalidate_from(episode, "script")
    for stage in ("tts", "asr", "subtitles", "storyboard", "continuity", "h3_prompts", "h3_clips", "render", "qc"):
        assert stage in result.stale_stages
~~~

- [ ] **Step 3: Run focused tests and confirm RED**

~~~powershell
uv run pytest tests/unit/test_config.py tests/unit/test_state.py -v
~~~

- [ ] **Step 4: Implement the minimal schema migration**

Remove the ambiguous final `video.duration_seconds`. Add the four video fields and four audio fields above. Keep Pydantic `extra="forbid"`; an old config containing `video.duration_seconds` must fail with a migration message instead of silently treating 15 seconds as final duration. Extend invalidation without deleting existing stale history.

- [ ] **Step 5: Update the example config and run regression tests**

~~~yaml
video:
  width: 1080
  height: 1920
  frame_rate: 30
  final_min_seconds: 45.0
  final_max_seconds: 60.0
  h3_segment_target_seconds: 15.0
  h3_segment_max_seconds: 15.5
audio:
  hard_min_seconds: 45.0
  ideal_min_seconds: 48.0
  ideal_max_seconds: 56.0
  hard_max_seconds: 60.0
~~~

~~~powershell
uv run pytest tests/unit/test_config.py tests/unit/test_state.py -v
uv run pytest -q
git add src/bv/config.py config.example.yaml src/bv/state/models.py src/bv/state/invalidation.py tests/unit/test_config.py tests/unit/test_state.py
git commit -m "refactor: adopt long-form multi-clip video contract"
~~~

---

### Task 12: Produce a claim-backed whole-book value synthesis

**Files:**
- Create: `src/bv/content/synthesis.py`
- Create: `prompts/book_synthesis/v2.md`
- Create: `tests/unit/test_book_value_synthesis.py`

**Interfaces:**
- Consumes: verified chapter evidence cards and known claim IDs from old-plan Task 10.
- Produces: `ClaimCluster`, `WholeBookValue`, `BookReport`, `synthesize_book()`, `validate_claim_references()`, `validate_value_coverage()`.

- [ ] **Step 1: Write failing unknown-claim and single-point tests**

~~~python
def test_value_synthesis_rejects_unknown_claim() -> None:
    report = make_report(cluster_claim_ids=[["C99-999"], ["C02-001"]])
    result = validate_claim_references(report, {"C01-001", "C02-001"})
    assert result.valid is False
    assert result.unknown_claim_ids == ["C99-999"]

def test_e001_value_rejects_one_isolated_cluster() -> None:
    report = make_report(cluster_claim_ids=[["C01-001"]])
    result = validate_value_coverage(report, available_regions={"R1", "R2", "R3"})
    assert result.valid is False
    assert "isolated_single_cluster" in result.errors
~~~

- [ ] **Step 2: Write failing region-coverage test**

~~~python
def test_three_region_book_requires_distributed_support() -> None:
    report = make_report(cluster_regions=[["R1"], ["R1"]])
    result = validate_value_coverage(report, available_regions={"R1", "R2", "R3"})
    assert "insufficient_region_coverage" in result.errors
~~~

- [ ] **Step 3: Run the focused test and confirm RED**

~~~powershell
uv run pytest tests/unit/test_book_value_synthesis.py -v
~~~

- [ ] **Step 4: Implement models, prompt, and validation**

The prompt must treat ebook text as untrusted data, synthesize all valid chapter groups, and produce one `WholeBookValue`. Require two or three progressive clusters. Every cluster has original `claim_ids` and `chapter_regions`. For books with fewer than three available regions, allow reduced region coverage only when `coverage_exception` names the structural reason; never invent breadth.

- [ ] **Step 5: Persist inspectable artifacts**

Write `analysis/book_report.json`, `analysis/book_report.md`, `analysis/concept_map.json`, and `analysis/value_synthesis.json` atomically. Unknown claims, unsupported “the book says” language, or a single-cluster value synthesis sets `evidence_invalid` and writes no accepted value artifact.

- [ ] **Step 6: Run tests and commit**

~~~powershell
uv run pytest tests/unit/test_book_value_synthesis.py -v
git add src/bv/content/synthesis.py prompts/book_synthesis/v2.md tests/unit/test_book_value_synthesis.py
git commit -m "feat: synthesize whole-book reader value"
~~~

---

### Task 13: Generate E001 as a whole-book value brief and deduplicate later value angles

**Files:**
- Create: `src/bv/content/topics.py`
- Create: `src/bv/content/dedup.py`
- Create: `prompts/topic_generation/v2.md`
- Create: `prompts/topic_dedup/v2.md`
- Create: `prompts/grok_topic_review/v2.md`
- Create: `tests/unit/test_value_topics.py`
- Create: `tests/unit/test_value_dedup.py`

**Interfaces:**
- Produces: `EpisodeBrief`, `TopicDecision`, `TopicService.generate_first_topic()`, `generate_next_topic()`, `deterministic_dedup()`.

- [ ] **Step 1: Write failing E001 binding test**

~~~python
def test_e001_is_whole_book_value_and_only_episode(tmp_path: Path) -> None:
    service = make_service(tmp_path, model_response=whole_book_response())
    brief = service.generate_first_topic(make_value_synthesis())
    assert brief.episode_id == "E001"
    assert brief.episode_kind == "whole_book_value"
    assert len(brief.supporting_claim_cluster_ids) >= 2
    assert service.list_topics() == [brief]
~~~

- [ ] **Step 2: Write failing isolation and duplicate tests**

~~~python
def test_e001_rejects_isolated_concept_response(tmp_path: Path) -> None:
    service = make_service(tmp_path, model_response=single_concept_response())
    with pytest.raises(EvidenceInvalid, match="whole-book value"):
        service.generate_first_topic(make_value_synthesis())

def test_later_angle_rejects_same_reader_value_and_takeaway() -> None:
    old = make_brief("E001")
    new = make_brief("E002", episode_kind="value_angle")
    assert deterministic_dedup(new, [old]).accepted is False
~~~

- [ ] **Step 3: Run tests and confirm RED**

~~~powershell
uv run pytest tests/unit/test_value_topics.py tests/unit/test_value_dedup.py -v
~~~

- [ ] **Step 4: Implement E001 without asking for a topic pool**

E001 is a structured editorial framing of `WholeBookValue`, not an arbitrary concept picker. Codex may improve the reader-facing `topic_name` and life connection, but it may not replace the value thesis or drop required clusters. Grok may critique reader clarity; its failure records `grok_review_skipped` and does not relax evidence gates.

- [ ] **Step 5: Implement five-layer later-Episode dedup**

For `bv add`, compare all reserved/approved/produced/published briefs across: value structure; claim-cluster/evidence overlap; life expression; Codex semantic takeaway; Grok repeat feeling. Reject claim overlap at or above 0.5, the same reader-before/reader-after transformation, or the same final takeaway even if scenery changes. Retry at most twice; the third failed candidate sets `insufficient_distinct_topic`.

E001 remains a comparison target for value structure, life expression, Codex semantics, and Grok repeat feeling. Because E001 intentionally covers every whole-book support cluster, apply the cluster/claim Jaccard threshold only between `value_angle` Episodes; otherwise every valid later angle would be rejected by construction.

- [ ] **Step 6: Run tests and commit**

~~~powershell
uv run pytest tests/unit/test_value_topics.py tests/unit/test_value_dedup.py -v
git add src/bv/content/topics.py src/bv/content/dedup.py prompts/topic_generation/v2.md prompts/topic_dedup/v2.md prompts/grok_topic_review/v2.md tests/unit/test_value_topics.py tests/unit/test_value_dedup.py
git commit -m "feat: frame and deduplicate book value episodes"
~~~

---

### Task 14: Draft, humanize, fact-check, and approve a 45–60-second value script

**Files:**
- Create: `src/bv/content/scripts.py`
- Create: `src/bv/content/reviews.py`
- Create: `prompts/script_draft/v2.md`
- Create: `prompts/script_humanize/v2.md`
- Create: `prompts/script_fact_diff/v2.md`
- Create: `prompts/grok_script_review/v2.md`
- Create: `tests/unit/test_value_scripts.py`
- Create: `tests/unit/test_script_approval.py`

**Interfaces:**
- Produces: `SemanticLock`, `ScriptPackage`, `FactDiffResult`, `validate_value_script()`, `build_script_review()`, `extract_review_voiceover()`, `approve_script()`.

- [ ] **Step 1: Write failing value-loss tests**

~~~python
def test_humanized_script_cannot_drop_value_thesis_or_cluster() -> None:
    lock = make_semantic_lock(required_cluster_ids=["change", "separation", "freedom"])
    result = validate_value_script(
        text="只要学会课题分离，你就自由了。",
        lock=lock,
        represented_cluster_ids=["separation"],
    )
    assert result.valid is False
    assert "missing_value_thesis" in result.violations
    assert "missing_required_cluster:change" in result.violations
    assert "scope_expanded" in result.violations
~~~

- [ ] **Step 2: Write failing review-marker test**

~~~python
def test_review_extracts_exactly_one_editable_voiceover() -> None:
    markdown = "<!-- BV:VOICEOVER:START -->\n口播\n<!-- BV:VOICEOVER:END -->"
    assert extract_review_voiceover(markdown) == "口播"
~~~

- [ ] **Step 3: Run focused tests and confirm RED**

~~~powershell
uv run pytest tests/unit/test_value_scripts.py tests/unit/test_script_approval.py -v
~~~

- [ ] **Step 4: Implement the six-pass pipeline**

Build the semantic lock; draft one recommended script plus two hooks; humanize with the installed `human-writing` Skill principles; perform deterministic scope/negation/number/cluster checks; request non-blocking Grok reader critique; let Codex accept/reject critique with recorded reasons. The narrative order is reader state, deeper book question, progressive clusters, life return, boundary and reading reason. Character count 180–240 is a drafting warning only; no duration pass occurs before real TTS.

- [ ] **Step 5: Implement approval and immutable hash**

The review contains value thesis, reader-before/after, cluster-to-claim trace, coverage regions, editable voiceover markers, Grok status, risks, character count, and estimated duration. Approval reruns all deterministic checks, writes `script/approved.txt` and `script/script_manifest.json`, appends `script_approved`, and invalidates all downstream artifacts when the approved hash changes.

- [ ] **Step 6: Run tests and commit**

~~~powershell
uv run pytest tests/unit/test_value_scripts.py tests/unit/test_script_approval.py -v
git add src/bv/content/scripts.py src/bv/content/reviews.py prompts/script_draft/v2.md prompts/script_humanize/v2.md prompts/script_fact_diff/v2.md prompts/grok_script_review/v2.md tests/unit/test_value_scripts.py tests/unit/test_script_approval.py
git commit -m "feat: create reviewable whole-book value scripts"
~~~

**Milestone gate:** A private complete book reaches `awaiting_script_review`; no TTS or cloud ASR runs before script approval.

---

### Task 15: Generate and validate the complete IndexTTS2 master narration

**Files:**
- Create: `src/bv/voice/indextts2.py`
- Create: `src/bv/voice/processing.py`
- Create: `tests/unit/test_indextts2.py`
- Create: `tests/integration/test_audio_processing.py`

**Interfaces:**
- Produces: `VoiceManifest`, `build_synth_argv()`, `synthesize_voice()`, `build_voice_master()`, `classify_narration_duration()`.

- [ ] **Step 1: Write failing exact-command and duration tests**

~~~python
def test_synth_uses_explicit_project_and_models(tmp_path: Path) -> None:
    argv = build_synth_argv("uv", Path(r"D:\Program Files (x86)\index-tts"), Path(r"D:\Program Files (x86)\index-tts\checkpoints"), tmp_path/"script.txt", tmp_path/"voice.wav", tmp_path/"raw.wav", "cuda")
    assert argv[:4] == ["uv", "run", "--project", r"D:\Program Files (x86)\index-tts"]
    assert ["--model-dir", r"D:\Program Files (x86)\index-tts\checkpoints"] == argv[argv.index("--model-dir"):argv.index("--model-dir") + 2]

@pytest.mark.parametrize("seconds,expected", [(44.9, "fail"), (45.0, "pass"), (52.0, "pass"), (60.0, "pass"), (60.1, "fail")])
def test_real_duration_controls_acceptance(seconds: float, expected: str) -> None:
    assert classify_narration_duration(seconds).level == expected
~~~

- [ ] **Step 2: Run tests and confirm RED**

~~~powershell
uv run pytest tests/unit/test_indextts2.py tests/integration/test_audio_processing.py -v
~~~

- [ ] **Step 3: Implement safe synthesis and processing**

Invoke `uv run --project <install-dir> indextts2 synth` with explicit model directory, authorized voice, unique temporary output, fp16, no deepspeed, and no CUDA kernel. Verify script/voice hashes, WAV decodability, and nonzero duration. Decode to PCM, trim only qualifying edge silence, two-pass loudnorm to -18 LUFS/-1.5 dBTP, and write `audio/voice_master.wav` atomically.

- [ ] **Step 4: Enforce the actual 45–60-second gate**

Below 45 or above 60 sets `tts_duration_out_of_range`; 45–48 and 56–60 pass with warning; 48–56 passes. Never speed-change audio to force acceptance. The real test runs only with `BV_RUN_REAL_TTS=1` and an authorized voice; automated tests use generated local WAVs.

- [ ] **Step 5: Run tests and commit**

~~~powershell
uv run pytest tests/unit/test_indextts2.py tests/integration/test_audio_processing.py -v
git add src/bv/voice/indextts2.py src/bv/voice/processing.py tests/unit/test_indextts2.py tests/integration/test_audio_processing.py
git commit -m "feat: synthesize complete IndexTTS2 narration"
~~~

---

### Task 16: Validate narration with Volcengine ASR and align approved text

**Files:**
- Create: `src/bv/asr/volcengine.py`
- Create: `src/bv/asr/alignment.py`
- Create: `tests/unit/test_volcengine_asr.py`
- Create: `tests/unit/test_alignment.py`

**Interfaces:**
- Produces: `VolcCredentials`, `AsrResult`, `WordTiming`, `AlignedCharacter`, `AlignedScript`, `PronunciationReport`, `recognize_flash()`, `align_approved_text()`.

- [ ] **Step 1: Write failing credential and key-term tests**

~~~python
def test_api_key_header_is_returned_as_secret() -> None:
    headers, secrets = build_headers(api_key="secret", app_key=None, access_key=None)
    assert headers["X-Api-Key"] == "secret"
    assert secrets == {"secret"}

def test_wrong_core_term_blocks_high_similarity() -> None:
    report = align_approved_text("这叫课题分离。", "这叫课题分类。", [], ["课题分离"])
    assert "key_term_mismatch:课题分离" in report.violations
~~~

- [ ] **Step 2: Run focused tests and confirm RED**

~~~powershell
uv run pytest tests/unit/test_volcengine_asr.py tests/unit/test_alignment.py -v
~~~

- [ ] **Step 3: Implement the official flash request and parser**

Use the configured official endpoint. Accept API key or a complete legacy app/access pair, never partial credentials. Require status `20000000` and word timings. Keep authorization values in the redaction set and out of logs, state, and exceptions.

- [ ] **Step 4: Implement approved-text alignment**

Use Unicode/punctuation-aware dynamic programming while preserving original approved characters. Key terms, names, numbers, and negations must match. Similarity at least 0.94 passes; 0.90–0.94 requires repair/manual review; below 0.90 fails. Missing usable timings blocks subtitle generation.

- [ ] **Step 5: Run tests and commit**

~~~powershell
uv run pytest tests/unit/test_volcengine_asr.py tests/unit/test_alignment.py -v
git add src/bv/asr/volcengine.py src/bv/asr/alignment.py tests/unit/test_volcengine_asr.py tests/unit/test_alignment.py
git commit -m "feat: align approved narration with Volcengine ASR"
~~~

---

### Task 17: Generate SRT and ASS from approved text and real timings

**Files:**
- Create: `src/bv/subtitles/generate.py`
- Create: `styles/subtitle-default.ass`
- Create: `tests/unit/test_subtitles.py`

**Interfaces:**
- Produces: `SubtitleCue`, `build_cues()`, `render_srt()`, `render_ass()`.

- [ ] **Step 1: Write failing coverage and protected-term tests**

~~~python
def test_cues_cover_all_approved_characters_once() -> None:
    aligned = make_aligned("读完这本书，你会重新理解自己的选择。")
    cues = build_cues(aligned, max_chars=14, min_duration_ms=600)
    assert flatten(cues) == aligned.approved_without_punctuation
    assert all(cue.end_ms > cue.start_ms for cue in cues)

def test_protected_book_term_is_not_split() -> None:
    cues = build_cues(make_aligned("这叫课题分离。"), protected_terms=["课题分离"])
    assert any("课题分离" in cue.text.replace("\n", "") for cue in cues)
~~~

- [ ] **Step 2: Run test and confirm RED**

~~~powershell
uv run pytest tests/unit/test_subtitles.py -v
~~~

- [ ] **Step 3: Implement cue segmentation and rendering**

Split on approved punctuation and semantic pauses, target 6–14 Chinese characters, at most two lines, minimum 600 ms, no one-character orphan when mergeable. Never substitute ASR text. Verify licensed font path/family before generating 1080×1920 ASS in the Douyin-safe area. Write SRT, ASS, and a manifest containing approved-script and ASR hashes.

- [ ] **Step 4: Run tests and commit**

~~~powershell
uv run pytest tests/unit/test_subtitles.py -v
git add src/bv/subtitles/generate.py styles/subtitle-default.ass tests/unit/test_subtitles.py
git commit -m "feat: generate full-length timed subtitles"
~~~

---

### Task 18: Split the master timeline into H3 segments and render continuity-locked prompts

**Files:**
- Create: `src/bv/storyboard/generate.py`
- Create: `prompts/storyboard/v2.md`
- Create: `tests/unit/test_storyboard_segments.py`

**Interfaces:**
- Produces: `ContinuityBible`, `H3Segment`, `Storyboard`, `validate_storyboard()`, `generate_storyboard()`, `render_h3_prompts()`.

- [ ] **Step 1: Write failing segment-coverage test**

~~~python
def test_storyboard_covers_master_timeline_with_four_or_five_segments() -> None:
    board = make_board(duration_ms=54000, boundaries=[0, 13000, 27000, 40500, 54000])
    result = validate_storyboard(board, master_duration_ms=54000)
    assert result.valid is True
    assert len(board.segments) == 4
    assert board.segments[0].start_ms == 0
    assert board.segments[-1].end_ms == 54000
    assert all(segment.end_ms - segment.start_ms <= 15000 for segment in board.segments)
~~~

- [ ] **Step 2: Write failing continuity and prompt-boundary tests**

~~~python
def test_each_prompt_contains_same_continuity_and_prohibitions() -> None:
    prompts = render_h3_prompts(make_board(), make_continuity())
    assert set(prompts) == {"S01", "S02", "S03", "S04"}
    for prompt in prompts.values():
        for phrase in ("人物不开口", "禁止字幕", "禁止书名文字", "禁止假书封", "禁止Logo和水印"):
            assert phrase in prompt
        assert "深色通勤外套" in prompt
~~~

- [ ] **Step 3: Run test and confirm RED**

~~~powershell
uv run pytest tests/unit/test_storyboard_segments.py -v
~~~

- [ ] **Step 4: Implement cue-boundary segmentation**

Use ASR/subtitle boundaries, not equal character division. Produce four segments by default and five only when no valid four-segment partition keeps every segment at or below 15 seconds. Segments must be continuous, non-overlapping, cover the complete master duration, and carry exact narration text. Reject abstract visual actions.

- [ ] **Step 5: Render artifacts**

Write `storyboard/storyboard.json`, `storyboard/storyboard.md`, `storyboard/continuity_bible.yaml`, and one `h3_prompt_<shot_id>.md` per required segment. S01 establishes reader state; middle segments carry progressive reframing; the last segment returns to life and reserves a clean cover area. Every prompt repeats the continuity lock and prohibitions.

- [ ] **Step 6: Run tests and commit**

~~~powershell
uv run pytest tests/unit/test_storyboard_segments.py -v
git add src/bv/storyboard/generate.py prompts/storyboard/v2.md tests/unit/test_storyboard_segments.py
git commit -m "feat: create continuity-locked H3 segment prompts"
~~~

**Milestone gate:** Approved script, real master narration, ASR, subtitles, storyboard, continuity bible, and all required H3 prompts exist; workflow stops at `awaiting_h3_generation`.

---

### Task 19: Import H3 segments by shot ID and approve the real cover

**Files:**
- Create: `src/bv/video/probe.py`
- Create: `src/bv/video/import_h3.py`
- Create: `src/bv/video/cover.py`
- Create: `tests/unit/test_h3_segment_import.py`
- Create: `tests/unit/test_cover.py`

**Interfaces:**
- Produces: `MediaProbe`, `H3ImportManifest`, `probe_media()`, `validate_h3_probe()`, `import_h3_video(book_id, episode_id, shot_id, source)`, `all_required_segments_present()`, `approve_cover()`.

- [ ] **Step 1: Write failing copy-only and partial-status tests**

~~~python
def test_s01_import_is_copy_only_and_not_complete(tmp_path: Path) -> None:
    source = make_video_file(tmp_path / "download.mp4")
    result = import_h3_video(make_context(tmp_path), "S01", source, probe=valid_probe)
    assert source.exists()
    assert result.original_copy.exists()
    assert result.status == "visual_sample_ready"
    assert result.all_required_segments is False
~~~

- [ ] **Step 2: Write failing identity and completion tests**

~~~python
def test_unknown_or_duplicate_shot_id_is_rejected(tmp_path: Path) -> None:
    context = make_context(tmp_path, required=["S01", "S02", "S03", "S04"])
    with pytest.raises(ValueError, match="unknown shot"):
        import_h3_video(context, "S09", make_video_file(tmp_path/"x.mp4"), probe=valid_probe)

def test_only_all_required_segments_set_h3_imported(tmp_path: Path) -> None:
    context = import_required_segments(tmp_path, ["S01", "S02", "S03", "S04"])
    assert all_required_segments_present(context) is True
    assert context.status == "h3_imported"
~~~

- [ ] **Step 3: Run tests and confirm RED**

~~~powershell
uv run pytest tests/unit/test_h3_segment_import.py tests/unit/test_cover.py -v
~~~

- [ ] **Step 4: Implement safe probe/import**

FFprobe must report decodability, duration, dimensions, frame rate, video/audio codecs, and streams. Require 9:16 and duration greater than zero and at most 15.5 seconds; compare against the segment target with at most 0.3-second shortfall for automatic freeze-frame repair. Copy atomically to `h3/<shot_id>/original-<sha256>.mp4`; never overwrite or move the source. Store one manifest per shot and an aggregate manifest.

- [ ] **Step 5: Implement cover approval**

Copy and decode-check the real cover, record SHA-256 and source type, require `matches_product_version=True`, and never accept a generated H3 cover as the product cover.

- [ ] **Step 6: Run tests and commit**

~~~powershell
uv run pytest tests/unit/test_h3_segment_import.py tests/unit/test_cover.py -v
git add src/bv/video/probe.py src/bv/video/import_h3.py src/bv/video/cover.py tests/unit/test_h3_segment_import.py tests/unit/test_cover.py
git commit -m "feat: import validated H3 segments by shot"
~~~

---

### Task 20: Concatenate H3 segments, render the final MP4, and run technical QC

**Files:**
- Create: `src/bv/video/render.py`
- Create: `src/bv/video/qc.py`
- Create: `tests/unit/test_multiclip_render.py`
- Create: `tests/integration/test_multiclip_render_pipeline.py`

**Interfaces:**
- Produces: `RenderInputs`, `build_segment_normalize_argv()`, `build_concat_manifest()`, `build_render_argv()`, `preflight_render()`, `render_final()`, `inspect_final()`.

- [ ] **Step 1: Write failing ordered-input and stale tests**

~~~python
def test_render_requires_every_segment_in_storyboard_order(tmp_path: Path) -> None:
    inputs = make_render_inputs(tmp_path, required=["S01", "S02", "S03", "S04"], present=["S01", "S03", "S04"])
    result = preflight_render(inputs, require_files=False)
    assert "missing_h3_segment:S02" in result.errors

def test_render_refuses_stale_continuity_or_script(tmp_path: Path) -> None:
    inputs = make_render_inputs(tmp_path, continuity_hash="new", prompt_continuity_hash="old")
    assert "stale_h3_prompts" in preflight_render(inputs, require_files=False).errors
~~~

- [ ] **Step 2: Run tests and confirm RED**

~~~powershell
uv run pytest tests/unit/test_multiclip_render.py tests/integration/test_multiclip_render_pipeline.py -v
~~~

- [ ] **Step 3: Implement deterministic segment normalization and concatenation**

Normalize every clip to 1080×1920, 30 fps, yuv420p, and its storyboard duration. Trim excess; permit at most 0.3 seconds of last-frame extension. Concatenate strictly by `shot_id` order. Discard generated narration; ambience is optional and separately mixed at checked low gain.

- [ ] **Step 4: Implement final render and QC**

Burn ASS subtitles, overlay approved cover only in storyboard-defined intervals, add complete `voice_master.wav`, normalize final audio to -16 LUFS/-1.5 dBTP, encode H.264/AAC stereo with faststart. QC requires 45–60 seconds, 1080×1920, expected codecs/pixel format, complete subtitle coverage, voice duration within video, all current hashes, no stale inputs, and no obvious clipping. QC must not claim aesthetic, face, or hand correctness.

- [ ] **Step 5: Add generated-media integration test**

Generate four short colored 9:16 clips and a sine-wave narration in a pytest temp directory, render a 45–60-second fixture with an ASS cue and generated cover rectangle, then inspect it with FFprobe. Commit no binary fixture.

- [ ] **Step 6: Run tests and commit**

~~~powershell
uv run pytest tests/unit/test_multiclip_render.py tests/integration/test_multiclip_render_pipeline.py -v
git add src/bv/video/render.py src/bv/video/qc.py tests/unit/test_multiclip_render.py tests/integration/test_multiclip_render_pipeline.py
git commit -m "feat: render and inspect multi-clip value videos"
~~~

---

### Task 21: Complete the stage-gated CMD workflow for multi-clip E001

**Files:**
- Create: `src/bv/workflow/stages.py`
- Create: `src/bv/workflow/orchestrator.py`
- Modify: `src/bv/cli.py`
- Modify: `tests/fakes.py`
- Create: `tests/unit/test_orchestrator.py`
- Create: `tests/integration/test_cli_value_workflow.py`

**Interfaces:**
- Produces: `Orchestrator.next()`, `retry()`, `status_view()`, `open_target()`, `approve_gate()`, `import_video(book_id, episode_id, shot_id, source)`.

- [ ] **Step 1: Write failing gate-transition test**

~~~python
def test_s01_does_not_unlock_render_but_all_segments_do(tmp_path: Path) -> None:
    workflow = make_orchestrator(tmp_path, required=["S01", "S02", "S03", "S04"])
    assert workflow.next("book-demo", "E001").status == "awaiting_script_review"
    workflow.approve("book-demo", "E001", "script")
    assert workflow.next("book-demo", "E001").status == "awaiting_h3_generation"
    workflow.import_video("book-demo", "E001", "S01", make_video(tmp_path, "S01"))
    assert workflow.status("book-demo", "E001").status == "visual_sample_ready"
    for shot in ("S02", "S03", "S04"):
        workflow.import_video("book-demo", "E001", shot, make_video(tmp_path, shot))
    assert workflow.next("book-demo", "E001").status == "awaiting_final_review"
~~~

- [ ] **Step 2: Run tests and confirm RED**

~~~powershell
uv run pytest tests/unit/test_orchestrator.py tests/integration/test_cli_value_workflow.py -v
~~~

- [ ] **Step 3: Implement explicit stage order and gates**

~~~python
STAGE_ORDER = [
    "parse_source", "analyze_chapters", "synthesize_book_value",
    "generate_episode_brief", "draft_script", "review_script",
    "tts", "asr", "subtitles", "storyboard", "await_h3_segments",
    "render", "qc",
]
GATES = {
    "review_script": "awaiting_script_review",
    "await_h3_segments": "awaiting_h3_generation",
    "qc": "awaiting_final_review",
}
~~~

Acquire an EpisodeLock, reload state, verify manifests, save after each stage, stop at the next gate, sanitize failures, and never skip blocking evidence/TTS/ASR/segment stages. Grok remains non-blocking; ASR and missing H3 segments block.

- [ ] **Step 4: Implement the nine public commands**

Keep the existing command set. Change only `import-video` arguments to `<book-id> <episode-id> <shot-id> <mp4>`. `open ... h3` opens the storyboard directory or manifest listing all prompt paths and import status. `status` shows required/present/missing shot IDs and the exact next command. `add` creates one `value_angle` Episode only after E001 exists and dedup passes.

- [ ] **Step 5: Run tests and commit**

~~~powershell
uv run pytest tests/unit/test_orchestrator.py tests/integration/test_cli_value_workflow.py -v
uv run bv --help
git add src/bv/workflow/stages.py src/bv/workflow/orchestrator.py src/bv/cli.py tests/fakes.py tests/unit/test_orchestrator.py tests/integration/test_cli_value_workflow.py
git commit -m "feat: orchestrate whole-book multi-clip workflow"
~~~

---

### Task 22: Add final review, operating guides, and automated end-to-end acceptance

**Files:**
- Create: `src/bv/content/final_review.py`
- Create: `docs/operations/first-run.md`
- Create: `docs/operations/private-acceptance.md`
- Modify: `README.md`
- Modify: `tests/fakes.py`
- Create: `tests/unit/test_final_review.py`
- Create: `tests/integration/test_end_to_end_value_video.py`

**Interfaces:**
- Produces: `build_final_review()`, deterministic fake-provider E2E, and private real-acceptance runbook.

- [ ] **Step 1: Write failing editorial and media checklist test**

~~~python
def test_final_review_checks_whole_book_value_and_multiclip_quality() -> None:
    markdown = build_final_review(make_review_inputs())
    for item in (
        "是否讲清整本书给目标读者的价值",
        "是否退化成一个孤立观点",
        "各片段人物、服装和环境是否连续",
        "有没有乱码、Logo或水印",
        "字幕是否被平台界面遮挡",
        "真实书封是否与商品版本一致",
        "发布时完成AI内容声明",
    ):
        assert item in markdown
~~~

- [ ] **Step 2: Write failing fake-provider E2E test**

The harness imports a generated TXT, injects validated chapter evidence and a whole-book value response, reaches script review, approves the script, injects a 52-second fake master timeline and ASR timings, creates four prompt manifests, imports S01 and verifies `visual_sample_ready`, imports S02–S04, renders a generated local fixture, reaches final review, and approves final. It makes no network/model call and asserts only E001 exists.

~~~python
def test_complete_value_video_pipeline_stops_at_all_gates(tmp_path: Path) -> None:
    harness = ValueVideoHarness(tmp_path, duration_ms=52000, shots=["S01", "S02", "S03", "S04"])
    book_id = harness.import_txt()
    assert harness.next(book_id).status == "awaiting_script_review"
    harness.approve_script(book_id)
    assert harness.next(book_id).status == "awaiting_h3_generation"
    harness.import_shot(book_id, "S01")
    assert harness.status(book_id).status == "visual_sample_ready"
    for shot in ("S02", "S03", "S04"):
        harness.import_shot(book_id, shot)
    assert harness.next(book_id).status == "awaiting_final_review"
    assert harness.approve_final(book_id).status == "final_approved"
~~~

- [ ] **Step 3: Run focused tests and confirm RED**

~~~powershell
uv run pytest tests/unit/test_final_review.py tests/integration/test_end_to_end_value_video.py -v
~~~

- [ ] **Step 4: Implement review and exact CMD guide**

The first-run guide must show `bv doctor`, import, script review/approval, H3 prompt opening, four explicit `import-video ... S0N` commands, final render/review, and approval. Mark `doctor --live`, Codex/Grok, and Volc calls as quota/network actions. Do not imply that S01 is a final video.

- [ ] **Step 5: Write private acceptance gates**

Require: private book untracked; authorized voice hash; real Codex analysis; Grok pass or visible skip; E001 whole-book value editorial approval; 45–60-second listened-to IndexTTS2 narration; Volc word timings and key terms; one S01 visual sample acceptance; all required H3 segment visual checks; multi-clip final QC; explicit final approval; clean Git secret/private-file audit. Do not run live/cloud/H3 actions without the user's explicit assets and authorization.

- [ ] **Step 6: Run complete automated verification and commit**

~~~powershell
uv run pytest -v
uv run pytest --cov=src/bv --cov-report=term-missing
uv run python -m compileall -q src tests
git diff --check
git status --short
git add src/bv/content/final_review.py README.md docs/operations/first-run.md docs/operations/private-acceptance.md tests/fakes.py tests/unit/test_final_review.py tests/integration/test_end_to_end_value_video.py
git commit -m "docs: add whole-book value video acceptance"
~~~

---

## Real acceptance sequence

Automated tests prove contracts, not editorial or visual quality. After Task 22 passes, run these gates in order and stop if a required asset or permission is missing:

1. `bv doctor` without `--live`.
2. User confirms credentials/quota before `bv doctor --live`.
3. One short authorized IndexTTS2 synthesis and Volc timing request.
4. One public-domain short book through parsing and evidence.
5. Private complete 《被讨厌的勇气》 through whole-book value synthesis and `awaiting_script_review`.
6. User edits and approves the 45–60-second E001 script.
7. Real complete IndexTTS2 narration, ASR, subtitles, storyboard, continuity bible, and prompt set.
8. Generate and manually approve S01 as the approximately 15-second visual sample.
9. Generate/import and manually inspect every remaining required H3 segment.
10. Render and technically inspect the 45–60-second final MP4.
11. User completes editorial, audio, subtitle, cover, continuity, and AI-artifact review; then approves final.
12. Confirm Git contains no ebook, voice, credential, log, H3 clip, or final media.
13. Only after E001 acceptance, run `bv add` once and verify E002 is a distinct `value_angle` rather than a repeated or isolated knowledge point.

## Plan self-review checklist

- Coverage: configuration migration, whole-book value, E001/later dedup, script, TTS, ASR, subtitles, multi-segment storyboard, H3 import, cover, render, QC, orchestration, docs, and acceptance all have task ownership.
- Duration consistency: 15 seconds appears only as an H3 segment/sample boundary; final duration is always 45–60 seconds.
- Content consistency: E001 is always `whole_book_value`; later Episodes are `value_angle`; no task asks for one isolated concept as E001.
- Timeline consistency: `voice_master.wav` drives ASR, subtitles, segment boundaries, and render.
- Completion consistency: S01 sets only `visual_sample_ready`; all required shot IDs are needed for `h3_imported` and render.
- Type consistency: `ClaimCluster`, `WholeBookValue`, `EpisodeBrief`, `H3Segment`, status names, and function signatures are reused unchanged.
- Security: no task logs credentials or commits private assets; model/cloud tests are opt-in.
- No placeholders: every task names files, interfaces, a failing test, minimal implementation, verification command, and commit.

## Execution checkpoints

- After old-plan Tasks 6–10: parser and evidence review.
- After Tasks 11–14: revised contract, whole-book value, Episode framing, and script review.
- After Tasks 15–18: complete narration, ASR, subtitles, and segmented H3 prompt review.
- After Tasks 19–21: multi-clip import, render/QC, and CLI workflow review.
- After Task 22: broad branch review, then user-controlled private acceptance.
