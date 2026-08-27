# Handdrawn Media Runtime and Final Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Wire the approved script through real local narration, ASR timing, hand-drawn visuals, FFmpeg composition, and final human review without adding a user-facing CLI.

**Architecture:** Focused stage runners reuse existing IndexTTS2, audio processing, Volc ASR, alignment, subtitle, render, and QC functions. MediaProductionService coordinates durable stages and approvals through StateStore; Codex calls this internal service during conversation, while external operations remain fail-closed behind explicit authorization.

**Tech Stack:** Python 3.11, existing StageRunner/StateStore contracts, IndexTTS2, Volc ASR, FFmpeg/FFprobe, Remotion adapter from the foundation plan, pytest 8.

**Spec:** docs/superpowers/specs/2026-08-22-handdrawn-book-video-media-v0.2-design.md

**Implementation status (2026-08-22):** Tasks 1–3 and Task 4 Steps 1–5 are implemented and covered by automated tests. Task 4 Steps 6–7 intentionally remain pending until the user provides the first real book and approves each media gate.

## Global Constraints

- Requires completion of both prior 2026-08-22 hand-drawn implementation plans.
- Keep approved script text immutable through TTS, ASR, subtitles, and illustration planning.
- Logical voice profile is 黑金3; resolve actual reference path and IndexTTS2 settings locally without logging private paths or audio.
- The final WAV is the sole authority for master_duration_ms.
- Technical-sample mode uses one contiguous, unmodified span from the approved script and a 10–20 second duration policy; full mode retains the existing 45–60 second policy.
- External ASR and Codex model calls require explicit authorization; representative approval authorizes only the remaining image jobs for that Episode.
- Final output is 1080×1920 H.264/AAC with burned ASS subtitles and a user-provided real book cover.
- Audio/visual duration mismatch above one frame fails closed.
- Do not add a user-facing CLI, web UI, GUI, BGM generator, or Douyin uploader.
- Do not delete old H3/Grok adapters; exclude them from the V0.2 default media runtime.

## File Map

- src/bv/workflow/media_stages.py: TTS, ASR, subtitle, style, planning, image, and visual-render runners.
- src/bv/workflow/media_runtime.py: conversation-facing MediaProductionService and bindings.
- src/bv/workflow/stages.py: provider-neutral media stage order and representative gate name.
- src/bv/workflow/runtime.py: optional construction of media bindings after script approval.
- src/bv/video/render.py: accept one validated silent hand-drawn visual segment without H3 assumptions.
- src/bv/video/qc.py: exact one-frame duration and final technical facts.
- docs/runbooks/handdrawn-codex-production.md: conversation runbook and acceptance evidence.

---

### Task 1: Implement durable TTS, ASR, and subtitle stages

**Files:**
- Create: src/bv/workflow/media_stages.py
- Modify: src/bv/voice/processing.py
- Modify: src/bv/config.py
- Modify: config.example.yaml
- Test: tests/unit/test_media_audio_stages.py
- Test: tests/integration/test_audio_processing.py
- Test: tests/integration/test_media_audio_pipeline.py

**Interfaces:**
- Consumes: script.approved.json, approved script SHA-256, configured 黑金3 reference voice, RuntimeAuthorization, and subtitle font.
- Produces: voice_master.wav, voice_manifest.json, alignment.json, subtitles.srt, subtitles.ass, and subtitle_manifest.json.

- [ ] **Step 1: Write failing stage tests**

~~~python
def test_tts_stage_publishes_only_verified_master_and_manifest(tmp_path: Path) -> None:
    outcome = TtsStage(fake_dependencies()).run(stage_context(tmp_path))
    assert set(outcome.outputs) == {"voice_master", "voice_manifest"}
    assert outcome.inputs["script_sha256"] == "a" * 64

def test_asr_stage_requires_external_authorization(tmp_path: Path) -> None:
    stage = AsrStage(fake_dependencies(), authorization=RuntimeAuthorization())
    with pytest.raises(ExternalAuthorizationError, match="external_not_authorized"):
        stage.run(stage_context(tmp_path))

def test_technical_sample_policy_accepts_fifteen_seconds_without_weakening_full_mode() -> None:
    assert classify_narration_duration(15.0, policy=TECHNICAL_SAMPLE_POLICY).level == "pass"
    assert classify_narration_duration(15.0, policy=FULL_EPISODE_POLICY).level == "fail"
~~~

- [ ] **Step 2: Run tests and verify missing stages**

Run: python -m pytest tests/unit/test_media_audio_stages.py -q

Expected: FAIL because media_stages.py does not exist.

- [ ] **Step 3: Implement TtsStage**

~~~python
class TtsStage:
    def run(self, context: StageContext) -> StageOutcome:
        script_path, script_sha256 = approved_narration_input(
            context,
            mode=self.mode,
        )
        raw_path = context.episode_root / "media/voice/voice_raw.wav"
        synthesized = synthesize_voice(
            episode_root=context.episode_root,
            install_dir=self.config.install_dir,
            model_dir=self.config.model_dir,
            script_path=script_path,
            approved_script_sha256=script_sha256,
            reference_voice_path=self.reference_voice_path,
            raw_output_path=raw_path,
            uv_command=self.config.uv_command,
            device=self.config.device,
        )
        master_path = context.episode_root / "media/voice/voice_master.wav"
        manifest = build_voice_master(
            episode_root=context.episode_root,
            raw_path=synthesized.raw_path,
            master_path=master_path,
            script_path=script_path,
            reference_voice_path=self.reference_voice_path,
            approved_script_sha256=script_sha256,
            voice_id=self.config.voice_id,
            ffmpeg_command=self.ffmpeg_command,
            duration_policy=policy_for(self.mode),
        )
        return StageOutcome(
            outputs={
                "voice_master": master_path,
                "voice_manifest": master_path.with_suffix(".json"),
            },
            inputs={"script_sha256": manifest.script_sha256},
        )
~~~

Use existing synthesize_voice and build_voice_master functions. Add the following policy contract to voice.processing; keep FULL_EPISODE_POLICY as the default so existing callers and tests retain 45–60 second behavior:

~~~python
@dataclass(frozen=True, slots=True)
class NarrationDurationPolicy:
    hard_min_seconds: float
    ideal_min_seconds: float
    ideal_max_seconds: float
    hard_max_seconds: float

TECHNICAL_SAMPLE_POLICY = NarrationDurationPolicy(10.0, 13.0, 17.0, 20.0)
FULL_EPISODE_POLICY = NarrationDurationPolicy(45.0, 48.0, 56.0, 60.0)

def classify_narration_duration(
    seconds: float,
    *,
    policy: NarrationDurationPolicy = FULL_EPISODE_POLICY,
) -> NarrationDuration:
    if seconds < policy.hard_min_seconds:
        return NarrationDuration(seconds=seconds, level="fail", band="too_short")
    if seconds < policy.ideal_min_seconds:
        return NarrationDuration(
            seconds=seconds, level="pass", band="edge_short",
            warnings=["edge_short_duration"],
        )
    if seconds <= policy.ideal_max_seconds:
        return NarrationDuration(seconds=seconds, level="pass", band="ideal")
    if seconds <= policy.hard_max_seconds:
        return NarrationDuration(
            seconds=seconds, level="pass", band="edge_long",
            warnings=["edge_long_duration"],
        )
    return NarrationDuration(seconds=seconds, level="fail", band="too_long")
~~~

Technical-sample mode writes sample_projection.json containing the exact source character span and its SHA-256; it never rewrites the approved text. Set config.example.yaml to indextts2.voice_id: 黑金3 and require the user's local voice_path to resolve to that authorized reference. Do not accept a process exit without a valid WAV and matching manifest.

- [ ] **Step 4: Implement AsrStage and SubtitleStage**

AsrStage invokes recognize_flash through invoke_external, then align_approved_text. It writes only the sanitized aligned contract and pronunciation report. SubtitleStage calls build_cues, render_srt, render_ass, and build_subtitle_manifest with the final audio hash and exact 1080×1920 frame size.

- [ ] **Step 5: Run unit and integration tests**

Run: python -m pytest tests/unit/test_media_audio_stages.py tests/unit/test_indextts2.py tests/unit/test_alignment.py tests/unit/test_subtitles.py tests/integration/test_audio_processing.py tests/integration/test_media_audio_pipeline.py -q

Expected: PASS using fakes for providers and real local WAV/FFmpeg processing in integration.

- [ ] **Step 6: Commit audio stages**

~~~powershell
git add -- src/bv/workflow/media_stages.py src/bv/voice/processing.py src/bv/config.py config.example.yaml tests/unit/test_media_audio_stages.py tests/integration/test_audio_processing.py tests/integration/test_media_audio_pipeline.py
git commit -m "feat: add durable narration and subtitle stages"
~~~

### Task 2: Implement the conversation-facing media service and representative gate

**Files:**
- Create: src/bv/workflow/media_runtime.py
- Modify: src/bv/workflow/stages.py
- Modify: src/bv/workflow/runtime.py
- Test: tests/unit/test_media_runtime.py
- Test: tests/integration/test_handdrawn_media_workflow.py

**Interfaces:**
- Consumes: StateStore, RuntimeBindings-compatible stages, illustration import service, and explicit authorizations.
- Produces: MediaProductionView and methods prepare(), import_image(), approve_representatives(), render(), approve_final().

- [ ] **Step 1: Write failing state-transition tests**

~~~python
def test_prepare_stops_at_three_representatives(tmp_path: Path) -> None:
    service = media_service(tmp_path)
    view = service.prepare("book-demo", "E001", mode="full")
    assert view.status == "representative_generation_running"
    for scene_id in view.missing_scene_ids:
        view = service.import_image(
            "book-demo", "E001", scene_id, generated_fixture(scene_id)
        )
    assert view.status == "awaiting_representative_review"
    assert view.present_scene_ids == ("S01", "S05", "S10")
    assert view.batch_unlocked is False

def test_approval_unlocks_only_current_manifest(tmp_path: Path) -> None:
    service = prepared_service(tmp_path)
    approval = service.approve_representatives("book-demo", "E001")
    assert approval.status == "representatives_approved"
    mutate_style_decision(service)
    with pytest.raises(MediaWorkflowError, match="representative_approval_stale"):
        service.prepare_batch("book-demo", "E001")
~~~

- [ ] **Step 2: Run tests and verify missing service**

Run: python -m pytest tests/unit/test_media_runtime.py -q

Expected: FAIL because MediaProductionService does not exist.

- [ ] **Step 3: Define the service interface**

~~~python
class MediaProductionView(BaseModel):
    book_id: str
    episode_id: str
    status: str
    next_action: str
    present_scene_ids: tuple[str, ...] = ()
    missing_scene_ids: tuple[str, ...] = ()
    batch_unlocked: bool = False

class MediaProductionService:
    def prepare(
        self,
        book_id: str,
        episode_id: str,
        *,
        mode: Literal["technical_sample", "full"],
    ) -> MediaProductionView:
        return self._advance_until_gate(book_id, episode_id, mode=mode)
    def import_image(
        self, book_id: str, episode_id: str, scene_id: str, source: Path
    ) -> MediaProductionView:
        self._images.import_master(book_id, episode_id, scene_id, source)
        return self.status(book_id, episode_id)
    def approve_representatives(
        self, book_id: str, episode_id: str
    ) -> MediaProductionView:
        self._approve_representative_gate(book_id, episode_id)
        return self.status(book_id, episode_id)
    def prepare_batch(self, book_id: str, episode_id: str) -> MediaProductionView:
        return self._unlock_current_batch(book_id, episode_id)
    def render(self, book_id: str, episode_id: str) -> MediaProductionView:
        return self._render_current_episode(book_id, episode_id)
    def approve_final(self, book_id: str, episode_id: str) -> MediaProductionView:
        return self._approve_final_gate(book_id, episode_id)

@dataclass(frozen=True, slots=True)
class MediaRuntimeBindings:
    service: MediaProductionService
    stages: Mapping[str, StageRunner]
    gate_approvers: Mapping[str, StageRunner]
~~~

Methods return next_action text for Codex, not shell commands. Every transition writes StageManifest data through StateStore and verifies artifact existence/hash before trusting completed status.

- [ ] **Step 4: Build media bindings without changing the default content boundary**

Add build_media_runtime_bindings(config: AppConfig, authorization: RuntimeAuthorization, store: StateStore, codex: StructuredModel) -> MediaRuntimeBindings separately from build_runtime_bindings. The existing content runtime still stops at script approval unless Codex explicitly constructs MediaProductionService. Do not add a new CLI command.

- [ ] **Step 5: Update stage names and preserve legacy migrations**

Add select_style, plan_illustrations, prepare_representatives, review_representatives, await_illustrations, visual_render, render, and qc. Keep old neutral video stage migration readable. The new service never asks VideoGateway for hand-drawn assets.

- [ ] **Step 6: Run focused and legacy orchestration tests**

Run: python -m pytest tests/unit/test_media_runtime.py tests/unit/test_orchestrator.py tests/unit/test_state_store.py tests/integration/test_handdrawn_media_workflow.py tests/integration/test_cli_value_workflow.py -q

Expected: PASS; existing CLI content behavior remains unchanged.

- [ ] **Step 7: Commit media runtime**

~~~powershell
git add -- src/bv/workflow/media_runtime.py src/bv/workflow/stages.py src/bv/workflow/runtime.py tests/unit/test_media_runtime.py tests/integration/test_handdrawn_media_workflow.py
git commit -m "feat: orchestrate handdrawn media from codex"
~~~

### Task 3: Adapt final rendering and QC to one silent visual track

**Files:**
- Modify: src/bv/video/render.py
- Modify: src/bv/video/qc.py
- Test: tests/unit/test_handdrawn_final_render.py
- Test: tests/integration/test_handdrawn_final_render_pipeline.py

**Interfaces:**
- Consumes: SilentRenderResult, VoiceManifest, SubtitleManifest/cues, ASS file, and approved real CoverOverlay.
- Produces: final.mp4, RenderManifest, and VideoQualityReport.

- [ ] **Step 1: Write failing one-track render tests**

~~~python
def test_handdrawn_render_uses_silent_track_without_concat(tmp_path: Path) -> None:
    argv = build_handdrawn_final_argv(final_inputs(tmp_path), temporary_path(tmp_path))
    assert "concat" not in " ".join(argv)
    assert "-map" in argv
    assert argv.count("-i") == 3

def test_qc_rejects_more_than_one_frame_duration_error(tmp_path: Path) -> None:
    facts = final_facts(duration_ms=10_035)
    with pytest.raises(QCError, match="final_duration_mismatch"):
        validate_handdrawn_contract(facts, final_inputs(duration_ms=10_000))
~~~

- [ ] **Step 2: Run tests and verify missing one-track path**

Run: python -m pytest tests/unit/test_handdrawn_final_render.py -q

Expected: FAIL because the hand-drawn final-render functions do not exist.

- [ ] **Step 3: Add a focused render request**

~~~python
class HanddrawnFinalInputs(BaseModel):
    book_id: str
    episode_id: str
    episode_root: Path
    render_root: Path
    visual: SilentRenderResult
    voice_master_path: Path
    voice_manifest: VoiceManifest
    ass_path: Path
    subtitle_manifest: SubtitleManifest
    subtitle_cues: tuple[SubtitleCue, ...]
    cover: CoverOverlay
    master_duration_ms: int
    script_sha256: str
    audio_sha256: str
    subtitle_sha256: str
    storyboard_sha256: str
    style_fingerprint: str
    representative_approval_sha256: str
    final_path: Path

def render_handdrawn_final(
    inputs: HanddrawnFinalInputs,
    *,
    ffmpeg_command: str | Path,
    runner: Runner,
    inspector: Inspector,
) -> RenderResult:
    preflight = preflight_handdrawn_final(inputs)
    temporary = create_owned_temporary_output(inputs.final_path)
    run_checked(
        build_handdrawn_final_argv(inputs, temporary),
        cwd=inputs.final_path.parent,
        runner=runner,
        error_code="final_render_failed",
    )
    report = inspector(temporary, inputs=inputs)
    require_current_inputs(inputs, preflight)
    return publish_verified_final(temporary, inputs=inputs, report=report)
~~~

Reuse existing path, hash, cover, voice, subtitle, atomic publication, and cleanup helpers. Do not route the visual track through H3 segment normalization or concat.

- [ ] **Step 4: Enforce exact final QC**

Require 1080×1920, 30 fps, H.264, AAC audio, no rotation, current hashes, full subtitle coverage, and abs(round(actual_duration_ms × 30 / 1000) - expected_total_frames) <= 1. Report aesthetic_review_required=True and never claim character/style quality automatically.

- [ ] **Step 5: Run unit, existing render regression, and real FFmpeg integration**

Run: python -m pytest tests/unit/test_handdrawn_final_render.py tests/unit/test_multiclip_render.py tests/integration/test_handdrawn_final_render_pipeline.py tests/integration/test_multiclip_render_pipeline.py -q

Expected: PASS for both new one-track and existing legacy multi-clip paths.

- [ ] **Step 6: Commit final render support**

~~~powershell
git add -- src/bv/video/render.py src/bv/video/qc.py tests/unit/test_handdrawn_final_render.py tests/integration/test_handdrawn_final_render_pipeline.py
git commit -m "feat: compose handdrawn visual track with narration"
~~~

### Task 4: Add acceptance runbook and execute the two real gates

**Files:**
- Create: docs/runbooks/handdrawn-codex-production.md
- Create during execution only: workspace/books/<book_id>/episodes/E001/media/
- Test: tests/integration/test_handdrawn_end_to_end_fake.py

**Interfaces:**
- Consumes: completed media runtime and the user's first real book.
- Produces: a repeatable Codex conversation runbook, a 15-second technical sample, and after approval one 45–60-second final.mp4.

- [ ] **Step 1: Write the fake end-to-end acceptance test**

The test must run approved script → fixture WAV → fake ASR → style decision → three representative fixture images → approval → remaining images → real Remotion silent render → real FFmpeg final render → QC.

Run: python -m pytest tests/integration/test_handdrawn_end_to_end_fake.py -q

Expected before wiring: FAIL at the first missing media-stage binding.

- [ ] **Step 2: Complete the runbook with exact conversation checkpoints**

The runbook must say:

1. User gives book title/author or file.
2. Codex produces and asks approval for the whole-book value script.
3. “开始制作” authorizes TTS, ASR, planning, and only three representative images.
4. Codex displays opening/middle/ending representatives and asks approval.
5. Approval authorizes remaining image jobs.
6. Codex renders and displays the 15-second technical sample.
7. After technical approval, Codex renders the full 45–60-second Episode.
8. Codex reports actual artifacts, hashes, FFprobe facts, and remaining aesthetic risks.
9. No automatic Douyin upload occurs.

- [ ] **Step 3: Run the fake full pipeline**

Run: python -m pytest tests/integration/test_handdrawn_end_to_end_fake.py -q

Expected: PASS with a real local MP4 and fake external providers only.

- [ ] **Step 4: Run the complete automated suite before real media**

Run: python -m pytest tests/unit -q

Expected: all unit tests PASS; only documented Windows symlink/reparse skips are allowed.

Run: python -m pytest tests/integration -q

Expected: all integration tests PASS.

Run: python -m compileall -q src tests

Expected: exit 0.

Run: git diff --check

Expected: no output and exit 0.

- [ ] **Step 5: Commit runbook and fake acceptance**

~~~powershell
git add -- docs/runbooks/handdrawn-codex-production.md tests/integration/test_handdrawn_end_to_end_fake.py
git commit -m "docs: add codex handdrawn production runbook"
~~~

- [ ] **Step 6: Produce the real 15-second technical sample**

Wait for the user to provide the first book. Use real 黑金3 IndexTTS2 audio, real Volc ASR, and three real Codex illustrations. Record the representative approval, render 1080×1920, and require:

~~~text
H.264 video, 1080×1920, 30 fps
AAC narration derived from the approved script
three hand-drawn scenes
no black bars, false cover, Logo, watermark, or image text
subtitle maximum two lines
audio/video duration difference <= 1 frame
render and QC manifests present
~~~

- [ ] **Step 7: Produce and review the first 45–60-second final**

Only after the technical sample passes, generate the full image batch, render final.mp4, inspect representative frames and the complete video, and ask the user for final approval. Do not mark V0.2 complete before that approval.
