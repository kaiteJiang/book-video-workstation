# Handdrawn Visual Planning and Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Turn an approved whole-book value script and final audio timings into one explainable style choice, a continuity-locked illustration storyboard, and an approved local image set.

**Architecture:** Deterministic rules shortlist three upstream styles, Codex selects one through the existing StructuredModel contract, and a strict planner maps approved narration spans to gapless scenes. Image generation remains Codex-conversation-driven: the program prepares hashed jobs, validates imported local images, and blocks batch generation until three representative frames are explicitly approved.

**Tech Stack:** Python 3.11, Pydantic 2, existing Codex CLI adapter, pytest 8, SHA-256 file contracts, FFmpeg for local grayscale derivatives.

**Spec:** docs/superpowers/specs/2026-08-22-handdrawn-book-video-media-v0.2-design.md

## Global Constraints

- Requires completion of docs/superpowers/plans/2026-08-22-handdrawn-renderer-foundation-implementation.md.
- Select exactly one of the 20 vendored styles for an Episode; never blend style recipes automatically.
- Rank styles from the full approved script and Episode value contract, not from one sentence.
- Default narrative target is about 60% recurring protagonist scenes and 40% metaphor scenes; deviations require a stored reason.
- Generate opening first, then middle and ending representatives using the opening identity; present all three as one review batch.
- Do not generate remaining images until representative approval binds exact image, storyboard, style, and character hashes.
- Each scene generates one 1024×1024 color master with no text; derive black-and-white locally.
- No API key or automatic API fallback; Codex conversation produces the image masters.
- A changed style, character lock, representative image, or scene prompt invalidates only the required downstream assets.

## File Map

- src/bv/illustration/style_selector.py: profile extraction, deterministic ranking, and Codex adjudication.
- prompts/illustration/style_select.md: bounded style-selection prompt.
- src/bv/illustration/planner.py: cue grouping, absolute frame allocation, key lines, and narrative validation.
- prompts/illustration/storyboard.md: bounded visual-planning prompt.
- src/bv/illustration/prompts.py: image prompt and dependency compilation.
- src/bv/illustration/assets.py: job manifests, safe import, grayscale derivation, and hash checks.
- src/bv/illustration/reviews.py: representative review artifact and approval binding.
- src/bv/state/models.py: illustration statuses.
- src/bv/state/invalidation.py: illustration dependency propagation.

---

### Task 1: Implement explainable automatic style selection

**Files:**
- Create: prompts/illustration/style_select.md
- Create: src/bv/illustration/style_selector.py
- Test: tests/unit/test_style_selector.py

**Interfaces:**
- Consumes: approved script, ReaderValueContract-compatible data, vendored style library, StructuredModel, request root.
- Produces: StyleDecision and its canonical SHA-256.

- [ ] **Step 1: Write failing ranking and adjudication tests**

~~~python
def test_emotional_relationship_script_shortlists_watercolor_before_whiteboard() -> None:
    profile = NarrativeProfile(
        narrative_types=("关系", "感悟"),
        emotional_temperature="克制",
        abstraction="现实",
        relationships=("伴侣",),
        era="现代",
        target_reader="正在关系中内耗的成年人",
    )
    ranked = rank_style_candidates(profile, load_style_catalog(VENDOR_LIBRARY))
    assert [item.style_id for item in ranked[:3]][0] == "emotional-watercolor-sketch"
    assert "whiteboard-explainer" not in [item.style_id for item in ranked[:3]]

def test_model_can_choose_only_from_three_candidates(tmp_path: Path) -> None:
    model = ScriptedStructuredModel([{"selected_style": "kid-crayon"}])
    with pytest.raises(StyleSelectionError, match="style_not_in_candidates"):
        select_style(
            profile=profile(),
            approved_script=script(),
            value_contract=value_contract(),
            candidates=three_candidates_without("kid-crayon"),
            model=model,
            request_root=tmp_path,
            prompt_asset=style_prompt_asset(),
        )
~~~

- [ ] **Step 2: Run tests and verify missing selector**

Run: python -m pytest tests/unit/test_style_selector.py -q

Expected: FAIL because style_selector.py does not exist.

- [ ] **Step 3: Implement catalog loading and deterministic ranking**

Required signatures:

~~~python
class StyleReference(BaseModel):
    path: str
    role: str

class StyleRecipe(BaseModel):
    style_id: str
    order: int
    name_zh: str
    group: str
    best_for: tuple[str, ...]
    summary: str
    caption_prompt: str
    color_hint: str
    avoid: str
    reference_images: tuple[StyleReference, ...]

class NarrativeProfile(BaseModel):
    narrative_types: tuple[str, ...]
    emotional_temperature: Literal["温暖", "克制", "沉重", "轻快"]
    abstraction: Literal["现实", "混合", "隐喻"]
    relationships: tuple[str, ...]
    era: str
    target_reader: str

class RankedStyle(BaseModel):
    style_id: str
    order: int
    score: int
    matched_dimensions: tuple[str, ...]
    rejected_dimensions: tuple[str, ...]

def load_style_catalog(path: Path) -> tuple[StyleRecipe, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        StyleRecipe.model_validate(
            {
                "style_id": item["id"],
                "order": item["order"],
                "name_zh": item["name_zh"],
                "group": item["group"],
                "best_for": item["best_for"],
                "summary": item["summary"],
                "caption_prompt": item["caption_prompt"],
                "color_hint": item["color_hint"],
                "avoid": item["avoid"],
                "reference_images": item.get("reference_images", []),
            }
        )
        for item in payload["styles"]
    )

def build_narrative_profile(
    approved_script: str,
    value_contract: ReaderValueContract,
) -> NarrativeProfile:
    return NarrativeProfile(
        narrative_types=classify_narrative_types(approved_script, value_contract),
        emotional_temperature=classify_emotional_temperature(
            approved_script, value_contract
        ),
        abstraction=classify_abstraction(approved_script),
        relationships=classify_relationships(approved_script),
        era=classify_era(approved_script),
        target_reader=value_contract.target_reader,
    )

def rank_style_candidates(
    profile: NarrativeProfile,
    catalog: Sequence[StyleRecipe],
) -> tuple[RankedStyle, ...]:
    ranked = [score_style(profile, style) for style in catalog]
    return tuple(sorted(ranked, key=lambda item: (-item.score, item.order)))
~~~

Keep scoring data-driven and deterministic. A stable tie breaks by upstream style order, not dictionary iteration.

- [ ] **Step 4: Implement bounded Codex selection**

~~~python
def select_style(
    profile: NarrativeProfile,
    approved_script: str,
    value_contract: ReaderValueContract,
    candidates: tuple[RankedStyle, RankedStyle, RankedStyle],
    model: StructuredModel,
    request_root: Path,
    prompt_asset: PromptAsset,
    manual_override: str | None = None,
) -> StyleDecision:
    selected = complete_style_selection(
        profile=profile,
        approved_script=approved_script,
        value_contract=value_contract,
        candidates=candidates,
        model=model,
        request_root=request_root,
        prompt_asset=prompt_asset,
        manual_override=manual_override,
    )
    return validate_style_decision(selected, candidates=candidates)
~~~

The response schema accepts only one of the supplied IDs, nonempty selection reasons, a rejection reason for each alternative, confidence from 0 to 1, the script hash, style-library version, and style fingerprint. Manual override must name a real style and record manual_override=True.

- [ ] **Step 5: Run focused tests**

Run: python -m pytest tests/unit/test_style_selector.py -q

Expected: PASS.

- [ ] **Step 6: Commit style selection**

~~~powershell
git add -- prompts/illustration/style_select.md src/bv/illustration/style_selector.py tests/unit/test_style_selector.py
git commit -m "feat: select one explainable handdrawn style"
~~~

### Task 2: Build the audio-timed illustration planner

**Files:**
- Create: prompts/illustration/storyboard.md
- Create: src/bv/illustration/planner.py
- Test: tests/unit/test_illustration_planner.py

**Interfaces:**
- Consumes: approved text, AlignedScript, SubtitleCue sequence, StyleDecision, semantic lock, and StructuredModel.
- Produces: CharacterLock and IllustrationStoryboard with 3 scenes for a 15-second sample or 8–12 scenes for a 45–60-second production narration.

- [ ] **Step 1: Write failing frame and narrative tests**

~~~python
def test_fifteen_second_plan_has_three_gapless_scenes() -> None:
    _, storyboard = plan_illustrations(**sample_inputs(duration_ms=15_000))
    assert len(storyboard.scenes) == 3
    assert storyboard.scenes[0].from_frame == 0
    assert storyboard.scenes[-1].to_frame == 450
    assert all(
        left.to_frame == right.from_frame
        for left, right in pairwise(storyboard.scenes)
    )

def test_key_lines_are_short_and_not_full_subtitles() -> None:
    _, storyboard = plan_illustrations(**sample_inputs())
    assert all(6 <= display_width(scene.key_line) <= 14 for scene in storyboard.scenes)
    assert all(scene.key_line != scene.narration for scene in storyboard.scenes)
~~~

- [ ] **Step 2: Run tests and verify missing planner**

Run: python -m pytest tests/unit/test_illustration_planner.py -q

Expected: FAIL because planner.py does not exist.

- [ ] **Step 3: Implement deterministic cue grouping and frame allocation**

~~~python
def partition_illustration_timeline(
    cues: Sequence[SubtitleCue],
    *,
    master_duration_ms: int,
    fps: int = 30,
) -> tuple[SceneWindow, ...]:
    target_count = 3 if master_duration_ms <= 20_000 else choose_scene_count(
        master_duration_ms, minimum=8, maximum=12
    )
    boundaries = choose_cue_boundaries(
        cues, target_count=target_count, master_duration_ms=master_duration_ms
    )
    return allocate_gapless_frames(
        boundaries, master_duration_ms=master_duration_ms, fps=fps
    )
~~~

Choose 3 scenes up to 20 seconds and 8–12 scenes from 45–60 seconds. Cut only at cue boundaries, prefer semantic punctuation and pauses, keep scenes generally 4–7 seconds, and assign all rounding remainder to the final scene so total frames stay exact.

- [ ] **Step 4: Implement constrained visual planning**

~~~python
def plan_illustrations(
    *,
    approved_text: str,
    aligned_script: AlignedScript,
    cues: Sequence[SubtitleCue],
    master_duration_ms: int,
    semantic_lock: SemanticLock,
    style_decision: StyleDecision,
    model: StructuredModel,
    request_root: Path,
    prompt_asset: PromptAsset,
) -> tuple[CharacterLock, IllustrationStoryboard]:
    windows = partition_illustration_timeline(
        cues, master_duration_ms=master_duration_ms
    )
    draft = complete_visual_plan(
        approved_text=approved_text,
        windows=windows,
        semantic_lock=semantic_lock,
        style_decision=style_decision,
        model=model,
        request_root=request_root,
        prompt_asset=prompt_asset,
    )
    return validate_visual_plan(
        draft,
        approved_text=approved_text,
        aligned_script=aligned_script,
        windows=windows,
        style_decision=style_decision,
    )
~~~

The model fills visual purpose, setting, character action, optional metaphor, composition, character references, and 6–14 character key line. Validation reconstructs exact approved narration spans, enforces one style, requires start/middle/end representative flags, prohibits prompt-injection strings and画内文字 requests, and checks the protagonist/metaphor ratio or a nonempty deviation_reason.

- [ ] **Step 5: Run focused and existing subtitle/storyboard regression tests**

Run: python -m pytest tests/unit/test_illustration_planner.py tests/unit/test_subtitles.py tests/unit/test_storyboard_segments.py -q

Expected: PASS.

- [ ] **Step 6: Commit the planner**

~~~powershell
git add -- prompts/illustration/storyboard.md src/bv/illustration/planner.py tests/unit/test_illustration_planner.py
git commit -m "feat: plan audio-timed handdrawn scenes"
~~~

### Task 3: Compile Codex image jobs and validate local assets

**Files:**
- Create: src/bv/illustration/prompts.py
- Create: src/bv/illustration/assets.py
- Test: tests/unit/test_illustration_assets.py
- Test: tests/integration/test_illustration_asset_pipeline.py

**Interfaces:**
- Consumes: IllustrationStoryboard, CharacterLock, vendored style recipe, Episode root.
- Produces: IllustrationManifest, ImageJob sequence, imported color masters, and local black-and-white derivatives.

- [ ] **Step 1: Write failing dependency and import tests**

~~~python
def test_representative_jobs_generate_opening_before_middle_and_end(tmp_path: Path) -> None:
    manifest = prepare_image_jobs(storyboard(), character_lock(), tmp_path)
    reps = [job for job in manifest.jobs if job.representative]
    assert reps[0].scene_id == "S01"
    assert reps[0].reference_scene_ids == ()
    assert reps[1].reference_scene_ids == ("S01",)
    assert reps[2].reference_scene_ids == ("S01",)

def test_import_rejects_wrong_dimensions(tmp_path: Path) -> None:
    with pytest.raises(IllustrationAssetError, match="image_contract_invalid"):
        import_image_master(job(), wrong_size_png(tmp_path), inspector=fake_inspector)
~~~

- [ ] **Step 2: Run tests and verify missing modules**

Run: python -m pytest tests/unit/test_illustration_assets.py -q

Expected: FAIL because prompts.py and assets.py do not exist.

- [ ] **Step 3: Implement prompt compilation**

~~~python
class CompiledImagePrompt(BaseModel):
    scene_id: str
    prompt: str
    prompt_sha256: str
    style_fingerprint: str
    character_lock_sha256: str
    reference_image_sha256s: tuple[str, ...]

def compile_image_prompt(
    scene: IllustrationScene,
    *,
    style: StyleRecipe,
    character_lock: CharacterLock,
) -> CompiledImagePrompt:
    prompt = render_image_prompt(scene, style=style, character_lock=character_lock)
    return CompiledImagePrompt(
        scene_id=scene.scene_id,
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        style_fingerprint=character_lock.style_fingerprint,
        character_lock_sha256=canonical_model_sha256(character_lock),
        reference_image_sha256s=(),
    )
~~~

Every prompt must request one 1024×1024 illustration, exact selected style, safe-contained figures, no text/letters/numbers/book-cover/Logo/watermark, and only the characters required by the current scene. Prompt SHA-256 includes the style fingerprint, character-lock hash, and referenced image hashes.

- [ ] **Step 4: Implement manifests and safe import**

~~~python
class ImageFacts(BaseModel):
    width: int
    height: int
    format: Literal["png", "jpeg"]

class ImageJob(BaseModel):
    scene_id: str
    representative: bool
    prompt: str
    prompt_sha256: str
    reference_scene_ids: tuple[str, ...]
    reference_image_sha256s: tuple[str, ...]
    output_master: Path
    output_bw: Path
    master_sha256: str | None = None
    bw_sha256: str | None = None
    status: Literal["planned", "generated", "approved"]

class IllustrationManifest(BaseModel):
    book_id: str
    episode_id: str
    storyboard_sha256: str
    style_fingerprint: str
    character_lock_sha256: str
    jobs: tuple[ImageJob, ...]

class ImportedImage(BaseModel):
    scene_id: str
    master_path: Path
    master_sha256: str
    bw_path: Path
    bw_sha256: str
    width: Literal[1024]
    height: Literal[1024]

def prepare_image_jobs(
    storyboard: IllustrationStoryboard,
    character_lock: CharacterLock,
    style: StyleRecipe,
    style_decision: StyleDecision,
    episode_root: Path,
) -> IllustrationManifest:
    jobs = compile_ordered_jobs(
        storyboard, character_lock=character_lock, style=style, episode_root=episode_root
    )
    return IllustrationManifest(
        book_id=storyboard.book_id,
        episode_id=storyboard.episode_id,
        storyboard_sha256=canonical_model_sha256(storyboard),
        style_fingerprint=style_decision.style_fingerprint,
        character_lock_sha256=canonical_model_sha256(character_lock),
        jobs=jobs,
    )

def import_image_master(
    job: ImageJob,
    source_path: Path,
    *,
    inspector: Callable[[Path], ImageFacts],
    ffmpeg_command: str | Path,
) -> ImportedImage:
    facts = inspector(source_path)
    if (facts.width, facts.height) != (1024, 1024):
        raise IllustrationAssetError("image_contract_invalid")
    return copy_validate_and_derive_bw(
        job,
        source_path,
        facts=facts,
        ffmpeg_command=ffmpeg_command,
    )
~~~

Copy through an Episode-local temporary file, verify regular non-reparse input, exact 1024×1024 dimensions, supported PNG/JPEG, expected job identity, and no existing destination. Derive Sxx_bw.png locally only after the color master is published. Do not claim automated semantic/aesthetic approval.

- [ ] **Step 5: Run unit and real local derivative tests**

Run: python -m pytest tests/unit/test_illustration_assets.py tests/integration/test_illustration_asset_pipeline.py -q

Expected: PASS; integration uses generated fixture PNGs and real FFmpeg grayscale output.

- [ ] **Step 6: Commit image jobs and import**

~~~powershell
git add -- src/bv/illustration/prompts.py src/bv/illustration/assets.py tests/unit/test_illustration_assets.py tests/integration/test_illustration_asset_pipeline.py
git commit -m "feat: prepare and import codex illustration jobs"
~~~

### Task 4: Add representative review and visual invalidation

**Files:**
- Create: src/bv/illustration/reviews.py
- Modify: src/bv/state/models.py
- Modify: src/bv/state/invalidation.py
- Test: tests/unit/test_representative_review.py
- Test: tests/unit/test_invalidation.py
- Test: tests/integration/test_representative_gate_pipeline.py

**Interfaces:**
- Consumes: three imported representative images plus storyboard/style/character hashes.
- Produces: RepresentativeReview and RepresentativeApproval; unlocks nonrepresentative ImageJobs only after exact binding.

- [ ] **Step 1: Write failing gate and stale-approval tests**

~~~python
def test_batch_jobs_remain_locked_without_exact_representative_approval() -> None:
    with pytest.raises(RepresentativeReviewError, match="representatives_not_approved"):
        unlock_batch_jobs(manifest(), approval=None)

def test_style_change_invalidates_representatives_and_all_visuals() -> None:
    episode = completed_visual_episode()
    invalidate_from(episode, "style_decision")
    assert "representative_review" in episode.stale_stages
    assert "illustration_images" in episode.stale_stages
    assert "visual_render" in episode.stale_stages
    assert "render" in episode.stale_stages
~~~

- [ ] **Step 2: Run tests and verify the gate is absent**

Run: python -m pytest tests/unit/test_representative_review.py tests/unit/test_invalidation.py -q

Expected: FAIL because review contracts and new dependency names do not exist.

- [ ] **Step 3: Implement review artifact and approval binding**

~~~python
class RepresentativeReview(BaseModel):
    scene_ids: tuple[str, str, str]
    image_sha256s: tuple[str, str, str]
    storyboard_sha256: str
    style_fingerprint: str
    character_lock_sha256: str

class RepresentativeApproval(BaseModel):
    review_sha256: str
    image_sha256s: tuple[str, str, str]
    storyboard_sha256: str
    style_fingerprint: str
    character_lock_sha256: str
    reviewer: Literal["user"]
    approved_at: datetime

def build_representative_review(
    storyboard: IllustrationStoryboard,
    manifest: IllustrationManifest,
) -> RepresentativeReview:
    representatives = approved_representative_jobs(manifest)
    return RepresentativeReview(
        scene_ids=tuple(job.scene_id for job in representatives),
        image_sha256s=tuple(job.master_sha256 for job in representatives),
        storyboard_sha256=manifest.storyboard_sha256,
        style_fingerprint=manifest.style_fingerprint,
        character_lock_sha256=manifest.character_lock_sha256,
    )

def approve_representatives(
    review: RepresentativeReview,
    *,
    approved_image_sha256s: tuple[str, str, str],
    reviewer: Literal["user"],
) -> RepresentativeApproval:
    require_exact_representative_hashes(review, approved_image_sha256s)
    return bind_representative_approval(review, reviewer=reviewer)

def unlock_batch_jobs(
    manifest: IllustrationManifest,
    approval: RepresentativeApproval,
) -> tuple[ImageJob, ...]:
    require_current_representative_approval(manifest, approval)
    return tuple(job for job in manifest.jobs if not job.representative)
~~~

Approval stores exactly three image hashes, storyboard hash, style fingerprint, character-lock hash, timestamp, and reviewer. Any mismatch raises representative_approval_stale.

- [ ] **Step 4: Extend state and invalidation without breaking old H3 histories**

Add the illustration statuses from the spec. Add dependency names style_decision, illustration_plan, representative_review, illustration_images, and visual_render. Keep legacy H3 status migrations and tests unchanged.

- [ ] **Step 5: Run focused and legacy state tests**

Run: python -m pytest tests/unit/test_representative_review.py tests/unit/test_invalidation.py tests/unit/test_state_store.py tests/unit/test_orchestrator.py tests/integration/test_representative_gate_pipeline.py -q

Expected: PASS, including legacy video-state tests.

- [ ] **Step 6: Commit the review gate**

~~~powershell
git add -- src/bv/illustration/reviews.py src/bv/state/models.py src/bv/state/invalidation.py tests/unit/test_representative_review.py tests/unit/test_invalidation.py tests/integration/test_representative_gate_pipeline.py
git commit -m "feat: gate illustration batches on representative review"
~~~
