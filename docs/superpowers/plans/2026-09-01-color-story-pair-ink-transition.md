# Color Story-Pair Ink Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace new book-video scenes' monochrome-to-color reveal with 3–4 pairs of full-color anchor and continuation images, revealed at an ASR-aligned semantic turn through a 45-frame multi-bloom ink mask.

**Architecture:** Add an explicit visual sequence mode and pair fields to the existing storyboard without rewriting legacy Episodes. The planner derives one semantic turn from approved narration and final-audio character timing; the asset layer creates ordered A/B jobs with an exact A-hash dependency; Remotion renders color A first and reveals color B through the existing soft radial mask. Existing `bw/color` storyboards remain readable and render through the legacy branch.

**Tech Stack:** Python 3.11, Pydantic 2, pytest 8, SHA-256 manifests, FFmpeg/ffprobe, TypeScript, React, Remotion 4, Node.js validator scripts.

**Spec:** `docs/superpowers/specs/2026-09-01-color-story-pair-ink-transition-design.md`

## Global Constraints

- New scenes use `color-story-pair`; a missing sequence field remains `legacy-monochrome-reveal`.
- New Episodes contain 3 or 4 scene pairs, defaulting to 4 for title-and-author production.
- Every new scene owns exactly two full-color images: `anchor` A and `continuation` B.
- B must bind the exact current A SHA-256 and use A as its first image reference.
- A is visible from local frame 0; B starts at the approved semantic turn and uses a 45-frame, five-or-more-bloom soft ink reveal.
- Cross-scene transition remains a 15-frame cross-dissolve from complete B to the next complete A.
- Title, author, Microsoft YaHei subtitles, audio clock, cover, QC, delivery, and final approval gates remain unchanged.
- Do not modify approved Episode assets, delivery packages, or final approval records.
- The worktree is already dirty. Before each edit, inspect the existing diff for that file; use `git add -p -- <path>` or a path-limited patch so unrelated pre-existing hunks never enter these commits.
- No test may make external model, ASR, image-generation, or publishing calls; use deterministic fixtures.

---

### Task 1: Add sequence mode and pair-aware storyboard contracts

**Files:**
- Modify: `src/bv/production/profile.py`
- Modify: `src/bv/illustration/contracts.py`
- Modify: `tests/unit/test_production_profile.py`
- Modify: `tests/unit/test_illustration_contracts.py`
- Modify: `tests/integration/test_title_author_creation.py`

**Interfaces:**
- Produces: `VisualSequenceMode = Literal["legacy-monochrome-reveal", "color-story-pair"]`.
- Produces: `VisualProfile.sequence_mode: VisualSequenceMode`.
- Produces: `IllustrationStoryboard.sequence_mode: VisualSequenceMode`.
- Produces on new scenes: `semantic_turn_span`, `semantic_turn_ms`, `semantic_turn_frame`, `continuation_action`, `continuation_prompt`, and `continuity_constraints`.
- Preserves: legacy storyboards that omit these fields.

- [ ] **Step 1: Write failing profile tests**

Add tests that require new title-and-author profiles to use the new mode while the legacy fallback remains unchanged:

```python
def test_short_book_profile_defaults_to_color_story_pairs() -> None:
    profile = ProductionProfile.short_book_default()
    assert profile.visual.sequence_mode == "color-story-pair"
    assert profile.visual.scene_count == 4


def test_missing_episode_profile_keeps_legacy_sequence_mode(tmp_path: Path) -> None:
    profile = load_production_profile(tmp_path)
    assert profile.visual.sequence_mode == "legacy-monochrome-reveal"
```

Update the title-and-author creation integration assertion to require `color-story-pair`.

- [ ] **Step 2: Run profile tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_production_profile.py tests/integration/test_title_author_creation.py -q
```

Expected: FAIL because `VisualProfile` has no `sequence_mode`.

- [ ] **Step 3: Implement the profile mode minimally**

Add the type and field:

```python
VisualSequenceMode = Literal[
    "legacy-monochrome-reveal",
    "color-story-pair",
]


class VisualProfile(_ProfileModel):
    sequence_mode: VisualSequenceMode = "legacy-monochrome-reveal"
```

Set `sequence_mode="color-story-pair"` in `short_book_default()` and `living_default()`. Keep `LEGACY_PRODUCTION_PROFILE` explicit as `legacy-monochrome-reveal`.

- [ ] **Step 4: Write failing storyboard contract tests**

Create a pair-mode scene fixture and assert it validates only with complete pair fields:

```python
pair_scene = legacy_scene.model_copy(update={
    "semantic_turn_span": (8, 16),
    "semantic_turn_ms": 3_000,
    "semantic_turn_frame": 90,
    "continuation_action": "母亲放慢脚步，主人公抬头看见她",
    "continuation_prompt": "同一园路同一机位，母亲放慢脚步，主人公抬头",
    "continuity_constraints": (
        "same characters",
        "same clothing",
        "same setting",
        "same camera direction",
    ),
})
storyboard = legacy_storyboard.model_copy(update={
    "sequence_mode": "color-story-pair",
    "scenes": (pair_scene,),
})
assert storyboard.scenes[0].semantic_turn_frame == 90
```

Also assert pair mode rejects missing continuation fields, rejects scene counts outside `{3, 4}`, rejects a turn that leaves fewer than 30 anchor frames or fewer than `45 + 30 + 15` continuation-side frames, and legacy mode still accepts old scenes unchanged.

- [ ] **Step 5: Run contract tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_illustration_contracts.py -q
```

Expected: FAIL because pair fields and conditional validation do not exist.

- [ ] **Step 6: Implement pair-aware Pydantic validation**

Add optional scene fields with legacy-safe defaults, then validate them when the parent storyboard mode is new. Put the cross-scene-mode checks in `IllustrationStoryboard._validate_timeline()` so legacy `IllustrationScene` instances remain constructible:

```python
sequence_mode: VisualSequenceMode = "legacy-monochrome-reveal"

if self.sequence_mode == "color-story-pair":
    if len(self.scenes) not in {3, 4}:
        raise ValueError("story_pair_scene_count_invalid")
    for scene in self.scenes:
        required = (
            scene.semantic_turn_span,
            scene.semantic_turn_ms,
            scene.semantic_turn_frame,
            scene.continuation_action,
            scene.continuation_prompt,
            scene.continuity_constraints,
        )
        if any(value is None or value == () or value == "" for value in required):
            raise ValueError("story_pair_fields_missing")
        local_turn = scene.semantic_turn_frame - scene.from_frame
        if local_turn < 30:
            raise ValueError("story_pair_anchor_too_short")
        if scene.to_frame - scene.semantic_turn_frame < 45 + 30 + 15:
            raise ValueError("story_pair_continuation_too_short")
```

Validate `semantic_turn_span` is strictly inside `narration_span`, `semantic_turn_ms` lies inside the scene millisecond range, and the turn frame equals `round(semantic_turn_ms * 30 / 1000)`.

- [ ] **Step 7: Run Task 1 tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_production_profile.py tests/unit/test_illustration_contracts.py tests/integration/test_title_author_creation.py -q
```

Expected: PASS.

Stage only Task 1 hunks and commit:

```powershell
git add -p -- src/bv/production/profile.py src/bv/illustration/contracts.py tests/unit/test_production_profile.py tests/unit/test_illustration_contracts.py tests/integration/test_title_author_creation.py
git commit -m "feat: add color story-pair contracts"
```

---

### Task 2: Derive an ASR-aligned semantic turn during illustration planning

**Files:**
- Modify: `src/bv/illustration/planner.py`
- Modify: `src/bv/workflow/media_stages.py`
- Modify: `prompts/illustration/storyboard.md`
- Modify: `tests/unit/test_illustration_planner.py`

**Interfaces:**
- Consumes: `VisualProfile.sequence_mode` and `AlignedScript.characters`.
- Produces: `_SceneDraft.semantic_turn_offset: int`, `continuation_action`, `continuation_prompt`, `continuity_constraints`.
- Produces: `resolve_semantic_turn(scene_span, offset, aligned_script) -> tuple[span, ms, frame]`.
- Preserves: legacy planning output when sequence mode is legacy.

- [ ] **Step 1: Write a failing semantic-turn resolver test**

Build an `AlignedScript` fixture where approved character indexes have known timings. Require an offset inside a scene narration to map to the first timed character at or after that boundary:

```python
span, turn_ms, turn_frame = resolve_semantic_turn(
    scene_span=(10, 30),
    semantic_turn_offset=8,
    aligned_script=aligned,
)
assert span == (10, 18)
assert turn_ms == aligned.characters[18].start_ms
assert turn_frame == round(turn_ms * 30 / 1000)
```

Add failure cases for offset `0`, offset at scene end, missing timing through the remainder of the scene, and a turn that violates the renderer dwell bounds.

- [ ] **Step 2: Run resolver test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_illustration_planner.py -k semantic_turn -q
```

Expected: FAIL because `resolve_semantic_turn` and draft fields do not exist.

- [ ] **Step 3: Implement deterministic timing resolution**

Implement a pure helper that uses approved-text character indexes only:

```python
def resolve_semantic_turn(
    *,
    scene_span: tuple[int, int],
    semantic_turn_offset: int,
    aligned_script: AlignedScript,
) -> tuple[tuple[int, int], int, int]:
    start, end = scene_span
    absolute = start + semantic_turn_offset
    if absolute <= start or absolute >= end:
        raise IllustrationPlanError("semantic_turn_outside_scene")
    timed = next(
        (
            character
            for character in aligned_script.characters[absolute:end]
            if character.start_ms is not None
        ),
        None,
    )
    if timed is None:
        raise IllustrationPlanError("semantic_turn_alignment_missing")
    assert timed.start_ms is not None
    return (start, absolute), timed.start_ms, round(timed.start_ms * 30 / 1000)
```

Do not introduce a 50% fallback.

- [ ] **Step 4: Write failing planner-model tests**

Extend `_draft()` fixtures to return the exact pair fields. Require the planner request payload to tell the model that the offset must identify a natural semantic transition and that B advances one physical action without changing identity, clothing, setting, camera direction, or style.

Assert `_validate_visual_plan()` copies these values into each new `IllustrationScene` and that pair mode returns exactly 3 or 4 scenes.

- [ ] **Step 5: Run planner tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_illustration_planner.py -q
```

Expected: FAIL because `_SceneDraft` and `plan_illustrations()` still describe one image per scene.

- [ ] **Step 6: Implement planner pair fields and stage wiring**

Add `sequence_mode` to `plan_illustrations()`. In pair mode, include it in the source payload and validate the new draft fields. Map the offset through `AlignedScript`, store the span/time/frame, and use the existing `character_action` and `image_prompt` as A.

In `IllustrationPlanningStage.run()`, pass `production_profile.visual.sequence_mode`.

Update the prompt asset with a positive output recipe:

```text
For each scene, return one anchor action and one continuation action.
The continuation occurs in the same place, time, camera direction, character identity, clothing and visual style.
semantic_turn_offset is the approved-narration character boundary where the continuation begins.
Advance exactly one visible action justified by the narration after that boundary.
```

- [ ] **Step 7: Run Task 2 tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_illustration_planner.py tests/unit/test_illustration_contracts.py -q
```

Expected: PASS.

Stage only Task 2 hunks and commit:

```powershell
git add -p -- src/bv/illustration/planner.py src/bv/workflow/media_stages.py prompts tests/unit/test_illustration_planner.py
git commit -m "feat: align story-pair turns to narration"
```

---

### Task 3: Create ordered A/B image jobs and pair-level representative approval

**Files:**
- Modify: `src/bv/illustration/assets.py`
- Modify: `src/bv/illustration/prompts.py`
- Modify: `src/bv/illustration/reviews.py`
- Modify: `src/bv/workflow/media_runtime.py`
- Modify: `src/bv/state/invalidation.py`
- Modify: `tests/unit/test_illustration_assets.py`
- Modify: `tests/unit/test_representative_review.py`
- Modify: `tests/unit/test_media_runtime.py`
- Modify: `tests/unit/test_invalidation.py`
- Modify: `tests/integration/test_illustration_asset_pipeline.py`
- Modify: `tests/integration/test_representative_gate_pipeline.py`

**Interfaces:**
- Produces: `ImagePhase = Literal["legacy", "anchor", "continuation"]`.
- Produces: `ImageJob.asset_id`, `phase`, and `anchor_sha256`.
- Produces: `import_image_asset(job, source, ...)` that imports one full-color asset and derives grayscale only for `legacy`.
- Produces: representative reviews containing six hashes ordered `S01-A, S01-B, middle-A, middle-B, final-A, final-B`.

- [ ] **Step 1: Write failing A/B job-order tests**

For a four-scene pair storyboard, require eight jobs in this exact order:

```python
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
```

Require continuation output `S01_continuation.png`, anchor output `S01_anchor.png`, no `_bw.png`, and B initially blocked until A has a current hash.

- [ ] **Step 2: Run asset tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_illustration_assets.py -k "story_pair or continuation" -q
```

Expected: FAIL because each scene still creates one master and one grayscale derivative.

- [ ] **Step 3: Implement pair-aware jobs and prompts**

Extend `ImageJob` without breaking legacy JSON:

```python
phase: Literal["legacy", "anchor", "continuation"] = "legacy"
asset_id: str | None = None
anchor_sha256: str | None = None
output_master: Path
output_bw: Path | None = None
```

In pair mode, build anchor first and continuation second. Compile B with A as the first `reference_scene_ids` entry and include `continuation_action` plus continuity constraints. Bind `anchor_sha256` only from the exact current generated anchor job.

Refactor import minimally so `legacy` still produces grayscale, while pair phases publish only the color PNG.

- [ ] **Step 4: Write failing representative-pair tests**

Require three representative scenes but six current image hashes. Approval must fail when any B is absent, when hash order is wrong, or when a B records a stale A hash:

```python
assert review.scene_ids == ("S01", "S03", "S04")
assert len(review.image_sha256s) == 6
assert review.asset_ids == (
    "S01-A", "S01-B", "S03-A", "S03-B", "S04-A", "S04-B",
)
```

For three scenes, assert all six jobs are representative and `unlock_batch_jobs()` returns an empty tuple. For four scenes, it returns exactly the remaining A/B pair.

- [ ] **Step 5: Run review tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_representative_review.py tests/integration/test_representative_gate_pipeline.py -q
```

Expected: FAIL because review models require three single-image hashes.

- [ ] **Step 6: Implement pair-level review, runtime status, and invalidation**

Change review tuples to variable-length tuples with explicit six-item validation in pair mode. Group jobs by representative scene and phase, validate A before B, and bind approval to both hashes.

Update `LocalIllustrationGateway.status()` and batch status to report asset IDs rather than treating one scene ID as one job. Preserve the conversation-facing scene grouping in the view by adding pair-aware missing IDs such as `S02-A` and `S02-B`.

Extend invalidation so:

- changed A invalidates its B plus representative review and all visual/final stages;
- changed B invalidates representative review and visual/final stages but does not invalidate A;
- style change invalidates all A/B jobs;
- legacy invalidation tests remain unchanged.

- [ ] **Step 7: Run Task 3 tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_illustration_assets.py tests/unit/test_representative_review.py tests/unit/test_media_runtime.py tests/unit/test_invalidation.py tests/integration/test_illustration_asset_pipeline.py tests/integration/test_representative_gate_pipeline.py -q
```

Expected: PASS.

Stage only Task 3 hunks and commit:

```powershell
git add -p -- src/bv/illustration/assets.py src/bv/illustration/prompts.py src/bv/illustration/reviews.py src/bv/workflow/media_runtime.py src/bv/state/invalidation.py tests/unit/test_illustration_assets.py tests/unit/test_representative_review.py tests/unit/test_media_runtime.py tests/unit/test_invalidation.py tests/integration/test_illustration_asset_pipeline.py tests/integration/test_representative_gate_pipeline.py
git commit -m "feat: gate paired illustration assets"
```

---

### Task 4: Render full-color A to full-color B through semantic ink bloom

**Files:**
- Modify: `src/bv/workflow/media_stages.py`
- Modify: `src/bv/illustration/renderer.py`
- Modify: `vendor/story_to_handdrawn_video/src/types.ts`
- Modify: `vendor/story_to_handdrawn_video/src/Scene.tsx`
- Modify: `vendor/story_to_handdrawn_video/src/LayerWipe.tsx`
- Modify: `vendor/story_to_handdrawn_video/scripts/validate-storyboard.mjs`
- Modify: `tests/unit/test_handdrawn_renderer.py`
- Modify: `tests/integration/test_handdrawn_renderer_pipeline.py`
- Modify: `tests/unit/test_renderer_title_overlay.py`

**Interfaces:**
- Consumes: validated pair jobs and `semantic_turn_frame`.
- Produces new render assets: `{anchor, continuation}` plus `sequence_mode` and absolute semantic frame.
- Preserves legacy render assets: `{bw, color}`.
- Produces: `LayerWipe.treatment` support for unfiltered story color.

- [ ] **Step 1: Write failing renderer-storyboard tests**

Require `build_renderer_storyboard()` to emit this shape for new scenes:

```python
assert payload["scenes"][0]["sequence_mode"] == "color-story-pair"
assert payload["scenes"][0]["assets"] == {
    "anchor": "media/illustration/images/S01_anchor.png",
    "continuation": "media/illustration/images/S01_continuation.png",
}
assert payload["scenes"][0]["semantic_turn_frame"] == expected_frame
assert payload["project"]["ink_reveal_frames"] == 45
```

Also keep an assertion that old manifests still emit `{bw, color}`.

- [ ] **Step 2: Run Python renderer tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_handdrawn_renderer.py tests/unit/test_renderer_title_overlay.py -q
```

Expected: FAIL because render payload supports only `bw/color`.

- [ ] **Step 3: Implement pair payload and safe asset copying**

Branch on `storyboard.sequence_mode`. In pair mode, locate both phase jobs, validate their hashes and the B-to-A dependency, and emit relative `anchor/continuation` paths. In legacy mode, leave the current payload unchanged.

Update `renderer.py` to copy whichever asset keys the validated mode requires into Remotion's isolated public directory.

- [ ] **Step 4: Write failing TypeScript validator and pixel tests**

Add a new fixture with a solid red A and solid blue B. Require:

- frame before semantic turn: center and corners are red;
- ink midpoint: sampled pixels include both red and blue regions;
- frame after 45 reveal frames: center and corners are blue;
- no sampled pixel is paper white during reveal;
- legacy fixture still holds monochrome through frame 59 and reveals color after frame 60.

Update validator tests so pair mode rejects missing A/B, rejects `bw/color` keys, rejects an out-of-range semantic frame, and accepts legacy unchanged.

- [ ] **Step 5: Run TypeScript/real-render tests and verify RED**

Run:

```powershell
npm --prefix vendor/story_to_handdrawn_video run check
.\.venv\Scripts\python.exe -m pytest tests/integration/test_handdrawn_renderer_pipeline.py -k "story_pair or legacy" -q
```

Expected: FAIL because `Scene.tsx` always starts with BW and starts color at `fps * 2`.

- [ ] **Step 6: Implement the Remotion pair branch**

Extend types as a discriminated union:

```typescript
type LegacySceneAssets = {bw: string; color: string};
type StoryPairAssets = {anchor: string; continuation: string};

type SceneData = LegacySceneData | StoryPairSceneData;
```

In `Scene.tsx`:

```tsx
if (scene.sequence_mode === 'color-story-pair') {
  const localTurn = scene.semantic_turn_frame - scene.from_frame;
  return (
    <>
      <LayerWipe src={scene.assets.anchor} startFrame={0} durationFrames={1}
        treatment="story-color" visibleFromStart zIndex={10} box={layout.illustration} />
      <LayerWipe src={scene.assets.continuation} startFrame={localTurn}
        durationFrames={45} treatment="story-color" zIndex={30} box={layout.illustration} />
    </>
  );
}
```

Keep the existing legacy branch intact. Add `story-color` with no grayscale and only the existing restrained brightness/contrast. Reuse the five radial blooms and final wash; do not add a linear directional wipe.

- [ ] **Step 7: Run Task 4 tests and commit**

Run:

```powershell
npm --prefix vendor/story_to_handdrawn_video run check
.\.venv\Scripts\python.exe -m pytest tests/unit/test_handdrawn_renderer.py tests/unit/test_renderer_title_overlay.py tests/integration/test_handdrawn_renderer_pipeline.py -q
```

Expected: PASS.

Stage only Task 4 hunks and commit:

```powershell
git add -p -- src/bv/workflow/media_stages.py src/bv/illustration/renderer.py vendor/story_to_handdrawn_video/src/types.ts vendor/story_to_handdrawn_video/src/Scene.tsx vendor/story_to_handdrawn_video/src/LayerWipe.tsx vendor/story_to_handdrawn_video/scripts/validate-storyboard.mjs tests/unit/test_handdrawn_renderer.py tests/unit/test_renderer_title_overlay.py tests/integration/test_handdrawn_renderer_pipeline.py
git commit -m "feat: reveal continuation art with ink bloom"
```

---

### Task 5: Prove new-mode workflow and legacy compatibility end to end

**Files:**
- Modify: `tests/integration/test_handdrawn_end_to_end_fake.py`
- Modify: `tests/integration/test_handdrawn_media_workflow.py`
- Modify: `tests/integration/test_doubao_longform_delivery_pipeline.py`
- Modify: `tests/unit/test_video_contracts.py`
- Modify only if failures require it: `src/bv/workflow/runtime.py`
- Modify only if failures require it: `src/bv/workflow/media_stages.py`

**Interfaces:**
- Consumes: all Task 1–4 contracts.
- Produces: a fully local new-mode candidate reaching `awaiting_final_review`.
- Preserves: an old `bw/color` fixture reaching the same state.

- [ ] **Step 1: Write a failing pair-mode end-to-end test**

Use a four-scene profile and deterministic local PNG fixtures. Drive:

```text
approved script
-> fixture WAV/ASR
-> pair storyboard
-> six representative A/B imports
-> representative approval
-> remaining A/B import
-> silent render
-> final render/QC
-> immutable candidate
-> awaiting_final_review
```

Assert eight unique color assets, no new `_bw.png`, three complete representative pairs, one remaining pair after approval, 45 reveal frames, 15 transition frames, exact final duration, and no final approval record before user approval.

- [ ] **Step 2: Run end-to-end test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_handdrawn_end_to_end_fake.py -k story_pair -q
```

Expected: FAIL at the first old single-image assumption.

- [ ] **Step 3: Make only required workflow adaptations**

Update runtime wiring only where the test demonstrates a mismatch. Do not redesign unrelated stage names or statuses. Keep `representative_generation_running`, `awaiting_representative_review`, `representatives_approved`, `illustration_batch_running`, `illustrations_ready`, and `awaiting_final_review` stable.

- [ ] **Step 4: Add explicit legacy regression coverage**

Run the existing legacy fixture without a production profile and assert:

- sequence mode resolves to legacy;
- one master plus one grayscale derivative per scene remains accepted;
- frame-59/frame-61 monochrome-to-color behavior remains unchanged;
- delivery and final approval hashes are not rewritten.

- [ ] **Step 5: Run workflow regression tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_handdrawn_end_to_end_fake.py tests/integration/test_handdrawn_media_workflow.py tests/integration/test_doubao_longform_delivery_pipeline.py tests/unit/test_video_contracts.py -q
```

Expected: PASS.

Stage only Task 5 hunks and commit:

```powershell
git add -p -- tests/integration/test_handdrawn_end_to_end_fake.py tests/integration/test_handdrawn_media_workflow.py tests/integration/test_doubao_longform_delivery_pipeline.py tests/unit/test_video_contracts.py src/bv/workflow/runtime.py src/bv/workflow/media_stages.py
git commit -m "test: cover story-pair workflow end to end"
```

---

### Task 6: Update the two production Skills from observed behavior

**Files:**
- Modify: `C:/Users/1/.codex/skills/bv-book-video-style-contract/SKILL.md`
- Modify: `C:/Users/1/.codex/skills/bv-book-video-style-contract/references/visual-contract.md`
- Modify: `C:/Users/1/.codex/skills/producing-book-handdrawn-videos/SKILL.md`
- Mirror if this repository copy is canonical for release: `skills/bv-book-video-style-contract/SKILL.md`
- Mirror if this repository copy is canonical for release: `skills/bv-book-video-style-contract/references/visual-contract.md`

**Interfaces:**
- Consumes: verified code behavior and the approved design spec.
- Produces: discoverable instructions for all future book-video styles.
- Preserves: research, writing, voice, subtitles, cover, QC, delivery, and approval permissions.

- [ ] **Step 1: Record a failing no-guidance baseline**

Use the realistic request below in a fresh context without the proposed new contract text and record whether the response incorrectly plans BW layers, four single masters, or a fixed midpoint:

```text
给《蛤蟆先生去看心理医生》换成暖色几何画风，其他流程不变，继续做代表图。
```

The baseline is failing if it does not require A/B pairs, an ASR-aligned semantic turn, B's exact A-hash reference, and pair-level approval.

- [ ] **Step 2: Update `bv-book-video-style-contract` minimally**

Replace the old stable layer with this positive recipe:

```text
Each scene is one full-color story pair. Show color A from local frame 0.
At the approved narration semantic turn, reveal color B over 45 frames through five or more soft ink blooms.
B continues one visible action in the same identity, clothing, setting, camera direction and style.
Changing style regenerates both A and B but preserves the semantic turn and story-pair meaning.
```

Update required evidence to pre-turn A, ink midpoint A+B, post-turn B, and all 15-frame cross-scene transitions. Keep a clearly labeled legacy compatibility paragraph.

- [ ] **Step 3: Update `producing-book-handdrawn-videos` minimally**

Change only the illustration and representative-gate instructions:

- 3–4 scene pairs, 6–8 color masters;
- A generated before B;
- B strongly references exact A;
- opening/middle/ending pairs are shown together as six representative images;
- remaining pair generation only after approval.

- [ ] **Step 4: Validate and forward-test both Skills**

Run:

```powershell
.\.venv\Scripts\python.exe "$env:USERPROFILE\.codex\skills\.system\skill-creator\scripts\quick_validate.py" "$env:USERPROFILE\.codex\skills\bv-book-video-style-contract"
.\.venv\Scripts\python.exe "$env:USERPROFILE\.codex\skills\.system\skill-creator\scripts\quick_validate.py" "$env:USERPROFILE\.codex\skills\producing-book-handdrawn-videos"
```

Repeat the baseline request with each updated Skill loaded. It passes only if the planned output is a style-independent A/B story pair flow and does not resurrect BW or fixed-50% timing.

- [ ] **Step 5: Commit repository Skill mirrors only**

External `$CODEX_HOME` Skill files are not part of this repository and must not be staged here. If repository mirrors are confirmed canonical, stage only those files:

```powershell
git add -p -- skills/bv-book-video-style-contract/SKILL.md skills/bv-book-video-style-contract/references/visual-contract.md
git commit -m "docs: update book-video story-pair contract"
```

---

### Task 7: Run focused, full, and visual verification

**Files:**
- Create: `.private/test-fixtures/story-pair-review/` only when the integration test owns and cleans it.
- Do not modify production Episodes or delivery directories.

**Interfaces:**
- Consumes: completed implementation and Skills.
- Produces: test logs, ffprobe facts, and review frames proving the new contract.

- [ ] **Step 1: Run focused Python tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_production_profile.py tests/unit/test_illustration_contracts.py tests/unit/test_illustration_planner.py tests/unit/test_illustration_assets.py tests/unit/test_representative_review.py tests/unit/test_media_runtime.py tests/unit/test_invalidation.py tests/unit/test_handdrawn_renderer.py tests/unit/test_renderer_title_overlay.py tests/integration/test_illustration_asset_pipeline.py tests/integration/test_representative_gate_pipeline.py tests/integration/test_handdrawn_renderer_pipeline.py tests/integration/test_handdrawn_end_to_end_fake.py -q
```

Expected: PASS with no network activity.

- [ ] **Step 2: Run vendor validation and tests**

Run:

```powershell
npm --prefix vendor/story_to_handdrawn_video run validate
npm --prefix vendor/story_to_handdrawn_video run check
```

Expected: PASS for both pair and legacy fixtures.

- [ ] **Step 3: Run the complete repository suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: PASS, excluding only pre-existing documented skips. If unrelated failures exist, compare them to the pre-change baseline and report them without modifying unrelated code.

- [ ] **Step 4: Inspect real rendered frames**

From the deterministic pair fixture, extract:

- one frame before semantic turn;
- frame `semantic_turn + 22` as ink midpoint;
- frame `semantic_turn + 45` as complete B;
- start/mid/end of each cross-scene transition;
- representative light and dark subtitle frames.

Verify full-color A, local A/B overlap, full-color B, no white flash, exact title/author geometry, Microsoft YaHei white outline subtitles, and no black subtitle box.

- [ ] **Step 5: Verify legacy and approved artifacts are untouched**

Run a read-only hash audit for all existing `*最终批准.json` records. Confirm each referenced video, cover, and delivery manifest still matches its stored hash and no approved Episode file modification time changed during tests.

- [ ] **Step 6: Final review and commit any test-only fixture metadata**

Run:

```powershell
git diff --check
git status --short
```

Do not stage `workspace/`, generated media, private requests, credentials, or unrelated dirty files. If a tracked deterministic fixture manifest was intentionally added, stage it explicitly and commit:

```powershell
git add -p -- tests vendor/story_to_handdrawn_video
git commit -m "test: verify color story-pair presentation"
```
