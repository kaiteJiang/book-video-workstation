# Handdrawn Renderer Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Vendor the approved upstream project and produce a deterministic 1080×1920 silent hand-drawn video from an Episode-local storyboard.

**Architecture:** Keep a pinned, attributed upstream snapshot under vendor/story_to_handdrawn_video and expose it through a narrow Python adapter. Remotion receives an explicit storyboard as input props, uses absolute frames, and emits one silent H.264 visual track without reading or overwriting production state implicitly.

**Tech Stack:** Python 3.11, Pydantic 2, pytest 8, Node.js 20+, TypeScript 5.9, React 19, Remotion 4.0.487, FFmpeg/FFprobe.

**Spec:** docs/superpowers/specs/2026-08-22-handdrawn-book-video-media-v0.2-design.md

## Global Constraints

- Pin upstream commit fbab5b27f4f0db61739d86f78000a39eeaa692d3; never pull latest at runtime.
- Preserve upstream MIT license, style-library license, font license, source URL, and local-change record.
- Formal visual output is exactly 1080×1920, 30 fps, H.264, yuv420p, and silent.
- Use absolute scene frames; total frames equal round(master_duration_ms × 30 / 1000).
- Page-flip/reveal animation occurs inside assigned scene frames and never shortens the master timeline.
- Theme text uses local font rendering; generated illustrations contain no text.
- Production inputs and outputs must remain under the current Episode root and must not follow symlinks or Windows reparse points.
- Do not add a user-facing CLI, GUI, API key, cloud fallback, or automatic GitHub update.
- Preserve the untracked .tmp_codex_session.json and unrelated user changes.

## File Map

- vendor/story_to_handdrawn_video/: pinned Remotion source and immutable upstream assets.
- vendor/story_to_handdrawn_video/UPSTREAM.json: source identity and local modification summary.
- src/bv/illustration/contracts.py: strict Python media contracts shared by all later plans.
- src/bv/illustration/renderer.py: shell-free Remotion invocation and silent-output validation.
- src/bv/config.py: hand-drawn renderer configuration.
- tests/unit/test_handdrawn_vendor.py: provenance and snapshot tests.
- tests/unit/test_illustration_contracts.py: contract/config tests.
- tests/unit/test_handdrawn_renderer.py: adapter argv, safety, and output checks.
- tests/integration/test_handdrawn_renderer_pipeline.py: real local 9:16 render.

---

### Task 1: Vendor the pinned upstream snapshot

**Files:**
- Create: vendor/story_to_handdrawn_video/UPSTREAM.json
- Create: vendor/story_to_handdrawn_video/LICENSE
- Create: vendor/story_to_handdrawn_video/package.json
- Create: vendor/story_to_handdrawn_video/package-lock.json
- Create: vendor/story_to_handdrawn_video/tsconfig.json
- Create: vendor/story_to_handdrawn_video/remotion.config.ts
- Create: vendor/story_to_handdrawn_video/src/
- Create: vendor/story_to_handdrawn_video/scripts/
- Create: vendor/story_to_handdrawn_video/public/
- Create: vendor/story_to_handdrawn_video/references/
- Test: tests/unit/test_handdrawn_vendor.py

**Interfaces:**
- Consumes: GitHub repository gnipbao/story-to-handdrawn-video at the fixed commit.
- Produces: a self-contained Node project and UPSTREAM.json with repository_url, commit, license, imported_at, and local_changes.

- [ ] **Step 1: Write the failing provenance test**

~~~python
def test_vendored_handdrawn_snapshot_is_pinned_and_attributed() -> None:
    root = Path("vendor/story_to_handdrawn_video")
    metadata = json.loads((root / "UPSTREAM.json").read_text("utf-8"))
    assert metadata["repository_url"] == "https://github.com/gnipbao/story-to-handdrawn-video"
    assert metadata["commit"] == "fbab5b27f4f0db61739d86f78000a39eeaa692d3"
    assert metadata["license"] == "MIT"
    assert (root / "LICENSE").read_text("utf-8").startswith("MIT License")
    styles = json.loads(
        (root / "references/handdrawn-style-library.json").read_text("utf-8")
    )
    assert len(styles["styles"]) == 20
~~~

- [ ] **Step 2: Run the test and verify the snapshot is absent**

Run: python -m pytest tests/unit/test_handdrawn_vendor.py -q

Expected: FAIL because vendor/story_to_handdrawn_video/UPSTREAM.json does not exist.

- [ ] **Step 3: Copy the exact upstream snapshot and create provenance**

Checkout only the approved commit in a temporary directory, copy the repository contents excluding .git, out, generated, prompts, and node_modules, then create:

~~~json
{
  "repository_url": "https://github.com/gnipbao/story-to-handdrawn-video",
  "commit": "fbab5b27f4f0db61739d86f78000a39eeaa692d3",
  "license": "MIT",
  "imported_at": "2026-08-22",
  "local_changes": [
    "9:16 output",
    "absolute-frame timeline",
    "font-rendered short key line",
    "explicit storyboard input props"
  ]
}
~~~

- [ ] **Step 4: Verify Python provenance and the untouched Node snapshot**

Run: python -m pytest tests/unit/test_handdrawn_vendor.py -q

Expected: PASS.

Run: npm ci

Working directory: vendor/story_to_handdrawn_video

Expected: exit 0 using package-lock.json.

Run: npm run check

Working directory: vendor/story_to_handdrawn_video

Expected: exit 0 before local renderer changes.

- [ ] **Step 5: Commit the pinned snapshot**

~~~powershell
git add -- vendor/story_to_handdrawn_video tests/unit/test_handdrawn_vendor.py
git commit -m "build: vendor pinned handdrawn renderer"
~~~

### Task 2: Add strict illustration contracts and configuration

**Files:**
- Create: src/bv/illustration/__init__.py
- Create: src/bv/illustration/contracts.py
- Modify: src/bv/config.py
- Modify: config.example.yaml
- Test: tests/unit/test_illustration_contracts.py
- Test: tests/unit/test_config.py

**Interfaces:**
- Consumes: approved script SHA-256, audio SHA-256, subtitle SHA-256, and Episode root.
- Produces: SafeArea, StyleDecision, CharacterLock, IllustrationScene, IllustrationStoryboard, SilentRenderRequest, and SilentRenderResult.

- [ ] **Step 1: Write failing model and config tests**

~~~python
def test_storyboard_requires_gapless_absolute_frames() -> None:
    with pytest.raises(ValidationError, match="scene_frame_gap"):
        IllustrationStoryboard(
            book_id="book-demo",
            episode_id="E001",
            width=1080,
            height=1920,
            fps=30,
            master_duration_ms=10_000,
            total_frames=300,
            script_sha256="a" * 64,
            audio_sha256="b" * 64,
            subtitle_sha256="c" * 64,
            style_decision_sha256="d" * 64,
            character_lock_sha256="e" * 64,
            scenes=(
                scene("S01", 0, 149),
                scene("S02", 151, 300),
            ),
        )

def test_handdrawn_config_defaults_are_douyin_native() -> None:
    config = AppConfig()
    assert config.handdrawn.width == 1080
    assert config.handdrawn.height == 1920
    assert config.handdrawn.fps == 30
    assert config.handdrawn.safe_area.subtitle_bottom == 1660
    assert config.handdrawn.representative_count == 3
~~~

- [ ] **Step 2: Run focused tests and verify missing imports**

Run: python -m pytest tests/unit/test_illustration_contracts.py tests/unit/test_config.py -q

Expected: FAIL because bv.illustration.contracts and AppConfig.handdrawn do not exist.

- [ ] **Step 3: Implement the minimal strict contracts**

Create frozen Pydantic models with extra="forbid". Required signatures:

~~~python
class SafeArea(BaseModel):
    top_reserved: int = 140
    keyline_top: int = 140
    keyline_bottom: int = 340
    illustration_top: int = 340
    illustration_bottom: int = 1460
    subtitle_top: int = 1460
    subtitle_bottom: int = 1660
    bottom_reserved: int = 260
    right_reserved: int = 180

class StyleDecision(BaseModel):
    library_version: str
    selected_style: str
    candidate_styles: tuple[str, str, str]
    selection_reasons: tuple[str, ...]
    rejected_reasons: dict[str, str]
    confidence: float = Field(ge=0.0, le=1.0)
    manual_override: bool
    source_script_sha256: str
    style_fingerprint: str

class CharacterLock(BaseModel):
    role: str
    age_range: str
    face: str
    hair: str
    body: str
    base_clothing: str
    allowed_variations: tuple[str, ...]
    color_markers: tuple[str, ...]
    personal_objects: tuple[str, ...]
    forbidden_changes: tuple[str, ...]
    source_script_sha256: str
    style_fingerprint: str

class IllustrationScene(BaseModel):
    scene_id: str
    start_ms: int
    end_ms: int
    from_frame: int
    to_frame: int
    narration: str
    narration_span: tuple[int, int]
    key_line: str
    visual_purpose: str
    setting: str
    character_action: str
    metaphor: str | None
    composition: str
    character_refs: tuple[str, ...]
    image_prompt: str
    negative_constraints: tuple[str, ...]
    representative_frame: bool
    asset_status: Literal["planned", "generated", "approved"]

class IllustrationStoryboard(BaseModel):
    book_id: str
    episode_id: str
    width: Literal[1080]
    height: Literal[1920]
    fps: Literal[30]
    master_duration_ms: int
    total_frames: int
    script_sha256: str
    audio_sha256: str
    subtitle_sha256: str
    style_decision_sha256: str
    character_lock_sha256: str
    scenes: tuple[IllustrationScene, ...]

class SilentRenderRequest(BaseModel):
    episode_root: Path
    storyboard_path: Path
    storyboard_sha256: str
    output_path: Path
    vendor_dir: Path

class SilentRenderResult(BaseModel):
    output_path: Path
    output_sha256: str
    duration_ms: int
    width: Literal[1080]
    height: Literal[1920]
    frame_rate: Literal[30.0]
    audio_present: Literal[False]
~~~

Validators must enforce safe identifiers, lowercase SHA-256 values, canonical Episode-contained paths, exactly 1080×1920 at 30 fps, total-frame formula, first frame 0, final frame total_frames, and gapless adjacent scenes.

- [ ] **Step 4: Add HanddrawnConfig and relative-path resolution**

~~~python
class HanddrawnConfig(_ConfigModel):
    vendor_dir: Path = Path("vendor/story_to_handdrawn_video")
    node_command: str = "npm"
    width: Literal[1080] = 1080
    height: Literal[1920] = 1920
    fps: Literal[30] = 30
    representative_count: Literal[3] = 3
    image_size: Literal["1024x1024"] = "1024x1024"
    text_mode: Literal["font"] = "font"
    transition: Literal["cut", "page-flip"] = "page-flip"
    safe_area: SafeArea = Field(default_factory=SafeArea)
~~~

Resolve vendor_dir relative to the configuration file. Reject blank node_command and all values outside the fixed V0.2 contract.

- [ ] **Step 5: Run focused and config regression tests**

Run: python -m pytest tests/unit/test_illustration_contracts.py tests/unit/test_config.py -q

Expected: PASS.

- [ ] **Step 6: Commit contracts and config**

~~~powershell
git add -- src/bv/illustration src/bv/config.py config.example.yaml tests/unit/test_illustration_contracts.py tests/unit/test_config.py
git commit -m "feat: define handdrawn illustration contracts"
~~~

### Task 3: Convert the vendored Remotion renderer to 9:16 absolute frames

**Files:**
- Modify: vendor/story_to_handdrawn_video/src/types.ts
- Create: vendor/story_to_handdrawn_video/src/layout.ts
- Create: vendor/story_to_handdrawn_video/src/timeline.ts
- Modify: vendor/story_to_handdrawn_video/src/Root.tsx
- Modify: vendor/story_to_handdrawn_video/src/StoryVideo.tsx
- Modify: vendor/story_to_handdrawn_video/src/Scene.tsx
- Modify: vendor/story_to_handdrawn_video/src/TextWipe.tsx
- Modify: vendor/story_to_handdrawn_video/scripts/validate-storyboard.mjs
- Create: vendor/story_to_handdrawn_video/storyboard.9x16.fixture.json
- Modify: vendor/story_to_handdrawn_video/package.json

**Interfaces:**
- Consumes: Storyboard passed with Remotion --props, including total_frames and absolute scene frames.
- Produces: PictureSilent composition with metadata derived from input props.

- [ ] **Step 1: Add a 9:16 fixture and make validation fail against the old schema**

The fixture must contain three scenes covering the half-open frame interval [0, 450) with no gap, 1080×1920, 30 fps, a 15,000 ms master, 6–14 character key lines, and existing local SVG assets.

Run: npm run validate -- --input storyboard.9x16.fixture.json

Working directory: vendor/story_to_handdrawn_video

Expected: FAIL because the old schema permits only ratio 3:4 and relative duration_sec.

- [ ] **Step 2: Implement timeline and layout primitives**

~~~ts
export type Storyboard = {
  project: {
    width: 1080;
    height: 1920;
    fps: 30;
    ratio: '9:16';
    total_frames: number;
    transition: 'cut' | 'page-flip';
    transition_frames: number;
  };
  safe_area: SafeArea;
  scenes: SceneData[];
};

export type SceneData = {
  id: string;
  start_ms: number;
  end_ms: number;
  from_frame: number;
  to_frame: number;
  key_line: string;
  narration: string;
  assets: {bw: string; color: string};
};

export const sceneDurationFrames = (scene: SceneData) =>
  scene.to_frame - scene.from_frame;
~~~

validateStoryboard must reject wrong dimensions, nonzero first frame, gaps, overlap, a final frame different from total_frames, more than 14 displayed key-line characters, missing assets, and any scene shorter than the reveal minimum.

- [ ] **Step 3: Make the composition accept explicit props**

StoryVideo receives value: Storyboard. Root uses calculateMetadata to set width, height, fps, and durationInFrames from the supplied storyboard. storyboard.json remains only a development default; production rendering always passes --props with an Episode-local JSON file.

- [ ] **Step 4: Implement the approved A layout**

Use layout.ts constants from safe_area. Scene renders:

~~~tsx
<AbsoluteFill style={{backgroundColor: paperColor}}>
  <KeyLineWipe text={scene.key_line} top={140} bottom={340} />
  <IllustrationReveal
    bw={scene.assets.bw}
    color={scene.assets.color}
    top={340}
    bottom={1460}
    objectFit="contain"
  />
</AbsoluteFill>
~~~

Do not render bottom narration subtitles in Remotion; FFmpeg burns final ASS subtitles later. Keep the 1460..1660 subtitle band visually quiet.

- [ ] **Step 5: Keep transitions inside assigned frames**

Page-flip begins near the end of the current scene but the next scene starts exactly at its own from_frame. Remove total-duration subtraction for overlaps. totalFrames must always equal project.total_frames.

- [ ] **Step 6: Run Node validation and type checks**

Run: npm run validate -- --input storyboard.9x16.fixture.json

Expected: PASS.

Run: npm run check

Expected: TypeScript and storyboard validation exit 0.

- [ ] **Step 7: Commit the 9:16 renderer**

~~~powershell
git add -- vendor/story_to_handdrawn_video
git commit -m "feat: render handdrawn stories in native 9x16"
~~~

### Task 4: Add the Python silent-render adapter and real render test

**Files:**
- Create: src/bv/illustration/renderer.py
- Test: tests/unit/test_handdrawn_renderer.py
- Test: tests/integration/test_handdrawn_renderer_pipeline.py

**Interfaces:**
- Consumes: SilentRenderRequest and configured npm/ffprobe executables.
- Produces: SilentRenderResult only after hash, dimensions, frame rate, duration, codec, and no-audio checks pass.

- [ ] **Step 1: Write failing argv and safety tests**

~~~python
def test_build_render_argv_uses_explicit_props_and_muted_output(tmp_path: Path) -> None:
    request = safe_request(tmp_path)
    argv = build_handdrawn_render_argv(request, npm_command="npm")
    assert argv[:3] == ["npm", "run", "render:props"]
    assert str(request.storyboard_path) in argv
    assert str(request.output_path) in argv

def test_render_rejects_output_outside_episode(tmp_path: Path) -> None:
    request = safe_request(tmp_path).model_copy(
        update={"output_path": tmp_path / "outside.mp4"}
    )
    with pytest.raises(HanddrawnRenderError, match="unsafe_render_path"):
        render_silent_story(request, runner=fake_runner, inspector=fake_inspector)
~~~

- [ ] **Step 2: Run tests and verify missing adapter**

Run: python -m pytest tests/unit/test_handdrawn_renderer.py -q

Expected: FAIL because bv.illustration.renderer does not exist.

- [ ] **Step 3: Implement shell-free invocation and validation**

Required public signatures:

~~~python
class HanddrawnRenderError(RuntimeError):
    error_code: str

def build_handdrawn_render_argv(
    request: SilentRenderRequest,
    *,
    npm_command: str = "npm",
) -> list[str]:
    return [
        npm_command,
        "run",
        "render:props",
        "--",
        "--props",
        str(request.storyboard_path),
        "--output",
        str(request.output_path),
    ]

def render_silent_story(
    request: SilentRenderRequest,
    *,
    npm_command: str = "npm",
    ffprobe_command: str = "ffprobe",
    runner: Callable[..., CommandResult] = run_command,
    inspector: Callable[..., MediaProbeFacts] = probe_media,
) -> SilentRenderResult:
    request = SilentRenderRequest.model_validate(request)
    if sha256_file(request.storyboard_path) != request.storyboard_sha256:
        raise HanddrawnRenderError("storyboard_hash_mismatch")
    result = runner(
        build_handdrawn_render_argv(request, npm_command=npm_command),
        cwd=request.vendor_dir,
        timeout=900,
    )
    if result.returncode != 0:
        raise HanddrawnRenderError("handdrawn_render_failed")
    facts = inspector(request.output_path, ffprobe_command=ffprobe_command)
    return validate_silent_render_result(request, facts)
~~~

The implementation must refuse existing outputs, redirected paths, stale storyboard hashes, nonzero npm exit, wrong codec/dimensions/frame rate, audio presence, duration beyond one frame, and output changes between probe and publish.

- [ ] **Step 4: Run unit tests**

Run: python -m pytest tests/unit/test_handdrawn_renderer.py -q

Expected: PASS.

- [ ] **Step 5: Add and run a real 15-second render integration test**

The test copies the 9:16 fixture into an Episode root, invokes the vendored renderer with local SVG/PNG fixtures, probes the output, and asserts:

~~~python
assert result.width == 1080
assert result.height == 1920
assert result.frame_rate == 30.0
assert result.audio_present is False
assert abs(round(result.duration_ms * 30 / 1000) - 450) <= 1
~~~

Run: python -m pytest tests/integration/test_handdrawn_renderer_pipeline.py -q

Expected: PASS with a real H.264 silent file.

- [ ] **Step 6: Run foundation regression suite**

Run: python -m pytest tests/unit/test_handdrawn_vendor.py tests/unit/test_illustration_contracts.py tests/unit/test_handdrawn_renderer.py tests/unit/test_config.py tests/integration/test_handdrawn_renderer_pipeline.py -q

Expected: all selected tests PASS.

- [ ] **Step 7: Commit the adapter**

~~~powershell
git add -- src/bv/illustration/renderer.py tests/unit/test_handdrawn_renderer.py tests/integration/test_handdrawn_renderer_pipeline.py
git commit -m "feat: add validated handdrawn render adapter"
~~~
