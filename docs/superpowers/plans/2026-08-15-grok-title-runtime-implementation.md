# Grok Title-Input Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing fail-closed BV Workstation accept a title and author, produce a reviewed whole-book value script through Grok, Codex, and `human-writing`, and generate validated 15-second vertical segments through the subscribed Grok CLI while retaining H3 manual import.

**Architecture:** Keep the existing durable orchestrator and artifact manifests. Add one title-only source mode, one mandatory web-enabled Grok research adapter, one explicit `human-writing` service, and a provider-neutral video boundary. Assemble those services through guarded stage runners so local work is always available and every network, quota, or cloud operation still requires the current command to carry `--allow-external`.

**Tech Stack:** Python 3.11, Pydantic 2, Typer, PyYAML, httpx, pytest 8, Codex CLI, Grok CLI 1.0.4+, IndexTTS2, Volcengine ASR, FFmpeg, FFprobe.

## Global Constraints

- Main implementation and first-pass tests are written by Grok CLI; Sol defines each bounded task, reviews every diff, runs independent tests, and owns acceptance.
- Work only in `<repo-root>\.worktrees\bv-workstation-v0.1` on `feature/bv-workstation-v0.1`.
- Do not add a GUI, web server, database, vector database, workflow engine, auto-publishing, xAI API adapter, or H3 API adapter.
- Do not install dependencies or change dependency versions unless a task explicitly names the change; this plan requires no new dependency.
- Never read, print, persist, or commit API keys, cookies, session tokens, authorization headers, or browser data.
- `--allow-external` authorizes only the current command. It is never written to configuration or state.
- No live Codex, Grok research, Grok media, Volcengine, or quota-consuming call is allowed in automated tests.
- The operator's publisher promotion rights are an accepted business premise. V0.1 has no copyright-proof, ISBN-approval, contract-storage, or edition-approval workflow.
- E001 remains a whole-book reader-value video, not an isolated opinion clip.
- The complete IndexTTS2 narration remains the only master timeline. Generated segment dialogue never becomes the narration.
- Generate and review S01 before allowing S02 through S04/S05.
- Preserve fail-closed path, hash, regular-file, reparse-point, stage-manifest, invalidation, and atomic-publication behavior.
- Existing file imports and existing H3 segment imports remain backward compatible.
- Use test-driven development and commit after each accepted task.

## File Structure

New focused modules:

- `src/bv/content/research.py`: title/author research schemas, prompt construction, validation, and durable research artifact.
- `src/bv/models/grok_research.py`: mandatory web-enabled structured Grok adapter; separate from optional non-blocking review.
- `src/bv/content/human_writing.py`: local Skill identity, Codex naturalization request, checker execution, and fact-lock return contract.
- `src/bv/video/contracts.py`: provider-neutral segment prompt, generation request/result, status, and gateway protocols.
- `src/bv/video/grok_cli.py`: restricted Grok CLI reference-image and `reference_to_video` adapter.
- `src/bv/workflow/content_runtime.py`: concrete runners from source preparation through the script gate.
- `src/bv/workflow/media_runtime.py`: concrete runners from TTS through QC plus video generation/import gateway.
- `src/bv/workflow/runtime.py`: small assembly layer and per-command external authorization.

Existing modules changed in place:

- `src/bv/state/models.py`: backward-compatible source mode and provider-neutral episode statuses.
- `src/bv/books/service.py`: metadata-only project creation beside file import.
- `src/bv/config.py` and `config.example.yaml`: Skill paths and video-provider configuration.
- `src/bv/content/scripts.py`: reorder Grok critique, adjudication, `human-writing`, and final fact checks.
- `src/bv/storyboard/generate.py`: neutral video prompt names with H3 compatibility aliases.
- `src/bv/video/import_h3.py`: provider-bound generic wrappers while preserving old public functions.
- `src/bv/workflow/stages.py` and `src/bv/workflow/orchestrator.py`: neutral video gate and generation method.
- `src/bv/cli.py`: title-only `new`, `--allow-external`, `generate-video`, and visual-sample approval.
- `src/bv/workflow/doctor.py`: report local Skill and Grok media readiness separately from live proof.

---

### Task 1: Metadata-Only Book Projects

**Files:**
- Modify: `src/bv/state/models.py`
- Modify: `src/bv/books/service.py`
- Modify: `src/bv/cli.py`
- Modify: `tests/unit/test_book_import.py`
- Modify: `tests/unit/test_cli_bootstrap.py`
- Create: `tests/integration/test_title_author_creation.py`

**Interfaces:**
- Consumes: `BookIdentity.from_metadata(title, authors)` and `StateStore`.
- Produces: `BookCreationReceipt`, `create_book_from_metadata(title: str, authors: Iterable[str], store: StateStore) -> BookCreationReceipt`, and `BookState.source_mode`.

- [ ] **Step 1: Write failing state compatibility tests**

Add tests proving old JSON still loads as file mode and a title-only state carries authors:

```python
def test_old_book_state_defaults_to_file_mode() -> None:
    state = BookState.model_validate({"book_id": "book-demo", "title": "Demo"})
    assert state.source_mode == "file"
    assert state.authors == []


def test_title_author_state_records_identity_without_source_file() -> None:
    state = BookState(
        book_id="book-demo",
        title="Demo",
        authors=["Writer"],
        source_mode="title_author",
    )
    assert state.source_ids == []
```

- [ ] **Step 2: Run the state tests and confirm failure**

Run:

```powershell
uv run pytest tests/unit/test_state_store.py tests/unit/test_book_import.py -q
```

Expected: failure because `BookState` has no `source_mode` or `authors` field.

- [ ] **Step 3: Add backward-compatible fields**

Implement exact fields:

```python
SourceMode = Literal["file", "title_author"]


class BookState(BaseModel):
    book_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    source_mode: SourceMode = "file"
    source_ids: list[str] = Field(default_factory=list)
```

Update `BookImportReceipt` construction so file imports persist `authors=identity.authors` and `source_mode="file"`.

- [ ] **Step 4: Write failing metadata-only creation tests**

Cover creation, idempotency, unsafe store redirection, E001 creation, and event recording:

```python
def test_create_book_from_metadata_is_idempotent(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "workspace")
    first = create_book_from_metadata("被讨厌的勇气", ["岸见一郎", "古贺史健"], store)
    second = create_book_from_metadata("被讨厌的勇气", ["岸见一郎", "古贺史健"], store)
    assert first.book_id == second.book_id
    assert second.source_mode == "title_author"
    assert second.source_ids == []
    assert store.load_episode(first.book_id, "E001").status == "source_imported"
```

- [ ] **Step 5: Implement metadata-only creation with existing path and lock rules**

Add `BookCreationReceipt(BookState)` with `episode_ids: list[str]` and `idempotent: bool`. Add the exact public signature `create_book_from_metadata(title: str, authors: Iterable[str] | str, store: StateStore) -> BookCreationReceipt`.

```python
class BookCreationReceipt(BookState):
    episode_ids: list[str]
    idempotent: bool
```

The implementation must use `BookIdentity`, `_build_paths`, `_import_lock`, `StateStore.save_book`, `StateStore.save_episode`, and the append-only event ledger. It must reject an empty normalized title or empty normalized author list with `ValueError` before writing.

- [ ] **Step 6: Add the dual-mode CLI contract**

Change `new` so the positional file is optional and these are valid:

```powershell
bv new "E:\books\demo.epub"
bv new --title "被讨厌的勇气" --author "岸见一郎" --author "古贺史健"
```

Reject these with exit code 2 and stable messages:

```text
book_source_required
book_author_required
```

Preserve the existing positional-file metadata override: a file may still be combined with `--title` and repeatable `--author`, and that path remains a file import whose output begins `Imported:`. Title-only creation is selected only when the positional file is absent.

- [ ] **Step 7: Run targeted and full tests**

Run:

```powershell
uv run pytest tests/unit/test_book_import.py tests/unit/test_cli_bootstrap.py tests/integration/test_title_author_creation.py -q
uv run pytest -q
```

Expected: all tests pass; no external process is invoked.

- [ ] **Step 8: Commit**

```powershell
git add src/bv/state/models.py src/bv/books/service.py src/bv/cli.py tests/unit/test_book_import.py tests/unit/test_cli_bootstrap.py tests/integration/test_title_author_creation.py
git commit -m "feat: create books from title and author"
```

---

### Task 2: Mandatory Grok Book Research

**Files:**
- Create: `src/bv/models/grok_research.py`
- Create: `src/bv/content/research.py`
- Create: `tests/unit/test_grok_research_adapter.py`
- Create: `tests/unit/test_book_research.py`

**Interfaces:**
- Consumes: `BookState`, `run_command`, `compose_source_prompt`, and an empty private request directory.
- Produces: `GrokResearchModel.complete(prompt, schema_type, request_dir)`, `BookResearch`, `BookResearchResult`, and `research_book(book: BookState, model: GrokResearchModel, output_root: Path, prompt_text: str) -> BookResearchResult` with the first four parameters keyword-only.

- [ ] **Step 1: Write the adapter argument tests**

Require a web-enabled, memoryless, non-editing invocation:

```python
def test_research_argv_enables_web_without_edit_permissions(tmp_path: Path) -> None:
    argv = build_grok_research_argv(tmp_path / "prompt.md", "{}", tmp_path)
    assert argv[:2] == ["grok", "--prompt-file"]
    assert "--disable-web-search" not in argv
    assert ["--no-memory", "--no-subagents"] == [
        argv[argv.index("--no-memory")],
        argv[argv.index("--no-subagents")],
    ]
    assert argv[argv.index("--permission-mode") + 1] == "plan"
```

Also test timeout, authentication failure, invalid JSON, extra files, redirected paths, and error redaction. Unlike optional `GrokCliModel`, every failure raises `GrokResearchError` with one of:

```text
book_research_timeout
book_research_auth_failed
book_research_process_failed
book_research_invalid
book_research_directory_unsafe
```

- [ ] **Step 2: Run the adapter tests and confirm failure**

Run:

```powershell
uv run pytest tests/unit/test_grok_research_adapter.py -q
```

Expected: import failure because `bv.models.grok_research` does not exist.

- [ ] **Step 3: Implement the mandatory adapter**

Implement:

```python
class GrokResearchError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)
```

Add `GrokResearchModel.__init__(*, command: str = "grok", timeout: float = 600.0) -> None` and `GrokResearchModel.complete(prompt: str, schema_type: type[ModelT], request_dir: Path) -> ModelT`. Write `prompt.md`, call Grok with `--json-schema` and `--output-format json`, parse either a direct schema object or the CLI envelope's `text` field, and accept only the expected prompt artifact in the request directory.

- [ ] **Step 4: Write the research schema tests**

Define strict Pydantic models with `extra="forbid"`:

```python
class ResearchSource(BaseModel):
    url: str
    source_type: Literal[
        "publisher", "author_interview", "official_sample", "table_of_contents",
        "academic", "professional_review", "reader_reception", "other",
    ]
    supports: tuple[str, ...]


class BookResearch(BaseModel):
    title: str
    authors: tuple[str, ...]
    identity_confidence: Literal["high", "medium"]
    central_question: str
    argument_or_narrative_arc: tuple[str, ...]
    key_ideas: tuple[str, ...]
    life_connections: tuple[str, ...]
    reader_value: tuple[str, ...]
    misunderstandings_and_boundaries: tuple[str, ...]
    sources: tuple[ResearchSource, ...]
    uncertainties: tuple[str, ...]
```

Tests must reject blank fields, zero sources, duplicate source URLs, unsupported URL schemes, fewer than two key ideas, and a result whose returned title/authors do not normalize to the requested identity.

- [ ] **Step 5: Implement research orchestration and atomic artifacts**

Implement:

```python
class BookResearchResult(BaseModel):
    status: Literal["accepted"]
    research: BookResearch
    json_path: Path
    markdown_path: Path
    prompt_sha256: str
```

Implement `research_book(*, book: BookState, model: GrokResearchModel, output_root: Path, prompt_text: str) -> BookResearchResult`. The trusted prompt must request the whole-book reader value, material identities, URLs, uncertainties, and no invented quotations. Encode title and author as untrusted JSON data. Publish `book_research.json` and `book_research.md` atomically only after validation.

- [ ] **Step 6: Run targeted and full tests**

```powershell
uv run pytest tests/unit/test_grok_research_adapter.py tests/unit/test_book_research.py -q
uv run pytest -q
```

- [ ] **Step 7: Commit**

```powershell
git add src/bv/models/grok_research.py src/bv/content/research.py tests/unit/test_grok_research_adapter.py tests/unit/test_book_research.py
git commit -m "feat: research books through Grok"
```

---

### Task 3: Human Writing Script Stage

**Files:**
- Create: `src/bv/content/human_writing.py`
- Modify: `src/bv/content/scripts.py`
- Modify: `src/bv/config.py`
- Modify: `config.example.yaml`
- Create: `tests/unit/test_human_writing.py`
- Modify: `tests/unit/test_value_scripts.py`
- Modify: `tests/unit/test_script_approval.py`

**Interfaces:**
- Consumes: `ScriptDraft`, `SemanticLock`, `GrokScriptReview`, `StructuredModel`, and the installed `human-writing` files.
- Produces: `HumanWritingService.naturalize(*, draft: str, semantic_lock: SemanticLock, coverage: VoiceoverCoverage, request_root: Path) -> HumanWritingResult` and a reordered `ScriptService.create(episode: EpisodeBrief, value: WholeBookValue) -> ScriptPackage`.

- [ ] **Step 1: Add configuration tests**

Require resolved paths without embedding user secrets:

```python
assert config.human_writing.skill_path.name == "SKILL.md"
assert config.human_writing.checker_path.name == "check_prose.py"
```

Add:

```python
class HumanWritingConfig(_ConfigModel):
    skill_path: Path = Path.home() / ".codex/skills/human-writing/SKILL.md"
    checker_path: Path = Path.home() / ".codex/skills/human-writing/scripts/check_prose.py"
```

Resolve both paths relative to the config file when they are not absolute.

- [ ] **Step 2: Write failing Skill identity and checker tests**

Cover missing files, redirected files, hashes over `SKILL.md`, `references/reality.md`, `references/formats.md`, `references/revision.md`, and `scripts/check_prose.py`, checker timeout, nonzero checker result, unexpected model output, and fact-lock mismatch.

Core test:

```python
def test_naturalize_runs_checker_then_returns_bound_result(skill_fixture: Path) -> None:
    service = HumanWritingService(
        model=fake_model,
        skill_path=skill_fixture / "SKILL.md",
        checker_path=skill_fixture / "scripts" / "check_prose.py",
        runner=fake_runner,
    )
    result = service.naturalize(
        draft="一段有材料的口播",
        semantic_lock=lock,
        coverage=coverage,
        request_root=request_root,
    )
    assert result.skill_sha256 == expected_combined_sha256
    assert result.checker_passed is True
    assert result.voiceover
```

- [ ] **Step 3: Implement Skill identity and naturalization**

Implement exact contracts:

```python
class HumanWritingResult(BaseModel):
    voiceover: str
    coverage: VoiceoverCoverage
    skill_sha256: str
    checker_passed: Literal[True]
```

Add `HumanWritingService.__init__(*, model: StructuredModel, skill_path: Path, checker_path: Path, runner: CommandRunner = run_command) -> None` and `HumanWritingService.naturalize(*, draft: str, semantic_lock: SemanticLock, coverage: VoiceoverCoverage, request_root: Path) -> HumanWritingResult`. The model prompt must explicitly invoke `$human-writing`, name the installed `SKILL.md`, require `reality.md` and `formats.md` for the first pass, require `revision.md` after drafting, and preserve every locked fact. Save the candidate only in the private request directory, run the checker through `sys.executable` with an argument list, then return validated structured output. Never include the electronic book itself.

- [ ] **Step 4: Reorder ScriptService with failing sequence tests**

Assert this exact model-service order:

```text
draft
grok_review
adjudicate_review
human_writing
fact_diff
final_validate
```

`GrokScriptReview` remains non-blocking. `human-writing`, fact diff, and final validation are blocking. Add `human_writing_sha256` and `human_writing_checker_passed` to `ScriptPackage` and the approval manifest.

- [ ] **Step 5: Implement the minimal reorder**

Inject `human_writing: HumanWritingService` into `ScriptService`. Remove the old generic `_humanize` pass from the accepted path without deleting compatibility parsing needed by old private artifacts. If Grok is skipped, the Skill still runs. If the final text differs after naturalization, recompute coverage and fact diff before returning.

- [ ] **Step 6: Run checker, script, approval, and full tests**

```powershell
uv run pytest tests/unit/test_human_writing.py tests/unit/test_value_scripts.py tests/unit/test_script_approval.py -q
uv run pytest -q
```

- [ ] **Step 7: Commit**

```powershell
git add src/bv/content/human_writing.py src/bv/content/scripts.py src/bv/config.py config.example.yaml tests/unit/test_human_writing.py tests/unit/test_value_scripts.py tests/unit/test_script_approval.py
git commit -m "feat: naturalize scripts with human writing"
```

---

### Task 4: Provider-Neutral Video Contracts and H3 Compatibility

**Files:**
- Create: `src/bv/video/contracts.py`
- Modify: `src/bv/storyboard/generate.py`
- Modify: `src/bv/video/import_h3.py`
- Modify: `src/bv/state/models.py`
- Modify: `src/bv/workflow/stages.py`
- Modify: `src/bv/workflow/orchestrator.py`
- Modify: `tests/fakes.py`
- Modify: `tests/unit/test_storyboard_segments.py`
- Modify: `tests/unit/test_h3_segment_import.py`
- Modify: `tests/unit/test_orchestrator.py`
- Modify: `tests/integration/test_cli_value_workflow.py`

**Interfaces:**
- Consumes: current `H3PromptArtifact`, import validation, and orchestrator video gate.
- Produces: `VideoPromptArtifact`, `VideoGateway`, generic video manifest provider identity, and backward-compatible H3 aliases/wrappers.

- [ ] **Step 1: Write neutral contract tests**

Define:

```python
VideoProviderName = Literal["grok_cli", "grok_manual", "h3_manual"]


class VideoGenerationRequest(BaseModel):
    book_id: str
    episode_id: str
    segment_id: str
    provider: VideoProviderName
    prompt: str
    prompt_sha256: str
    expected_duration_ms: int
    aspect_ratio: Literal["9:16"] = "9:16"
    resolution: Literal["480p", "720p"] = "480p"
    reference_images: tuple[Path, ...] = ()
    reference_image_specs: tuple[str, ...] = ()


class VideoGenerationResult(BaseModel):
    provider: VideoProviderName
    segment_id: str
    source_path: Path
    source_sha256: str


class VideoGateway(Protocol):
    def generate(self, book_id: str, episode_id: str, shot_id: str) -> object:
        raise NotImplementedError

    def import_video(self, book_id: str, episode_id: str, shot_id: str, source: Path) -> object:
        raise NotImplementedError

    def status(self, book_id: str, episode_id: str) -> object:
        raise NotImplementedError
```

Reject bad identifiers, non-SHA hashes, empty prompts, duplicate references, provided-but-nonexistent references, references outside the episode workspace, more than seven references, and duration over 15,000 ms. Require at least one existing reference image or one non-empty reference-image specification. S01 normally starts from specifications; later shots reuse accepted shared reference paths and add one scene specification.

- [ ] **Step 2: Run contract tests and confirm failure**

```powershell
uv run pytest tests/unit/test_storyboard_segments.py tests/unit/test_h3_segment_import.py tests/unit/test_orchestrator.py -q
```

- [ ] **Step 3: Add neutral names with compatibility aliases**

Expose neutral aliases and make `render_video_prompts(storyboard: Storyboard) -> tuple[VideoPromptArtifact, ...]` the primary renderer:

```python
VideoSegment = H3Segment
VideoPromptArtifact = H3PromptArtifact
render_h3_prompts = render_video_prompts
```

Do not duplicate prompt-generation logic. New artifacts use `video_prompt_S01.md`; readers continue accepting old `h3_prompt_S01.md` while existing projects are loaded.

- [ ] **Step 4: Bind imported media to a provider**

Expose `VideoImportResult = H3ImportResult` and `VideoSegmentSetStatus = H3SegmentSetStatus` during the compatibility migration. Add `provider: VideoProviderName = "h3_manual"` to segment and aggregate manifests so old JSON remains valid. Implement `inspect_video_segments(*, book_id: str, episode_id: str, prompt_artifacts: Sequence[VideoPromptArtifact], episode_media_root: Path, provider: VideoProviderName) -> VideoSegmentSetStatus` and `import_video_segment(*, book_id: str, episode_id: str, segment_id: str, prompt_artifacts: Sequence[VideoPromptArtifact], source_path: Path, episode_media_root: Path, provider: VideoProviderName, ffprobe_runner: Callable[..., CommandResult] | None = None) -> VideoImportResult`.

Keep `inspect_h3_segments` and `import_h3_segment` as wrappers passing `h3_manual`. Provider changes must make existing media stale instead of accepting mixed manifests.

- [ ] **Step 5: Neutralize orchestrator behavior**

Use `video_gateway` as the primary constructor argument and retain deprecated `h3_gateway` only as a mutually exclusive compatibility keyword. New states and commands are:

```text
awaiting_video_generation
video_partial
video_imported
video_gateway_not_configured
bv open book-demo E001 video
```

Continue loading old H3 statuses and map them to the equivalent new behavior on the next save. Preserve `visual_sample_ready`.

- [ ] **Step 6: Run targeted and full tests**

```powershell
uv run pytest tests/unit/test_storyboard_segments.py tests/unit/test_h3_segment_import.py tests/unit/test_orchestrator.py tests/integration/test_cli_value_workflow.py -q
uv run pytest -q
```

- [ ] **Step 7: Commit**

```powershell
git add src/bv/video/contracts.py src/bv/storyboard/generate.py src/bv/video/import_h3.py src/bv/state/models.py src/bv/workflow/stages.py src/bv/workflow/orchestrator.py tests/fakes.py tests/unit/test_storyboard_segments.py tests/unit/test_h3_segment_import.py tests/unit/test_orchestrator.py tests/integration/test_cli_value_workflow.py
git commit -m "refactor: generalize video segment providers"
```

---

### Task 5: Restricted Grok CLI Video Adapter

**Files:**
- Create: `src/bv/video/grok_cli.py`
- Modify: `src/bv/config.py`
- Modify: `config.example.yaml`
- Create: `tests/unit/test_grok_video_adapter.py`

**Interfaces:**
- Consumes: `VideoGenerationRequest`, Grok CLI login, and generated reference images.
- Produces: `GrokCliVideoProvider.generate(request) -> VideoGenerationResult`.

- [ ] **Step 1: Add provider configuration tests**

Require:

```python
class VideoGenerationConfig(_ConfigModel):
    provider: Literal["grok_cli", "grok_manual", "h3_manual"] = "grok_cli"
    resolution: Literal["480p", "720p"] = "480p"
    command: str = "grok"
    timeout_seconds: int = 900
```

Replace `video.h3_segment_target_seconds` and `video.h3_segment_max_seconds` with neutral names while accepting the old YAML keys through validation aliases.

- [ ] **Step 2: Write the CLI and parser tests**

Require a no-memory, no-subagent, restricted tool invocation:

```python
def test_video_argv_restricts_tools(tmp_path: Path) -> None:
    argv = build_grok_video_argv(tmp_path / "prompt.md", tmp_path)
    assert argv[0] == "grok"
    assert argv[argv.index("--tools") + 1] == "image_gen,reference_to_video"
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert "--no-memory" in argv
    assert "--no-subagents" in argv
```

Fixture outputs must cover direct final JSON, CLI envelope JSON, missing `text`, malformed paths, nonexistent files, oversized files, redirected files, output outside the request directory, process timeout, authentication failure, and redaction of the full prompt.

- [ ] **Step 3: Implement request prompts and safe result parsing**

Implement:

```python
class GrokVideoError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _GrokVideoResponse(BaseModel):
    segment_id: str
    reference_image_paths: tuple[Path, ...]
    video_path: Path
```

Add `GrokCliVideoProvider.__init__(*, command: str = "grok", timeout: float = 900.0, runner: CommandRunner = run_command) -> None` and `GrokCliVideoProvider.generate(request: VideoGenerationRequest) -> VideoGenerationResult`. For every missing reference described by `reference_image_specs`, the prompt must call `image_gen` and retain the resulting local path. It then calls `reference_to_video` exactly once with the supplied and newly generated references, uses 9:16 and the exact requested duration/resolution, creates no dialogue or on-screen text, and returns only `_GrokVideoResponse`. The adapter accepts only regular, non-reparse files inside the isolated request directory, snapshots the accepted reference images and final MP4, computes SHA-256 itself, and never trusts a model-reported hash. Shared references created for S01 become hash-bound inputs to later segment requests.

- [ ] **Step 4: Add ffprobe validation**

Probe the generated MP4 before returning. Require one video stream, portrait aspect near 9:16, duration within the existing shortfall/tolerance contract, and size at most 500 MiB. Map failures to safe codes:

```text
video_generation_timeout
video_generation_auth_failed
video_generation_failed
video_output_missing
video_output_unsafe
video_output_invalid
```

- [ ] **Step 5: Run targeted and full tests**

```powershell
uv run pytest tests/unit/test_grok_video_adapter.py tests/unit/test_config.py -q
uv run pytest -q
```

- [ ] **Step 6: Commit**

```powershell
git add src/bv/video/grok_cli.py src/bv/config.py config.example.yaml tests/unit/test_grok_video_adapter.py tests/unit/test_config.py
git commit -m "feat: generate segments through Grok CLI"
```

---

### Task 6: Content Runtime Assembly Through Script Approval

**Files:**
- Create: `src/bv/workflow/content_runtime.py`
- Create: `src/bv/workflow/runtime.py`
- Modify: `src/bv/cli.py`
- Create: `tests/unit/test_content_runtime.py`
- Create: `tests/integration/test_title_to_script_gate.py`

**Interfaces:**
- Consumes: Tasks 1 through 3, existing parse/analyze/synthesis/topic/script services, and `StageRunner`.
- Produces: `RuntimeAuthorization`, `build_runtime_bindings(config, authorization)`, and a real `build_orchestrator(config, authorization)` through the script gate.

- [ ] **Step 1: Write authorization tests**

Define:

```python
@dataclass(frozen=True)
class RuntimeAuthorization:
    allow_external: bool = False


class ExternalAuthorizationError(RuntimeError):
    error_code = "external_not_authorized"
```

Without permission, title research, Codex generation, Grok critique, and any cloud call raise before invoking a process. Local source parsing and deterministic validation continue to run.

- [ ] **Step 2: Write stage artifact tests**

For title mode, `parse_source` writes a metadata snapshot and `analyze_chapters` writes accepted `book_research.json`/Markdown through Grok. For file mode, preserve the current parser and chapter analysis. Both modes must feed the existing whole-book synthesis contract and arrive at `awaiting_script_review` with:

```text
book_value_card.json
fact_lock.json
script_package.json
script/review.md
```

Every `StageOutcome.inputs` value is a SHA-256. Empty output, stale prompt, changed Skill hash, or mismatched title identity fails closed.

- [ ] **Step 3: Implement focused stage runners**

Implement classes with only a `run(context: StageContext) -> StageOutcome` public method:

```python
TitleOrFileParseStage
BookAnalysisStage
BookValueStage
EpisodeBriefStage
ScriptDraftStage
ScriptApprovalStage
```

Each runner loads current upstream Pydantic artifacts, invokes one existing service, writes only its own output directory, then returns exact hashes. It never calls the next stage directly.

- [ ] **Step 4: Assemble the real content runtime**

`build_runtime_bindings` returns stage and gate-approver mappings. `build_orchestrator` must no longer return an empty stage mapping. Add `--allow-external` to `bv next` and `bv approve`; pass it only into the current command's authorization object.

Expected behavior:

```powershell
bv next book-demo
# Error: external_not_authorized at the first external stage

bv next book-demo --allow-external
# Advances until awaiting_script_review
```

- [ ] **Step 5: Run fake integration and full tests**

```powershell
uv run pytest tests/unit/test_content_runtime.py tests/integration/test_title_to_script_gate.py tests/integration/test_cli_value_workflow.py -q
uv run pytest -q
```

Tests inject fake models; no live CLI is called.

- [ ] **Step 6: Commit**

```powershell
git add src/bv/workflow/content_runtime.py src/bv/workflow/runtime.py src/bv/cli.py tests/unit/test_content_runtime.py tests/integration/test_title_to_script_gate.py
git commit -m "feat: wire title research to script review"
```

---

### Task 7: Media Runtime, Video Generation Command, and Final Render

**Files:**
- Create: `src/bv/workflow/media_runtime.py`
- Modify: `src/bv/workflow/runtime.py`
- Modify: `src/bv/workflow/orchestrator.py`
- Modify: `src/bv/cli.py`
- Create: `tests/unit/test_media_runtime.py`
- Create: `tests/integration/test_grok_video_workflow.py`
- Modify: `tests/integration/test_multiclip_render_pipeline.py`

**Interfaces:**
- Consumes: approved script, IndexTTS2, Volcengine ASR, subtitles, storyboard, Task 4 gateway, Task 5 provider, renderer, and QC.
- Produces: concrete stages from TTS through QC, `Orchestrator.generate_video(book_id: str, episode_id: str, shot_id: str) -> WorkflowView`, `bv generate-video`, and visual-sample approval.

- [ ] **Step 1: Write media-stage authorization and ordering tests**

Local IndexTTS2 and FFmpeg do not require `--allow-external`. Volcengine ASR and Grok video do. After script approval, `bv next` advances until an unauthorized external stage and reports its exact next command instead of running it.

Required sequence:

```text
tts
asr
subtitles
storyboard
await_video_segments
render
qc
```

- [ ] **Step 2: Implement media stage runners**

Implement:

```python
TtsStage
AsrStage
SubtitleStage
StoryboardStage
RenderStage
QcStage
RuntimeVideoGateway
```

`RuntimeVideoGateway` loads the current neutral prompt artifact, reference set, and provider configuration. `generate` calls only `grok_cli`; manual providers return `video_manual_import_required`. All generated outputs pass through the same provider-bound import validation before becoming present segments.

- [ ] **Step 3: Add visual-sample state behavior**

Implement `Orchestrator.generate_video(self, book_id: str, episode_id: str, shot_id: str) -> WorkflowView` under the existing episode lock.

Rules:

- Only the exact missing shot can be generated.
- S01 is the only allowed first shot.
- After S01 import, status is `visual_sample_ready`.
- S02 through S04/S05 return `visual_sample_not_approved` until `bv approve book-demo E001 visual-sample` succeeds.
- Approval is invalidated if S01, its prompt, reference images, script, audio, subtitle, storyboard, provider, or config changes.

- [ ] **Step 4: Add the CMD commands**

Implement:

```powershell
bv generate-video book-demo E001 S01 --allow-external
bv approve book-demo E001 visual-sample
bv open book-demo E001 video
```

`generate-video` without `--allow-external` returns `video_generation_not_authorized` without starting Grok. Manual import remains:

```powershell
bv import-video book-demo E001 S01 D:\Downloads\S01.mp4
```

- [ ] **Step 5: Wire final render and QC**

All required imported segments, approved narration, subtitle, cover, and current hashes are mandatory. Preserve existing FFmpeg normalization, concat, subtitle, cover, loudness, and publication behavior. Generated segment audio is optional ambience only; it cannot replace the master narration.

- [ ] **Step 6: Run media integration and full tests**

```powershell
uv run pytest tests/unit/test_media_runtime.py tests/integration/test_grok_video_workflow.py tests/integration/test_multiclip_render_pipeline.py -q
uv run pytest -q
```

All media providers are fake in automated tests.

- [ ] **Step 7: Commit**

```powershell
git add src/bv/workflow/media_runtime.py src/bv/workflow/runtime.py src/bv/workflow/orchestrator.py src/bv/cli.py tests/unit/test_media_runtime.py tests/integration/test_grok_video_workflow.py tests/integration/test_multiclip_render_pipeline.py
git commit -m "feat: wire Grok segments to final render"
```

---

### Task 8: Doctor, Operating Guide, Adversarial Review, and Private Gate

**Files:**
- Modify: `src/bv/workflow/doctor.py`
- Modify: `tests/unit/test_doctor.py`
- Modify: `README.md`
- Create: `docs/operations/title-to-video.md`
- Create: `tests/integration/test_runtime_fail_closed.py`

**Interfaces:**
- Consumes: every accepted task.
- Produces: honest readiness reporting, operator commands, full adversarial suite, and the boundary before the user's first book.

- [ ] **Step 1: Add doctor tests**

`bv doctor` must distinguish:

```text
Grok executable found
Grok login unverified or ready
Grok native media tools unverified or ready
human-writing files ready or missing
IndexTTS2 project/model/check ready or missing
IndexTTS2 real WAV unverified or ready
Volcengine credentials present or missing
FFmpeg and FFprobe ready or missing
```

Only `doctor --live` may use external probes. A live Grok media probe still must not generate a paid/quota video; real generation remains the explicit S01 private gate.

- [ ] **Step 2: Add adversarial runtime tests**

Cover at minimum:

- missing `--allow-external` performs zero process calls;
- malicious title/author cannot alter commands or paths;
- model output cannot inject an output path;
- stale Skill files invalidate scripts;
- changed provider invalidates segments;
- forged S01 approval cannot unlock later shots;
- generated file outside the request directory is rejected;
- redirected reference image or video is rejected;
- empty research, script, audio, subtitle, storyboard, video, render, or QC output never completes a stage;
- exceptions and CLI output never reveal prompts, book text, credentials, cookies, or tokens.

- [ ] **Step 3: Write exact CMD operating instructions**

Document the title-only happy path with concrete commands and status meanings. Clearly label four levels:

```text
configured
fake-tested
real provider connected
private end-to-end accepted
```

State that automated tests do not prove a live Grok video, a real IndexTTS2 voice, a real Volcengine response, or a final private book video.

- [ ] **Step 4: Run static, targeted, and full verification**

```powershell
python -m compileall -q src tests
uv run pytest tests/integration/test_runtime_fail_closed.py -q
uv run pytest -q
git diff --check
git status --short
```

Expected before the documentation commit: compile succeeds, all tests pass, diff check is empty, and only Task 8 files are modified.

- [ ] **Step 5: Commit**

```powershell
git add src/bv/workflow/doctor.py tests/unit/test_doctor.py README.md docs/operations/title-to-video.md tests/integration/test_runtime_fail_closed.py
git commit -m "docs: finish title-to-video runtime"
```

- [ ] **Step 6: Sol performs independent final review**

Sol reviews all commits since `e0ac674`, reruns the full suite, checks every public command, scans for secret-like values, and probes the fail-closed paths. Any substantive defect is sent back to Grok CLI as a bounded repair task; Sol only makes narrow hardening edits and records them in a separate commit.

- [ ] **Step 7: Stop at the private-input gate**

Do not invent a sample book and do not spend video quota. Report the exact ready state and ask the user for the first title and author. The subsequent private acceptance begins with:

```powershell
bv new --title "用户提供的真实书名" --author "用户提供的真实作者"
```

The pipeline is complete only after the user's real S01 and final MP4 pass the private acceptance checklist in the approved design.
