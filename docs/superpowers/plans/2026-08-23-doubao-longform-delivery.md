# Doubao Long-Form Book Video Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留旧版 45–60 秒 IndexTTS2 流程的同时，为单期节目增加豆包异步长文本旁白、120–150 秒手绘视频、独立作品封面和北京时间日期交付，并以余华《活着》完成真实端到端验收。

**Architecture:** Episode 根目录中的 `production_profile.json` 是本期时长、TTS、画面和交付参数的唯一持久来源；缺少该文件的旧 Episode 走完全相同的旧默认值。TTS 通过统一 Provider 接口路由，本地标准化 WAV 后仍以火山 ASR 作为字幕权威时间轴；技术 QC 通过后再把稳定内部产物原子复制为不可覆盖的日期交付包，用户最终批准只新增独立批准记录。

**Tech Stack:** Python 3.11、Pydantic 2、httpx、pytest、FFmpeg/ffprobe、Remotion、火山 ASR、豆包语音异步长文本 API、JSON/SHA-256 文件合同。

**Spec:** `docs/superpowers/specs/2026-08-23-doubao-longform-delivery-design.md`

> **2026-08-24 protocol amendment:** Task 3 的 v1 请求示例已被豆包大模型异步长文本 v3 协议取代，不得再作为实现依据。当前合同以 `src/bv/voice/doubao.py` 及其测试为准：`POST /api/v3/tts/submit`、`POST /api/v3/tts/query`，使用 `x-api-key`、`X-Api-Resource-Id: seed-tts-2.0` 和 `X-Api-Request-Id`，不要求 APP ID；请求正文使用 `req_params`，响应读取 `code=20000000` 下的 `data`，状态 1/2/3 分别表示处理中/成功/失败。旧版 `volc.tts_async.default`、`Resource-Id`、`Bearer;`、APP ID + Access Token 及 0/1/2 状态示例均已废止。

## Global Constraints

- 不新增 CMD、网页、GUI、BGM、平台上传或第二/第三主题。
- 旧 Episode 没有 `production_profile.json` 时仍使用 IndexTTS2 黑金3与 45/48/56/60 秒时长带。
- 《活着》正式成片硬范围为 120–150 秒，理想范围为 128–142 秒，目标约 135 秒。
- 豆包正式旁白只允许一次提交；查询不计提交；不自动重试、不自动改音色、不回退 IndexTTS2。
- 豆包音色必须先以同一段 15–25 秒批准文稿试听，候选最多三个；用户批准前 `voice_type` 必须为 `null`。
- 凭据只从 `BV_DOUBAO_TTS_API_KEY` 读取；公开信息只能是 `SET`/`UNSET`。
- 不持久化 Token、Authorization Header、完整 Provider 响应、签名下载 URL 或原始错误正文。
- 正式旁白统一生成 48 kHz、单声道 PCM WAV；最终视频为 1080×1920、30 fps、H.264、`yuv420p`，音频 AAC 48 kHz 双声道。
- 字幕一行展示、去标点、Microsoft YaHei 粗体、半透明深色底框；画面使用 15 帧交叉叠化，不出现纸白空场。
- 代表插画只生成开头、中段、结尾三张，用户批准后才批量生成其余插画。
- 日期按 `Asia/Shanghai` 计算；同日重复导出整包升级为 V2/V3，任何已有文件、符号链接、junction 或重解析目标都拒绝覆盖。
- 文稿必须讲整本书给读者的价值，约 60% 当代生活联系、40% 原著人物与情节；不伪造原句、页码、读者来信、作者亲历或人物心理。
- 外部调用仍受一次性 `RuntimeAuthorization` 约束；真实付费调用必须在对应人工门后执行。
- 所有私有试听、回执和历史文件留在 Episode 的 `.private` 中，不提交 Git。

---

### Task 1: Episode Production Profile and Dynamic Script Bounds

**Files:**
- Create: `src/bv/production/__init__.py`
- Create: `src/bv/production/profile.py`
- Modify: `src/bv/content/scripts.py:267-430`
- Modify: `prompts/script_draft/v2.md`
- Modify: `prompts/script_humanize/v2.md`
- Test: `tests/unit/test_production_profile.py`
- Test: `tests/unit/test_value_scripts.py`

**Interfaces:**
- Produces: `ProductionProfile`, `DurationProfile`, `TtsProfile`, `VisualProfile`, `DeliveryProfile`.
- Produces: `load_production_profile(episode_root: Path) -> ProductionProfile` and `production_profile_sha256(episode_root: Path) -> str`.
- Produces: `validate_value_script(text: str, *, semantic_lock: SemanticLock, target_duration: DurationProfile | None = None) -> ScriptValidation` while preserving the legacy call form.
- Consumes: the exact schema in the approved spec; later tasks receive the immutable `ProductionProfile` instance rather than reading loose dictionaries.

- [ ] **Step 1: Add profile parsing and backward-compatibility tests**

```python
def test_missing_profile_uses_legacy_defaults(tmp_path: Path) -> None:
    profile = load_production_profile(tmp_path)
    assert profile.tts.provider == "indextts2"
    assert profile.tts.voice_type == "黑金3"
    assert profile.duration.model_dump() == {
        "hard_min_seconds": 45.0,
        "ideal_min_seconds": 48.0,
        "ideal_max_seconds": 56.0,
        "hard_max_seconds": 60.0,
    }


def test_doubao_profile_accepts_null_voice_before_audition(tmp_path: Path) -> None:
    write_profile(tmp_path, provider="doubao", voice_type=None)
    assert load_production_profile(tmp_path).tts.voice_type is None


def test_doubao_profile_can_persist_the_approved_voice_id(tmp_path: Path) -> None:
    write_profile(tmp_path, provider="doubao", voice_type="reader_voice")
    assert load_production_profile(tmp_path).tts.voice_type == "reader_voice"


def test_profile_hash_changes_when_duration_changes(tmp_path: Path) -> None:
    write_living_profile(tmp_path)
    first = production_profile_sha256(tmp_path)
    data = json.loads((tmp_path / "production_profile.json").read_text("utf-8"))
    data["duration"]["ideal_max_seconds"] = 141.0
    (tmp_path / "production_profile.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    assert production_profile_sha256(tmp_path) != first
```

- [ ] **Step 2: Run the focused tests and confirm the new module is absent**

Run: `python -m pytest tests/unit/test_production_profile.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'bv.production'`.

- [ ] **Step 3: Implement the strict profile models and loader**

```python
class DurationProfile(_ProfileModel):
    hard_min_seconds: float
    ideal_min_seconds: float
    ideal_max_seconds: float
    hard_max_seconds: float

    @model_validator(mode="after")
    def validate_order(self) -> "DurationProfile":
        if not (
            0 < self.hard_min_seconds <= self.ideal_min_seconds
            <= self.ideal_max_seconds <= self.hard_max_seconds
        ):
            raise ValueError("invalid_duration_profile")
        return self


class TtsProfile(_ProfileModel):
    provider: Literal["indextts2", "doubao"]
    resource_id: str
    voice_type: str | None
    speed: float = Field(gt=0.5, le=2.0)


class ProductionProfile(_ProfileModel):
    schema_version: Literal[1]
    duration: DurationProfile
    tts: TtsProfile
    visual: VisualProfile
    delivery: DeliveryProfile


def load_production_profile(episode_root: Path) -> ProductionProfile:
    path = episode_root / "production_profile.json"
    if not path.exists():
        return LEGACY_PRODUCTION_PROFILE
    return ProductionProfile.model_validate_json(path.read_text(encoding="utf-8"))
```

The loader must reject a symlink/reparse profile, unknown fields, unsupported schema versions, scene minimum above scene maximum, representative counts other than 3, timezones other than `Asia/Shanghai`, and any `include` list other than the six approved artifact kinds. The profile parser accepts a null Doubao voice before audition and a nonblank voice after approval; Task 4 owns the cross-file rule that a formal request requires a current `voice_profile.json` matching the persisted voice ID.

- [ ] **Step 4: Make script warnings and prompts profile-aware**

```python
def estimated_script_character_band(duration: DurationProfile) -> tuple[int, int]:
    return (
        round(duration.ideal_min_seconds * 3.0),
        round(duration.ideal_max_seconds * 4.5),
    )


def validate_value_script(
    text: str,
    *,
    semantic_lock: SemanticLock,
    target_duration: DurationProfile | None = None,
) -> ScriptValidation:
    warning_min, warning_max = (
        (180, 240)
        if target_duration is None
        else estimated_script_character_band(target_duration)
    )
```

Pass a rendered `production_context` into both script prompts containing the four exact duration values, target audience pain points, the 60/40 meaning-unit target, and the prohibitions from spec section 3. The prompt must explicitly label facts, interpretations and illustration metaphors as three distinct categories.

- [ ] **Step 5: Run profile and script tests**

Run: `python -m pytest tests/unit/test_production_profile.py tests/unit/test_value_scripts.py -q`

Expected: all selected tests pass; the pre-existing 180–240 character warning assertions remain unchanged when no profile is supplied.

- [ ] **Step 6: Commit the profile boundary**

```powershell
git add src/bv/production src/bv/content/scripts.py prompts/script_draft/v2.md prompts/script_humanize/v2.md tests/unit/test_production_profile.py tests/unit/test_value_scripts.py
git commit -m "feat: add episode production profiles"
```

---

### Task 2: Provider-Neutral Narration Contracts and Legacy Manifest Migration

**Files:**
- Create: `src/bv/voice/providers.py`
- Modify: `src/bv/voice/processing.py:59-240`
- Modify: `src/bv/voice/indextts2.py`
- Modify: `src/bv/workflow/runtime.py:25-60`
- Test: `tests/unit/test_voice_providers.py`
- Test: `tests/unit/test_media_audio_stages.py`
- Test: `tests/integration/test_audio_processing.py`

**Interfaces:**
- Consumes: `ProductionProfile` from Task 1.
- Produces: `NarrationRequest`, `NarrationArtifact`, `ProviderReceipt`, and the `NarrationSynthesizer` protocol.
- Produces: `IndexTTS2NarrationSynthesizer.synthesize(request, authorization) -> NarrationArtifact`.
- Produces: `RuntimeAuthorization.assert_tts_submission(book_id, episode_id, script_sha256, provider_voice_id, kind) -> None`.
- Updates: `VoiceManifest` with `provider`, `provider_voice_id`, nullable `reference_sha256`, and nullable `provider_receipt_sha256`.

- [ ] **Step 1: Write provider and manifest migration tests**

```python
def test_legacy_voice_manifest_migrates_to_indextts2() -> None:
    manifest = VoiceManifest.model_validate(legacy_manifest_payload())
    assert manifest.provider == "indextts2"
    assert manifest.provider_voice_id == manifest.voice_id == "黑金3"
    assert manifest.reference_sha256 is not None
    assert manifest.provider_receipt_sha256 is None


@pytest.mark.parametrize(
    ("provider", "reference_sha256", "receipt_sha256", "error"),
    [
        ("indextts2", None, None, "indextts2_reference_required"),
        ("doubao", "a" * 64, "b" * 64, "doubao_reference_forbidden"),
        ("doubao", None, None, "doubao_receipt_required"),
    ],
)
def test_manifest_enforces_provider_specific_fields(
    provider: str,
    reference_sha256: str | None,
    receipt_sha256: str | None,
    error: str,
) -> None:
    with pytest.raises(ValidationError, match=error):
        VoiceManifest.model_validate(
            manifest_payload(provider, reference_sha256, receipt_sha256)
        )
```

- [ ] **Step 2: Run the focused tests and verify the contract does not exist**

Run: `python -m pytest tests/unit/test_voice_providers.py tests/integration/test_audio_processing.py -q`

Expected: tests fail because `bv.voice.providers` and the new manifest fields are absent.

- [ ] **Step 3: Add immutable provider contracts**

```python
class ProviderReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: Literal["indextts2", "doubao"]
    request_sha256: Sha256
    task_id_sha256: Sha256 | None
    response_sha256: Sha256 | None
    status: Literal["local_complete", "submitted", "working", "success", "failed"]
    error_code: str | None = None


class NarrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    book_id: SafeIdentifier
    episode_id: SafeIdentifier
    approved_script_path: Path
    approved_script_sha256: Sha256
    provider_voice_id: str
    speed: float
    output_dir: Path
    kind: Literal["audition", "formal"]


class NarrationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    raw_audio_path: Path
    raw_audio_sha256: Sha256
    receipt_path: Path
    receipt_sha256: Sha256


class NarrationSynthesizer(Protocol):
    provider: Literal["indextts2", "doubao"]

    def synthesize(
        self,
        request: NarrationRequest,
        authorization: RuntimeAuthorization,
    ) -> NarrationArtifact:
        raise NotImplementedError
```

`RuntimeAuthorization` must include exact optional scope fields: `book_id`, `episode_id`, `script_sha256`, `provider_voice_id`, `max_audition_submissions=0`, `max_formal_submissions=0`, and `allow_fallback=False`. Validation compares every field before an external submit; a query of an existing task does not consume a submission allowance.

- [ ] **Step 4: Migrate `VoiceManifest` and `build_voice_master()`**

Keep `voice_id` as a read-compatible alias for one release. Add a `model_validator(mode="before")` that maps an old payload to `provider="indextts2"` and copies `voice_id` to `provider_voice_id`. Change `build_voice_master()` to accept:

```python
def build_voice_master(
    *,
    episode_root: Path,
    raw_path: Path,
    master_path: Path,
    script_path: Path,
    approved_script_sha256: str,
    provider: Literal["indextts2", "doubao"],
    provider_voice_id: str,
    provider_receipt_path: Path | None,
    reference_voice_path: Path | None,
    ffmpeg_command: Path | str,
    duration_policy: NarrationDurationPolicy,
    expected_reference_sha256: str | None = None,
    **processing_options: object,
) -> VoiceManifest:
```

For IndexTTS2, require and hash the authorized reference WAV. For Doubao, reject a reference path and require a safe receipt file inside the Episode. Include provider, voice ID and receipt hash in the processing identity so cache reuse cannot cross providers.

- [ ] **Step 5: Wrap the existing local synthesizer without changing its inference command**

`IndexTTS2NarrationSynthesizer` calls the existing `synthesize_voice()` and writes a minimal `ProviderReceipt(status="local_complete")`; it does not move, rename or reinterpret the approved reference voice. Preserve the old `TtsStage` behavior through this adapter before adding Doubao routing.

- [ ] **Step 6: Run voice regressions**

Run: `python -m pytest tests/unit/test_voice_providers.py tests/unit/test_media_audio_stages.py tests/integration/test_audio_processing.py tests/integration/test_media_audio_pipeline.py -q`

Expected: all selected tests pass, including legacy VoiceManifest loading and existing IndexTTS2 audio processing.

- [ ] **Step 7: Commit provider-neutral contracts**

```powershell
git add src/bv/voice/providers.py src/bv/voice/processing.py src/bv/voice/indextts2.py src/bv/workflow/runtime.py tests/unit/test_voice_providers.py tests/unit/test_media_audio_stages.py tests/integration/test_audio_processing.py
git commit -m "refactor: make narration provider neutral"
```

---

### Task 3: Doubao Async Long-Text Client with Durable Resume and Redaction

**Files:**
- Modify: `src/bv/config.py:34-175`
- Create: `src/bv/voice/doubao.py`
- Test: `tests/unit/test_doubao_tts.py`

**Interfaces:**
- Consumes: `NarrationRequest`, `NarrationArtifact`, `ProviderReceipt`, `RuntimeAuthorization`.
- Produces: `DoubaoTtsConfig`, `DoubaoSubmitRequest`, `DoubaoTask`, `DoubaoAsyncClient.submit()`, `DoubaoAsyncClient.query()`, `DoubaoAsyncClient.download()`, and `DoubaoNarrationSynthesizer.synthesize()`.
- Persists: `.private/tts/doubao-formal-receipt.json` or the corresponding audition receipt; no URL or secret appears in persisted JSON.

- [ ] **Step 1: Write request, resume, redaction and no-retry tests using `httpx.MockTransport`**

```python
def test_formal_submit_uses_required_headers_without_logging_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen = capture_submit_request(status_code=200, json_body=submit_success())
    artifact = make_synthesizer(seen).synthesize(formal_request(), formal_auth())
    assert seen.headers["Resource-Id"] == "volc.tts_async.default"
    assert seen.headers["Authorization"].startswith("Bearer; ")
    assert "secret-token" not in caplog.text
    assert "Authorization" not in artifact.receipt_path.read_text("utf-8")


def test_existing_submitted_receipt_is_queried_without_second_submit() -> None:
    transport = submit_forbidden_query_success_transport()
    artifact = make_synthesizer(transport, existing_receipt=True).synthesize(
        formal_request(), query_only_auth()
    )
    assert transport.submit_count == 0
    assert transport.query_count >= 1
    assert artifact.raw_audio_path.is_file()


@pytest.mark.parametrize("status_code", [400, 403, 500])
def test_provider_failure_has_one_request_and_no_fallback(status_code: int) -> None:
    transport = failing_transport(status_code)
    with pytest.raises(DoubaoTtsError, match="doubao_submit_failed"):
        make_synthesizer(transport).synthesize(formal_request(), formal_auth())
    assert transport.request_count == 1
```

- [ ] **Step 2: Run the tests and verify the Doubao module is absent**

Run: `python -m pytest tests/unit/test_doubao_tts.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'bv.voice.doubao'`.

- [ ] **Step 3: Add configuration containing names, endpoints and timeouts only**

```python
class DoubaoTtsConfig(_ConfigModel):
    submit_endpoint: str = "https://openspeech.bytedance.com/api/v3/tts/submit"
    query_endpoint: str = "https://openspeech.bytedance.com/api/v3/tts/query"
    api_key_env: str = "BV_DOUBAO_TTS_API_KEY"
    poll_interval_seconds: float = Field(default=2.0, ge=0.2, le=30.0)
    timeout_seconds: float = Field(default=900.0, ge=10.0, le=3600.0)
```

Add `doubao_tts: DoubaoTtsConfig` to `AppConfig`. Do not place app IDs or tokens in YAML defaults, examples or tests.

- [ ] **Step 4: Implement a three-operation client with stable public errors**

```python
class DoubaoTtsError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class DoubaoSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    app_id: str
    request_id: UUID
    text: str
    resource_id: str
    voice_type: str
    speed: float

    def provider_payload(self) -> dict[str, object]:
        return {
            "appid": self.app_id,
            "reqid": str(self.request_id),
            "text": self.text,
            "format": "wav",
            "voice_type": self.voice_type,
            "sample_rate": 48_000,
            "speed": self.speed,
            "enable_subtitle": True,
        }


class DoubaoTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    task_status: Literal[0, 1, 2]
    text_length: int = Field(ge=0)
    audio_url: str | None = None


class DoubaoAsyncClient:
    def submit(self, request: DoubaoSubmitRequest) -> DoubaoTask:
        response = self.http.post(
            self.config.submit_endpoint,
            headers=self.request_headers(request.resource_id),
            json=request.provider_payload(),
        )
        return self.parse_task_response(response, operation="submit")

    def query(self, *, app_id: str, task_id: str, resource_id: str) -> DoubaoTask:
        response = self.http.get(
            self.config.query_endpoint,
            headers=self.request_headers(resource_id),
            params={"appid": app_id, "task_id": task_id},
        )
        return self.parse_task_response(response, operation="query")

    def download(self, audio_url: str, destination: Path) -> str:
        response = self.http.get(audio_url)
        if response.status_code != 200 or not response.content:
            raise DoubaoTtsError("doubao_download_failed")
        atomic_write_bytes(destination, response.content)
        return sha256_file(destination)
```

The submit body uses `appid`, a UUID4 `reqid`, approved script text, `format="wav"`, `voice_type`, `sample_rate=48000`, configured `speed`, and `enable_subtitle=True`. Treat task status 0 as working, 1 as success, and 2 as failure. Hash the complete in-memory response before discarding it; persist only the task ID, status, input hash, response hash, timestamps and stable error code. Never interpolate response text or headers into an exception.

- [ ] **Step 5: Implement submit-once/resume-first synthesis**

On entry, validate the script hash and exact authorization scope. If a safe receipt exists, query its task ID. If no receipt exists and the authorization allows one matching submission, submit once and atomically persist the redacted receipt before polling. On success, download the one-hour URL immediately into a temporary file, validate non-empty content, hash it, atomically publish `raw.wav`, then rewrite the redacted receipt with `status="success"`; never save the URL. A timeout leaves `status="working"` so the next run resumes it.

- [ ] **Step 6: Run client and secret-safety tests**

Run: `python -m pytest tests/unit/test_doubao_tts.py tests/unit/test_config.py -q`

Expected: all selected tests pass; `rg -n "secret-token|Bearer; secret" . --glob '!tests/**'` returns no matches.

- [ ] **Step 7: Commit the Doubao client**

```powershell
git add src/bv/config.py src/bv/voice/doubao.py tests/unit/test_doubao_tts.py tests/unit/test_config.py
git commit -m "feat: add resumable doubao long text tts"
```

---

### Task 4: Voice Audition Gate and TTS Runtime Routing

**Files:**
- Create: `src/bv/voice/audition.py`
- Modify: `src/bv/workflow/media_stages.py:145-240`
- Modify: `src/bv/workflow/runtime.py:169-240`
- Modify: `src/bv/workflow/media_runtime.py:93-335`
- Modify: `src/bv/state/models.py:1-230`
- Modify: `src/bv/state/invalidation.py`
- Test: `tests/unit/test_voice_audition.py`
- Test: `tests/unit/test_media_runtime.py`

**Interfaces:**
- Consumes: Provider synthesizers from Tasks 2–3 and `ProductionProfile` from Task 1.
- Produces: `VoiceAuditionService.prepare_candidates()`, `VoiceAuditionService.approve_candidate()`.
- Produces: `VoiceProfile` persisted as `voice_profile.json` with provider, voice ID, speed, excerpt hash, approved WAV hash and approval timestamp.
- Updates: `MediaProductionService.prepare_voice_audition()` and `approve_voice_audition()`; existing `prepare()` routes only after the voice profile gate is current.

- [ ] **Step 1: Write gate and routing tests**

```python
def test_doubao_full_tts_is_blocked_before_voice_approval(runtime) -> None:
    episode = living_episode(status="script_approved")
    runtime.store.save_episode(episode)
    with pytest.raises(MediaWorkflowError, match="voice_profile_not_approved"):
        runtime.prepare("B001", "E001", mode="full")


def test_audition_uses_same_excerpt_and_at_most_three_candidates(service) -> None:
    view = service.prepare_candidates(
        context(), voice_ids=("voice-a", "voice-b", "voice-c")
    )
    manifest = load_audition_manifest(view.manifest_path)
    assert len(manifest.candidates) == 3
    assert len({item.excerpt_sha256 for item in manifest.candidates}) == 1
    assert len({item.speed for item in manifest.candidates}) == 1


def test_approval_sets_profile_and_invalidates_downstream(service) -> None:
    result = service.approve_candidate(context(), "candidate-02")
    assert result.provider == "doubao"
    assert result.approved_audio_sha256 == sha256(candidate_02_path())
    assert result.provider_voice_id == "voice-b"
```

- [ ] **Step 2: Run the focused tests and confirm the gate is absent**

Run: `python -m pytest tests/unit/test_voice_audition.py tests/unit/test_media_runtime.py -q`

Expected: new audition imports fail and the runtime currently proceeds directly to the IndexTTS2-only stage.

- [ ] **Step 3: Implement safe audition storage and approval**

Use `.private/candidates/tts-audition-YYYYMMDD/`. Validate the approved excerpt is a continuous 15–25 second section of the approved script, based on the same candidate voice speed and provider estimate. Reject more than three voice IDs, duplicates, blank IDs and IDs not supplied by the account-supported async long-text voice discovery result. Candidate generation consumes the exact `max_audition_submissions` scope and never starts the formal task.

Approval atomically writes both durable documents: it updates `production_profile.json.tts.voice_type` to the exact approved ID and creates `voice_profile.json` as the approval evidence. A formal stage is current only when the profile hash, audition manifest hash and approved WAV hash all match. `VoiceProfile` is frozen and contains:

```python
class VoiceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: Literal["indextts2", "doubao"]
    provider_voice_id: str
    speed: float
    excerpt_sha256: Sha256
    approved_audio_sha256: Sha256
    audition_manifest_sha256: Sha256
    approved_at: datetime
```

- [ ] **Step 4: Replace `TtsStage` provider conditionals with injected synthesizers**

```python
@dataclass(frozen=True)
class TtsStage:
    synthesizers: Mapping[str, NarrationSynthesizer]
    ffmpeg_command: Path
    audio_config: AudioConfig

    def __call__(self, context: StageContext) -> StageResult:
        profile = load_production_profile(
            context.episode_root,
        )
        voice_profile = require_current_voice_profile(context, profile)
        synthesizer = self.synthesizers[profile.tts.provider]
        artifact = synthesizer.synthesize(
            narration_request_from_context(context, profile),
            context.authorization,
        )
        manifest = build_voice_master(
            provider=profile.tts.provider,
            provider_voice_id=voice_profile.provider_voice_id,
            provider_receipt_path=artifact.receipt_path,
            duration_policy=duration_policy_from(profile.duration),
            **voice_master_inputs(context, artifact),
        )
        return stage_result_from_voice_manifest(manifest, profile)
```

The actual implementation may preserve the existing dataclass fields needed by tests, but routing decisions must depend only on the production profile and approved voice profile. IndexTTS2 legacy Episodes synthesize without a Doubao audition gate.

- [ ] **Step 5: Add durable states/evidence and invalidation edges**

Persist `awaiting_voice_audition_review` and `voice_profile_approved` where the current state model accepts status literals. A change in approved script, Provider, voice ID or speed invalidates `tts` through `final_qc`, archives old private audition/audio artifacts under `.private/history/<UTC timestamp>-voice-change`, and never deletes old evidence.

- [ ] **Step 6: Run runtime tests**

Run: `python -m pytest tests/unit/test_voice_audition.py tests/unit/test_media_runtime.py tests/unit/test_media_audio_stages.py tests/unit/test_state_store.py -q`

Expected: all selected tests pass; legacy IndexTTS2 test fixtures do not require a new voice gate.

- [ ] **Step 7: Commit the audition gate**

```powershell
git add src/bv/voice/audition.py src/bv/workflow/media_stages.py src/bv/workflow/runtime.py src/bv/workflow/media_runtime.py src/bv/state/models.py src/bv/state/invalidation.py tests/unit/test_voice_audition.py tests/unit/test_media_runtime.py tests/unit/test_media_audio_stages.py tests/unit/test_state_store.py
git commit -m "feat: add approved voice audition gate"
```

---

### Task 5: Per-Episode Duration in Audio, Render and QC

**Files:**
- Modify: `src/bv/voice/processing.py:89-240`
- Modify: `src/bv/workflow/media_stages.py:607-690`
- Modify: `src/bv/video/qc.py`
- Modify: `src/bv/state/invalidation.py`
- Test: `tests/unit/test_media_audio_stages.py`
- Test: `tests/integration/test_audio_processing.py`
- Test: `tests/integration/test_handdrawn_final_render_pipeline.py`

**Interfaces:**
- Consumes: `DurationProfile` and production profile hash.
- Produces: `duration_policy_from(profile: DurationProfile) -> NarrationDurationPolicy`.
- Updates: TTS, render and QC StageManifest inputs include `production_profile_sha256`; QC validates the profile-specific hard band.

- [ ] **Step 1: Add exact boundary and invalidation tests**

```python
@pytest.mark.parametrize(
    ("seconds", "level"),
    [(119.9, "fail"), (120.0, "pass"), (135.0, "pass"),
     (150.0, "pass"), (150.1, "fail")],
)
def test_living_duration_boundaries(seconds: float, level: str) -> None:
    result = classify_narration_duration(seconds, policy=LIVING_POLICY)
    assert result.level == level


def test_profile_change_marks_tts_through_delivery_stale(episode) -> None:
    changed = invalidate_from(episode, "production_profile")
    assert {"tts", "asr", "subtitles", "plan_illustrations", "render", "final_qc"} \
        <= set(changed.stale_stages)
```

- [ ] **Step 2: Run the focused tests and observe hardcoded full policy failure**

Run: `python -m pytest tests/unit/test_media_audio_stages.py tests/integration/test_audio_processing.py -q`

Expected: 120–150 second assertions fail because `FULL_EPISODE_POLICY` is 45/48/56/60.

- [ ] **Step 3: Route the profile policy through every duration check**

Remove `policy_for("full")` as the source of truth for profiled Episodes. Keep `TECHNICAL_SAMPLE_POLICY` unchanged. For full mode, construct `NarrationDurationPolicy` from the Episode profile and include the profile hash in voice processing settings, `final.render.json` inputs and `final.qc.json` inputs.

The automatic tempo factor stays capped at 1.08. A 119.9 or 150.1 second result fails instead of applying a larger speed correction.

- [ ] **Step 4: Make final QC accept the profile explicitly**

```python
def validate_final_media(
    *,
    video_path: Path,
    render_manifest_path: Path,
    production_profile: ProductionProfile,
    ffprobe_command: Path | str,
    runner: Runner = run_command,
) -> FinalQcManifest:
```

Verify 1080×1920, 30 fps, H.264, yuv420p, AAC 48 kHz stereo, duration inside the hard band and audio/video end-time difference no greater than `1 / 30` seconds. Preserve every existing legacy QC assertion.

- [ ] **Step 5: Run duration, render and QC tests**

Run: `python -m pytest tests/unit/test_media_audio_stages.py tests/integration/test_audio_processing.py tests/integration/test_handdrawn_final_render_pipeline.py -q`

Expected: all selected tests pass for both the old short fixture and a generated 135-second fixture.

- [ ] **Step 6: Commit per-Episode duration support**

```powershell
git add src/bv/voice/processing.py src/bv/workflow/media_stages.py src/bv/video/qc.py src/bv/state/invalidation.py tests/unit/test_media_audio_stages.py tests/integration/test_audio_processing.py tests/integration/test_handdrawn_final_render_pipeline.py
git commit -m "feat: apply episode duration profiles to media qc"
```

---

### Task 6: Long-Form Multi-Character Storyboard for Living

**Files:**
- Modify: `src/bv/illustration/contracts.py:80-180`
- Modify: `src/bv/illustration/planner.py:60-300`
- Modify: `src/bv/illustration/prompts.py`
- Modify: `src/bv/illustration/assets.py`
- Modify: `src/bv/illustration/reviews.py`
- Modify: `src/bv/workflow/media_stages.py:401-605`
- Modify: `prompts/illustration/storyboard.md`
- Test: `tests/unit/test_illustration_contracts.py`
- Test: `tests/unit/test_illustration_planner.py`
- Test: `tests/unit/test_illustration_assets.py`
- Test: `tests/integration/test_handdrawn_end_to_end_fake.py`

**Interfaces:**
- Consumes: final WAV duration, ASR cues, approved script, `VisualProfile` and style decision.
- Produces: backward-compatible `CharacterBible` containing 1–3 `CharacterLock` entries with stable IDs.
- Updates: `partition_illustration_timeline(cues: Sequence[SubtitleCue], *, master_duration_ms: int, fps: int = 30, seconds_per_scene_min: float | None = None, seconds_per_scene_max: float | None = None) -> tuple[SceneWindow, ...]`.
- Updates: `IllustrationStoryboard.character_bible_sha256`; old `character_lock_sha256` payloads remain readable.

- [ ] **Step 1: Add scene-count, role separation and old-contract tests**

```python
def test_135_seconds_uses_profile_density_not_legacy_cap(cues_135s) -> None:
    windows = partition_illustration_timeline(
        cues_135s,
        master_duration_ms=135_000,
        seconds_per_scene_min=7.0,
        seconds_per_scene_max=11.0,
    )
    assert 13 <= len(windows) <= 19
    assert windows[0].from_frame == 0
    assert windows[-1].to_frame == 135_000 * 30 // 1000


def test_modern_reader_and_fugui_are_distinct_character_ids(storyboard) -> None:
    ids = {lock.character_id for lock in storyboard.character_bible.characters}
    assert "reader-01" in ids
    assert "source-protagonist" in ids
    assert all(set(scene.character_refs) <= ids for scene in storyboard.scenes)


def test_legacy_single_character_lock_loads_as_reader(legacy_lock_json) -> None:
    bible = load_character_bible(legacy_lock_json)
    assert [item.character_id for item in bible.characters] == ["reader-01"]
```

- [ ] **Step 2: Run tests and verify the 12-scene cap and single lock fail**

Run: `python -m pytest tests/unit/test_illustration_contracts.py tests/unit/test_illustration_planner.py -q`

Expected: the 135-second scene test returns 12 scenes and `CharacterBible` imports fail.

- [ ] **Step 3: Add a bounded multi-character contract**

```python
class CharacterLock(_IllustrationModel):
    character_id: SafeIdentifier = "reader-01"
    role: str
    # keep every existing visual identity field unchanged


class CharacterBible(_IllustrationModel):
    source_script_sha256: Sha256
    style_fingerprint: Sha256
    characters: tuple[CharacterLock, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "CharacterBible":
        ids = [item.character_id for item in self.characters]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate_character_id")
        allowed = {"reader-01", "source-protagonist", "source-family-01"}
        if not set(ids) <= allowed:
            raise ValueError("unsupported_character_id")
        return self
```

Migrate old `character_lock.json` in memory to a one-character bible and continue validating its original hash. New Episodes persist `character_bible.json`; each scene may reference only IDs present in the bible.

- [ ] **Step 4: Replace the legacy count formula with an interval-derived count**

```python
def _profile_scene_count(duration_seconds: float, low: float, high: float) -> int:
    minimum_count = math.ceil(duration_seconds / high)
    maximum_count = math.floor(duration_seconds / low)
    target_count = round(duration_seconds / ((low + high) / 2))
    return min(max(target_count, minimum_count), maximum_count)
```

Use this only when both profile bounds are supplied. Preserve exactly `3` scenes at 20 seconds or less and `min(12, max(8, round(ms / 5500)))` for legacy calls. If subtitle cue boundaries cannot provide the count, fail with `insufficient_cue_boundaries` rather than inventing timecodes.

- [ ] **Step 5: Generalize the prompt and validate the 60/40 visual allocation**

Remove hardcoded references to《我与地坛》、史铁生和母亲. The prompt receives book title, approved narration, evidence-backed source roles, target reader context, and allowed character IDs. It requests approximately 60% current-life reader scenes and 40% source-book people/settings or controlled metaphors, while forbidding film-actor likenesses, fake covers, readable text and sensational suffering.

The planner must reject `source-protagonist` unless the approved narration contains the approved source role name such as“福贵”. It must reject a scene where `reader-01` and `source-protagonist` are described as the same physical person.

- [ ] **Step 6: Keep the three-frame human gate and update asset prompts**

Select representatives by timeline position: first scene, scene nearest 50%, and final scene. Image prompt assembly includes only the locks referenced by that scene and repeats the low-saturation Chinese rural watercolor/pencil constraints. Batch preparation remains unavailable until the representative approval manifest matches storyboard, style and character-bible hashes.

- [ ] **Step 7: Run illustration and fake end-to-end tests**

Run: `python -m pytest tests/unit/test_illustration_contracts.py tests/unit/test_illustration_planner.py tests/unit/test_illustration_assets.py tests/integration/test_handdrawn_end_to_end_fake.py -q`

Expected: all selected tests pass; old《我与地坛》 fixtures load without rewriting their files.

- [ ] **Step 8: Commit long-form storyboard support**

```powershell
git add src/bv/illustration/contracts.py src/bv/illustration/planner.py src/bv/illustration/prompts.py src/bv/illustration/assets.py src/bv/illustration/reviews.py src/bv/workflow/media_stages.py prompts/illustration/storyboard.md tests/unit/test_illustration_contracts.py tests/unit/test_illustration_planner.py tests/unit/test_illustration_assets.py tests/integration/test_handdrawn_end_to_end_fake.py
git commit -m "feat: plan long form multi character illustrations"
```

---

### Task 7: Punk-Cover-Guided 1080×1920 Social Cover

**Files:**
- Create: `src/bv/video/cover_brief.py`
- Create: `src/bv/video/social_cover.py`
- Modify: `src/bv/workflow/media_stages.py`
- Modify: `src/bv/workflow/stages.py`
- Modify: `src/bv/workflow/runtime.py`
- Test: `tests/unit/test_cover_brief.py`
- Test: `tests/unit/test_social_cover.py`
- Test: `tests/integration/test_handdrawn_final_render_pipeline.py`

**Interfaces:**
- Consumes: approved script, semantic lock, production profile, exact original book title and the explicitly selected single `punk-cover` style.
- Produces: `CoverBriefBuilder.build(request: CoverBriefRequest) -> CoverBrief` and the Skill-owned `punk-assets/punk-cover/{slug}/prompts/cover.md`.
- Accepts only the exact current-run image artifact returned by the authorized image-generation call; it never scans broad output directories and never falls back to a representative illustration.
- Produces: `SocialCoverRenderer.normalize(request: SocialCoverRequest) -> SocialCoverManifest`.
- Produces: `media/final/social_cover.png` and `media/final/social_cover.json` at stable internal paths.

- [ ] **Step 1: Write cover-brief, artifact-binding and reparse-safety tests**

```python
def test_cover_brief_uses_one_eligible_style_and_approved_meaning(tmp_path: Path) -> None:
    brief = builder(tmp_path).build(living_cover_request())
    assert brief.style_id == "french-minimal-ink-poster"
    assert brief.title == "活着"
    assert brief.script_sha256 == sha256(approved_script())
    assert len(brief.visual_metaphors) == 1
    assert "{{" not in brief.prompt_text


def test_social_cover_is_vertical_and_binds_exact_current_run_artifact(tmp_path: Path) -> None:
    manifest = renderer(tmp_path).normalize(cover_request(title="活着"))
    assert probe_image(manifest.output_path) == (1080, 1920)
    assert manifest.title == "活着"
    assert manifest.style_id == "french-minimal-ink-poster"
    assert manifest.prompt_sha256 == sha256(saved_punk_cover_prompt())
    assert manifest.source_image_sha256 == sha256(explicit_current_run_artifact())


def test_cover_refuses_existing_or_redirected_output(tmp_path: Path) -> None:
    output = tmp_path / "media/final/social_cover.png"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing")
    with pytest.raises(SocialCoverError, match="cover_output_exists"):
        renderer(tmp_path).normalize(cover_request(title="活着"))
```

- [ ] **Step 2: Run tests and confirm the cover components are absent**

Run: `python -m pytest tests/unit/test_cover_brief.py tests/unit/test_social_cover.py -q`

Expected: collection fails because the cover brief and social-cover modules do not exist.

- [ ] **Step 3: Implement the Skill-compatible cover brief and prompt contract**

Build only concise derived fields from the approved script and semantic lock: exact title, optional short subtitle, 1–3 sentence context summary, visual subject, target audience, mood, one visual metaphor and banned elements. Select exactly one eligible style. For《活着》E001 use `french-minimal-ink-poster`; read only its `META.md` and `STYLE.md`, then compile those anchors through the `punk-cover` blueprint into one integrated prompt. Preserve the complete title“活着”, portrait `9:16`, centered enlarged title hierarchy, quiet hand-drawn ink texture and one meaning-bearing metaphor. Reject unresolved placeholders, mixed style IDs, copied long-form source text, fake book covers, actor likenesses, sensational suffering and generic inspirational iconography.

Save the complete prompt at `punk-assets/punk-cover/huo-zhe-e001/prompts/cover.md` before image generation. The prompt file and `CoverBrief` manifest bind the approved script, semantic lock, production profile, selected style metadata/style hashes and exact prompt hash. Prompt preparation is local; the real image-generation call remains an explicit major authorization gate.

- [ ] **Step 4: Import and normalize only the exact generated artifact**

Accept the generated cover only from an explicit artifact returned by the current authorized call. If the tool exposes only an inline preview and no retrievable artifact, do not create a fake `cover.png` and do not mark the stage complete. Reject links/reparse points, changed source identity, stale prompt hashes and pre-existing targets.

Normalize the authorized 9:16 source to 1080×1920 without inventing a new title layer or cropping the title safe area. Write to a sibling temporary PNG, verify dimensions, hash it and atomically publish it. The manifest binds title, style ID, script/semantic-lock/profile/prompt/style hashes, exact source image hash, normalization arguments and output hash. Title accuracy and visual meaning are human-review fields; technical success cannot approve them.

- [ ] **Step 5: Add the stages after script approval and before delivery**

`CoverPromptStage` can run after script approval and remains local. `SocialCoverStage` can run only when the prompt manifest and exact current-run generated artifact are current. A changed title, script, semantic lock, style file, prompt, source image or profile invalidates social cover and delivery. It does not silently regenerate an image and does not invalidate approved audio unless the audio's own inputs changed.

- [ ] **Step 6: Run cover and render integration tests**

Run: `python -m pytest tests/unit/test_cover_brief.py tests/unit/test_social_cover.py tests/integration/test_handdrawn_final_render_pipeline.py -q`

Expected: all selected tests pass, stale or ambiguous artifacts are rejected, and the exact fixture cover probes as 1080×1920.

- [ ] **Step 7: Commit the punk-cover-guided social cover stage**

```powershell
git add src/bv/video/cover_brief.py src/bv/video/social_cover.py src/bv/workflow/media_stages.py src/bv/workflow/stages.py src/bv/workflow/runtime.py tests/unit/test_cover_brief.py tests/unit/test_social_cover.py tests/integration/test_handdrawn_final_render_pipeline.py
git commit -m "feat: add punk cover production stage"
```

---

### Task 8: Atomic Date Delivery and Immutable Final Approval

**Files:**
- Create: `src/bv/delivery/__init__.py`
- Create: `src/bv/delivery/exporter.py`
- Modify: `src/bv/workflow/media_runtime.py:250-335`
- Modify: `src/bv/workflow/stages.py`
- Modify: `src/bv/state/invalidation.py`
- Test: `tests/unit/test_delivery_exporter.py`
- Test: `tests/unit/test_media_runtime.py`

**Interfaces:**
- Consumes: current script, profile, voice manifest, final render/QC, social cover and subtitles.
- Produces: `DeliveryExporter.export_candidate(request: DeliveryRequest) -> DeliveryBundle`.
- Produces: `DeliveryExporter.record_final_approval(bundle: DeliveryBundle, approved_at: datetime) -> ApprovalRecord`.
- Updates: `MediaProductionService.render()` exports a candidate after current QC and cover; `approve_final()` adds the approval record without modifying the delivery manifest.

- [ ] **Step 1: Write date, V2, hash, redirect and immutable-approval tests**

```python
def test_delivery_uses_shanghai_date_and_expected_names(tmp_path: Path) -> None:
    bundle = exporter(clock=fixed_utc("2026-08-22T16:30:00Z")).export_candidate(
        delivery_request(tmp_path, title="活着")
    )
    assert bundle.version == 1
    assert bundle.root.name == "2026-08-23"
    assert (bundle.root / "2026-08-23-活着-成片.mp4").is_file()
    assert (bundle.root / "2026-08-23-活着-作品封面.png").is_file()


def test_any_same_day_collision_advances_whole_bundle_to_v2(tmp_path: Path) -> None:
    first = exporter().export_candidate(delivery_request(tmp_path))
    second = exporter().export_candidate(delivery_request(tmp_path))
    assert first.version == 1
    assert second.version == 2
    assert all("-V2" in item.name for item in second.files)
    assert second.manifest_path.name == "2026-08-23-活着-交付清单-V2.json"


def test_final_approval_does_not_rewrite_delivery_manifest(tmp_path: Path) -> None:
    bundle = exporter().export_candidate(delivery_request(tmp_path))
    before = sha256(bundle.manifest_path)
    record = exporter().record_final_approval(bundle, approved_at=fixed_datetime())
    assert sha256(bundle.manifest_path) == before
    assert record.delivery_manifest_sha256 == before
```

- [ ] **Step 2: Run tests and confirm the package is absent**

Run: `python -m pytest tests/unit/test_delivery_exporter.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'bv.delivery'`.

- [ ] **Step 3: Implement filename safety and deterministic bundle selection**

Normalize the book title by replacing Windows-reserved characters `<>:"/\\|?*` and control characters with `_`, trimming trailing spaces/dots, and rejecting an empty or device-reserved result. Determine the Shanghai date with `ZoneInfo("Asia/Shanghai")`. If the date directory is absent, use version 1; if it exists or any target path exists, scan complete numeric suffixes and select the next whole-bundle version.

- [ ] **Step 4: Implement atomic export with source and destination hash checks**

```python
class DeliveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    episode_root: Path
    book_id: SafeIdentifier
    episode_id: SafeIdentifier
    title: str
    author: str
    script_path: Path
    voice_path: Path
    subtitle_path: Path
    video_path: Path
    cover_path: Path
    production_profile_path: Path
    voice_manifest_path: Path
    render_manifest_path: Path
    qc_manifest_path: Path


class DeliveryBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    root: Path
    version: int
    manifest_path: Path
    manifest_sha256: Sha256
    files: tuple[Path, ...]


class ApprovalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    delivery_manifest_sha256: Sha256
    video_sha256: Sha256
    cover_sha256: Sha256
    approved_at: datetime
    status: Literal["final_approved"] = "final_approved"
    path: Path
```

Resolve and validate every source under the Episode root, reject links/reparse points in every existing ancestor, snapshot source size/hash, copy to a unique temporary directory beside the date directory, fsync/close, then re-check source identity and destination hash. The visible contract remains the approved flat `deliveries/YYYY-MM-DD/` layout. Under an exclusive delivery lock, move the six no-replace files from staging into that directory and publish the candidate manifest last as the package commit marker; readers and approval code treat a version as nonexistent until that manifest appears. If publication stops before the marker, the next run moves the incomplete files into `.recovery/delivery-<timestamp>` and uses the next whole-bundle version. Never replace a pre-existing target.

The candidate manifest records original title/author, IDs, Shanghai date, version, timezone, script/profile hashes, Provider and voice ID, relative output paths/sizes/hashes, render/QC manifest hashes and fixed state `awaiting_final_review`.

- [ ] **Step 5: Implement immutable approval recording**

`record_final_approval()` re-reads and validates the candidate manifest, video and cover hashes, rejects an existing approval path, writes a sibling temporary JSON containing candidate-manifest hash, video hash, cover hash, Shanghai approval time and `final_approved`, and publishes it atomically. It never edits the candidate manifest or media files.

- [ ] **Step 6: Wire delivery to current QC and final approval**

`render()` completes render, QC and social cover, then exports the candidate while Episode status stays `awaiting_final_review`; the returned view exposes delivery root/version. `approve_final()` validates the exact candidate package, adds the approval record, then changes Episode status to `final_approved`. A delivery export failure leaves the stable internal QC result current and does not mark final approval.

- [ ] **Step 7: Run exporter and runtime tests**

Run: `python -m pytest tests/unit/test_delivery_exporter.py tests/unit/test_media_runtime.py tests/unit/test_state_store.py -q`

Expected: all selected tests pass, including V2 atomicity and no-replace behavior.

- [ ] **Step 8: Commit date delivery**

```powershell
git add src/bv/delivery src/bv/workflow/media_runtime.py src/bv/workflow/stages.py src/bv/state/invalidation.py tests/unit/test_delivery_exporter.py tests/unit/test_media_runtime.py tests/unit/test_state_store.py
git commit -m "feat: export immutable dated delivery bundles"
```

---

### Task 9: Full Fake Pipeline, Regression Suite and Production Skill Update

**Files:**
- Create: `tests/integration/test_doubao_longform_delivery_pipeline.py`
- Modify: `tests/integration/test_handdrawn_end_to_end_fake.py`
- Modify: `tests/integration/test_handdrawn_final_render_pipeline.py`
- Modify: `README.md`
- Modify after repository verification: `C:/Users/1/.codex/skills/producing-book-handdrawn-videos/SKILL.md`
- Create after repository verification: `C:/Users/1/.codex/skills/producing-book-handdrawn-videos/references/doubao-longform-delivery.md`

**Interfaces:**
- Consumes: all Tasks 1–8.
- Produces: one deterministic fake 135-second end-to-end acceptance test and an updated reusable production Skill.
- Preserves: no CLI entry point and all existing《我与地坛》 fixtures/artifacts.

- [ ] **Step 1: Add a fake 135-second acceptance test**

```python
def test_doubao_longform_pipeline_requires_all_four_human_gates(
    longform_runtime,
) -> None:
    episode = seed_approved_living_script(longform_runtime)
    with pytest.raises(MediaWorkflowError, match="voice_profile_not_approved"):
        longform_runtime.prepare(episode.book_id, episode.episode_id, mode="full")

    auditions = longform_runtime.prepare_voice_audition(
        episode.book_id, episode.episode_id, ("reader-a", "reader-b")
    )
    longform_runtime.approve_voice_audition(
        episode.book_id, episode.episode_id, auditions.candidates[0].candidate_id
    )
    representatives = longform_runtime.prepare(
        episode.book_id, episode.episode_id, mode="full"
    )
    assert len(representatives.missing_scene_ids) == 3

    import_representatives(longform_runtime, episode)
    longform_runtime.approve_representatives(episode.book_id, episode.episode_id)
    import_remaining_images(longform_runtime, episode)
    candidate = longform_runtime.render(episode.book_id, episode.episode_id)
    assert candidate.status == "awaiting_final_review"
    assert candidate.delivery_version == 1
    approved = longform_runtime.approve_final(episode.book_id, episode.episode_id)
    assert approved.status == "final_approved"
    assert approved.approval_record_path.is_file()
```

The fake providers must create deterministic WAV/image/video fixtures locally; they must not inspect clipboard or environment secrets and must not make network calls.

- [ ] **Step 2: Run the new integration test before final wiring**

Run: `python -m pytest tests/integration/test_doubao_longform_delivery_pipeline.py -q`

Expected: fail at the first missing integration field or stage, giving the final wiring point to correct without weakening gates.

- [ ] **Step 3: Complete only the wiring exposed by the failing acceptance test**

Ensure StageManifest inputs contain script, profile, voice profile, audio, ASR, subtitles, storyboard, character bible, representative approval, render, QC and cover hashes at their dependency boundaries. Ensure a modified profile marks every dependent stage stale. Do not add a new command or automatic external call.

- [ ] **Step 4: Run repository and vendor verification**

Run: `python -m pytest -q`

Expected: the full Python suite passes with no regression in legacy Episodes.

Run: `npm run check`

Working directory: `vendor/story_to_handdrawn_video`

Expected: TypeScript/Remotion checks pass.

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 5: Update the repository operating notes**

Document the exact no-CLI conversation workflow: inspect title/author evidence, approve script, inspect TTS credential state as `SET`/`UNSET`, audition up to three account-supported long-text voices, approve one voice, submit one formal task, resume its receipt, run ASR, approve three illustrations, render/QC, inspect the dated candidate bundle, approve final, add approval record. Include the official API URLs from spec section 14 and state that fields, entitlements and pricing must be checked live before a real request.

- [ ] **Step 6: Update and verify the reusable production Skill**

Only after Step 4 passes, use `superpowers:writing-skills` because this step edits a Skill. Change the Skill from hardcoded 45–60 seconds/IndexTTS2 to profile-driven branching, preserving the old default. Add `references/doubao-longform-delivery.md` containing the four approval gates, submit-once/resume-first rule, SET/UNSET credential reporting, date bundle contract and failure recovery table. Run every validation command required by that Skill’s own `SKILL.md`.

- [ ] **Step 7: Commit tested code and repository docs**

```powershell
git add tests/integration/test_doubao_longform_delivery_pipeline.py tests/integration/test_handdrawn_end_to_end_fake.py tests/integration/test_handdrawn_final_render_pipeline.py README.md
git commit -m "test: verify long form book video delivery"
```

Do not add files under `.private`, `episodes/*/deliveries`, generated media, credentials, `.superpowers/` or `.tmp_codex_session.json`. The global Skill files are outside the repository and are reported separately with their validation result.

---

### Task 10: Real Production of Yu Hua's Living

**Files:**
- Create locally, never commit: `workspace/books/B001/episodes/E001/production_profile.json`
- Create locally, never commit: `workspace/books/B001/episodes/E001/voice_profile.json`
- Create locally, never commit: `workspace/books/B001/episodes/E001/.private/**`
- Create locally, never commit: `workspace/books/B001/episodes/E001/media/**`
- Create locally, never commit: `workspace/books/B001/episodes/E001/deliveries/YYYY-MM-DD/**`
- Append reusable sanitized failures only: `docs/production-notes/living-e001-runbook.md`

**Interfaces:**
- Consumes: tested code from Tasks 1–9, title“活着”, author“余华”, current account voice entitlements and explicit external authorization at each paid gate.
- Produces: one 120–150 second candidate video and cover, then an immutable approval record only after the user says the final work is approved.
- Produces: a sanitized runbook containing stable error codes, root causes, fixes, verification commands and recovery steps; no secrets, signed URLs or private response bodies.

- [ ] **Step 1: Create the Episode profile and perform a read-only preflight**

Write the exact JSON from spec section 4 with `voice_type: null`. Verify Python 3.11, FFmpeg/ffprobe, Remotion dependencies, image-generation availability, Volc ASR credential state and Doubao TTS credential state. Report only executable/version paths and `SET`/`UNSET`; do not read the clipboard into logs or touch unrelated API files.

- [ ] **Step 2: Research and draft the whole-book value script**

Use title+author research mode and separate verified facts from interpretation. Draft for the approved 135-second target with the four-part timing in spec section 3.3. Apply `human-writing`; validate that modern reader contexts dominate without turning the piece into generic encouragement, and that necessary福贵 plot material remains factually grounded. Stop at the script gate and show the complete script plus evidence notes to the user.

- [ ] **Step 3: Persist the approved script hash**

After explicit script approval, write the approved text and semantic lock, calculate SHA-256, and invalidate any older E001 downstream artifacts whose input hash differs. Do not call TTS, ASR or image generation before this hash is fixed.

- [ ] **Step 4: Discover eligible voices without synthesizing audio**

Re-check the current official long-text service documentation and the account’s actually enabled voice IDs. Present at most three calm audiobook candidates that all support the same async service and resource ID. No candidate name alone counts as support evidence.

- [ ] **Step 5: Generate and approve the short voice audition**

After explicit authorization scoped to the approved script hash and up to three audition submissions, generate the same 15–25 second excerpt at speed 1.0 with no BGM. Show the candidate WAVs and audition manifest. Stop until the user selects one voice.

- [ ] **Step 6: Submit or resume the one formal narration task**

Write the selected exact `voice_type` into `production_profile.json`, write the matching approval evidence to `voice_profile.json`, and regenerate the profile hash. After explicit formal authorization scoped to one submit and zero fallback, first check for an existing redacted receipt; query it if present, otherwise submit once. Normalize the successful download to `voice_master.wav`, verify 120–150 seconds, loudness and intelligibility, then run Volc ASR against that final WAV.

- [ ] **Step 7: Build subtitles and the long-form storyboard**

Generate single-line punctuation-free ASS subtitles from final-WAV ASR. Plan 7–11 seconds per scene without a fixed image count, create separate `reader-01` and `source-protagonist` locks, and produce the opening/middle/ending representative prompts. Verify the 60/40 visual intent, no actor likeness, no readable in-image text and no sensational suffering.

- [ ] **Step 8: Generate and approve three representative illustrations**

After the scoped image-generation action is authorized, generate only the three representatives. Present all three with style decision and character bible. Stop until the user approves style,人物一致性 and emotional fit.

- [ ] **Step 9: Generate remaining illustrations and render the candidate**

Generate only missing storyboard assets after representative approval. Render 1080×1920 at 30 fps with 15-frame cross-dissolves, approved narration and subtitles. Compile the `punk-cover` prompt from the approved《活着》script using the single selected style, obtain explicit authorization for the cover image call, accept only its exact returned artifact, normalize it to 1080×1920, then run technical QC. Export the entire candidate bundle under the actual Shanghai date and next safe version while Episode status remains `awaiting_final_review`.

- [ ] **Step 10: Present final artifacts and wait for the final gate**

Show the dated video, cover, script, voice, subtitles and candidate manifest paths with QC summary. Do not call `approve_final()` based on technical success or silence; wait for the user’s explicit final approval.

- [ ] **Step 11: Record approval and complete the sanitized runbook**

After explicit approval, add the immutable approval JSON and change status to `final_approved`. Record every encountered error as: stage, stable code, observable symptom, root cause, minimal fix, verification result and future recovery action. Confirm no credential, signed URL, complete Provider response or private source text entered Git.

---

## Approved Spec Coverage

| Spec section | Implementation task |
|---|---|
| 1–3 scope, whole-book value, audience and script structure | Tasks 1 and 10 |
| 4 single-Episode production profile | Tasks 1 and 5 |
| 5 provider boundary, auditions, submit/query and credential privacy | Tasks 2–4 |
| 6 VoiceManifest compatibility | Task 2 |
| 7 duration, scene density, render and QC | Tasks 5–7 |
| 8 dated immutable delivery | Task 8 |
| 9 human gates and durable evidence | Tasks 4, 8 and 10 |
| 10 failure recovery | Tasks 3, 4, 8 and 10 |
| 11 automated and real tests | Tasks 9 and 10 |
| 12 explicit non-goals | Global Constraints and Tasks 1–10 |
| 13 implementation order | Tasks 1–10 in dependency order |
| 14 live official API verification | Tasks 9 and 10 |

Self-review result: every approved section has an implementation owner; no independent subsystem is left without a test and human acceptance gate.

## Final Verification Matrix

| Requirement | Automated evidence | Human evidence |
|---|---|---|
| Old IndexTTS2 45–60 sec compatibility | legacy unit/integration suite | existing《我与地坛》 state remains readable |
| Doubao voice approval before full submit | audition/runtime unit tests | approved candidate WAV |
| One formal submit, resume after restart | MockTransport request counters | redacted receipt and run log |
| 120–150 sec final duration | audio/QC boundary tests and ffprobe | narration pacing review |
| 7–11 sec dynamic scenes | planner unit test | three-frame then batch approval |
| Reader and福贵 remain distinct | character-bible validation | visual consistency review |
| Date naming and V2/V3 no overwrite | exporter unit tests | dated candidate folder |
| Candidate manifest immutable | approval unit test | final approval record |
| No secrets or signed URLs | redaction tests and repository scan | SET/UNSET-only preflight |
| No CLI/GUI/upload added | Git diff review | conversation-driven production |

## Execution Completion Criteria

Implementation is complete only when the full Python suite, Remotion check and `git diff --check` pass; the global production Skill validates; a real《活着》 candidate bundle is 120–150 seconds and has passed technical QC; the user has watched and explicitly approved both video and independent cover; and the immutable approval record binds the exact candidate-manifest, video and cover hashes.
