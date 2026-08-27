"""RED tests for the local human-writing naturalization service.

Production module under test: bv.content.human_writing (absent in this pass).

Combined Skill hash framing (collision-safe, deterministic):
For each required relative POSIX path in fixed order, append big-endian
uint64 path length, UTF-8 path bytes, big-endian uint64 content length,
and the exact file bytes. The final SHA-256 digests that concatenation.
Required order:
  1. SKILL.md
  2. references/reality.md
  3. references/formats.md
  4. references/revision.md
  5. scripts/check_prose.py
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bv.content.scripts import (
    HumanizedScript,
    SemanticLock,
    VoiceoverCoverage,
    build_semantic_lock,
)
from bv.content.synthesis import ClaimCluster, WholeBookValue
from bv.content.topics import EpisodeBrief
from bv.core.process import CommandResult
from bv.models.contracts import ModelCompletionError, ModelInvalidResponseError


# ---------------------------------------------------------------------------
# Production import (lazy so collection succeeds while the module is missing)
# ---------------------------------------------------------------------------


def _load_human_writing_api() -> tuple[Any, ...]:
    """Import production symbols only inside tests so collection stays green."""
    from bv.content.human_writing import (
        HumanWritingError,
        HumanWritingResult,
        HumanWritingService,
    )

    return HumanWritingError, HumanWritingResult, HumanWritingService


# ---------------------------------------------------------------------------
# Combined Skill hash (test oracle; production must match this framing)
# ---------------------------------------------------------------------------

_REQUIRED_RELATIVE_PATHS: tuple[str, ...] = (
    "SKILL.md",
    "references/reality.md",
    "references/formats.md",
    "references/revision.md",
    "scripts/check_prose.py",
)


def combined_skill_sha256(skill_root: Path) -> str:
    """Collision-safe combined hash over path+bytes for the five Skill assets.

    Framing for each required relative POSIX path, in order:
      struct.pack(">Q", len(path_utf8)) + path_utf8
      + struct.pack(">Q", len(file_bytes)) + file_bytes
    Final digest is SHA-256 of the full concatenation, lowercase hex.
    """
    hasher = hashlib.sha256()
    for relative in _REQUIRED_RELATIVE_PATHS:
        path_bytes = relative.encode("utf-8")
        content = (skill_root / Path(relative)).read_bytes()
        hasher.update(struct.pack(">Q", len(path_bytes)))
        hasher.update(path_bytes)
        hasher.update(struct.pack(">Q", len(content)))
        hasher.update(content)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


def _value() -> WholeBookValue:
    return WholeBookValue(
        value_thesis="这本书帮助读者重看选择、关系与责任的边界",
        target_reader="经常为了评价而犹豫的成年人",
        reader_before="把他人的评价当成选择是否正确的证明",
        reader_after="能区分自己的选择和他人的评价",
        central_life_tension="想按自己的判断生活又害怕不被认可",
        supporting_claim_clusters=[
            ClaimCluster(
                cluster_id="problem",
                summary="先看见以评价替代判断的问题",
                narrative_role="problem",
                claim_ids=["C03-001"],
                chapter_regions=["R3"],
            ),
            ClaimCluster(
                cluster_id="reframe",
                summary="再区分自己的选择和他人的评价",
                narrative_role="reframe",
                claim_ids=["C01-001", "C02-001"],
                chapter_regions=["R1", "R2"],
            ),
            ClaimCluster(
                cluster_id="application",
                summary="最后把判断落到可承担的行动",
                narrative_role="application",
                claim_ids=["C04-001"],
                chapter_regions=["R4"],
            ),
        ],
        practical_value="提供重新理解选择和关系的视角，不承诺消除痛苦，也不保证立刻改变",
        reading_reason="视频只呈现价值主线，完整论证和边界需要回到原书",
    )


def _brief(value: WholeBookValue) -> EpisodeBrief:
    return EpisodeBrief(
        episode_id="E001",
        episode_kind="whole_book_value",
        topic_name="重新拿回选择权",
        book_value_thesis=value.value_thesis,
        target_reader=value.target_reader,
        reader_before=value.reader_before,
        reader_after=value.reader_after,
        central_life_tension=value.central_life_tension,
        supporting_claim_cluster_ids=["problem", "reframe", "application"],
        source_claim_ids=["C01-001", "C02-001", "C03-001", "C04-001"],
        life_connection="从反复解释和迎合转向承担自己的选择",
        practical_value=value.practical_value,
        reading_reason=value.reading_reason,
        constructed_scene=True,
        status="reserved",
    )


def _lock() -> SemanticLock:
    value = _value()
    return build_semantic_lock(_brief(value), value)


def _voiceover() -> str:
    return (
        "你准备回复那条消息时，又把自己的决定交给别人的评价。"
        "这本书帮助读者重看选择、关系与责任的边界。"
        "它先让人看见，先看见以评价替代判断的问题，会把责任推给别人。"
        "再区分自己的选择和他人的评价，判断才有落点。"
        "最后把判断落到可承担的行动，关系也不必靠迎合维持。"
        "从反复解释和迎合转向承担自己的选择。"
        "你能区分自己的选择和他人的评价。"
        "它提供重新理解选择和关系的视角，不承诺消除痛苦，也不保证立刻改变。"
        "视频只呈现价值主线，完整论证和边界需要回到原书。"
    )


def _coverage(text: str | None = None) -> VoiceoverCoverage:
    text = text or _voiceover()
    return VoiceoverCoverage(
        thesis_span="这本书帮助读者重看选择、关系与责任的边界",
        reader_after_span="你能区分自己的选择和他人的评价",
        life_connection_span="从反复解释和迎合转向承担自己的选择",
        boundary_span="它提供重新理解选择和关系的视角，不承诺消除痛苦，也不保证立刻改变",
        reading_reason_span="视频只呈现价值主线，完整论证和边界需要回到原书",
        cluster_spans={
            "problem": "先看见以评价替代判断的问题",
            "reframe": "再区分自己的选择和他人的评价",
            "application": "最后把判断落到可承担的行动",
        },
    )


def _draft() -> str:
    return _voiceover()


def _skill_fixture(tmp_path: Path) -> Path:
    """Create the five required Skill assets under a local fixture root."""
    root = tmp_path / "human-writing-skill"
    (root / "references").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    files = {
        "SKILL.md": "# human-writing fixture\nUse $human-writing for oral naturalization.\n",
        "references/reality.md": "# reality\nPreserve locked facts only.\n",
        "references/formats.md": "# formats\nOral voiceover format rules.\n",
        "references/revision.md": "# revision\nRevise after drafting.\n",
        "scripts/check_prose.py": (
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "print('fixture-checker')\n"
            "sys.exit(0)\n"
        ),
    }
    for relative, body in files.items():
        path = root / relative
        path.write_text(body, encoding="utf-8", newline="\n")
    return root


def _make_symlink_or_skip(
    link: Path, target: Path, *, target_is_directory: bool
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError:
        pytest.skip("symlink creation requires additional privilege")
    if not link.exists() and not link.is_symlink():
        pytest.skip("symlink creation is unavailable on this platform")


class _FakeModel:
    """Records complete() calls and returns a scripted HumanizedScript (never a process)."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(self, prompt: str, schema_type: type, request_dir: Path) -> object:
        request_path = Path(request_dir)
        self.calls.append(
            {
                "prompt": prompt,
                "schema_type": schema_type,
                "request_dir": request_path,
            }
        )
        assert request_path.is_dir()
        # Request dir must start empty; production may write the candidate after.
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _RecordingRunner:
    """Fake process runner that records argv/cwd/timeout/secrets and returns a result."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "ok",
        stderr: str = "",
        side_effect: Exception | None = None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.side_effect = side_effect
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: Path | str | None = None,
        timeout: float | None = None,
        secrets: object = (),
        stdin_text: str | None = None,
        **kwargs: object,
    ) -> CommandResult:
        recorded = {
            "argv": list(argv),
            "cwd": Path(cwd) if cwd is not None else None,
            "timeout": timeout,
            "secrets": secrets,
            "stdin_text": stdin_text,
            "kwargs": kwargs,
        }
        self.calls.append(recorded)
        if self.side_effect is not None:
            raise self.side_effect
        return CommandResult(
            argv=list(argv),
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def _service(
    skill_root: Path,
    model: _FakeModel,
    runner: _RecordingRunner,
    *,
    checker_timeout: float = 60.0,
) -> Any:
    _, _, HumanWritingService = _load_human_writing_api()
    return HumanWritingService(
        model=model,
        skill_path=skill_root / "SKILL.md",
        checker_path=skill_root / "scripts" / "check_prose.py",
        runner=runner,
        checker_timeout=checker_timeout,
    )


def _good_humanized(voiceover: str | None = None) -> HumanizedScript:
    text = voiceover or _voiceover()
    return HumanizedScript(voiceover=text, coverage=_coverage(text))


def _assert_safe_error(error: Exception, code: str) -> None:
    assert getattr(error, "error_code", None) == code
    rendered = str(error)
    assert rendered == code
    # Private process / model bodies must never surface in the public error.
    for forbidden in (
        "PRIVATE_MODEL_BODY",
        "fixture-checker",
        "candidate-secret-body",
        "SECRET_STDOUT",
        "SECRET_STDERR",
        "traceback",
    ):
        assert forbidden not in rendered


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


def test_human_writing_error_exposes_code_only_str() -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    error = HumanWritingError("human_writing_skill_invalid")
    assert error.error_code == "human_writing_skill_invalid"
    assert str(error) == "human_writing_skill_invalid"


def test_human_writing_result_requires_nonblank_voiceover_and_true_checker(
    tmp_path: Path,
) -> None:
    _, HumanWritingResult, _ = _load_human_writing_api()
    coverage = _coverage()
    skill_hash = "a" * 64
    ok = HumanWritingResult(
        voiceover=_voiceover(),
        coverage=coverage,
        skill_sha256=skill_hash,
        checker_passed=True,
    )
    assert ok.checker_passed is True
    assert ok.skill_sha256 == skill_hash

    with pytest.raises(ValidationError):
        HumanWritingResult(
            voiceover="   ",
            coverage=coverage,
            skill_sha256=skill_hash,
            checker_passed=True,
        )
    with pytest.raises(ValidationError):
        HumanWritingResult(
            voiceover=_voiceover(),
            coverage=coverage,
            skill_sha256="ABC" + "a" * 61,
            checker_passed=True,
        )
    with pytest.raises(ValidationError):
        HumanWritingResult(
            voiceover=_voiceover(),
            coverage=coverage,
            skill_sha256=skill_hash,
            checker_passed=False,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Successful naturalization
# ---------------------------------------------------------------------------


def test_naturalize_runs_checker_and_returns_bound_result(tmp_path: Path) -> None:
    skill_root = _skill_fixture(tmp_path)
    expected_hash = combined_skill_sha256(skill_root)
    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner(returncode=0, stdout="ok")
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    result = service.naturalize(
        draft=_draft(),
        semantic_lock=_lock(),
        coverage=_coverage(),
        request_root=request_root,
    )

    assert result.voiceover == _voiceover()
    assert result.coverage == _coverage()
    assert result.skill_sha256 == expected_hash
    assert result.skill_sha256 == result.skill_sha256.lower()
    assert len(result.skill_sha256) == 64
    assert result.checker_passed is True

    assert len(model.calls) == 1
    request_dir = model.calls[0]["request_dir"]
    assert isinstance(request_dir, Path)
    assert request_dir.is_relative_to(request_root)
    assert request_dir != request_root
    assert request_dir.is_dir()

    # Candidate must live only under the unique request directory.
    candidate_files = [
        path
        for path in request_dir.rglob("*")
        if path.is_file() and path.suffix in {".txt", ".md", ".candidate"}
    ]
    assert candidate_files, "expected a candidate prose file under the request directory"
    for path in candidate_files:
        assert request_root not in {path}  # not written at the root itself
        assert path.is_relative_to(request_dir)
        assert not path.is_relative_to(skill_root)

    assert len(runner.calls) == 1
    call = runner.calls[0]
    argv = call["argv"]
    assert isinstance(argv, list)
    assert argv[0] == sys.executable
    assert argv[1] == str(skill_root / "scripts" / "check_prose.py")
    candidate_arg = Path(argv[2])
    assert candidate_arg.is_relative_to(request_dir)
    assert candidate_arg.is_file()
    assert call["cwd"] == request_dir
    assert call["timeout"] == 60.0
    # secrets may be empty or a sequence; presence of the keyword is required.
    assert "secrets" in call


def test_trusted_prompt_invokes_skill_and_keeps_only_source_inputs_untrusted(
    tmp_path: Path,
) -> None:
    skill_root = _skill_fixture(tmp_path)
    skill_path = skill_root / "SKILL.md"
    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()
    lock = _lock()
    draft = _draft()
    coverage = _coverage()

    service.naturalize(
        draft=draft,
        semantic_lock=lock,
        coverage=coverage,
        request_root=request_root,
    )

    prompt = model.calls[0]["prompt"]
    assert isinstance(prompt, str)
    assert prompt.count("BEGIN_SOURCE_DATA") == prompt.count("END_SOURCE_DATA") == 1
    trusted, encoded_source = prompt.split("BEGIN_SOURCE_DATA\n", maxsplit=1)
    source_raw, _ = encoded_source.split("END_SOURCE_DATA", maxsplit=1)

    assert "$human-writing" in trusted
    assert str(skill_path) in trusted or skill_path.as_posix() in trusted
    assert "SKILL.md" in trusted
    assert "reality.md" in trusted
    assert "formats.md" in trusted
    assert "revision.md" in trusted
    # reality + formats before first pass; revision after drafting.
    reality_pos = trusted.find("reality.md")
    formats_pos = trusted.find("formats.md")
    revision_pos = trusted.find("revision.md")
    assert 0 <= reality_pos < revision_pos
    assert 0 <= formats_pos < revision_pos
    assert "preserve" in trusted.casefold() or "保留" in trusted
    for token in ("locked", "fact", "事实", "semantic"):
        if token in trusted.casefold() or token in trusted:
            break
    else:
        pytest.fail("trusted prompt must require preserving locked facts")

    decoded = json.loads(source_raw)
    if isinstance(decoded, str):
        payload = json.loads(decoded)
    else:
        payload = decoded
    assert isinstance(payload, dict)
    # Only draft / semantic lock / coverage belong in the untrusted block.
    allowed = {"draft", "semantic_lock", "coverage"}
    assert set(payload) == allowed
    assert payload["draft"] == draft


def test_naturalize_keeps_production_context_in_untrusted_payload(
    tmp_path: Path,
) -> None:
    skill_root = _skill_fixture(tmp_path)
    model = _FakeModel(_good_humanized())
    service = _service(skill_root, model, _RecordingRunner())
    request_root = tmp_path / "requests"
    request_root.mkdir()
    production_context = {
        "duration": {
            "hard_min_seconds": 120.0,
            "ideal_min_seconds": 128.0,
            "ideal_max_seconds": 142.0,
            "hard_max_seconds": 150.0,
        },
        "meaning_unit_target": {"current_life": 0.6, "source_book": 0.4},
        "fact_categories": [
            "verified_fact",
            "interpretation",
            "illustration_metaphor",
        ],
    }

    service.naturalize(
        draft=_draft(),
        semantic_lock=_lock(),
        coverage=_coverage(),
        request_root=request_root,
        production_context=production_context,
    )

    prompt = model.calls[0]["prompt"]
    assert isinstance(prompt, str)
    trusted, encoded_source = prompt.split("BEGIN_SOURCE_DATA\n", maxsplit=1)
    payload = json.loads(json.loads(encoded_source.split("END_SOURCE_DATA", 1)[0]))
    assert "production_context" in trusted
    assert "duration" in trusted
    assert "verified fact" in trusted.casefold()
    assert "interpretation" in trusted.casefold()
    assert "illustration metaphor" in trusted.casefold()
    assert payload["production_context"] == production_context


def test_prompt_and_candidate_never_include_ebook_body_or_source_path(
    tmp_path: Path,
) -> None:
    skill_root = _skill_fixture(tmp_path)
    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    service.naturalize(
        draft=_draft(),
        semantic_lock=_lock(),
        coverage=_coverage(),
        request_root=request_root,
    )

    prompt = model.calls[0]["prompt"]
    assert isinstance(prompt, str)
    forbidden_snippets = (
        "chapter text",
        "电子书",
        "ebook body",
        "E:\\books\\",
        "/books/",
        "source.epub",
        "paragraph_id",
    )
    for snippet in forbidden_snippets:
        assert snippet.casefold() not in prompt.casefold()

    request_dir = model.calls[0]["request_dir"]
    assert isinstance(request_dir, Path)
    for path in request_dir.rglob("*"):
        if path.is_file():
            body = path.read_text(encoding="utf-8", errors="replace")
            for snippet in forbidden_snippets:
                assert snippet.casefold() not in body.casefold()


# ---------------------------------------------------------------------------
# Skill / checker / request-root preflight failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_skill",
        "missing_reality",
        "skill_is_directory",
        "missing_checker",
        "checker_is_directory",
    ],
)
def test_invalid_skill_or_checker_fails_before_model(
    tmp_path: Path, mutation: str
) -> None:
    HumanWritingError, _, HumanWritingService = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner()

    skill_path = skill_root / "SKILL.md"
    checker_path = skill_root / "scripts" / "check_prose.py"
    if mutation == "missing_skill":
        skill_path.unlink()
    elif mutation == "missing_reality":
        (skill_root / "references" / "reality.md").unlink()
    elif mutation == "skill_is_directory":
        skill_path.unlink()
        skill_path.mkdir()
    elif mutation == "missing_checker":
        checker_path.unlink()
    elif mutation == "checker_is_directory":
        checker_path.unlink()
        checker_path.mkdir()

    service = HumanWritingService(
        model=model,
        skill_path=skill_path,
        checker_path=checker_path,
        runner=runner,
    )
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    _assert_safe_error(raised.value, "human_writing_skill_invalid")
    assert model.calls == []
    assert runner.calls == []


def test_redirected_skill_asset_fails_before_model(tmp_path: Path) -> None:
    HumanWritingError, _, HumanWritingService = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    outside = tmp_path / "outside-reality.md"
    outside.write_text("redirected", encoding="utf-8")
    reality = skill_root / "references" / "reality.md"
    reality.unlink()
    _make_symlink_or_skip(reality, outside, target_is_directory=False)

    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner()
    service = HumanWritingService(
        model=model,
        skill_path=skill_root / "SKILL.md",
        checker_path=skill_root / "scripts" / "check_prose.py",
        runner=runner,
    )
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    _assert_safe_error(raised.value, "human_writing_skill_invalid")
    assert model.calls == []
    assert runner.calls == []


def test_redirected_parent_chain_fails_before_model(tmp_path: Path) -> None:
    HumanWritingError, _, HumanWritingService = _load_human_writing_api()
    real_skill = _skill_fixture(tmp_path / "real")
    linked_parent = tmp_path / "linked-skill"
    _make_symlink_or_skip(linked_parent, real_skill, target_is_directory=True)

    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner()
    service = HumanWritingService(
        model=model,
        skill_path=linked_parent / "SKILL.md",
        checker_path=linked_parent / "scripts" / "check_prose.py",
        runner=runner,
    )
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    _assert_safe_error(raised.value, "human_writing_skill_invalid")
    assert model.calls == []
    assert runner.calls == []


@pytest.mark.parametrize(
    "bad_root",
    ["missing", "file", "symlink"],
)
def test_unsafe_request_root_fails_before_model(tmp_path: Path, bad_root: str) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)

    if bad_root == "missing":
        request_root = tmp_path / "does-not-exist"
    elif bad_root == "file":
        request_root = tmp_path / "not-a-dir.txt"
        request_root.write_text("x", encoding="utf-8")
    else:
        real = tmp_path / "real-requests"
        real.mkdir()
        request_root = tmp_path / "linked-requests"
        _make_symlink_or_skip(request_root, real, target_is_directory=True)

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    assert raised.value.error_code in {
        "human_writing_request_root_invalid",
        "unsafe_request_directory",
    }
    assert str(raised.value) == raised.value.error_code
    assert model.calls == []
    assert runner.calls == []


def test_blank_draft_fails_before_model(tmp_path: Path) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft="  \n\t  ",
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    assert raised.value.error_code == "human_writing_draft_invalid"
    assert str(raised.value) == "human_writing_draft_invalid"
    assert model.calls == []
    assert runner.calls == []


def test_tampered_semantic_lock_fails_before_model(tmp_path: Path) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    lock = _lock()
    # Nested mutation breaks the integrity hash without rebinding sha256.
    lock.allowed_claim_ids_by_cluster["problem"] = ()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=lock,
            coverage=_coverage(),
            request_root=request_root,
        )

    assert raised.value.error_code in {
        "human_writing_lock_invalid",
        "semantic_lock_integrity_failed",
    }
    assert str(raised.value) == raised.value.error_code
    assert model.calls == []
    assert runner.calls == []


# ---------------------------------------------------------------------------
# Model failures
# ---------------------------------------------------------------------------


def test_model_failure_maps_to_stable_safe_error(tmp_path: Path) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    private = "PRIVATE_MODEL_BODY: secret stack and path C:\\Users\\1\\secrets"
    model = _FakeModel(ModelCompletionError("provider_failed", private))
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    _assert_safe_error(raised.value, "human_writing_model_failed")
    assert private not in str(raised.value)
    assert runner.calls == []


def test_unexpected_model_output_maps_to_stable_safe_error(tmp_path: Path) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    model = _FakeModel({"voiceover": "not a HumanizedScript"})
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    _assert_safe_error(raised.value, "human_writing_invalid_output")
    assert runner.calls == []


def test_invalid_model_response_error_does_not_leak_private_body(
    tmp_path: Path,
) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    private = '{"voiceover":"PRIVATE_MODEL_BODY"}'
    model = _FakeModel(
        ModelInvalidResponseError("invalid_json", "bad", private_response=private)
    )
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    assert raised.value.error_code in {
        "human_writing_invalid_output",
        "human_writing_model_failed",
    }
    assert str(raised.value) == raised.value.error_code
    assert "PRIVATE_MODEL_BODY" not in str(raised.value)
    assert runner.calls == []


def test_blank_humanized_voiceover_is_rejected_before_checker(tmp_path: Path) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    bad = HumanizedScript(voiceover="   \n", coverage=_coverage())
    model = _FakeModel(bad)
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    _assert_safe_error(raised.value, "human_writing_invalid_output")
    assert runner.calls == []


# ---------------------------------------------------------------------------
# Fact-lock mismatches (before checker)
# ---------------------------------------------------------------------------


def test_fact_lock_mismatch_on_mutated_locked_fields_before_checker(
    tmp_path: Path,
) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    lock = _lock()
    # Drop/mutate thesis, reader transformation, life connection, boundary,
    # reading reason, and one cluster span.
    mutated_text = (
        "你准备回复那条消息时，又把自己的决定交给别人的评价。"
        "这是一条被改写的主线。"
        "它先让人看见，先看见以评价替代判断的问题，会把责任推给别人。"
        "再区分自己的选择和他人的评价，判断才有落点。"
        "最后把判断落到可承担的行动，关系也不必靠迎合维持。"
        "从别处找出口。"
        "你会彻底变成另一个人。"
        "它保证立刻消除痛苦。"
        "不必回到原书。"
    )
    mutated_coverage = VoiceoverCoverage(
        thesis_span="这是一条被改写的主线",
        reader_after_span="你会彻底变成另一个人",
        life_connection_span="从别处找出口",
        boundary_span="它保证立刻消除痛苦",
        reading_reason_span="不必回到原书",
        cluster_spans={
            "problem": "先看见以评价替代判断的问题",
            "reframe": "再区分自己的选择和他人的评价",
            # application cluster dropped / rewritten
            "application": "改成无关的另一句话",
        },
    )
    model = _FakeModel(
        HumanizedScript(voiceover=mutated_text, coverage=mutated_coverage)
    )
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=lock,
            coverage=_coverage(),
            request_root=request_root,
        )

    _assert_safe_error(raised.value, "human_writing_fact_lock_mismatch")
    assert runner.calls == []
    assert len(model.calls) == 1


def test_fact_lock_mismatch_on_unknown_number_or_negation_before_checker(
    tmp_path: Path,
) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    lock = _lock()
    # Inject a number and a negation not present in the lock's allowed sets.
    polluted = _voiceover() + "答案是99，而且没有任何人能反驳。"
    coverage = _coverage()
    model = _FakeModel(HumanizedScript(voiceover=polluted, coverage=coverage))
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=lock,
            coverage=coverage,
            request_root=request_root,
        )

    _assert_safe_error(raised.value, "human_writing_fact_lock_mismatch")
    assert runner.calls == []


# ---------------------------------------------------------------------------
# Candidate write and checker outcomes
# ---------------------------------------------------------------------------


def test_candidate_write_failure_is_stable_and_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    import bv.content.human_writing as hw_module

    skill_root = _skill_fixture(tmp_path)
    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner()
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    def _fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full: candidate-secret-body")

    # Prefer production write helpers if present; otherwise patch Path.write_text
    # only for candidate paths under the request root.
    if hasattr(hw_module, "_atomic_write_text"):
        monkeypatch.setattr(hw_module, "_atomic_write_text", _fail_write)
    elif hasattr(hw_module, "atomic_write_text"):
        monkeypatch.setattr(hw_module, "atomic_write_text", _fail_write)
    else:
        original_write_text = Path.write_text

        def _guarded_write_text(self: Path, data: str, *args: object, **kwargs: object):
            if self.is_relative_to(request_root) and self != request_root:
                raise OSError("disk full: candidate-secret-body")
            return original_write_text(self, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", _guarded_write_text)

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    _assert_safe_error(raised.value, "human_writing_candidate_write_failed")
    assert "candidate-secret-body" not in str(raised.value)
    assert runner.calls == []


def test_checker_timeout_maps_to_timeout_code_without_leaking_output(
    tmp_path: Path,
) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner(
        returncode=-9,
        stdout="SECRET_STDOUT candidate-secret-body",
        stderr="SECRET_STDERR command timed out after 60 seconds",
    )
    service = _service(skill_root, model, runner, checker_timeout=12.5)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    _assert_safe_error(raised.value, "human_writing_checker_timeout")
    assert "SECRET_STDOUT" not in str(raised.value)
    assert "SECRET_STDERR" not in str(raised.value)
    assert "candidate-secret-body" not in str(raised.value)
    assert len(runner.calls) == 1
    assert runner.calls[0]["timeout"] == 12.5


def test_checker_nonzero_maps_to_failed_code_without_leaking_output(
    tmp_path: Path,
) -> None:
    HumanWritingError, _, _ = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner(
        returncode=2,
        stdout="SECRET_STDOUT banned pattern hit",
        stderr="SECRET_STDERR checker details",
    )
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    with pytest.raises(HumanWritingError) as raised:
        service.naturalize(
            draft=_draft(),
            semantic_lock=_lock(),
            coverage=_coverage(),
            request_root=request_root,
        )

    _assert_safe_error(raised.value, "human_writing_checker_failed")
    assert "SECRET_STDOUT" not in str(raised.value)
    assert "SECRET_STDERR" not in str(raised.value)


def test_checker_runs_after_model_and_after_candidate_exists(tmp_path: Path) -> None:
    skill_root = _skill_fixture(tmp_path)
    events: list[str] = []

    class _OrderedModel(_FakeModel):
        def complete(self, prompt: str, schema_type: type, request_dir: Path) -> object:
            events.append("model")
            return super().complete(prompt, schema_type, request_dir)

    class _OrderedRunner(_RecordingRunner):
        def __call__(self, argv, **kwargs):  # type: ignore[no-untyped-def]
            events.append("checker")
            request_dir = Path(kwargs["cwd"])
            # Candidate must already exist when the checker starts.
            candidates = [p for p in request_dir.rglob("*") if p.is_file()]
            assert candidates, "checker ran before candidate file existed"
            assert Path(argv[2]).is_file()
            return super().__call__(argv, **kwargs)

    model = _OrderedModel(_good_humanized())
    runner = _OrderedRunner(returncode=0)
    service = _service(skill_root, model, runner)
    request_root = tmp_path / "requests"
    request_root.mkdir()

    result = service.naturalize(
        draft=_draft(),
        semantic_lock=_lock(),
        coverage=_coverage(),
        request_root=request_root,
    )

    assert result.checker_passed is True
    assert events == ["model", "checker"]


def test_service_constructor_and_naturalize_are_keyword_only(tmp_path: Path) -> None:
    _, _, HumanWritingService = _load_human_writing_api()
    skill_root = _skill_fixture(tmp_path)
    model = _FakeModel(_good_humanized())
    runner = _RecordingRunner()

    with pytest.raises(TypeError):
        HumanWritingService(  # type: ignore[misc]
            model,
            skill_root / "SKILL.md",
            skill_root / "scripts" / "check_prose.py",
        )

    service = HumanWritingService(
        model=model,
        skill_path=skill_root / "SKILL.md",
        checker_path=skill_root / "scripts" / "check_prose.py",
        runner=runner,
    )
    request_root = tmp_path / "requests"
    request_root.mkdir()
    with pytest.raises(TypeError):
        service.naturalize(  # type: ignore[misc]
            _draft(),
            _lock(),
            _coverage(),
            request_root,
        )
