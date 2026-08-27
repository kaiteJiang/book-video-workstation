from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from bv.core.process import CommandResult


class ResearchSnippet(BaseModel):
    claim_id: str
    confidence: float


def _load_research_api() -> tuple[type[Exception], type[Any], Any]:
    """Import production symbols only inside tests so collection stays green."""
    from bv.models.grok_research import (
        GrokResearchError,
        GrokResearchModel,
        build_grok_research_argv,
    )

    return GrokResearchError, GrokResearchModel, build_grok_research_argv


def _result(
    argv: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> CommandResult:
    return CommandResult(argv=argv, returncode=returncode, stdout=stdout, stderr=stderr)


def _valid_stdout() -> str:
    return ResearchSnippet(claim_id="research-1", confidence=0.9).model_dump_json()


def _empty_request_dir(tmp_path: Path, name: str = "research-request") -> Path:
    request_dir = tmp_path / name
    request_dir.mkdir()
    return request_dir


def _make_symlink_or_skip(
    link: Path, target: Path, *, target_is_directory: bool
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError:
        pytest.skip("symlink creation requires additional privilege")
    if not link.exists() and not link.is_symlink():
        pytest.skip("symlink creation is unavailable on this platform")


def test_research_argv_enables_web_without_edit_permissions(tmp_path: Path) -> None:
    _, _, build_grok_research_argv = _load_research_api()
    prompt_path = tmp_path / "prompt.md"
    schema_json = '{"type":"object"}'

    argv = build_grok_research_argv(prompt_path, schema_json, tmp_path)

    assert argv[0] == "grok"
    assert argv[:3] == ["grok", "--prompt-file", str(prompt_path)]
    assert "--json-schema" in argv
    assert argv[argv.index("--json-schema") + 1] == schema_json
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--disable-web-search" not in argv
    assert argv[argv.index("--no-memory")] == "--no-memory"
    assert argv[argv.index("--no-subagents")] == "--no-subagents"
    assert ["--no-memory", "--no-subagents"] == [
        argv[argv.index("--no-memory")],
        argv[argv.index("--no-subagents")],
    ]
    assert argv[argv.index("--permission-mode") + 1] == "plan"
    assert argv[argv.index("--cwd") + 1] == str(tmp_path)


def test_research_argv_honors_custom_command(tmp_path: Path) -> None:
    _, _, build_grok_research_argv = _load_research_api()

    argv = build_grok_research_argv(
        tmp_path / "prompt.md",
        "{}",
        tmp_path,
        command=r"C:\tools\grok.cmd",
    )

    assert argv[0] == r"C:\tools\grok.cmd"
    assert "--disable-web-search" not in argv


def test_grok_research_error_exposes_only_stable_error_code() -> None:
    GrokResearchError, _, _ = _load_research_api()

    error = GrokResearchError("book_research_process_failed")

    assert error.error_code == "book_research_process_failed"
    assert str(error) == "book_research_process_failed"
    assert "TOP_SECRET" not in str(error)


def test_complete_parses_direct_schema_json_and_injects_runner(
    tmp_path: Path,
) -> None:
    _, GrokResearchModel, _ = _load_research_api()
    request_dir = _empty_request_dir(tmp_path)
    prompt = "research this TOP_SECRET_DO_NOT_ECHO title"
    calls: list[dict[str, object]] = []

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        calls.append({"argv": argv, **kwargs})
        return _result(argv, stdout=_valid_stdout())

    model = GrokResearchModel(command="grok", timeout=600.0, runner=fake_runner)
    result = model.complete(prompt, ResearchSnippet, request_dir)

    assert result == ResearchSnippet(claim_id="research-1", confidence=0.9)
    assert len(calls) == 1
    call = calls[0]
    argv = call["argv"]
    assert isinstance(argv, list)
    assert argv[0] == "grok"
    assert "--disable-web-search" not in argv
    prompt_path = Path(argv[argv.index("--prompt-file") + 1])
    assert prompt_path == request_dir / "prompt.md"
    assert prompt_path.read_text(encoding="utf-8") == prompt
    assert call["cwd"] == request_dir
    assert call["timeout"] == 600.0
    assert call["secrets"] == (prompt,)
    assert "stdin_text" not in call
    assert list(request_dir.iterdir()) == [prompt_path]


def test_complete_honors_custom_command_and_timeout(tmp_path: Path) -> None:
    _, GrokResearchModel, _ = _load_research_api()
    request_dir = _empty_request_dir(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        calls.append({"argv": argv, **kwargs})
        return _result(argv, stdout=_valid_stdout())

    model = GrokResearchModel(
        command=r"D:\bin\custom-grok.exe",
        timeout=12.5,
        runner=fake_runner,
    )
    model.complete("research prompt", ResearchSnippet, request_dir)

    assert calls[0]["argv"][0] == r"D:\bin\custom-grok.exe"
    assert calls[0]["timeout"] == 12.5


def test_complete_accepts_cli_envelope_with_text_schema_json(tmp_path: Path) -> None:
    _, GrokResearchModel, _ = _load_research_api()
    request_dir = _empty_request_dir(tmp_path)
    envelope = json.dumps({"text": _valid_stdout(), "session_id": "private-session"})

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        return _result(argv, stdout=envelope)

    result = GrokResearchModel(runner=fake_runner).complete(
        "research",
        ResearchSnippet,
        request_dir,
    )

    assert result == ResearchSnippet(claim_id="research-1", confidence=0.9)


@pytest.mark.parametrize(
    ("request_setup",),
    [
        ("missing",),
        ("file",),
        ("nonempty",),
    ],
)
def test_complete_rejects_unsafe_request_directory_before_runner(
    tmp_path: Path,
    request_setup: str,
) -> None:
    GrokResearchError, GrokResearchModel, _ = _load_research_api()
    calls: list[list[str]] = []

    if request_setup == "missing":
        request_dir = tmp_path / "absent-request"
    elif request_setup == "file":
        request_dir = tmp_path / "not-a-dir"
        request_dir.write_text("file", encoding="utf-8")
    else:
        request_dir = _empty_request_dir(tmp_path)
        (request_dir / "stale.txt").write_text("stale", encoding="utf-8")

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        calls.append(argv)
        return _result(argv, stdout=_valid_stdout())

    with pytest.raises(GrokResearchError) as raised:
        GrokResearchModel(runner=fake_runner).complete(
            "research",
            ResearchSnippet,
            request_dir,
        )

    assert raised.value.error_code == "book_research_directory_unsafe"
    assert str(raised.value) == "book_research_directory_unsafe"
    assert calls == []


def test_complete_rejects_redirected_request_directory_before_runner(
    tmp_path: Path,
) -> None:
    GrokResearchError, GrokResearchModel, _ = _load_research_api()
    target = tmp_path / "real-request"
    target.mkdir()
    request_dir = tmp_path / "redirected-request"
    _make_symlink_or_skip(request_dir, target, target_is_directory=True)
    calls: list[list[str]] = []

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        calls.append(argv)
        return _result(argv, stdout=_valid_stdout())

    with pytest.raises(GrokResearchError) as raised:
        GrokResearchModel(runner=fake_runner).complete(
            "research",
            ResearchSnippet,
            request_dir,
        )

    assert raised.value.error_code == "book_research_directory_unsafe"
    assert calls == []


def test_complete_rejects_unexpected_request_artifacts_after_runner(
    tmp_path: Path,
) -> None:
    GrokResearchError, GrokResearchModel, _ = _load_research_api()
    request_dir = _empty_request_dir(tmp_path)

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        (request_dir / "unexpected.txt").write_text("leak", encoding="utf-8")
        return _result(argv, stdout=_valid_stdout())

    with pytest.raises(GrokResearchError) as raised:
        GrokResearchModel(runner=fake_runner).complete(
            "research",
            ResearchSnippet,
            request_dir,
        )

    assert raised.value.error_code == "book_research_directory_unsafe"
    assert str(raised.value) == "book_research_directory_unsafe"


def test_complete_rejects_redirected_prompt_artifact_after_runner(
    tmp_path: Path,
) -> None:
    GrokResearchError, GrokResearchModel, _ = _load_research_api()
    request_dir = _empty_request_dir(tmp_path)
    outside = tmp_path / "outside-prompt.md"
    outside.write_text("outside", encoding="utf-8")

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        prompt_path = request_dir / "prompt.md"
        if prompt_path.exists() or prompt_path.is_symlink():
            prompt_path.unlink()
        _make_symlink_or_skip(prompt_path, outside, target_is_directory=False)
        return _result(argv, stdout=_valid_stdout())

    with pytest.raises(GrokResearchError) as raised:
        GrokResearchModel(runner=fake_runner).complete(
            "research",
            ResearchSnippet,
            request_dir,
        )

    assert raised.value.error_code == "book_research_directory_unsafe"


@pytest.mark.parametrize(
    ("returncode", "stderr", "error_code"),
    [
        (-9, "timed out", "book_research_timeout"),
        (1, "command timed out after 600 seconds", "book_research_timeout"),
        (1, "authentication failed", "book_research_auth_failed"),
        (1, "Authorization token rejected", "book_research_auth_failed"),
        (17, "ordinary failure", "book_research_process_failed"),
    ],
)
def test_complete_maps_process_failures_to_stable_codes(
    tmp_path: Path,
    returncode: int,
    stderr: str,
    error_code: str,
) -> None:
    GrokResearchError, GrokResearchModel, _ = _load_research_api()
    request_dir = _empty_request_dir(tmp_path)
    secret_prompt = "research this TOP_SECRET_DO_NOT_ECHO"

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        return _result(
            argv,
            returncode=returncode,
            stdout=f"stdout leaks {secret_prompt}",
            stderr=stderr,
        )

    with pytest.raises(GrokResearchError) as raised:
        GrokResearchModel(runner=fake_runner).complete(
            secret_prompt,
            ResearchSnippet,
            request_dir,
        )

    assert raised.value.error_code == error_code
    assert str(raised.value) == error_code
    assert secret_prompt not in str(raised.value)
    assert "authentication" not in str(raised.value).casefold()
    assert "timed out" not in str(raised.value).casefold()
    assert "ordinary failure" not in str(raised.value)


def test_complete_does_not_treat_author_diagnostics_as_auth_failure(
    tmp_path: Path,
) -> None:
    """Ordinary process failures mentioning 'author' are not auth failures."""
    GrokResearchError, GrokResearchModel, _ = _load_research_api()
    request_dir = _empty_request_dir(tmp_path)

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        return _result(
            argv,
            returncode=1,
            stderr="could not resolve author from authoritative source",
        )

    with pytest.raises(GrokResearchError) as raised:
        GrokResearchModel(runner=fake_runner).complete(
            "research",
            ResearchSnippet,
            request_dir,
        )

    assert raised.value.error_code == "book_research_process_failed"
    assert str(raised.value) == "book_research_process_failed"


@pytest.mark.parametrize(
    ("stdout",),
    [
        ("not json",),
        ('{"text": {"nested": true}}',),
        ('{"text": 12}',),
        ('{"message": "missing text field"}',),
        ('{"claim_id":"only-one-field"}',),
        ('{"text": "not json either"}',),
        ('{"text": "{\\"claim_id\\": 1, \\"confidence\\": \\"bad\\"}"}',),
    ],
)
def test_complete_maps_invalid_outputs_to_book_research_invalid(
    tmp_path: Path,
    stdout: str,
) -> None:
    GrokResearchError, GrokResearchModel, _ = _load_research_api()
    request_dir = _empty_request_dir(tmp_path)
    secret_prompt = "research this TOP_SECRET_DO_NOT_ECHO"

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        return _result(argv, stdout=stdout)

    with pytest.raises(GrokResearchError) as raised:
        GrokResearchModel(runner=fake_runner).complete(
            secret_prompt,
            ResearchSnippet,
            request_dir,
        )

    assert raised.value.error_code == "book_research_invalid"
    assert str(raised.value) == "book_research_invalid"
    assert secret_prompt not in str(raised.value)
    assert stdout not in str(raised.value)


def test_complete_error_never_exposes_private_diagnostics(tmp_path: Path) -> None:
    GrokResearchError, GrokResearchModel, _ = _load_research_api()
    request_dir = _empty_request_dir(tmp_path)
    secret_prompt = "PROMPT_SECRET_VALUE_DO_NOT_ECHO"
    secret_stdout = "STDOUT_SECRET_VALUE_DO_NOT_ECHO"
    secret_stderr = "STDERR_AUTH_TOKEN=private-token-xyz"

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        return _result(
            argv,
            returncode=1,
            stdout=secret_stdout,
            stderr=secret_stderr,
        )

    with pytest.raises(GrokResearchError) as raised:
        GrokResearchModel(runner=fake_runner).complete(
            secret_prompt,
            ResearchSnippet,
            request_dir,
        )

    rendered = str(raised.value)
    assert raised.value.error_code == "book_research_auth_failed"
    assert rendered == "book_research_auth_failed"
    assert secret_prompt not in rendered
    assert secret_stdout not in rendered
    assert secret_stderr not in rendered
    assert "private-token-xyz" not in rendered
    assert "PROMPT_SECRET" not in rendered
