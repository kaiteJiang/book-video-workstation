from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from bv.core.process import CommandResult, run_command
from bv.models.contracts import (
    empty_request_directory as _empty_request_directory,
    only_expected_artifacts as _only_expected_artifacts,
)


ModelT = TypeVar("ModelT", bound=BaseModel)

_AUTH_MARKERS = (
    "authentication",
    "authorization",
    "unauthorized",
    "auth failure",
    "auth fail",
    "auth token",
    "auth_token",
    "auth-token",
    "forbidden",
    "login",
    "credential",
)
_TOKEN_REJECTION_MARKERS = (
    "reject",
    "invalid",
    "denied",
    "expired",
    "missing",
    "fail",
)


class GrokResearchError(RuntimeError):
    """A safe, stable failure from the mandatory Grok book-research adapter."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


def build_grok_research_argv(
    prompt_path: Path,
    schema_json: str,
    empty_request_dir: Path,
    *,
    command: str = "grok",
) -> list[str]:
    """Build a web-enabled, non-editing Grok research invocation."""
    return [
        command,
        "--prompt-file",
        str(prompt_path),
        "--json-schema",
        schema_json,
        "--output-format",
        "json",
        "--no-memory",
        "--no-subagents",
        "--permission-mode",
        "plan",
        "--cwd",
        str(empty_request_dir),
    ]


def _is_auth_failure(diagnostics: str) -> bool:
    if any(marker in diagnostics for marker in _AUTH_MARKERS):
        return True
    if "token" in diagnostics and any(
        marker in diagnostics for marker in _TOKEN_REJECTION_MARKERS
    ):
        return True
    return False


def _process_error_code(result: CommandResult) -> str:
    diagnostics = f"{result.stdout}\n{result.stderr}".casefold()
    if result.returncode == -9 or "timed out" in diagnostics:
        return "book_research_timeout"
    if _is_auth_failure(diagnostics):
        return "book_research_auth_failed"
    return "book_research_process_failed"


def _parse_research_output(stdout: str, schema_type: type[ModelT]) -> ModelT:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        raise GrokResearchError("book_research_invalid") from None

    if not isinstance(payload, dict):
        raise GrokResearchError("book_research_invalid")

    if "text" in payload:
        text = payload["text"]
        if not isinstance(text, str):
            raise GrokResearchError("book_research_invalid")
        try:
            return schema_type.model_validate_json(text)
        except (ValidationError, ValueError):
            raise GrokResearchError("book_research_invalid") from None

    try:
        return schema_type.model_validate(payload)
    except ValidationError:
        raise GrokResearchError("book_research_invalid") from None


class GrokResearchModel:
    """Mandatory local Grok CLI adapter for structured book research."""

    def __init__(
        self,
        *,
        command: str = "grok",
        timeout: float = 600.0,
        runner: Callable[..., CommandResult] | None = None,
    ) -> None:
        self.command = command
        self.timeout = timeout
        self.runner = run_command if runner is None else runner

    def complete(
        self, prompt: str, schema_type: type[ModelT], request_dir: Path
    ) -> ModelT:
        request_dir = Path(request_dir)
        if not _empty_request_directory(request_dir):
            raise GrokResearchError("book_research_directory_unsafe")

        prompt_path = request_dir / "prompt.md"
        try:
            prompt_path.write_text(prompt, encoding="utf-8")
        except OSError:
            raise GrokResearchError("book_research_directory_unsafe") from None

        schema_json = json.dumps(
            schema_type.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
        )
        result = self.runner(
            build_grok_research_argv(
                prompt_path,
                schema_json,
                request_dir,
                command=self.command,
            ),
            cwd=request_dir,
            timeout=self.timeout,
            secrets=(prompt,),
        )

        if not _only_expected_artifacts(request_dir, {prompt_path}):
            raise GrokResearchError("book_research_directory_unsafe")

        if result.returncode != 0:
            raise GrokResearchError(_process_error_code(result))

        return _parse_research_output(result.stdout, schema_type)
