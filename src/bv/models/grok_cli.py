from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from bv.core.process import CommandResult, run_command
from bv.models.contracts import (
    ReviewSkipped,
    empty_request_directory as _empty_request_directory,
    only_expected_artifacts as _only_expected_artifacts,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


def build_grok_argv(
    prompt_path: Path,
    schema_json: str,
    empty_request_dir: Path,
    *,
    command: str = "grok",
) -> list[str]:
    return [
        command,
        "--prompt-file",
        str(prompt_path),
        "--json-schema",
        schema_json,
        "--output-format",
        "json",
        "--disable-web-search",
        "--no-memory",
        "--no-subagents",
        "--permission-mode",
        "plan",
        "--cwd",
        str(empty_request_dir),
    ]


def _review_skip_for_process(result: CommandResult) -> ReviewSkipped:
    diagnostics = f"{result.stdout}\n{result.stderr}".casefold()
    if result.returncode == -9 or "timed out" in diagnostics:
        return ReviewSkipped(
            error_code="review_timeout",
            user_message="Grok review was skipped because the request timed out.",
        )
    if "auth" in diagnostics:
        return ReviewSkipped(
            error_code="review_auth_failed",
            user_message="Grok review was skipped because authentication failed.",
        )
    return ReviewSkipped(
        error_code="review_process_failed",
        user_message="Grok review was skipped because the request failed.",
    )


class GrokCliModel:
    """Optional local Grok CLI review adapter; failures are explicitly non-blocking."""

    def __init__(self, *, command: str = "grok", timeout: float = 60.0) -> None:
        self.command = command
        self.timeout = timeout

    def complete(
        self, prompt: str, schema_type: type[ModelT], request_dir: Path
    ) -> ModelT | ReviewSkipped:
        request_dir = Path(request_dir)
        if not _empty_request_directory(request_dir):
            return ReviewSkipped(
                error_code="review_request_directory_invalid",
                user_message="Grok review was skipped because the request directory is unsafe.",
            )

        prompt_path = request_dir / "prompt.md"
        try:
            prompt_path.write_text(prompt, encoding="utf-8")
        except OSError:
            return ReviewSkipped(
                error_code="review_request_directory_invalid",
                user_message="Grok review was skipped because the request directory is unsafe.",
            )

        schema_json = json.dumps(schema_type.model_json_schema(), ensure_ascii=False, sort_keys=True)
        result = run_command(
            build_grok_argv(
                prompt_path, schema_json, request_dir, command=self.command
            ),
            cwd=request_dir,
            timeout=self.timeout,
            secrets=(prompt,),
        )
        if result.returncode != 0:
            return _review_skip_for_process(result)
        if not _only_expected_artifacts(request_dir, {prompt_path}):
            return ReviewSkipped(
                error_code="review_request_directory_invalid",
                user_message="Grok review was skipped because the request directory is unsafe.",
            )
        try:
            return schema_type.model_validate_json(result.stdout)
        except (ValidationError, ValueError):
            return ReviewSkipped(
                error_code="review_invalid_output",
                user_message="Grok review was skipped because the response was invalid.",
            )
