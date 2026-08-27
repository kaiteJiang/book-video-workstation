from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from bv.core.process import run_command
from bv.models.contracts import (
    ModelCompletionError,
    ModelInvalidResponseError,
    empty_request_directory as _empty_request_directory,
    is_redirected as _is_redirected,
    only_expected_artifacts as _only_expected_artifacts,
)


ModelT = TypeVar("ModelT", bound=BaseModel)

_SCHEMA_NAME = "schema.json"
_OUTPUT_NAME = "output.json"
_BEGIN_SOURCE = "\nBEGIN_SOURCE_DATA\n"
_END_SOURCE = "\nEND_SOURCE_DATA\n"


def _resolved_command(command: str) -> str:
    if os.name != "nt":
        return command
    return shutil.which(command) or command


def build_codex_argv(
    request_dir: Path,
    schema_path: Path,
    output_path: Path,
    *,
    command: str = "codex",
) -> list[str]:
    return [
        command,
        "exec",
        "--ephemeral",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--cd",
        str(request_dir),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "--color",
        "never",
        "-",
    ]


def _validated_prompt(prompt: str) -> str:
    if prompt.count(_BEGIN_SOURCE) != 1 or prompt.count(_END_SOURCE) != 1:
        raise ModelCompletionError(
            "codex_prompt_invalid",
            "Codex prompt source boundaries are invalid.",
        )
    begin = prompt.index(_BEGIN_SOURCE)
    end = prompt.index(_END_SOURCE)
    if begin == 0 or end <= begin + len(_BEGIN_SOURCE):
        raise ModelCompletionError(
            "codex_prompt_invalid",
            "Codex prompt source boundaries are invalid.",
        )
    return prompt


class CodexCliModel:
    """Constrained local Codex CLI adapter for primary structured generation."""

    def __init__(self, *, command: str = "codex", timeout: float = 60.0) -> None:
        self.command = command
        self.timeout = timeout

    def complete(self, prompt: str, schema_type: type[ModelT], request_dir: Path) -> ModelT:
        request_dir = Path(request_dir)
        if not _empty_request_directory(request_dir):
            raise ModelCompletionError(
                "codex_request_directory_invalid", "Codex request directory is unsafe."
            )
        validated_prompt = _validated_prompt(prompt)

        schema_path = request_dir / _SCHEMA_NAME
        output_path = request_dir / _OUTPUT_NAME
        try:
            schema_path.write_text(
                json.dumps(
                    schema_type.model_json_schema(),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except OSError:
            raise ModelCompletionError(
                "codex_request_directory_invalid",
                "Codex request directory is unsafe.",
            ) from None
        result = run_command(
            build_codex_argv(
                request_dir,
                schema_path,
                output_path,
                command=_resolved_command(self.command),
            ),
            cwd=request_dir,
            stdin_text=validated_prompt,
            timeout=self.timeout,
            secrets=(prompt,),
        )
        if result.returncode != 0:
            raise ModelCompletionError("codex_process_failed", "Codex request failed.")
        if _is_redirected(output_path):
            raise ModelCompletionError(
                "codex_output_redirected", "Codex response path is unsafe."
            )
        if not output_path.is_file():
            raise ModelCompletionError(
                "codex_output_missing", "Codex response was not produced."
            )
        if not _only_expected_artifacts(request_dir, {schema_path, output_path}):
            raise ModelCompletionError(
                "codex_request_artifacts", "Codex request produced unexpected files."
            )
        try:
            private_response = output_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                private_response = output_path.read_bytes().decode(
                    "utf-8",
                    errors="replace",
                )
            except OSError:
                raise ModelCompletionError(
                    "codex_output_invalid", "Codex response was invalid."
                ) from None
        except OSError:
            raise ModelCompletionError(
                "codex_output_invalid", "Codex response was invalid."
            ) from None
        try:
            return schema_type.model_validate_json(private_response)
        except (ValidationError, ValueError):
            raise ModelInvalidResponseError(
                "codex_output_invalid",
                "Codex response was invalid.",
                private_response,
            ) from None
