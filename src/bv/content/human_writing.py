"""Local human-writing Skill naturalization for oral voiceover drafts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from bv.content.scripts import HumanizedScript, SemanticLock, VoiceoverCoverage
from bv.core.process import CommandResult, run_command
from bv.models.contracts import (
    ModelCompletionError,
    ModelInvalidResponseError,
    StructuredModel,
    empty_request_directory,
    is_redirected,
)
from bv.models.prompts import compose_source_prompt

_REQUIRED_RELATIVE_PATHS: tuple[str, ...] = (
    "SKILL.md",
    "references/reality.md",
    "references/formats.md",
    "references/revision.md",
    "scripts/check_prose.py",
)
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
_NEGATION_TOKENS = ("不", "没有", "无", "未")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CANDIDATE_NAME = "candidate.txt"

CommandRunner = Callable[..., CommandResult]


class HumanWritingError(RuntimeError):
    """A safe, code-only failure from human-writing naturalization."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class HumanWritingResult(BaseModel):
    """Bound naturalized voiceover accepted by the local Skill checker."""

    model_config = ConfigDict(extra="forbid")

    voiceover: str
    coverage: VoiceoverCoverage
    skill_sha256: str
    checker_passed: Literal[True]

    @field_validator("voiceover")
    @classmethod
    def _require_nonblank_voiceover(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("voiceover must not be blank")
        return value

    @field_validator("skill_sha256")
    @classmethod
    def _require_lowercase_sha256(cls, value: str) -> str:
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("skill_sha256 must be a lowercase hex SHA-256 digest")
        return value


class HumanWritingService:
    """Naturalize a draft voiceover through the installed human-writing Skill."""

    def __init__(
        self,
        *,
        model: StructuredModel,
        skill_path: Path,
        checker_path: Path,
        runner: CommandRunner | None = None,
        checker_timeout: float = 60.0,
    ) -> None:
        if checker_timeout <= 0:
            raise ValueError("checker_timeout must be positive")
        self.model = model
        self.skill_path = Path(skill_path)
        self.checker_path = Path(checker_path)
        self.runner: CommandRunner = run_command if runner is None else runner
        self.checker_timeout = float(checker_timeout)

    def naturalize(
        self,
        *,
        draft: str,
        semantic_lock: SemanticLock,
        coverage: VoiceoverCoverage,
        request_root: Path,
        production_context: dict[str, object] | None = None,
    ) -> HumanWritingResult:
        if not isinstance(draft, str) or not draft.strip():
            raise HumanWritingError("human_writing_draft_invalid")
        if not _semantic_lock_is_intact(semantic_lock):
            raise HumanWritingError("human_writing_lock_invalid")

        skill_root, skill_sha256 = _validate_skill_identity(
            skill_path=self.skill_path,
            checker_path=self.checker_path,
        )
        request_root = _validate_request_root(request_root)
        request_dir = _create_request_directory(request_root)

        prompt = _build_prompt(
            skill_path=self.skill_path,
            draft=draft,
            semantic_lock=semantic_lock,
            coverage=coverage,
            production_context=production_context,
        )
        candidate = _complete_humanized(
            model=self.model,
            prompt=prompt,
            request_dir=request_dir,
        )
        _validate_fact_lock(candidate=candidate, semantic_lock=semantic_lock)

        candidate_path = request_dir / _CANDIDATE_NAME
        try:
            _atomic_write_text(candidate_path, candidate.voiceover)
        except OSError as error:
            raise HumanWritingError("human_writing_candidate_write_failed") from error
        if is_redirected(candidate_path) or not candidate_path.is_file():
            raise HumanWritingError("human_writing_candidate_write_failed")
        if not candidate_path.is_relative_to(request_dir):
            raise HumanWritingError("human_writing_candidate_write_failed")

        _run_checker(
            runner=self.runner,
            checker_path=skill_root / "scripts" / "check_prose.py",
            candidate_path=candidate_path,
            request_dir=request_dir,
            timeout=self.checker_timeout,
            secrets=(draft, candidate.voiceover),
        )
        return HumanWritingResult(
            voiceover=candidate.voiceover,
            coverage=candidate.coverage,
            skill_sha256=skill_sha256,
            checker_passed=True,
        )


def combined_skill_sha256(skill_root: Path) -> str:
    """Collision-safe combined hash over path+bytes for the five Skill assets."""
    hasher = hashlib.sha256()
    for relative in _REQUIRED_RELATIVE_PATHS:
        path_bytes = relative.encode("utf-8")
        content = (skill_root / Path(relative)).read_bytes()
        hasher.update(struct.pack(">Q", len(path_bytes)))
        hasher.update(path_bytes)
        hasher.update(struct.pack(">Q", len(content)))
        hasher.update(content)
    return hasher.hexdigest()


def check_human_writing_candidate(
    *,
    voiceover: str,
    skill_path: Path,
    request_root: Path,
    runner: CommandRunner | None = None,
    checker_timeout: float = 60.0,
) -> str:
    """Run the installed Skill checker against an explicitly reviewed script."""
    if not isinstance(voiceover, str) or not voiceover.strip():
        raise HumanWritingError("human_writing_draft_invalid")
    skill_path = Path(skill_path)
    skill_root, skill_sha256 = _validate_skill_identity(
        skill_path=skill_path,
        checker_path=skill_path.parent / "scripts" / "check_prose.py",
    )
    request_dir = _create_request_directory(_validate_request_root(request_root))
    candidate_path = request_dir / _CANDIDATE_NAME
    try:
        _atomic_write_text(candidate_path, voiceover)
    except OSError as error:
        raise HumanWritingError("human_writing_candidate_write_failed") from error
    if is_redirected(candidate_path) or not candidate_path.is_file():
        raise HumanWritingError("human_writing_candidate_write_failed")
    _run_checker(
        runner=run_command if runner is None else runner,
        checker_path=skill_root / "scripts" / "check_prose.py",
        candidate_path=candidate_path,
        request_dir=request_dir,
        timeout=checker_timeout,
        secrets=(voiceover,),
    )
    return skill_sha256


def _atomic_write_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _semantic_lock_is_intact(lock: SemanticLock) -> bool:
    try:
        fields = lock.model_dump(mode="json", exclude={"sha256"})
        digest = hashlib.sha256(
            json.dumps(
                fields,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return digest == lock.sha256
    except (TypeError, ValueError):
        return False


def _validate_skill_identity(
    *,
    skill_path: Path,
    checker_path: Path,
) -> tuple[Path, str]:
    skill_path = Path(skill_path)
    checker_path = Path(checker_path)
    skill_root = skill_path.parent
    expected_checker = skill_root / "scripts" / "check_prose.py"

    if skill_path.name != "SKILL.md":
        raise HumanWritingError("human_writing_skill_invalid")
    if checker_path != expected_checker:
        raise HumanWritingError("human_writing_skill_invalid")
    if _redirect_in_chain(skill_root) or is_redirected(skill_root):
        raise HumanWritingError("human_writing_skill_invalid")

    for relative in _REQUIRED_RELATIVE_PATHS:
        asset = skill_root / Path(relative)
        if _redirect_in_chain(asset) or not _is_real_file(asset):
            raise HumanWritingError("human_writing_skill_invalid")
    if not _is_real_file(skill_path) or not _is_real_file(checker_path):
        raise HumanWritingError("human_writing_skill_invalid")

    try:
        digest = combined_skill_sha256(skill_root)
    except OSError as error:
        raise HumanWritingError("human_writing_skill_invalid") from error
    return skill_root, digest


def _validate_request_root(request_root: Path) -> Path:
    root = Path(request_root)
    try:
        if (
            not root.exists()
            or not root.is_dir()
            or is_redirected(root)
            or _redirect_in_chain(root)
        ):
            raise HumanWritingError("human_writing_request_root_invalid")
    except OSError as error:
        raise HumanWritingError("human_writing_request_root_invalid") from error
    return root


def _create_request_directory(request_root: Path) -> Path:
    try:
        request_dir = Path(
            tempfile.mkdtemp(prefix="human-writing-", dir=request_root)
        )
    except OSError as error:
        raise HumanWritingError("human_writing_request_root_invalid") from error
    if (
        not empty_request_directory(request_dir)
        or is_redirected(request_dir)
        or not request_dir.is_relative_to(request_root)
        or request_dir == request_root
    ):
        raise HumanWritingError("human_writing_request_root_invalid")
    return request_dir


def _build_prompt(
    *,
    skill_path: Path,
    draft: str,
    semantic_lock: SemanticLock,
    coverage: VoiceoverCoverage,
    production_context: dict[str, object] | None = None,
) -> str:
    skill_display = str(skill_path)
    trusted = (
        "Use the installed $human-writing Skill for oral naturalization of the "
        "draft voiceover.\n"
        f"Skill entry: {skill_display}\n"
        "Read SKILL.md first.\n"
        "Before the first writing pass, also read references/reality.md and "
        "references/formats.md so locked facts and oral format rules are preserved.\n"
        "After drafting, read references/revision.md and revise.\n"
        "Preserve every locked fact, semantic boundary, number, and negation from "
        "the semantic lock. Do not invent electronic-book body text or source paths.\n"
        "When production_context is present, use its duration and reader-life balance "
        "as writing constraints. Keep verified fact, interpretation, and illustration "
        "metaphor distinct; never promote the latter two into book facts.\n"
        "Return only a HumanizedScript with the naturalized voiceover and coverage "
        "spans bound to the lock."
    )
    payload = {
        "draft": draft,
        "semantic_lock": semantic_lock.model_dump(mode="json"),
        "coverage": coverage.model_dump(mode="json"),
    }
    if production_context is not None:
        payload["production_context"] = production_context
    return compose_source_prompt(
        trusted,
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )


def _complete_humanized(
    *,
    model: StructuredModel,
    prompt: str,
    request_dir: Path,
) -> HumanizedScript:
    try:
        response = model.complete(prompt, HumanizedScript, request_dir)
    except ModelInvalidResponseError as error:
        raise HumanWritingError("human_writing_invalid_output") from error
    except ModelCompletionError as error:
        raise HumanWritingError("human_writing_model_failed") from error
    except Exception as error:
        raise HumanWritingError("human_writing_model_failed") from error

    if not isinstance(response, HumanizedScript):
        raise HumanWritingError("human_writing_invalid_output")
    if not isinstance(response.voiceover, str) or not response.voiceover.strip():
        raise HumanWritingError("human_writing_invalid_output")
    return response


def _validate_fact_lock(
    *,
    candidate: HumanizedScript,
    semantic_lock: SemanticLock,
) -> None:
    voiceover = candidate.voiceover
    coverage = candidate.coverage
    required_clusters = set(semantic_lock.required_cluster_ids)

    required_bindings = (
        (semantic_lock.value_thesis, coverage.thesis_span),
        (semantic_lock.reader_after, coverage.reader_after_span),
        (semantic_lock.life_connection, coverage.life_connection_span),
        (semantic_lock.practical_boundary, coverage.boundary_span),
        (semantic_lock.reading_reason, coverage.reading_reason_span),
    )
    for locked, span in required_bindings:
        if (
            not isinstance(span, str)
            or not span.strip()
            or locked not in span
            or span not in voiceover
        ):
            raise HumanWritingError("human_writing_fact_lock_mismatch")

    if set(coverage.cluster_spans) != required_clusters:
        raise HumanWritingError("human_writing_fact_lock_mismatch")
    for cluster_id in semantic_lock.required_cluster_ids:
        span = coverage.cluster_spans.get(cluster_id, "")
        summary = semantic_lock.cluster_coverage_terms.get(cluster_id, "")
        if (
            not isinstance(span, str)
            or not span.strip()
            or span not in voiceover
            or (summary and summary not in span)
        ):
            raise HumanWritingError("human_writing_fact_lock_mismatch")

    for number in _NUMBER_PATTERN.findall(voiceover):
        if number not in semantic_lock.allowed_numbers:
            raise HumanWritingError("human_writing_fact_lock_mismatch")
    for token in _NEGATION_TOKENS:
        if token in voiceover and token not in semantic_lock.allowed_negations:
            raise HumanWritingError("human_writing_fact_lock_mismatch")


def _run_checker(
    *,
    runner: CommandRunner,
    checker_path: Path,
    candidate_path: Path,
    request_dir: Path,
    timeout: float,
    secrets: Sequence[str],
) -> None:
    argv = [sys.executable, str(checker_path), str(candidate_path)]
    secret_values = tuple(secret for secret in secrets if secret)
    try:
        result = runner(
            argv,
            cwd=request_dir,
            timeout=timeout,
            secrets=secret_values,
        )
    except Exception as error:
        raise HumanWritingError("human_writing_checker_failed") from error

    if not isinstance(result, CommandResult):
        raise HumanWritingError("human_writing_checker_failed")
    if result.returncode == -9:
        raise HumanWritingError("human_writing_checker_timeout")
    if result.returncode != 0:
        raise HumanWritingError("human_writing_checker_failed")


def _is_real_file(path: Path) -> bool:
    try:
        if is_redirected(path):
            return False
        info = path.stat(follow_symlinks=False)
        return stat.S_ISREG(info.st_mode)
    except OSError:
        return False


def _redirect_in_chain(path: Path) -> bool:
    current = Path(path)
    while True:
        try:
            if current.is_symlink() or current.exists():
                if is_redirected(current):
                    return True
        except OSError:
            pass
        parent = current.parent
        if parent == current:
            return False
        current = parent
