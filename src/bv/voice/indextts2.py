"""Safe, explicit IndexTTS2 command construction and one-shot synthesis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import wave
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from bv.core.process import CommandResult, run_command


_CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REVIEW_MARKER_PATTERN = re.compile(
    r"\[(?:REVIEW|TODO|TBD)[^\]]*\]|<!-- BV:VOICEOVER:(?:START|END) -->",
    re.IGNORECASE,
)


class VoiceSynthesisError(RuntimeError):
    """A stable, privacy-safe synthesis failure."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class SynthesizedVoice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_path: Path
    raw_sha256: str


Runner = Callable[..., CommandResult]


def build_synth_argv(
    *,
    install_dir: Path,
    model_dir: Path,
    script_path: Path,
    reference_voice_path: Path,
    raw_output_path: Path,
    uv_command: str = "uv",
    device: str = "cuda",
) -> list[str]:
    """Return the supported CLI argv without exposing text in a process listing."""

    return [
        str(uv_command), "run", "--project", str(install_dir), "indextts2", "synth",
        "--text-file", str(script_path), "--voice", str(reference_voice_path),
        "--output", str(raw_output_path), "--model-dir", str(model_dir),
        "--device", str(device), "--fp16", "--no-deepspeed", "--no-cuda-kernel",
        "--no-accel", "--no-torch-compile",
    ]


def build_batch_synth_argv(
    *,
    install_dir: Path,
    model_dir: Path,
    batch_file_path: Path,
    reference_voice_path: Path,
    raw_output_path: Path,
    uv_command: str = "uv",
    device: str = "cuda",
) -> list[str]:
    direct_cli = Path(install_dir) / ".venv" / "Scripts" / "indextts2.exe"
    prefix = (
        [str(direct_cli)]
        if direct_cli.is_file() and not _redirect_in_existing_chain(direct_cli)
        else [str(uv_command), "run", "--project", str(install_dir), "indextts2"]
    )
    return [
        *prefix,
        "batch",
        "--batch-file",
        str(batch_file_path),
        "--voice",
        str(reference_voice_path),
        "--concat",
        "--output",
        str(raw_output_path),
        "--model-dir",
        str(model_dir),
        "--device",
        str(device),
        "--fp16",
        "--no-deepspeed",
        "--no-cuda-kernel",
        "--no-accel",
        "--no-torch-compile",
    ]


def synthesize_voice(
    *,
    episode_root: Path,
    install_dir: Path,
    model_dir: Path,
    script_path: Path,
    approved_script_sha256: str,
    reference_voice_path: Path,
    raw_output_path: Path,
    expected_reference_sha256: str | None = None,
    uv_command: str = "uv",
    device: str = "cuda",
    runner: Runner = run_command,
    timeout: float = 3_600.0,
) -> SynthesizedVoice:
    """Run one verified synthesis command; never initialize a model in-process."""

    root = Path(episode_root)
    install = Path(install_dir)
    model = Path(model_dir)
    script = Path(script_path)
    reference = Path(reference_voice_path)
    raw = Path(raw_output_path)
    _require_safe_existing_directory(root, "unsafe_voice_path")
    for path in (install, model):
        _require_safe_existing_directory(path, "unsafe_voice_path")
    for path in (script, reference):
        _require_safe_existing_file(path, "unsafe_voice_path")
    _require_child(root, script)
    _require_child(root, raw)
    if raw.exists() or raw.is_symlink():
        raise VoiceSynthesisError("raw_output_exists")
    if raw.suffix.lower() != ".wav" or _redirect_in_existing_chain(raw) or _has_non_directory_ancestor(raw):
        raise VoiceSynthesisError("unsafe_voice_path")

    script_hash = _read_approved_script(script, approved_script_sha256)
    reference_hash = _sha256(reference)
    if expected_reference_sha256 is not None and reference_hash != expected_reference_sha256:
        raise VoiceSynthesisError("reference_hash_mismatch")
    if not _is_nonempty_wav(reference):
        raise VoiceSynthesisError("invalid_reference_wav")

    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", script.read_text(encoding="utf-8")) if item.strip()]
    batch_path: Path | None = None
    if len(paragraphs) > 1:
        batch_path = raw.with_suffix(".batch.jsonl")
        if batch_path.is_symlink() or _redirect_in_existing_chain(batch_path):
            raise VoiceSynthesisError("unsafe_voice_path")
        _write_batch_manifest(batch_path, paragraphs)
        argv = build_batch_synth_argv(
            install_dir=install,
            model_dir=model,
            batch_file_path=batch_path,
            reference_voice_path=reference,
            raw_output_path=raw,
            uv_command=uv_command,
            device=device,
        )
    else:
        argv = build_synth_argv(
            install_dir=install, model_dir=model, script_path=script,
            reference_voice_path=reference, raw_output_path=raw, uv_command=uv_command,
            device=device,
        )
    try:
        result = runner(
            argv=argv,
            cwd=install,
            timeout=timeout,
            secrets=tuple(
                str(path)
                for path in (script, reference, raw, batch_path)
                if path is not None
            ),
        )
        if result.returncode == -9:
            raise VoiceSynthesisError("synthesis_timeout")
        if result.returncode != 0:
            raise VoiceSynthesisError("synthesis_failed")
        if not raw.is_file() or raw.is_symlink() or _redirect_in_existing_chain(raw):
            raise VoiceSynthesisError("missing_raw_output")
        if not _is_nonempty_wav(raw):
            raise VoiceSynthesisError("invalid_raw_output")
        if _sha256(script) != script_hash or _sha256(reference) != reference_hash:
            raise VoiceSynthesisError("input_hash_changed")
        return SynthesizedVoice(raw_path=raw, raw_sha256=_sha256(raw))
    except VoiceSynthesisError:
        _remove_task_owned_raw(raw)
        raise
    except (OSError, ValueError, TypeError):
        _remove_task_owned_raw(raw)
        raise VoiceSynthesisError("synthesis_runner_error") from None


def _read_approved_script(path: Path, expected_sha256: str) -> str:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        raise VoiceSynthesisError("invalid_approved_script") from None
    digest = hashlib.sha256(raw).hexdigest()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or digest != expected_sha256
        or not text.strip()
        or _CONTROL_PATTERN.search(text)
        or _REVIEW_MARKER_PATTERN.search(text)
    ):
        raise VoiceSynthesisError("invalid_approved_script")
    return digest


def _write_batch_manifest(path: Path, paragraphs: list[str]) -> None:
    payload = "\n".join(
        json.dumps(
            {
                "text": paragraph,
                "silence_after_ms": 280 if index < len(paragraphs) - 1 else 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for index, paragraph in enumerate(paragraphs)
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_nonempty_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as stream:
            return stream.getnframes() > 0 and stream.getnchannels() > 0 and stream.getsampwidth() > 0
    except (OSError, EOFError, wave.Error):
        return False


def _require_child(root: Path, path: Path) -> None:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        raise VoiceSynthesisError("unsafe_voice_path") from None


def _require_safe_existing_directory(path: Path, error_code: str) -> None:
    if not path.is_dir() or path.is_symlink() or _redirect_in_existing_chain(path):
        raise VoiceSynthesisError(error_code)


def _require_safe_existing_file(path: Path, error_code: str) -> None:
    if not path.is_file() or path.is_symlink() or _redirect_in_existing_chain(path):
        raise VoiceSynthesisError(error_code)


def _remove_task_owned_raw(path: Path) -> None:
    try:
        if path.is_file() and not path.is_symlink():
            path.unlink()
    except OSError:
        pass


def _redirect_in_existing_chain(path: Path) -> bool:
    candidate = Path(path)
    while True:
        if candidate.exists() or candidate.is_symlink():
            if _is_redirected(candidate):
                return True
        parent = candidate.parent
        if parent == candidate:
            return False
        candidate = parent


def _has_non_directory_ancestor(path: Path) -> bool:
    candidate = Path(path).parent
    while True:
        if candidate.exists() and not candidate.is_dir():
            return True
        parent = candidate.parent
        if parent == candidate:
            return False
        candidate = parent


def _is_redirected(path: Path) -> bool:
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        attributes = 0
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
