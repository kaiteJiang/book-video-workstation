"""Safe, profile-aware subtitle line-break inputs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from bv.production.profile import load_production_profile


_MAX_SUBTITLE_BREAK_BYTES = 256 * 1024


class SubtitleBreakInputError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SubtitleBreakInput:
    sequence_mode: str
    present: bool
    sha256: str | None
    chunks: tuple[str, ...]

    def reconstructs(self, approved_text: str) -> bool:
        if not self.present:
            return self.sequence_mode == "legacy-monochrome-reveal"
        reconstructed = "".join(
            character
            for chunk in self.chunks
            for character in chunk
            if not character.isspace()
        )
        approved_visible = "".join(
            character for character in approved_text if not character.isspace()
        )
        return bool(approved_visible) and reconstructed == approved_visible


def load_subtitle_break_input(episode_root: Path) -> SubtitleBreakInput:
    root = Path(episode_root)
    path = root / "script" / "subtitle_breaks.txt"
    sequence_mode = _resolve_sequence_mode(root)
    required = sequence_mode != "legacy-monochrome-reveal"
    try:
        canonical_path = Path(os.path.abspath(path))
        if _has_reparse_or_symlink_in_existing_chain(canonical_path):
            raise SubtitleBreakInputError("unsafe_subtitle_break_path")
        if not os.path.lexists(canonical_path):
            if required:
                raise SubtitleBreakInputError("subtitle_breaks_required")
            return SubtitleBreakInput(
                sequence_mode=sequence_mode,
                present=False,
                sha256=None,
                chunks=(),
            )
        raw = _read_safe_bytes(
            root,
            path,
            max_bytes=_MAX_SUBTITLE_BREAK_BYTES,
            unsafe_code="unsafe_subtitle_break_path",
            invalid_code="invalid_subtitle_break_file",
        )
        text = raw.decode("utf-8")
        chunks = tuple(line.strip() for line in text.splitlines() if line.strip())
        if not chunks:
            raise SubtitleBreakInputError("invalid_subtitle_break_file")
        return SubtitleBreakInput(
            sequence_mode=sequence_mode,
            present=True,
            sha256=hashlib.sha256(raw).hexdigest(),
            chunks=chunks,
        )
    except SubtitleBreakInputError:
        raise
    except (OSError, UnicodeError):
        raise SubtitleBreakInputError("invalid_subtitle_break_file") from None


def load_subtitle_approved_text(episode_root: Path) -> str:
    raw = _read_safe_bytes(
        Path(episode_root),
        Path(episode_root) / "script" / "approved.txt",
        max_bytes=4 * 1024 * 1024,
        unsafe_code="unsafe_subtitle_approved_path",
        invalid_code="subtitle_approved_invalid",
    )
    try:
        return raw.decode("utf-8")
    except UnicodeError:
        raise SubtitleBreakInputError("subtitle_approved_invalid") from None


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        if not os.path.lexists(path):
            return False
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
        return path.is_symlink() or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    except OSError:
        return True


def _has_reparse_or_symlink_in_existing_chain(path: Path) -> bool:
    current = Path(path)
    while True:
        if os.path.lexists(current) and _is_reparse_or_symlink(current):
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _resolve_sequence_mode(root: Path) -> str:
    profile_path = root / "production_profile.json"
    if os.path.lexists(profile_path):
        try:
            return load_production_profile(root).visual.sequence_mode
        except Exception:
            raise SubtitleBreakInputError("subtitle_profile_invalid") from None
    storyboard_path = root / "media" / "illustration" / "illustration_storyboard.json"
    if not os.path.lexists(storyboard_path):
        return "legacy-monochrome-reveal"
    try:
        raw = _read_safe_bytes(
            root,
            storyboard_path,
            max_bytes=4 * 1024 * 1024,
            unsafe_code="unsafe_subtitle_storyboard_path",
            invalid_code="subtitle_storyboard_invalid",
        )
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise SubtitleBreakInputError("subtitle_storyboard_invalid")
        sequence_mode = payload.get(
            "sequence_mode",
            "legacy-monochrome-reveal",
        )
        if sequence_mode not in {
            "legacy-monochrome-reveal",
            "color-story-pair",
        }:
            raise SubtitleBreakInputError("subtitle_storyboard_invalid")
        return sequence_mode
    except SubtitleBreakInputError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise SubtitleBreakInputError("subtitle_storyboard_invalid") from None


def _read_safe_bytes(
    root: Path,
    path: Path,
    *,
    max_bytes: int,
    unsafe_code: str,
    invalid_code: str,
) -> bytes:
    try:
        canonical_root = Path(os.path.abspath(root))
        canonical_path = Path(os.path.abspath(path))
        if (
            not canonical_path.is_relative_to(canonical_root)
            or _has_reparse_or_symlink_in_existing_chain(canonical_path)
        ):
            raise SubtitleBreakInputError(unsafe_code)
        info = canonical_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise SubtitleBreakInputError(invalid_code)
        resolved_root = canonical_root.resolve(strict=True)
        resolved_path = canonical_path.resolve(strict=True)
        if not resolved_path.is_relative_to(resolved_root):
            raise SubtitleBreakInputError(unsafe_code)
        raw = canonical_path.read_bytes()
        if len(raw) > max_bytes:
            raise SubtitleBreakInputError(invalid_code)
        return raw
    except SubtitleBreakInputError:
        raise
    except OSError:
        raise SubtitleBreakInputError(invalid_code) from None
