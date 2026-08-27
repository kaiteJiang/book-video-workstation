from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bv.core.atomic import atomic_write_json
from bv.core.hashing import sha256_file


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,127}")


class CoverBriefError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class _CoverModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CoverDerivedFields(_CoverModel):
    title: str
    subtitle: str = ""
    summary: str
    visual_subject: str
    audience: str
    mood: str
    visual_metaphor: str
    banned_elements: tuple[str, ...] = ()

    @field_validator(
        "title",
        "summary",
        "visual_subject",
        "audience",
        "mood",
        "visual_metaphor",
    )
    @classmethod
    def _nonblank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 500:
            raise ValueError("cover_field_invalid")
        return normalized


class CoverBrief(_CoverModel):
    schema_version: int = 1
    title: str
    style_id: str
    aspect_ratio: Literal["3:4"] = "3:4"
    width: Literal[1080] = 1080
    height: Literal[1440] = 1440
    approved_script_sha256: str
    semantic_lock_sha256: str
    production_profile_sha256: str
    style_meta_sha256: str
    style_atom_sha256: str
    blueprint_sha256: str
    approved_script_path: Path
    semantic_lock_path: Path
    style_meta_path: Path
    style_atom_path: Path
    blueprint_path: Path
    derived_fields: CoverDerivedFields
    prompt_path: Path
    prompt_sha256: str
    manifest_path: Path

    @field_validator(
        "approved_script_sha256",
        "semantic_lock_sha256",
        "production_profile_sha256",
        "style_meta_sha256",
        "style_atom_sha256",
        "blueprint_sha256",
        "prompt_sha256",
    )
    @classmethod
    def _hash(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("cover_hash_invalid")
        return value


class CoverBriefBuilder:
    def __init__(self, *, blueprint_path: Path) -> None:
        self.blueprint_path = Path(blueprint_path)

    def build(
        self,
        *,
        episode_root: Path,
        slug: str,
        fields: CoverDerivedFields,
        compiled_prompt: str,
        approved_script_path: Path,
        approved_script_sha256: str,
        semantic_lock_path: Path,
        semantic_lock_sha256: str,
        production_profile_sha256: str,
        style_id: str,
        style_meta_path: Path,
        style_atom_path: Path,
    ) -> CoverBrief:
        root = Path(episode_root).absolute()
        if _SLUG.fullmatch(slug) is None or _SHA256.fullmatch(
            production_profile_sha256
        ) is None:
            raise CoverBriefError("cover_brief_invalid")
        script = Path(approved_script_path)
        semantic_lock = Path(semantic_lock_path)
        for path, expected in (
            (script, approved_script_sha256),
            (semantic_lock, semantic_lock_sha256),
        ):
            _require_safe_regular(path)
            if not path.absolute().is_relative_to(root) or sha256_file(path) != expected:
                raise CoverBriefError("cover_source_stale")
        meta_path = Path(style_meta_path)
        atom_path = Path(style_atom_path)
        blueprint_path = self.blueprint_path
        for path in (meta_path, atom_path, blueprint_path):
            _require_safe_regular(path)
        metadata = _style_metadata(meta_path)
        if metadata.get("id") != style_id:
            raise CoverBriefError("cover_style_mismatch")
        outputs = metadata.get("outputs")
        if not isinstance(outputs, list) or not {"cover", "poster"}.intersection(
            {str(item) for item in outputs}
        ):
            raise CoverBriefError("cover_style_ineligible")

        prompt = compiled_prompt.strip() + "\n"
        script_text = script.read_text(encoding="utf-8").strip()
        if script_text and script_text in prompt:
            raise CoverBriefError("cover_prompt_copies_source")
        if (
            not prompt.strip()
            or "{{" in prompt
            or "}}" in prompt
            or fields.title not in prompt
            or style_id not in prompt
            or "3:4" not in prompt
        ):
            raise CoverBriefError("cover_prompt_invalid")

        prompt_path = (
            root
            / "punk-assets"
            / "punk-cover"
            / slug
            / "prompts"
            / "cover.md"
        )
        manifest_path = prompt_path.parents[1] / "cover-brief.json"
        _require_safe_destination(root, prompt_path)
        _require_safe_destination(root, manifest_path)
        if (
            prompt_path.exists()
            or prompt_path.is_symlink()
            or manifest_path.exists()
            or manifest_path.is_symlink()
        ):
            raise CoverBriefError("cover_brief_output_exists")
        _atomic_write_text(prompt_path, prompt)
        brief = CoverBrief(
            title=fields.title,
            style_id=style_id,
            approved_script_sha256=approved_script_sha256,
            semantic_lock_sha256=semantic_lock_sha256,
            production_profile_sha256=production_profile_sha256,
            style_meta_sha256=sha256_file(meta_path),
            style_atom_sha256=sha256_file(atom_path),
            blueprint_sha256=sha256_file(blueprint_path),
            approved_script_path=script.absolute(),
            semantic_lock_path=semantic_lock.absolute(),
            style_meta_path=meta_path.absolute(),
            style_atom_path=atom_path.absolute(),
            blueprint_path=blueprint_path.absolute(),
            derived_fields=fields,
            prompt_path=prompt_path,
            prompt_sha256=sha256_file(prompt_path),
            manifest_path=manifest_path,
        )
        atomic_write_json(manifest_path, brief.model_dump(mode="json"))
        return brief


def _style_metadata(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
        match = re.search(r"```yaml\s*(.*?)\s*```", text, flags=re.DOTALL)
        if match is None:
            raise ValueError
        payload = yaml.safe_load(match.group(1))
        if not isinstance(payload, dict):
            raise ValueError
        return payload
    except Exception:
        raise CoverBriefError("cover_style_metadata_invalid") from None


def _require_safe_regular(path: Path) -> None:
    try:
        if _redirect_in_existing_chain(path):
            raise CoverBriefError("unsafe_cover_path")
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise CoverBriefError("unsafe_cover_path")
    except CoverBriefError:
        raise
    except OSError:
        raise CoverBriefError("unsafe_cover_path") from None


def _require_safe_destination(root: Path, path: Path) -> None:
    if not path.absolute().is_relative_to(root) or _redirect_in_existing_chain(path):
        raise CoverBriefError("unsafe_cover_path")


def _redirect_in_existing_chain(path: Path) -> bool:
    candidate = Path(path)
    while True:
        if candidate.exists() or candidate.is_symlink():
            attributes = getattr(
                candidate.stat(follow_symlinks=False), "st_file_attributes", 0
            )
            if candidate.is_symlink() or attributes & 0x400:
                return True
        if candidate.parent == candidate:
            return False
        candidate = candidate.parent


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=path.parent
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
