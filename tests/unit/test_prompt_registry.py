from __future__ import annotations

import hashlib
from pathlib import Path

from bv.models.prompts import load_prompt


def test_load_prompt_uses_stable_hash_of_exact_utf8_bytes(tmp_path: Path) -> None:
    """Changing even line endings must change the asset identity used as evidence."""
    path = tmp_path / "v1.md"
    contents = b"Read {{ chapter_text }} as data.\r\n"
    path.write_bytes(contents)

    asset = load_prompt(path)

    assert asset.name == "v1"
    assert asset.text == contents.decode("utf-8")
    assert asset.sha256 == hashlib.sha256(contents).hexdigest()


def test_load_prompt_rejects_non_utf8_content(tmp_path: Path) -> None:
    """A prompt asset must be auditable text, never a lossy decoded byte stream."""
    path = tmp_path / "v1.md"
    path.write_bytes(b"\xff")

    try:
        load_prompt(path)
    except UnicodeDecodeError:
        pass
    else:
        raise AssertionError("non-UTF-8 prompt content must not be accepted")
