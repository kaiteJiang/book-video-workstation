from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import bv.workflow.media_stages as module
from bv.workflow.media_stages import MediaStageError, _optional_handdrawn_cover
from bv.workflow.stages import StageContext
from bv.video.cover import CoverManifest


def _context(tmp_path: Path) -> StageContext:
    episode = tmp_path / "workspace" / "books" / "book-01" / "episodes" / "E001"
    episode.mkdir(parents=True)
    return StageContext(book_id="book-01", episode_id="E001", episode_root=episode)


def test_optional_handdrawn_cover_is_none_only_when_entry_is_truly_absent(
    tmp_path: Path,
) -> None:
    assert _optional_handdrawn_cover(_context(tmp_path), 15_000) is None


def test_optional_handdrawn_cover_rejects_partial_existing_directory(tmp_path: Path) -> None:
    context = _context(tmp_path)
    (context.episode_root / "cover").mkdir()

    with pytest.raises(MediaStageError, match="final_render_input_invalid"):
        _optional_handdrawn_cover(context, 15_000)


def test_optional_handdrawn_cover_rejects_redirect_even_when_exists_is_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    cover_root = context.episode_root / "cover"
    original = module._is_reparse_or_symlink
    monkeypatch.setattr(
        module,
        "_is_reparse_or_symlink",
        lambda path: Path(path) == cover_root or original(Path(path)),
    )

    with pytest.raises(MediaStageError, match="final_render_input_invalid"):
        _optional_handdrawn_cover(context, 15_000)


@pytest.mark.parametrize("unsafe_name", ["manifest.json", "book.png"])
def test_optional_handdrawn_cover_rejects_redirected_manifest_or_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_name: str,
) -> None:
    context = _context(tmp_path)
    cover_root = context.episode_root / "cover"
    cover_root.mkdir()
    cover_path = cover_root / "book.png"
    cover_path.write_bytes(b"cover")
    digest = hashlib.sha256(b"cover").hexdigest()
    manifest_path = cover_root / "manifest.json"
    manifest_path.write_text(
        CoverManifest(
            book_id="book-01",
            episode_id="E001",
            source_sha256=digest,
            copied_sha256=digest,
            source_type="user_provided",
            matches_product_version=True,
            width=240,
            height=360,
            image_format="PNG",
        ).model_dump_json(),
        encoding="utf-8",
    )
    unsafe_path = cover_root / unsafe_name
    original = module._is_reparse_or_symlink
    monkeypatch.setattr(
        module,
        "_is_reparse_or_symlink",
        lambda path: Path(path) == unsafe_path or original(Path(path)),
    )

    with pytest.raises(MediaStageError, match="final_render_input_invalid"):
        _optional_handdrawn_cover(context, 15_000)


def test_reparse_detector_checks_lexists_before_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "dangling-junction"
    monkeypatch.setattr(module.os.path, "lexists", lambda _path: True)
    monkeypatch.setattr(Path, "exists", lambda _path: False)
    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)
    monkeypatch.setattr(
        Path,
        "stat",
        lambda _path, *, follow_symlinks=False: SimpleNamespace(
            st_file_attributes=0x400
        ),
    )

    assert module._is_reparse_or_symlink(candidate) is True
