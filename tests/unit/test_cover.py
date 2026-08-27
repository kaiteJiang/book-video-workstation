from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pymupdf
import pytest

from bv.video.cover import CoverApprovalError, CoverImageFacts, approve_cover


_FAKE_DECODER = object()


def _source(tmp_path: Path, name: str = "cover.jpg", data: bytes = b"image fixture") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _decoder(*, width: int = 600, height: int = 900, image_format: str = "JPEG"):
    def decode(raw: bytes, suffix: str, *, max_pixels: int) -> CoverImageFacts:
        assert raw and suffix == ".jpg" and max_pixels > width * height
        return CoverImageFacts(width=width, height=height, image_format=image_format)
    return decode


def _approve(
    tmp_path: Path, *, source_type: str = "user_provided", source: Path | None = None,
    matches_product_version: bool = True, decoder: object = _FAKE_DECODER,
):
    arguments = dict(
        book_id="book-01", episode_id="E001", source_path=source or _source(tmp_path),
        episode_cover_root=tmp_path / "book" / "episodes" / "E001" / "cover",
        source_type=source_type, matches_product_version=matches_product_version,
    )
    arguments["decoder"] = _decoder() if decoder is _FAKE_DECODER else decoder
    return approve_cover(**arguments)


@pytest.mark.parametrize("source_type", ["user_provided", "ebook_extracted", "official_product_image"])
def test_cover_approves_real_allowed_source_types_with_product_version_assertion(
    tmp_path: Path, source_type: str,
) -> None:
    source = _source(tmp_path)
    before = source.read_bytes()

    result = _approve(tmp_path, source_type=source_type, source=source)

    assert source.read_bytes() == before
    assert result.idempotent is False and result.cover_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "book_id": "book-01", "episode_id": "E001", "source_sha256": hashlib.sha256(before).hexdigest(),
        "copied_sha256": hashlib.sha256(before).hexdigest(), "source_type": source_type,
        "matches_product_version": True, "width": 600, "height": 900, "image_format": "JPEG",
    }
    assert "source_path" not in result.manifest_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("source_type", ["h3_generated", "ai_generated", "unknown", ""])
def test_cover_refuses_fake_generated_or_unknown_product_truth(tmp_path: Path, source_type: str) -> None:
    with pytest.raises(CoverApprovalError, match="invalid_cover_source_type"):
        _approve(tmp_path, source_type=source_type)

    assert not (tmp_path / "book").exists()


def test_cover_refuses_missing_explicit_product_version_match(tmp_path: Path) -> None:
    with pytest.raises(CoverApprovalError, match="product_version_not_confirmed"):
        _approve(tmp_path, matches_product_version=False)


@pytest.mark.parametrize(
    ("name", "decoder", "code"),
    [
        ("cover.gif", _decoder(), "unsupported_cover_extension"),
        ("cover.jpg", lambda raw, suffix, *, max_pixels: (_ for _ in ()).throw(ValueError("invalid")), "invalid_cover_image"),
        ("cover.jpg", _decoder(width=10_001, height=1), "cover_dimensions_out_of_bounds"),
    ],
)
def test_cover_rejects_invalid_or_excessive_decoded_images_before_copy(
    tmp_path: Path, name: str, decoder, code: str,
) -> None:
    source = _source(tmp_path, name)
    with pytest.raises(CoverApprovalError, match=code):
        _approve(tmp_path, source=source, decoder=decoder)

    assert not (tmp_path / "book").exists()


@pytest.mark.parametrize("field", ["book_id", "episode_id"])
def test_cover_rejects_unsafe_book_and_episode_identifiers_before_copy(tmp_path: Path, field: str) -> None:
    values = {"book_id": "book-01", "episode_id": "E001"}
    values[field] = "../unsafe"
    with pytest.raises(CoverApprovalError, match="unsafe_identifier"):
        approve_cover(
            **values, source_path=_source(tmp_path),
            episode_cover_root=tmp_path / "book" / "episodes" / "E001" / "cover",
            source_type="user_provided", matches_product_version=True, decoder=_decoder(),
        )


def test_cover_default_pymupdf_decoder_accepts_a_real_generated_temporary_png(tmp_path: Path) -> None:
    source = tmp_path / "cover.png"
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, 4, 6, bytes([35, 80, 160]) * 24, False)
    pixmap.save(str(source))

    result = _approve(tmp_path, source=source, decoder=None)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert (manifest["width"], manifest["height"], manifest["image_format"]) == (4, 6, "PNG")


def test_cover_rejects_oversized_snapshot_before_decoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bv.video.cover as module

    source = _source(tmp_path, data=b"x" * 33)
    monkeypatch.setattr(module, "_MAX_COVER_BYTES", 32)
    called = False

    def decoder(raw: bytes, suffix: str, *, max_pixels: int) -> CoverImageFacts:
        nonlocal called
        called = True
        return CoverImageFacts(width=1, height=1, image_format="JPEG")

    with pytest.raises(CoverApprovalError, match="cover_source_too_large"):
        _approve(tmp_path, source=source, decoder=decoder)

    assert called is False


def test_cover_identical_retry_is_idempotent_and_corruption_fails_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    first = _approve(tmp_path, source=source)
    retry = _approve(tmp_path, source=source)
    assert retry.idempotent is True and retry.cover_path == first.cover_path

    first.cover_path.write_bytes(b"corrupt")
    with pytest.raises(CoverApprovalError, match="existing_cover_integrity_invalid"):
        _approve(tmp_path, source=source)


def test_manifest_failure_rolls_back_only_new_cover_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bv.video.cover as module

    source = _source(tmp_path)
    original = source.read_bytes()
    monkeypatch.setattr(module, "_write_new_manifest", lambda path, manifest: (_ for _ in ()).throw(CoverApprovalError("manifest_write_failed")))
    with pytest.raises(CoverApprovalError, match="manifest_write_failed"):
        _approve(tmp_path, source=source)

    root = tmp_path / "book" / "episodes" / "E001" / "cover"
    assert source.read_bytes() == original
    assert not (root / "manifest.json").exists()
    assert not list(root.glob("original-*"))


def test_cover_rejects_stale_manifest_and_extra_foreign_file(tmp_path: Path) -> None:
    source = _source(tmp_path)
    first = _approve(tmp_path, source=source)
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    manifest["source_type"] = "official_product_image"
    first.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CoverApprovalError, match="existing_cover_integrity_invalid"):
        _approve(tmp_path, source=source)

    clean = tmp_path / "clean"
    next_source = _source(clean)
    approved = _approve(clean, source=next_source)
    (approved.cover_path.parent / "foreign.txt").write_text("foreign", encoding="utf-8")
    with pytest.raises(CoverApprovalError, match="existing_cover_integrity_invalid"):
        _approve(clean, source=next_source)


def test_cover_detects_source_mutation_after_snapshot_before_publication(tmp_path: Path) -> None:
    source = _source(tmp_path)

    def mutating_decoder(raw: bytes, suffix: str, *, max_pixels: int) -> CoverImageFacts:
        source.write_bytes(b"mutated after snapshot")
        return CoverImageFacts(width=600, height=900, image_format="JPEG")

    with pytest.raises(CoverApprovalError, match="cover_source_changed_during_import"):
        _approve(tmp_path, source=source, decoder=mutating_decoder)

    assert not (tmp_path / "book").exists()


def test_cover_rejects_redirected_source_or_destination_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bv.video.cover as module

    source = _source(tmp_path)
    root = tmp_path / "book" / "episodes" / "E001" / "cover"
    monkeypatch.setattr(module, "_is_redirected", lambda path: Path(path) in {source.parent, root})

    with pytest.raises(CoverApprovalError, match="unsafe_cover_path"):
        _approve(tmp_path, source=source)

    assert not root.exists()


def test_cover_rejects_real_windows_junction_destination_and_leaves_external_target_empty(tmp_path: Path) -> None:
    source = _source(tmp_path)
    root = tmp_path / "book" / "episodes" / "E001" / "cover"
    outside = tmp_path / "outside"
    root.parent.mkdir(parents=True)
    outside.mkdir()
    junction = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root), str(outside)],
        shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if junction.returncode != 0:
        pytest.skip("Windows Junction unavailable")
    try:
        with pytest.raises(CoverApprovalError, match="unsafe_cover_path"):
            _approve(tmp_path, source=source)
        assert list(outside.iterdir()) == []
    finally:
        if os.path.lexists(root):
            os.rmdir(root)


def test_cover_rejects_unrelated_existing_target_without_overwrite(tmp_path: Path) -> None:
    source = _source(tmp_path)
    copied_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    root = tmp_path / "book" / "episodes" / "E001" / "cover"
    root.mkdir(parents=True)
    target = root / f"original-{copied_sha}.jpg"
    target.write_bytes(b"unrelated existing content")

    with pytest.raises(CoverApprovalError, match="existing_cover_integrity_invalid"):
        _approve(tmp_path, source=source)

    assert target.read_bytes() == b"unrelated existing content"
