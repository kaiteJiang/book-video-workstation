"""RED tests for provider-neutral video contracts.

Production module under test: bv.video.contracts (absent in this pass).
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, get_type_hints

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Production import (lazy so collection succeeds while the module is missing)
# ---------------------------------------------------------------------------


def _load_contracts_api() -> dict[str, Any]:
    """Import production symbols only inside tests so collection stays green."""
    from bv.video.contracts import (
        VideoGateway,
        VideoGenerationRequest,
        VideoGenerationResult,
        VideoProviderName,
    )

    return {
        "VideoGateway": VideoGateway,
        "VideoGenerationRequest": VideoGenerationRequest,
        "VideoGenerationResult": VideoGenerationResult,
        "VideoProviderName": VideoProviderName,
    }


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_BOOK_ID = "book-01"
_EPISODE_ID = "E001"
_PROMPT = "Quiet reader sits at a wooden desk, adult East Asian face, no dialogue, no on-screen text."
_TINY_IMAGE = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
_TINY_VIDEO = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32


def _sha(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _episode_root(tmp_path: Path, *, book_id: str = _BOOK_ID, episode_id: str = _EPISODE_ID) -> Path:
    root = tmp_path / "workspace" / "books" / book_id / "episodes" / episode_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_image(path: Path, data: bytes = _TINY_IMAGE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _hash_images(images: Any) -> tuple[str, ...]:
    hashes: list[str] = []
    for item in images:
        try:
            hashes.append(_sha(Path(item).read_bytes()))
        except OSError:
            hashes.append("0" * 64)
    return tuple(hashes)


def _write_video(path: Path, data: bytes = _TINY_VIDEO) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _make_symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError:
        pytest.skip("symlink creation requires additional privilege")
    if not link.exists() and not link.is_symlink():
        pytest.skip("symlink creation is unavailable on this platform")


def _request_kwargs(
    tmp_path: Path,
    *,
    segment_id: str = "S01",
    provider: str = "grok_cli",
    prompt: str = _PROMPT,
    prompt_sha256: str | None = None,
    expected_duration_ms: int = 12_000,
    aspect_ratio: str = "9:16",
    resolution: str = "480p",
    reference_images: tuple[Path, ...] | list[Path] | None = None,
    reference_image_specs: tuple[str, ...] | list[str] | None = None,
    reference_image_sha256s: tuple[str, ...] | list[str] | None = None,
    episode_root: Path | str | None = None,
    book_id: str = _BOOK_ID,
    episode_id: str = _EPISODE_ID,
) -> dict[str, Any]:
    specs = reference_image_specs
    images = reference_images
    if specs is None and images is None:
        specs = ("Adult East Asian reader, short black hair, navy shirt, calm posture",)
        images = ()
    elif specs is None:
        specs = ()
    elif images is None:
        images = ()
    if episode_root is None:
        episode_root = _episode_root(tmp_path, book_id=book_id, episode_id=episode_id)
    if reference_image_sha256s is None:
        reference_image_sha256s = _hash_images(images)
    return {
        "book_id": book_id,
        "episode_id": episode_id,
        "segment_id": segment_id,
        "provider": provider,
        "prompt": prompt,
        "prompt_sha256": _sha(prompt) if prompt_sha256 is None else prompt_sha256,
        "expected_duration_ms": expected_duration_ms,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "episode_root": episode_root,
        "reference_images": images,
        "reference_image_sha256s": reference_image_sha256s,
        "reference_image_specs": specs,
    }


def _valid_request(tmp_path: Path, **overrides: Any) -> Any:
    api = _load_contracts_api()
    payload = _request_kwargs(tmp_path, **overrides)
    return api["VideoGenerationRequest"].model_validate(payload)


def _valid_s02_request(tmp_path: Path, **overrides: Any) -> Any:
    episode = _episode_root(tmp_path)
    image = _write_image(episode / "references" / "character.png")
    defaults: dict[str, Any] = {
        "segment_id": "S02",
        "episode_root": episode,
        "reference_images": (image,),
        "reference_image_sha256s": (_sha(_TINY_IMAGE),),
        "reference_image_specs": ("Same character opens a notebook at the desk",),
    }
    defaults.update(overrides)
    return _valid_request(tmp_path, **defaults)


def _result_kwargs(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    source = overrides.pop("source_path", None)
    if source is None:
        source = _write_video(tmp_path / "imports" / "S01.mp4")
    payload: dict[str, Any] = {
        "provider": "grok_cli",
        "segment_id": "S01",
        "source_path": source,
    }
    if "source_sha256" not in overrides:
        payload["source_sha256"] = _sha(Path(source).read_bytes())
    payload.update(overrides)
    return payload


def _valid_result(tmp_path: Path, **overrides: Any) -> Any:
    api = _load_contracts_api()
    return api["VideoGenerationResult"].model_validate(_result_kwargs(tmp_path, **overrides))


# ---------------------------------------------------------------------------
# Provider literals
# ---------------------------------------------------------------------------


def test_video_provider_name_accepts_only_declared_literals(tmp_path: Path) -> None:
    api = _load_contracts_api()
    allowed = {"grok_cli", "grok_manual", "h3_manual"}
    assert set(getattr(api["VideoProviderName"], "__args__", ())) == allowed

    for provider in sorted(allowed):
        request = _valid_request(tmp_path, provider=provider)
        assert request.provider == provider

    with pytest.raises(ValidationError):
        _valid_request(tmp_path, provider="openai")


# ---------------------------------------------------------------------------
# Happy paths: S01 spec-only, S02 image + scene spec
# ---------------------------------------------------------------------------


def test_s01_accepts_spec_only_request(tmp_path: Path) -> None:
    request = _valid_request(
        tmp_path,
        segment_id="S01",
        reference_images=(),
        reference_image_sha256s=(),
        reference_image_specs=(
            "Adult East Asian reader, short black hair, navy shirt, calm posture",
        ),
    )

    assert request.segment_id == "S01"
    assert request.reference_images == ()
    assert request.reference_image_sha256s == ()
    assert request.episode_root == _episode_root(tmp_path)
    assert request.episode_root.parts[-4:] == ("books", _BOOK_ID, "episodes", _EPISODE_ID)
    assert len(request.reference_image_specs) == 1
    assert request.aspect_ratio == "9:16"
    assert request.resolution == "480p"
    assert request.prompt_sha256 == _sha(_PROMPT)
    assert request.model_config.get("frozen") is True


def test_s02_accepts_episode_local_reference_plus_scene_spec(tmp_path: Path) -> None:
    request = _valid_s02_request(tmp_path)

    assert request.segment_id == "S02"
    assert len(request.reference_images) == 1
    assert request.reference_images[0].is_file()
    assert request.episode_root == _episode_root(tmp_path)
    assert request.reference_images[0].is_relative_to(request.episode_root)
    assert f"books/{_BOOK_ID}/episodes/{_EPISODE_ID}" in request.reference_images[0].as_posix()
    assert request.reference_image_sha256s == (_sha(_TINY_IMAGE),)
    assert request.reference_image_specs == ("Same character opens a notebook at the desk",)


# ---------------------------------------------------------------------------
# Identifiers, prompt hash, duration, aspect, resolution
# ---------------------------------------------------------------------------


def test_request_rejects_unsafe_identifiers(tmp_path: Path) -> None:
    for field, bad_value in (
        ("book_id", "../escape"),
        ("episode_id", "has space"),
        ("segment_id", "bad/id"),
        ("segment_id", "-leading"),
        ("book_id", "a" * 65),
        ("episode_id", ""),
    ):
        with pytest.raises(ValidationError):
            _valid_request(tmp_path, **{field: bad_value})


def test_request_rejects_blank_prompt_or_prompt_hash_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _valid_request(tmp_path, prompt="   ")

    with pytest.raises(ValidationError):
        _valid_request(tmp_path, prompt=_PROMPT, prompt_sha256="0" * 64)


def test_request_enforces_duration_aspect_and_resolution(tmp_path: Path) -> None:
    low = _valid_request(tmp_path, expected_duration_ms=1, resolution="480p")
    high = _valid_request(tmp_path, expected_duration_ms=15_000, resolution="720p")
    assert low.expected_duration_ms == 1
    assert high.expected_duration_ms == 15_000
    assert high.resolution == "720p"

    for duration in (0, -1, 15_001):
        with pytest.raises(ValidationError):
            _valid_request(tmp_path, expected_duration_ms=duration)

    with pytest.raises(ValidationError):
        _valid_request(tmp_path, aspect_ratio="16:9")
    with pytest.raises(ValidationError):
        _valid_request(tmp_path, resolution="1080p")


def test_request_rejects_string_duration_and_list_reference_coercions(tmp_path: Path) -> None:
    api = _load_contracts_api()
    request_cls = api["VideoGenerationRequest"]
    spec = "Adult East Asian reader, short black hair, navy shirt, calm posture"

    with pytest.raises(ValidationError):
        _valid_request(tmp_path, expected_duration_ms="12000")

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            reference_images=(),
            reference_image_specs=[spec],
        )

    episode = _episode_root(tmp_path)
    image = _write_image(episode / "references" / "character.png")
    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=[image],
            reference_image_specs=("Same character opens a notebook at the desk",),
        )

    accepted = _valid_request(
        tmp_path,
        expected_duration_ms=12_000,
        reference_images=(),
        reference_image_specs=(spec,),
    )
    assert accepted.expected_duration_ms == 12_000
    assert accepted.reference_image_specs == (spec,)

    with pytest.raises(ValidationError):
        _valid_request(tmp_path, episode_root=str(episode))

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(image,),
            reference_image_sha256s=[_sha(_TINY_IMAGE)],
            reference_image_specs=("Same character opens a notebook at the desk",),
        )

    s02 = _valid_request(
        tmp_path,
        segment_id="S02",
        reference_images=(image,),
        reference_image_specs=("Same character opens a notebook at the desk",),
    )
    assert s02.reference_images == (image,)
    assert isinstance(s02.reference_images[0], Path)
    assert s02.reference_image_sha256s == (_sha(_TINY_IMAGE),)
    assert s02.episode_root == episode
    assert isinstance(s02.episode_root, Path)
    assert request_cls.model_config.get("strict") is True


# ---------------------------------------------------------------------------
# Reference count, content, duplicates, presence, workspace, symlink
# ---------------------------------------------------------------------------


def test_request_requires_at_least_one_image_or_spec(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _valid_request(tmp_path, reference_images=(), reference_image_specs=())


def test_request_rejects_blank_or_duplicate_reference_specs(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            reference_images=(),
            reference_image_specs=("  ", "valid scene"),
        )

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            reference_images=(),
            reference_image_specs=("same character", "same character"),
        )


def test_request_rejects_more_than_seven_total_references(tmp_path: Path) -> None:
    episode = _episode_root(tmp_path)
    images = tuple(
        _write_image(episode / "references" / f"ref-{index}.png") for index in range(4)
    )
    specs = tuple(f"scene-{index}" for index in range(4))
    assert len(images) + len(specs) == 8

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=images,
            reference_image_specs=specs,
        )


def test_request_rejects_missing_or_duplicate_reference_images(tmp_path: Path) -> None:
    episode = _episode_root(tmp_path)
    missing = episode / "references" / "absent.png"
    present = _write_image(episode / "references" / "character.png")

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(missing,),
            reference_image_specs=("scene continues",),
        )

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(present, present),
            reference_image_specs=("scene continues",),
        )


def test_request_rejects_reference_image_outside_episode_workspace(tmp_path: Path) -> None:
    outside = _write_image(tmp_path / "elsewhere" / "character.png")
    wrong_book = _write_image(
        tmp_path / "workspace" / "books" / "other-book" / "episodes" / _EPISODE_ID / "ref.png"
    )

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(outside,),
            reference_image_specs=("scene continues",),
        )

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(wrong_book,),
            reference_image_specs=("scene continues",),
        )


def test_request_rejects_symlink_reference_images(tmp_path: Path) -> None:
    episode = _episode_root(tmp_path)
    target = _write_image(tmp_path / "real" / "character.png")
    link = episode / "references" / "character-link.png"
    link.parent.mkdir(parents=True, exist_ok=True)
    _make_symlink_or_skip(link, target)

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(link,),
            reference_image_specs=("scene continues",),
        )


def test_request_rejects_relative_or_noncanonical_reference_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _episode_root(tmp_path)
    image = _write_image(episode / "references" / "character.png")
    nested = episode / "references" / "nested"
    nested.mkdir()
    aliased = nested / ".." / "character.png"
    escaped = episode / "references" / ".." / ".." / ".." / ".." / "elsewhere" / "character.png"
    _write_image(tmp_path / "workspace" / "books" / "elsewhere" / "character.png")

    monkeypatch.chdir(image.parent)
    relative = Path(image.name)
    assert relative.is_file()
    assert aliased.is_file()
    assert escaped.is_file()
    assert f"books/{_BOOK_ID}/episodes/{_EPISODE_ID}" in escaped.as_posix()

    for bad in (relative, aliased, escaped):
        with pytest.raises(ValidationError):
            _valid_request(
                tmp_path,
                segment_id="S02",
                reference_images=(bad,),
                reference_image_specs=("scene continues",),
            )


def test_request_rejects_reference_image_when_ancestor_is_marked_redirected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bv.video import contracts as module

    episode = _episode_root(tmp_path)
    image = _write_image(episode / "references" / "character.png")
    original = module._is_redirected
    monkeypatch.setattr(
        module,
        "_is_redirected",
        lambda path: Path(path) == image.parent or original(path),
    )

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(image,),
            reference_image_specs=("scene continues",),
        )


def test_request_rejects_reference_image_reached_through_redirect_ancestor(
    tmp_path: Path,
) -> None:
    episode = _episode_root(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _write_image(outside / "character.png")
    link_dir = episode / "references"
    _make_symlink_or_skip(link_dir, outside, target_is_directory=True)
    image = link_dir / "character.png"
    assert image.is_file()
    assert not image.is_symlink()

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(image,),
            reference_image_specs=("scene continues",),
        )


def test_s02_plus_requires_episode_local_image_and_scene_spec(tmp_path: Path) -> None:
    episode = _episode_root(tmp_path)
    image = _write_image(episode / "references" / "character.png")

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(),
            reference_image_specs=("scene only is not enough after S01",),
        )

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S03",
            reference_images=(image,),
            reference_image_specs=(),
        )


# ---------------------------------------------------------------------------
# Trusted episode root and hash-bound reference identity
# ---------------------------------------------------------------------------


def test_request_requires_canonical_episode_root_matching_book_and_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _load_contracts_api()
    request_cls = api["VideoGenerationRequest"]
    real = _episode_root(tmp_path)

    accepted = _valid_request(tmp_path, episode_root=real)
    assert accepted.episode_root == real
    assert accepted.episode_root.is_dir()
    assert accepted.episode_root.is_absolute()
    assert accepted.episode_root.parts[-4:] == ("books", _BOOK_ID, "episodes", _EPISODE_ID)

    omitted = dict(_request_kwargs(tmp_path, episode_root=real))
    del omitted["episode_root"]
    with pytest.raises(ValidationError):
        request_cls.model_validate(omitted)

    with pytest.raises(ValidationError):
        _valid_request(tmp_path, episode_root=str(real))

    missing = tmp_path / "workspace" / "books" / _BOOK_ID / "episodes" / "E404"
    with pytest.raises(ValidationError):
        _valid_request(tmp_path, episode_id="E404", episode_root=missing)

    as_file = tmp_path / "workspace" / "books" / _BOOK_ID / "episodes" / "Efile"
    as_file.write_bytes(b"not-a-directory")
    with pytest.raises(ValidationError):
        _valid_request(tmp_path, episode_id="Efile", episode_root=as_file)

    monkeypatch.chdir(real.parent)
    relative = Path(real.name)
    assert relative.is_dir()
    with pytest.raises(ValidationError):
        _valid_request(tmp_path, episode_root=relative)

    (real / "references").mkdir(exist_ok=True)
    aliased = real / "references" / ".."
    assert aliased.is_dir()
    with pytest.raises(ValidationError):
        _valid_request(tmp_path, episode_root=aliased)

    with pytest.raises(ValidationError):
        _valid_request(tmp_path, episode_root=real / "references")

    other = _episode_root(tmp_path, book_id="other-book")
    with pytest.raises(ValidationError):
        _valid_request(tmp_path, episode_root=other)


def test_request_rejects_episode_root_with_redirect_in_chain(tmp_path: Path) -> None:
    control = _episode_root(tmp_path)
    accepted = _valid_request(tmp_path, episode_root=control)
    assert accepted.episode_root == control

    real = tmp_path / "real" / "books" / _BOOK_ID / "episodes" / _EPISODE_ID
    real.mkdir(parents=True)
    link_parent = tmp_path / "linked" / "books" / _BOOK_ID / "episodes"
    link_parent.mkdir(parents=True)
    link = link_parent / _EPISODE_ID
    _make_symlink_or_skip(link, real, target_is_directory=True)

    with pytest.raises(ValidationError):
        _valid_request(tmp_path, episode_root=link)

    elsewhere = tmp_path / "elsewhere-workspace"
    elsewhere_episode = elsewhere / "books" / _BOOK_ID / "episodes" / _EPISODE_ID
    elsewhere_episode.mkdir(parents=True)
    alias = tmp_path / "alias-workspace"
    _make_symlink_or_skip(alias, elsewhere, target_is_directory=True)
    aliased_root = alias / "books" / _BOOK_ID / "episodes" / _EPISODE_ID
    assert aliased_root.is_dir()

    with pytest.raises(ValidationError):
        _valid_request(tmp_path, episode_root=aliased_root)


def test_request_rejects_episode_root_when_ancestor_is_marked_redirected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bv.video import contracts as module

    episode = _episode_root(tmp_path)
    accepted = _valid_request(tmp_path, episode_root=episode)
    assert accepted.episode_root == episode

    original = module._is_redirected
    monkeypatch.setattr(
        module,
        "_is_redirected",
        lambda path: Path(path) == episode.parent or original(path),
    )

    with pytest.raises(ValidationError):
        _valid_request(tmp_path, episode_root=episode)


def test_request_rejects_same_shaped_image_outside_supplied_episode_root(
    tmp_path: Path,
) -> None:
    real_root = _episode_root(tmp_path)
    contained = _write_image(real_root / "references" / "character.png")
    digest = _sha(_TINY_IMAGE)

    accepted = _valid_request(
        tmp_path,
        episode_root=real_root,
        segment_id="S02",
        reference_images=(contained,),
        reference_image_sha256s=(digest,),
        reference_image_specs=("scene continues",),
    )
    assert accepted.episode_root == real_root
    assert accepted.reference_images == (contained,)
    assert contained.resolve().is_relative_to(real_root.resolve())

    fake_image = _write_image(
        tmp_path
        / "external"
        / "books"
        / _BOOK_ID
        / "episodes"
        / _EPISODE_ID
        / "references"
        / "character.png"
    )
    assert f"books/{_BOOK_ID}/episodes/{_EPISODE_ID}" in fake_image.as_posix()
    assert not fake_image.resolve().is_relative_to(real_root.resolve())

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            episode_root=real_root,
            segment_id="S02",
            reference_images=(fake_image,),
            reference_image_sha256s=(digest,),
            reference_image_specs=("scene continues",),
        )


def test_request_requires_one_to_one_lowercase_reference_image_hashes(
    tmp_path: Path,
) -> None:
    api = _load_contracts_api()
    request_cls = api["VideoGenerationRequest"]
    episode = _episode_root(tmp_path)
    first_bytes = _TINY_IMAGE
    second_bytes = _TINY_IMAGE + b"\x01"
    first = _write_image(episode / "references" / "first.png", first_bytes)
    second = _write_image(episode / "references" / "second.png", second_bytes)
    first_digest = _sha(first_bytes)
    second_digest = _sha(second_bytes)

    accepted = _valid_request(
        tmp_path,
        segment_id="S02",
        episode_root=episode,
        reference_images=(first, second),
        reference_image_sha256s=(first_digest, second_digest),
        reference_image_specs=("scene continues",),
    )
    assert accepted.reference_image_sha256s == (first_digest, second_digest)

    omitted = dict(
        _request_kwargs(
            tmp_path,
            segment_id="S02",
            episode_root=episode,
            reference_images=(first,),
            reference_image_specs=("scene continues",),
        )
    )
    del omitted["reference_image_sha256s"]
    with pytest.raises(ValidationError):
        request_cls.model_validate(omitted)

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(first,),
            reference_image_sha256s=(),
            reference_image_specs=("scene continues",),
        )

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(first,),
            reference_image_sha256s=(first_digest, second_digest),
            reference_image_specs=("scene continues",),
        )

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(first, second),
            reference_image_sha256s=(second_digest, first_digest),
            reference_image_specs=("scene continues",),
        )

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(first,),
            reference_image_sha256s=("0" * 64,),
            reference_image_specs=("scene continues",),
        )

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(first,),
            reference_image_sha256s=(first_digest.upper(),),
            reference_image_specs=("scene continues",),
        )

    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(first,),
            reference_image_sha256s=("not-a-sha256",),
            reference_image_specs=("scene continues",),
        )

    duplicate = _write_image(episode / "references" / "duplicate.png", first_bytes)
    with pytest.raises(ValidationError):
        _valid_request(
            tmp_path,
            segment_id="S02",
            reference_images=(first, duplicate),
            reference_image_sha256s=(first_digest, first_digest),
            reference_image_specs=("scene continues",),
        )


def test_verify_reference_images_rechecks_paths_and_hashes_before_provider_use(
    tmp_path: Path,
) -> None:
    spec_only = _valid_request(
        tmp_path,
        segment_id="S01",
        reference_images=(),
        reference_image_sha256s=(),
        reference_image_specs=("Adult East Asian reader, calm posture",),
    )
    assert callable(getattr(spec_only, "verify_reference_images", None))
    spec_only.verify_reference_images()

    request = _valid_s02_request(tmp_path)
    verify = getattr(type(request), "verify_reference_images")
    assert list(inspect.signature(verify).parameters) == ["self"]
    request.verify_reference_images()

    image = request.reference_images[0]
    image.write_bytes(b"mutated-after-construction" + b"\x00" * 8)
    with pytest.raises(ValidationError):
        request.verify_reference_images()


def test_verify_reference_images_rejects_path_replaced_after_construction(
    tmp_path: Path,
) -> None:
    request = _valid_s02_request(tmp_path)
    request.verify_reference_images()

    image = request.reference_images[0]
    image.unlink()
    outside = _write_image(tmp_path / "elsewhere" / "character.png")
    _make_symlink_or_skip(image, outside)
    with pytest.raises(ValidationError):
        request.verify_reference_images()


def test_verify_reference_images_wraps_filesystem_race_as_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bv.video.contracts as module

    request = _valid_s02_request(tmp_path)
    monkeypatch.setattr(
        module,
        "sha256_file",
        lambda path: (_ for _ in ()).throw(OSError("simulated read race")),
    )

    with pytest.raises(ValidationError):
        request.verify_reference_images()


# ---------------------------------------------------------------------------
# Result validation
# ---------------------------------------------------------------------------


def test_result_accepts_regular_source_with_matching_hash(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    assert result.provider == "grok_cli"
    assert result.segment_id == "S01"
    assert result.source_path.is_file()
    assert result.source_sha256 == _sha(result.source_path.read_bytes())
    assert result.reference_images == ()
    assert result.reference_image_sha256s == ()
    assert result.model_config.get("frozen") is True


def test_result_defaults_empty_generated_reference_images_when_omitted(
    tmp_path: Path,
) -> None:
    api = _load_contracts_api()
    payload = _result_kwargs(tmp_path)
    assert "reference_images" not in payload
    assert "reference_image_sha256s" not in payload

    result = api["VideoGenerationResult"].model_validate(payload)

    assert result.reference_images == ()
    assert result.reference_image_sha256s == ()


def test_result_requires_one_to_one_lowercase_generated_reference_hashes(
    tmp_path: Path,
) -> None:
    api = _load_contracts_api()
    result_cls = api["VideoGenerationResult"]
    first_bytes = _TINY_IMAGE
    second_bytes = _TINY_IMAGE + b"\x01"
    first = _write_image(tmp_path / "accepted" / "ref-a.png", first_bytes)
    second = _write_image(tmp_path / "accepted" / "ref-b.png", second_bytes)
    first_digest = _sha(first_bytes)
    second_digest = _sha(second_bytes)
    source = _write_video(tmp_path / "accepted" / "S01.mp4")

    accepted = _valid_result(
        tmp_path,
        source_path=source,
        reference_images=(first, second),
        reference_image_sha256s=(first_digest, second_digest),
    )
    assert accepted.reference_images == (first, second)
    assert accepted.reference_image_sha256s == (first_digest, second_digest)
    assert all(path.is_file() for path in accepted.reference_images)

    missing = tmp_path / "accepted" / "absent.png"
    with pytest.raises(ValidationError):
        _valid_result(
            tmp_path,
            source_path=source,
            reference_images=(missing,),
            reference_image_sha256s=(_sha(first_bytes),),
        )

    with pytest.raises(ValidationError):
        _valid_result(
            tmp_path,
            source_path=source,
            reference_images=(first,),
            reference_image_sha256s=(),
        )

    with pytest.raises(ValidationError):
        _valid_result(
            tmp_path,
            source_path=source,
            reference_images=(first, second),
            reference_image_sha256s=(second_digest, first_digest),
        )

    with pytest.raises(ValidationError):
        _valid_result(
            tmp_path,
            source_path=source,
            reference_images=(first,),
            reference_image_sha256s=(first_digest.upper(),),
        )

    with pytest.raises(ValidationError):
        _valid_result(
            tmp_path,
            source_path=source,
            reference_images=(first,),
            reference_image_sha256s=("0" * 64,),
        )

    listed = _result_kwargs(
        tmp_path,
        source_path=source,
        reference_images=[first],
        reference_image_sha256s=[first_digest],
    )
    with pytest.raises(ValidationError):
        result_cls.model_validate(listed)


def test_result_rejects_missing_source_or_hash_mismatch(tmp_path: Path) -> None:
    missing = tmp_path / "imports" / "missing.mp4"
    with pytest.raises(ValidationError):
        _valid_result(tmp_path, source_path=missing, source_sha256="a" * 64)

    source = _write_video(tmp_path / "imports" / "S01.mp4")
    with pytest.raises(ValidationError):
        _valid_result(tmp_path, source_path=source, source_sha256="b" * 64)


def test_result_rejects_string_source_path_coercion(tmp_path: Path) -> None:
    api = _load_contracts_api()
    result_cls = api["VideoGenerationResult"]
    source = _write_video(tmp_path / "imports" / "S01.mp4")
    payload = _result_kwargs(tmp_path, source_path=source)
    payload["source_path"] = str(source)

    with pytest.raises(ValidationError):
        result_cls.model_validate(payload)

    result = _valid_result(tmp_path, source_path=source)
    assert result.source_path == source
    assert isinstance(result.source_path, Path)
    assert result_cls.model_config.get("strict") is True


def test_result_rejects_symlink_source(tmp_path: Path) -> None:
    real = _write_video(tmp_path / "real" / "S01.mp4")
    link = tmp_path / "imports" / "S01-link.mp4"
    link.parent.mkdir(parents=True, exist_ok=True)
    _make_symlink_or_skip(link, real)

    with pytest.raises(ValidationError):
        _valid_result(
            tmp_path,
            source_path=link,
            source_sha256=_sha(real.read_bytes()),
        )


def test_result_validation_streams_source_hash_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video(tmp_path / "imports" / "S01.mp4")
    digest = _sha(_TINY_VIDEO)

    def _forbid_read_bytes(self: Path) -> bytes:
        raise AssertionError("source hashing must stream from disk")

    monkeypatch.setattr(Path, "read_bytes", _forbid_read_bytes)
    result = _valid_result(tmp_path, source_path=source, source_sha256=digest)
    assert result.source_path == source
    assert result.source_sha256 == digest


def test_result_rejects_relative_or_noncanonical_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video(tmp_path / "imports" / "S01.mp4")
    digest = _sha(_TINY_VIDEO)
    nested = tmp_path / "imports" / "nested"
    nested.mkdir()
    aliased = nested / ".." / "S01.mp4"

    monkeypatch.chdir(source.parent)
    relative = Path(source.name)
    assert relative.is_file()
    assert aliased.is_file()

    for bad in (relative, aliased):
        with pytest.raises(ValidationError):
            _valid_result(tmp_path, source_path=bad, source_sha256=digest)


def test_result_rejects_source_when_ancestor_is_marked_redirected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bv.video import contracts as module

    source = _write_video(tmp_path / "imports" / "S01.mp4")
    original = module._is_redirected
    monkeypatch.setattr(
        module,
        "_is_redirected",
        lambda path: Path(path) == source.parent or original(path),
    )

    with pytest.raises(ValidationError):
        _valid_result(
            tmp_path,
            source_path=source,
            source_sha256=_sha(_TINY_VIDEO),
        )


def test_result_rejects_source_reached_through_redirect_ancestor(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real = _write_video(real_dir / "S01.mp4")
    link_dir = tmp_path / "imports"
    _make_symlink_or_skip(link_dir, real_dir, target_is_directory=True)
    source = link_dir / "S01.mp4"
    assert source.is_file()
    assert not source.is_symlink()

    with pytest.raises(ValidationError):
        _valid_result(
            tmp_path,
            source_path=source,
            source_sha256=_sha(real.read_bytes()),
        )


# ---------------------------------------------------------------------------
# Structural VideoGateway protocol
# ---------------------------------------------------------------------------


def test_video_gateway_protocol_exposes_generate_import_and_status() -> None:
    api = _load_contracts_api()
    gateway = api["VideoGateway"]

    for name in ("generate", "import_video", "status"):
        assert hasattr(gateway, name), name
        assert callable(getattr(gateway, name)), name

    generate = inspect.signature(gateway.generate)
    import_video = inspect.signature(gateway.import_video)
    status = inspect.signature(gateway.status)

    assert list(generate.parameters) == ["self", "book_id", "episode_id", "shot_id"]
    assert list(import_video.parameters) == ["self", "book_id", "episode_id", "shot_id", "source"]
    assert list(status.parameters) == ["self", "book_id", "episode_id"]

    hints = get_type_hints(gateway.import_video)
    assert hints.get("source") is Path
