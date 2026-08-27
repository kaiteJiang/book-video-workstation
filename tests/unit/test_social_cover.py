from __future__ import annotations

import hashlib
import json
import shutil
import struct
import zlib
from pathlib import Path

import pytest

from bv.illustration.assets import inspect_image
from bv.core.process import run_command
from bv.video.cover_brief import CoverBriefBuilder, CoverDerivedFields
from bv.video.social_cover import (
    SocialCoverError,
    SocialCoverRenderer,
    SocialCoverRequest,
)
from bv.workflow.media_runtime import LocalSocialCoverGateway
from bv.workflow.stages import StageContext


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_png(path: Path, *, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = b"\x00" + (b"\xe8\xdf\xcc" * width)
    raw = row * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=1))
        + chunk(b"IEND", b"")
    )


def _cover_request(tmp_path: Path, *, width: int = 540, height: int = 720) -> SocialCoverRequest:
    episode = tmp_path / "episode"
    script = episode / "script" / "approved.txt"
    semantic_lock = episode / "script" / "semantic-lock.json"
    script.parent.mkdir(parents=True)
    script.write_text("生活不断失去以后 人仍然要把一天交给下一天", encoding="utf-8")
    semantic_lock.write_text('{"value_thesis":"把一天交给下一天"}', encoding="utf-8")

    style_root = tmp_path / "style"
    style_root.mkdir()
    meta = style_root / "META.md"
    atom = style_root / "STYLE.md"
    blueprint = tmp_path / "cover-prompt-blueprint.md"
    meta.write_text(
        "```yaml\nid: french-minimal-ink-poster\noutputs: [cover, poster]\n```",
        encoding="utf-8",
    )
    atom.write_text("warm ivory paper and sparse black ink", encoding="utf-8")
    blueprint.write_text("single integrated cover prompt", encoding="utf-8")
    prompt = """Create one 3:4 cover using french-minimal-ink-poster.
Main title: 活着. Keep the title large and centered.
Show one old man and one old ox as a restrained metaphor.
"""
    brief = CoverBriefBuilder(blueprint_path=blueprint).build(
        episode_root=episode,
        slug="huo-zhe-e001",
        fields=CoverDerivedFields(
            title="活着",
            summary="失去以后仍然活下去",
            visual_subject="老人和老牛",
            audience="承受现实压力的成年人",
            mood="克制沉静",
            visual_metaphor="老人和老牛并肩站立",
        ),
        compiled_prompt=prompt,
        approved_script_path=script,
        approved_script_sha256=_sha(script),
        semantic_lock_path=semantic_lock,
        semantic_lock_sha256=_sha(semantic_lock),
        production_profile_sha256="a" * 64,
        style_id="french-minimal-ink-poster",
        style_meta_path=meta,
        style_atom_path=atom,
    )
    source = tmp_path / "current-run" / "generated-cover.png"
    _write_png(source, width=width, height=height)
    return SocialCoverRequest(
        episode_root=episode,
        cover_brief_path=brief.manifest_path,
        source_image_path=source,
        source_image_sha256=_sha(source),
    )


def _renderer() -> SocialCoverRenderer:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for the social-cover contract test")
    return SocialCoverRenderer(ffmpeg_command=ffmpeg)


def test_social_cover_is_vertical_and_binds_exact_current_run_artifact(
    tmp_path: Path,
) -> None:
    request = _cover_request(tmp_path)

    manifest = _renderer().normalize(request)

    facts = inspect_image(manifest.output_path)
    assert (facts.width, facts.height, facts.format) == (1080, 1440, "png")
    assert (manifest.width, manifest.height) == (1080, 1440)
    assert manifest.title == "活着"
    assert manifest.style_id == "french-minimal-ink-poster"
    assert manifest.source_image_sha256 == request.source_image_sha256
    assert manifest.prompt_sha256 == _sha(
        request.episode_root
        / "punk-assets/punk-cover/huo-zhe-e001/prompts/cover.md"
    )
    assert manifest.output_sha256 == _sha(manifest.output_path)


def test_social_cover_refuses_existing_output(tmp_path: Path) -> None:
    request = _cover_request(tmp_path)
    output = request.episode_root / "media" / "final" / "social_cover.png"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing")

    with pytest.raises(SocialCoverError, match="cover_output_exists"):
        _renderer().normalize(request)


def test_social_cover_refuses_changed_source_or_stale_prompt(tmp_path: Path) -> None:
    request = _cover_request(tmp_path)
    request.source_image_path.write_bytes(request.source_image_path.read_bytes() + b"changed")
    with pytest.raises(SocialCoverError, match="cover_source_changed"):
        _renderer().normalize(request)

    second = _cover_request(tmp_path / "second")
    brief_prompt = (
        second.episode_root
        / "punk-assets/punk-cover/huo-zhe-e001/prompts/cover.md"
    )
    brief_prompt.write_text("stale prompt", encoding="utf-8")
    with pytest.raises(SocialCoverError, match="cover_prompt_stale"):
        _renderer().normalize(second)


def test_social_cover_refuses_non_three_by_four_source(tmp_path: Path) -> None:
    request = _cover_request(tmp_path, width=600, height=960)

    with pytest.raises(SocialCoverError, match="cover_aspect_ratio_invalid"):
        _renderer().normalize(request)


def test_social_cover_refuses_style_changed_after_prompt_was_compiled(
    tmp_path: Path,
) -> None:
    request = _cover_request(tmp_path)
    (tmp_path / "style" / "STYLE.md").write_text(
        "a different visual language",
        encoding="utf-8",
    )

    with pytest.raises(SocialCoverError, match="cover_dependency_stale"):
        _renderer().normalize(request)


def test_social_cover_refuses_cover_brief_changed_during_render(tmp_path: Path) -> None:
    request = _cover_request(tmp_path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for the social-cover contract test")

    def mutating_runner(argv: list[str], *, timeout: float):
        result = run_command(argv, timeout=timeout)
        request.cover_brief_path.write_text("{}", encoding="utf-8")
        return result

    renderer = SocialCoverRenderer(ffmpeg_command=ffmpeg, runner=mutating_runner)
    with pytest.raises(SocialCoverError, match="cover_brief_changed"):
        renderer.normalize(request)


def test_social_cover_gateway_marks_changed_prompt_stale(tmp_path: Path) -> None:
    request = _cover_request(tmp_path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for the social-cover contract test")
    context = StageContext(
        book_id="book-demo",
        episode_id="E001",
        episode_root=request.episode_root,
    )
    gateway = LocalSocialCoverGateway(ffmpeg_command=ffmpeg)
    gateway.import_current_run(
        context,
        cover_brief_path=request.cover_brief_path,
        source=request.source_image_path,
        source_sha256=request.source_image_sha256,
    )
    assert gateway.ready(context) is True

    prompt = (
        request.episode_root
        / "punk-assets/punk-cover/huo-zhe-e001/prompts/cover.md"
    )
    prompt.write_text("changed after normalization", encoding="utf-8")

    assert gateway.ready(context) is False


def test_social_cover_gateway_rejects_manifest_with_wrong_dimensions(
    tmp_path: Path,
) -> None:
    request = _cover_request(tmp_path)
    context = StageContext(
        book_id="book-demo",
        episode_id="E001",
        episode_root=request.episode_root,
    )
    gateway = LocalSocialCoverGateway(
        ffmpeg_command=shutil.which("ffmpeg") or "ffmpeg"
    )
    manifest = gateway.import_current_run(
        context,
        cover_brief_path=request.cover_brief_path,
        source=request.source_image_path,
        source_sha256=request.source_image_sha256,
    ).outputs["social_cover_manifest"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["height"] = 1920
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert gateway.ready(context) is False
