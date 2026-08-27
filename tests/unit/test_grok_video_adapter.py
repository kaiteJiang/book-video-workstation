from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bv.core.hashing import sha256_file
from bv.core.process import CommandResult

_BOOK_ID = "book-01"
_EPISODE_ID = "E001"
_PROMPT = (
    "Quiet reader sits at a wooden desk, adult East Asian face, no dialogue, no on-screen text."
)
_TINY_IMAGE = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
_S01_SPEC = "Adult East Asian reader, short black hair, navy shirt, calm posture"
_S02_SPEC = "Same character opens a notebook at the desk"


def _sha(data: bytes | str) -> str:
    payload = data if isinstance(data, bytes) else data.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _episode_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace" / "books" / _BOOK_ID / "episodes" / _EPISODE_ID
    root.mkdir(parents=True, exist_ok=True)
    return root


def _valid_request(tmp_path: Path, **overrides: Any) -> Any:
    from bv.video.contracts import VideoGenerationRequest

    episode = overrides.pop("episode_root", None) or _episode_root(tmp_path)
    prompt = overrides.pop("prompt", _PROMPT)
    payload = {
        "book_id": _BOOK_ID,
        "episode_id": _EPISODE_ID,
        "segment_id": "S01",
        "provider": "grok_cli",
        "prompt": prompt,
        "prompt_sha256": _sha(prompt),
        "expected_duration_ms": 12_000,
        "aspect_ratio": "9:16",
        "resolution": "480p",
        "episode_root": episode,
        "reference_images": (),
        "reference_image_sha256s": (),
        "reference_image_specs": (_S01_SPEC,),
    }
    payload.update(overrides)
    return VideoGenerationRequest.model_validate(payload)


def _valid_s02_request(tmp_path: Path) -> Any:
    episode = _episode_root(tmp_path)
    image = episode / "references" / "character.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(_TINY_IMAGE)
    return _valid_request(
        tmp_path,
        segment_id="S02",
        episode_root=episode,
        reference_images=(image,),
        reference_image_sha256s=(_sha(_TINY_IMAGE),),
        reference_image_specs=(_S02_SPEC,),
    )


def _load_video_api() -> tuple[type[Exception], type[Any], Any, Any, Any]:
    """Import production symbols only inside tests so collection stays green."""
    from bv.video.grok_cli import (
        GrokVideoError,
        _GrokVideoResponse,
        build_grok_video_argv,
        map_grok_video_process_result,
        parse_grok_video_response,
    )

    return (
        GrokVideoError,
        _GrokVideoResponse,
        build_grok_video_argv,
        parse_grok_video_response,
        map_grok_video_process_result,
    )


def _result(
    argv: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> CommandResult:
    return CommandResult(argv=argv, returncode=returncode, stdout=stdout, stderr=stderr)


def _payload(
    *,
    segment_id: str = "S01",
    refs: list[str] | None = None,
    video: str | None = None,
) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "reference_image_paths": refs
        if refs is not None
        else [r"E:\req\ref-a.png"],
        "video_path": video if video is not None else r"E:\req\out.mp4",
    }


def test_grok_video_error_exposes_only_stable_error_code() -> None:
    GrokVideoError, *_ = _load_video_api()

    error = GrokVideoError("video_generation_failed")

    assert error.error_code == "video_generation_failed"
    assert str(error) == "video_generation_failed"
    assert "TOP_SECRET" not in str(error)


def test_grok_video_response_is_strict_and_frozen() -> None:
    _, Response, *_ = _load_video_api()
    video = Path(r"E:\req\out.mp4")
    refs = (Path(r"E:\req\ref-a.png"),)

    parsed = Response(
        segment_id="S01",
        reference_image_paths=refs,
        video_path=video,
    )

    assert parsed.segment_id == "S01"
    assert parsed.reference_image_paths == refs
    assert parsed.video_path == video
    with pytest.raises(Exception):
        parsed.segment_id = "S02"  # type: ignore[misc]
    with pytest.raises(Exception):
        Response(
            segment_id="S01",
            reference_image_paths=refs,
            video_path=video,
            extra_field="nope",  # type: ignore[call-arg]
        )


def test_video_argv_restricts_tools_and_cwd(tmp_path: Path) -> None:
    _, _, build_grok_video_argv, *_ = _load_video_api()
    prompt = tmp_path / "prompt.md"
    schema = '{"type":"object"}'

    argv = build_grok_video_argv(prompt, schema, tmp_path)

    assert argv[0] == "grok"
    assert argv[:3] == ["grok", "--prompt-file", str(prompt)]
    assert argv[argv.index("--json-schema") + 1] == schema
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--tools") + 1] == "image_gen,reference_to_video"
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert "--no-memory" in argv
    assert "--no-subagents" in argv
    assert "--disable-web-search" in argv
    assert argv[argv.index("--cwd") + 1] == str(tmp_path)


def test_video_argv_honors_custom_command_without_shell_edit_or_web_grants(
    tmp_path: Path,
) -> None:
    _, _, build_grok_video_argv, *_ = _load_video_api()

    argv = build_grok_video_argv(
        tmp_path / "prompt.md",
        "{}",
        tmp_path,
        command=r"C:\tools\grok.cmd",
    )

    joined = " ".join(argv)
    assert argv[0] == r"C:\tools\grok.cmd"
    assert "--disable-web-search" in argv
    assert "shell" not in joined.casefold()
    assert "--edit" not in argv
    assert "web_search" not in joined
    assert argv[argv.index("--tools") + 1] == "image_gen,reference_to_video"


def test_parse_accepts_direct_json_and_text_envelope() -> None:
    _, Response, _, parse_grok_video_response, _ = _load_video_api()
    body = _payload()

    direct = parse_grok_video_response(json.dumps(body), expected_segment_id="S01")
    envelope = parse_grok_video_response(
        json.dumps({"text": json.dumps(body), "session_id": "private"}),
        expected_segment_id="S01",
    )

    expected = Response(
        segment_id="S01",
        reference_image_paths=(Path(r"E:\req\ref-a.png"),),
        video_path=Path(r"E:\req\out.mp4"),
    )
    assert direct == expected
    assert envelope == expected


@pytest.mark.parametrize(
    "stdout",
    [
        "not json",
        '{"text": {"nested": true}}',
        '{"text": 12}',
        '{"message": "missing text"}',
        '{"segment_id": "S01"}',
        '{"text": "not json either"}',
    ],
)
def test_parse_rejects_malformed_and_missing_fields(stdout: str) -> None:
    GrokVideoError, _, _, parse_grok_video_response, _ = _load_video_api()

    with pytest.raises(GrokVideoError) as raised:
        parse_grok_video_response(stdout, expected_segment_id="S01")

    assert raised.value.error_code == "video_output_invalid"
    assert str(raised.value) == "video_output_invalid"
    assert stdout not in str(raised.value)


def test_parse_rejects_extra_fields_and_wrong_types() -> None:
    GrokVideoError, _, _, parse_grok_video_response, _ = _load_video_api()
    extra = _payload()
    extra["hash"] = "trusted-by-model"
    wrong = _payload()
    wrong["reference_image_paths"] = r"E:\req\ref-a.png"
    wrong["video_path"] = 12

    for stdout in (json.dumps(extra), json.dumps(wrong)):
        with pytest.raises(GrokVideoError) as raised:
            parse_grok_video_response(stdout, expected_segment_id="S01")
        assert str(raised.value) == "video_output_invalid"


def test_parse_rejects_relative_and_traversal_paths() -> None:
    GrokVideoError, _, _, parse_grok_video_response, _ = _load_video_api()
    relative = _payload(video="out.mp4")
    traversal = _payload(refs=[r"E:\req\..\secret.png"])

    for payload in (relative, traversal):
        with pytest.raises(GrokVideoError) as raised:
            parse_grok_video_response(
                json.dumps(payload),
                expected_segment_id="S01",
            )
        assert str(raised.value) == "video_output_invalid"


def test_parse_rejects_segment_mismatch() -> None:
    GrokVideoError, _, _, parse_grok_video_response, _ = _load_video_api()

    with pytest.raises(GrokVideoError) as raised:
        parse_grok_video_response(
            json.dumps(_payload(segment_id="S02")),
            expected_segment_id="S01",
        )

    assert str(raised.value) == "video_output_invalid"


def test_parse_rejects_oversized_payload() -> None:
    GrokVideoError, _, _, parse_grok_video_response, _ = _load_video_api()
    huge = _payload(video=r"E:\req\\" + ("x" * 200_000) + ".mp4")

    with pytest.raises(GrokVideoError) as raised:
        parse_grok_video_response(json.dumps(huge), expected_segment_id="S01")

    assert str(raised.value) == "video_output_invalid"


@pytest.mark.parametrize(
    ("returncode", "stderr", "error_code"),
    [
        (-9, "timed out", "video_generation_timeout"),
        (1, "command timed out after 900 seconds", "video_generation_timeout"),
        (1, "authentication failed", "video_generation_auth_failed"),
        (1, "login required", "video_generation_auth_failed"),
        (1, "Authorization token rejected", "video_generation_auth_failed"),
        (17, "ordinary failure", "video_generation_failed"),
    ],
)
def test_process_mapping_uses_stable_codes_without_diagnostics(
    returncode: int,
    stderr: str,
    error_code: str,
) -> None:
    GrokVideoError, _, _, _, map_grok_video_process_result = _load_video_api()
    secret = "TOP_SECRET_DO_NOT_ECHO"

    with pytest.raises(GrokVideoError) as raised:
        map_grok_video_process_result(
            _result(
                ["grok"],
                returncode=returncode,
                stdout=f"stdout leaks {secret}",
                stderr=stderr,
            )
        )

    assert raised.value.error_code == error_code
    assert str(raised.value) == error_code
    assert secret not in str(raised.value)
    assert "authentication" not in str(raised.value).casefold()
    assert "timed out" not in str(raised.value).casefold()
    assert "ordinary failure" not in str(raised.value)


def test_process_mapping_does_not_treat_author_as_auth_failure() -> None:
    GrokVideoError, _, _, _, map_grok_video_process_result = _load_video_api()

    with pytest.raises(GrokVideoError) as raised:
        map_grok_video_process_result(
            _result(
                ["grok"],
                returncode=1,
                stderr="could not resolve author from authoritative source",
            )
        )

    assert raised.value.error_code == "video_generation_failed"
    assert str(raised.value) == "video_generation_failed"
    assert "author" not in str(raised.value)


def test_process_mapping_returns_normally_for_success() -> None:
    _, _, _, _, map_grok_video_process_result = _load_video_api()

    mapped = map_grok_video_process_result(
        _result(["grok"], returncode=0, stdout=_payload_json())
    )

    assert mapped is None


def _payload_json(*, segment_id: str = "S01", refs: list[str] | None = None, video: str | None = None) -> str:
    return json.dumps(_payload(segment_id=segment_id, refs=refs, video=video))


def _load_provider() -> type[Any]:
    from bv.video.grok_cli import GrokCliVideoProvider

    return GrokCliVideoProvider


def _ffprobe_payload(*, duration_s: str = "12.000") -> dict[str, object]:
    return {
        "format": {"duration": duration_s},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 480,
                "height": 854,
                "r_frame_rate": "30/1",
            }
        ],
    }


_STABLE_VIDEO_CODES = frozenset(
    {
        "video_generation_timeout",
        "video_generation_auth_failed",
        "video_generation_failed",
        "video_output_missing",
        "video_output_unsafe",
        "video_output_invalid",
    }
)
_SECRET = "E:\\secrets\\do-not-leak\\TOP_SECRET.mp4"


def _make_symlink_or_skip(
    link: Path, target: Path, *, target_is_directory: bool = False
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError:
        pytest.skip("symlink creation requires additional privilege")
    if not link.exists() and not link.is_symlink():
        pytest.skip("symlink creation is unavailable on this platform")


def _assert_prompt_constraints(prompt: str, request: Any) -> None:
    assert request.prompt in prompt
    assert request.segment_id in prompt
    assert str(request.expected_duration_ms) in prompt
    assert request.resolution in prompt
    assert "9:16" in prompt
    assert prompt.count("image_gen") == len(request.reference_image_specs)
    assert prompt.count("reference_to_video") == 1
    lowered = prompt.casefold()
    for banned in (
        "dialogue",
        "lip-sync",
        "lip sync",
        "narration",
        "music",
        "subtitle",
        "readable text",
        "logo",
        "watermark",
    ):
        assert banned in lowered


def _write_listed_outputs(
    cwd: Path,
    *,
    ref_names: tuple[str, ...] = ("generated-ref-0.png",),
    video_name: str = "out.mp4",
) -> tuple[list[str], str]:
    refs: list[str] = []
    for index, name in enumerate(ref_names):
        path = cwd / name
        path.write_bytes(b"generated-ref" + bytes([index]) + b"\x00" * 8)
        refs.append(str(path))
    video = cwd / video_name
    video.write_bytes(b"generated-mp4" + b"\x00" * 16)
    return refs, str(video)


def _provider(
    *,
    runner: Any,
    ffprobe_runner: Any,
    command: str = "grok",
    ffprobe_command: str = "ffprobe",
) -> Any:
    return _load_provider()(
        command=command,
        timeout_seconds=900,
        ffprobe_command=ffprobe_command,
        runner=runner,
        ffprobe_runner=ffprobe_runner,
    )


def _ok_ffprobe(
    *,
    width: int = 480,
    height: int = 854,
    duration_s: str = "12.000",
) -> Any:
    payload = _ffprobe_payload(duration_s=duration_s)
    streams = list(payload["streams"])  # type: ignore[arg-type]
    video_stream = dict(streams[0])  # type: ignore[arg-type]
    video_stream["width"] = width
    video_stream["height"] = height
    payload["streams"] = [video_stream]

    def runner(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        return _result(argv, stdout=json.dumps(payload))

    return runner


def _accepted_under(episode_root: Path) -> list[Path]:
    return [
        path
        for path in episode_root.rglob("*")
        if path.is_file() and "accepted" in path.parts
    ]


def _assert_stable_code(error: Exception, *, expected: str | None = None) -> None:
    code = getattr(error, "error_code", None)
    assert code in _STABLE_VIDEO_CODES
    assert str(error) == code
    assert _SECRET not in str(error)
    assert "TOP_SECRET" not in str(error)
    if expected is not None:
        assert code == expected


def test_generate_s01_spec_only_writes_private_request_and_accepted_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    GrokCliVideoProvider = _load_provider()
    request = _valid_request(tmp_path, segment_id="S01")
    grok_calls: list[dict[str, object]] = []
    generated = b"generated-s01-ref" + b"\x00" * 8
    video_bytes = b"generated-s01-mp4" + b"\x00" * 16

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        grok_calls.append({"argv": argv, **kwargs})
        prompt = cwd / "prompt.md"
        assert prompt.is_file()
        _assert_prompt_constraints(prompt.read_text(encoding="utf-8"), request)
        ref = cwd / "generated-ref-0.png"
        video = cwd / "out.mp4"
        ref.write_bytes(generated)
        video.write_bytes(video_bytes)
        return _result(
            argv,
            stdout=_payload_json(
                segment_id="S01",
                refs=[str(ref)],
                video=str(video),
            ),
        )

    def fake_ffprobe(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        assert argv[0] == "ffprobe"
        return _result(argv, stdout=json.dumps(_ffprobe_payload()))

    def _forbid_read_bytes(self: Path) -> bytes:
        raise AssertionError("accepted hashes must stream from disk")

    monkeypatch.setattr(Path, "read_bytes", _forbid_read_bytes)
    provider = GrokCliVideoProvider(
        command="grok",
        timeout_seconds=900,
        ffprobe_command="ffprobe",
        runner=fake_runner,
        ffprobe_runner=fake_ffprobe,
    )
    result = provider.generate(request)

    assert len(grok_calls) == 1
    call = grok_calls[0]
    request_dir = Path(str(call["cwd"]))
    private_root = request.episode_root / ".private" / "grok-video-requests"
    assert request_dir.is_dir()
    assert request_dir.is_relative_to(private_root)
    assert request_dir != private_root
    prompt_path = request_dir / "prompt.md"
    assert prompt_path.is_file()
    assert call["argv"][0] == "grok"
    assert call.get("timeout") == 900 or call.get("timeout_seconds") == 900

    raw_video = request_dir / "out.mp4"
    raw_ref = request_dir / "generated-ref-0.png"
    assert result.provider == "grok_cli"
    assert result.segment_id == "S01"
    assert result.source_path.is_file()
    assert result.source_path.is_relative_to(request.episode_root)
    assert "accepted" in result.source_path.parts
    assert result.source_path != raw_video
    assert result.source_sha256 == sha256_file(result.source_path)
    assert len(result.reference_images) == 1
    accepted_ref = result.reference_images[0]
    assert accepted_ref.is_file()
    assert accepted_ref.is_relative_to(request.episode_root)
    assert "accepted" in accepted_ref.parts
    assert accepted_ref != raw_ref
    assert result.reference_image_sha256s == (sha256_file(accepted_ref),)


def test_generate_s02_snapshots_supplied_image_and_never_names_original(
    tmp_path: Path,
) -> None:
    GrokCliVideoProvider = _load_provider()
    request = _valid_s02_request(tmp_path)
    original = request.reference_images[0]
    grok_calls: list[dict[str, object]] = []
    generated = b"generated-s02-ref" + b"\x01" * 8
    video_bytes = b"generated-s02-mp4" + b"\x01" * 16

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        grok_calls.append({"argv": argv, **kwargs})
        prompt = (cwd / "prompt.md").read_text(encoding="utf-8")
        _assert_prompt_constraints(prompt, request)
        snapshots = [
            path
            for path in cwd.rglob("*")
            if path.is_file() and path.name != "prompt.md"
        ]
        assert snapshots, "supplied images must be snapshotted before the runner"
        assert all(path.is_relative_to(cwd) for path in snapshots)
        assert original.as_posix() not in prompt
        assert str(original) not in prompt
        assert any(path.as_posix() in prompt or path.name in prompt for path in snapshots)
        ref = cwd / "generated-ref-0.png"
        video = cwd / "out.mp4"
        ref.write_bytes(generated)
        video.write_bytes(video_bytes)
        return _result(
            argv,
            stdout=_payload_json(
                segment_id="S02",
                refs=[str(ref)],
                video=str(video),
            ),
        )

    def fake_ffprobe(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        return _result(argv, stdout=json.dumps(_ffprobe_payload()))

    provider = GrokCliVideoProvider(
        command="grok",
        timeout_seconds=900,
        ffprobe_command="ffprobe",
        runner=fake_runner,
        ffprobe_runner=fake_ffprobe,
    )
    result = provider.generate(request)

    request_dir = Path(str(grok_calls[0]["cwd"]))
    assert request_dir.is_relative_to(request.episode_root / ".private" / "grok-video-requests")
    assert result.provider == "grok_cli"
    assert result.segment_id == "S02"
    assert result.source_path.is_file()
    assert result.source_path.is_relative_to(request.episode_root)
    assert result.source_path != request_dir / "out.mp4"
    assert result.reference_images[0] != request_dir / "generated-ref-0.png"
    assert result.reference_images[0].is_relative_to(request.episode_root)
    assert original.read_bytes() == _TINY_IMAGE


def test_prompt_includes_approved_text_and_never_names_original_path(
    tmp_path: Path,
) -> None:
    request = _valid_s02_request(tmp_path)
    original = request.reference_images[0]
    seen: list[str] = []

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        prompt = (cwd / "prompt.md").read_text(encoding="utf-8")
        seen.append(prompt)
        _assert_prompt_constraints(prompt, request)
        assert original.as_posix() not in prompt
        assert str(original) not in prompt
        refs, video = _write_listed_outputs(cwd)
        return _result(argv, stdout=_payload_json(segment_id="S02", refs=refs, video=video))

    result = _provider(runner=fake_runner, ffprobe_runner=_ok_ffprobe()).generate(request)
    assert seen and request.prompt in seen[0]
    assert result.segment_id == "S02"


@pytest.mark.parametrize(
    ("resolution", "width", "height", "duration_s", "accept"),
    [
        ("480p", 480, 854, "12.000", True),
        ("720p", 720, 1280, "12.000", True),
        ("480p", 480, 854, "11.700", True),
        ("480p", 480, 854, "12.050", True),
        ("480p", 480, 854, "11.699", False),
        ("480p", 480, 854, "12.051", False),
        ("480p", 720, 1280, "12.000", False),
        ("720p", 480, 854, "12.000", False),
        ("480p", 640, 480, "12.000", False),
        ("720p", 1280, 720, "12.000", False),
    ],
)
def test_media_acceptance_resolution_aspect_and_duration_window(
    tmp_path: Path,
    resolution: str,
    width: int,
    height: int,
    duration_s: str,
    accept: bool,
) -> None:
    request = _valid_request(tmp_path, resolution=resolution)

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        refs, video = _write_listed_outputs(cwd)
        return _result(argv, stdout=_payload_json(refs=refs, video=video))

    provider = _provider(
        runner=fake_runner,
        ffprobe_runner=_ok_ffprobe(width=width, height=height, duration_s=duration_s),
    )
    if accept:
        result = provider.generate(request)
        assert result.source_path.is_file()
        assert "accepted" in result.source_path.parts
        return

    with pytest.raises(Exception) as raised:
        provider.generate(request)
    _assert_stable_code(raised.value, expected="video_output_invalid")
    assert _accepted_under(request.episode_root) == []


def _adversarial_payload(
    kind: str, cwd: Path, tmp_path: Path, request: Any
) -> tuple[list[str], str]:
    refs, video = _write_listed_outputs(cwd)
    if kind == "outside":
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"outside-bytes")
        return refs, str(outside)
    if kind == "traversal":
        return refs, str((cwd / ".." / ".." / "escaped.mp4").resolve())
    if kind == "missing":
        return [str(cwd / "missing-ref.png")], str(cwd / "missing.mp4")
    if kind == "duplicate":
        return [refs[0], refs[0]], video
    if kind == "returned_prompt":
        return refs, str(cwd / "prompt.md")
    if kind == "returned_input":
        inputs = list((cwd / "inputs").glob("ref-*"))
        return refs, str(inputs[0] if inputs else cwd / "prompt.md")
    if kind == "wrong_image_ext":
        return _write_listed_outputs(cwd, ref_names=("generated-ref-0.txt",))
    if kind == "wrong_video_ext":
        return _write_listed_outputs(cwd, video_name="out.webm")
    if kind == "wrong_ref_count":
        return [], video
    if kind == "symlink_output":
        target = tmp_path / "link-target.mp4"
        target.write_bytes(b"linked-video")
        link = cwd / "linked.mp4"
        _make_symlink_or_skip(link, target)
        return refs, str(link)
    if kind == "oversized":
        Path(video).write_bytes(b"x" * 64)
        return refs, video
    if kind == "unlisted":
        (cwd / "unlisted.bin").write_bytes(b"model-side-channel")
        return refs, video
    raise AssertionError(kind)


@pytest.mark.parametrize(
    ("kind", "error_code"),
    [
        ("outside", "video_output_unsafe"),
        ("traversal", "video_output_unsafe"),
        ("missing", "video_output_missing"),
        ("duplicate", "video_output_unsafe"),
        ("returned_prompt", "video_output_unsafe"),
        ("returned_input", "video_output_unsafe"),
        ("wrong_image_ext", "video_output_invalid"),
        ("wrong_video_ext", "video_output_invalid"),
        ("wrong_ref_count", "video_output_invalid"),
        ("symlink_output", "video_output_unsafe"),
        ("oversized", "video_output_unsafe"),
        ("unlisted", "video_output_unsafe"),
    ],
)
def test_reject_adversarial_outputs_before_accepted_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    error_code: str,
) -> None:
    request = _valid_s02_request(tmp_path)
    grok_cli = __import__("bv.video.grok_cli", fromlist=["_MAX_OUTPUT_BYTES"])
    monkeypatch.setattr(grok_cli, "_MAX_OUTPUT_BYTES", 8, raising=False)

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        refs, video = _adversarial_payload(kind, cwd, tmp_path, request)
        return _result(
            argv,
            stdout=_payload_json(segment_id="S02", refs=refs, video=video),
        )

    with pytest.raises(Exception) as raised:
        _provider(runner=fake_runner, ffprobe_runner=_ok_ffprobe()).generate(request)

    _assert_stable_code(raised.value, expected=error_code)
    assert _accepted_under(request.episode_root) == []


def test_changed_request_reference_rejects_before_grok_runner(tmp_path: Path) -> None:
    request = _valid_s02_request(tmp_path)
    request.reference_images[0].write_bytes(b"mutated-after-validation")
    ran: list[int] = []

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        ran.append(1)
        raise AssertionError("Grok runner must not start after a mutated reference")

    with pytest.raises(Exception) as raised:
        _provider(runner=fake_runner, ffprobe_runner=_ok_ffprobe()).generate(request)

    _assert_stable_code(raised.value)
    assert ran == []


def test_output_video_change_during_ffprobe_is_not_published(tmp_path: Path) -> None:
    request = _valid_request(tmp_path)
    mutated = b"mutated-during-ffprobe" + b"\xff" * 8

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        refs, video = _write_listed_outputs(cwd)
        fake_runner.video = Path(video)  # type: ignore[attr-defined]
        return _result(argv, stdout=_payload_json(refs=refs, video=video))

    def mutating_ffprobe(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        Path(argv[-1]).write_bytes(mutated)
        return _ok_ffprobe()(argv)

    with pytest.raises(Exception) as raised:
        _provider(runner=fake_runner, ffprobe_runner=mutating_ffprobe).generate(request)

    _assert_stable_code(raised.value)
    for path in _accepted_under(request.episode_root):
        assert path.read_bytes() != mutated


@pytest.mark.parametrize(
    "mode",
    [
        "runner_exception",
        "runner_non_result",
        "ffprobe_malformed",
        "ffprobe_failed",
        "bad_aspect",
        "bad_duration",
    ],
)
def test_failures_map_to_stable_codes_without_leaking_paths(
    tmp_path: Path, mode: str
) -> None:
    request = _valid_request(tmp_path)

    def fake_runner(argv: list[str], **kwargs: object) -> object:
        if mode == "runner_exception":
            raise RuntimeError(f"grok exploded at {_SECRET}")
        if mode == "runner_non_result":
            return {"stdout": _SECRET}
        cwd = Path(str(kwargs["cwd"]))
        refs, video = _write_listed_outputs(cwd)
        return _result(argv, stdout=_payload_json(refs=refs, video=video))

    def fake_ffprobe(argv: list[str], **kwargs: object) -> CommandResult:
        del kwargs
        if mode == "ffprobe_malformed":
            return _result(argv, stdout=f"not-json {_SECRET}")
        if mode == "ffprobe_failed":
            return _result(argv, returncode=1, stderr=f"probe failed {_SECRET}")
        if mode == "bad_aspect":
            return _ok_ffprobe(width=1920, height=1080)(argv)
        if mode == "bad_duration":
            return _ok_ffprobe(duration_s="30.000")(argv)
        return _ok_ffprobe()(argv)

    with pytest.raises(Exception) as raised:
        _provider(runner=fake_runner, ffprobe_runner=fake_ffprobe).generate(request)

    _assert_stable_code(raised.value)
    assert _SECRET not in str(raised.value)
    assert "exploded" not in str(raised.value)
    assert "not-json" not in str(raised.value)
    if mode in {"bad_aspect", "bad_duration", "ffprobe_malformed", "ffprobe_failed"}:
        assert raised.value.error_code == "video_output_invalid"
    if mode in {"runner_exception", "runner_non_result"}:
        assert raised.value.error_code == "video_generation_failed"


def _paths_with_accepted_component(episode_root: Path) -> list[Path]:
    return [path for path in episode_root.rglob("*") if "accepted" in path.parts]


def test_mutated_request_local_input_copy_rejects_without_accepted(
    tmp_path: Path,
) -> None:
    request = _valid_s02_request(tmp_path)

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        copies = list((cwd / "inputs").glob("ref-*"))
        assert copies, "request-local input copies must exist before the runner"
        copies[0].write_bytes(b"mutated-request-local-input" + b"\xaa" * 8)
        refs, video = _write_listed_outputs(cwd)
        return _result(
            argv,
            stdout=_payload_json(segment_id="S02", refs=refs, video=video),
        )

    with pytest.raises(Exception) as raised:
        _provider(runner=fake_runner, ffprobe_runner=_ok_ffprobe()).generate(request)

    _assert_stable_code(raised.value)
    assert _accepted_under(request.episode_root) == []


def test_stream_copy_failure_after_accepted_ref_leaves_no_accepted_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _valid_request(tmp_path, segment_id="S01")
    grok_cli = __import__("bv.video.grok_cli", fromlist=["_stream_copy"])
    original = grok_cli._stream_copy

    def failing_stream_copy(source: Path, destination: Path) -> None:
        parent = destination.parent
        if (
            destination.name == "video.mp4"
            and parent.name.startswith("publish-")
            and "accepted" not in destination.parts
            and parent.parent.is_dir()
            and destination.is_relative_to(parent.parent)
        ):
            raise OSError("simulated staging video copy failure")
        original(source, destination)

    monkeypatch.setattr(grok_cli, "_stream_copy", failing_stream_copy)

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        refs, video = _write_listed_outputs(cwd)
        return _result(argv, stdout=_payload_json(refs=refs, video=video))

    with pytest.raises(Exception) as raised:
        _provider(runner=fake_runner, ffprobe_runner=_ok_ffprobe()).generate(request)

    _assert_stable_code(raised.value)
    assert _paths_with_accepted_component(request.episode_root) == []


def test_duplicate_generated_reference_bytes_reject_before_accepted(
    tmp_path: Path,
) -> None:
    request = _valid_request(
        tmp_path,
        segment_id="S01",
        reference_image_specs=(_S01_SPEC, _S02_SPEC),
    )
    identical = b"same-generated-ref" + b"\x00" * 8

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        first = cwd / "generated-ref-0.png"
        second = cwd / "generated-ref-1.png"
        first.write_bytes(identical)
        second.write_bytes(identical)
        video = cwd / "out.mp4"
        video.write_bytes(b"generated-mp4" + b"\x00" * 16)
        return _result(
            argv,
            stdout=_payload_json(
                refs=[str(first), str(second)],
                video=str(video),
            ),
        )

    with pytest.raises(Exception) as raised:
        _provider(runner=fake_runner, ffprobe_runner=_ok_ffprobe()).generate(request)

    _assert_stable_code(raised.value)
    assert _accepted_under(request.episode_root) == []
    assert _paths_with_accepted_component(request.episode_root) == []


def test_token_budget_is_not_auth_failure_while_authorization_token_is() -> None:
    GrokVideoError, _, _, _, map_grok_video_process_result = _load_video_api()

    with pytest.raises(GrokVideoError) as budget:
        map_grok_video_process_result(
            _result(
                ["grok"],
                returncode=1,
                stderr="token budget exceeded for author analysis",
            )
        )
    assert budget.value.error_code == "video_generation_failed"
    assert str(budget.value) == "video_generation_failed"

    with pytest.raises(GrokVideoError) as auth:
        map_grok_video_process_result(
            _result(
                ["grok"],
                returncode=1,
                stderr="authorization token rejected",
            )
        )
    assert auth.value.error_code == "video_generation_auth_failed"
    assert str(auth.value) == "video_generation_auth_failed"


def test_accepted_bytes_come_from_pre_probe_snapshot_not_mutated_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _valid_request(tmp_path)
    grok_cli = __import__("bv.video.grok_cli", fromlist=["_require_probe"])
    original_probe = grok_cli._require_probe
    snap_ref = b"pre-probe-ref" + b"\x11" * 8
    snap_video = b"pre-probe-vid" + b"\x22" * 16
    mutated_ref = b"post-probe-ref" + b"\xee" * 8
    mutated_video = b"post-probe-vid" + b"\xff" * 16
    raw: dict[str, Path] = {}

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        ref = cwd / "generated-ref-0.png"
        video = cwd / "out.mp4"
        ref.write_bytes(snap_ref)
        video.write_bytes(snap_video)
        raw["ref"] = ref
        raw["video"] = video
        return _result(argv, stdout=_payload_json(refs=[str(ref)], video=str(video)))

    def mutating_probe(*args: object, **kwargs: object) -> None:
        original_probe(*args, **kwargs)
        raw["ref"].write_bytes(mutated_ref)
        raw["video"].write_bytes(mutated_video)

    monkeypatch.setattr(grok_cli, "_require_probe", mutating_probe)
    result = _provider(runner=fake_runner, ffprobe_runner=_ok_ffprobe()).generate(request)

    assert result.source_path.read_bytes() == snap_video
    assert result.reference_images[0].read_bytes() == snap_ref
    assert result.source_path.read_bytes() != mutated_video
    assert result.reference_images[0].read_bytes() != mutated_ref


def test_prompt_md_mutation_during_runner_rejects_without_accepted(
    tmp_path: Path,
) -> None:
    request = _valid_request(tmp_path)

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        prompt = cwd / "prompt.md"
        prompt.write_text(prompt.read_text(encoding="utf-8") + "\ninjected", encoding="utf-8")
        refs, video = _write_listed_outputs(cwd)
        return _result(argv, stdout=_payload_json(refs=refs, video=video))

    with pytest.raises(Exception) as raised:
        _provider(runner=fake_runner, ffprobe_runner=_ok_ffprobe()).generate(request)

    _assert_stable_code(raised.value)
    assert _accepted_under(request.episode_root) == []


def test_runner_receives_approved_prompt_as_redaction_secret(tmp_path: Path) -> None:
    request = _valid_request(tmp_path)
    seen: list[object] = []

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        seen.append(kwargs.get("secrets"))
        cwd = Path(str(kwargs["cwd"]))
        refs, video = _write_listed_outputs(cwd)
        return _result(argv, stdout=_payload_json(refs=refs, video=video))

    result = _provider(runner=fake_runner, ffprobe_runner=_ok_ffprobe()).generate(request)

    assert seen == [(request.prompt,)]
    assert result.segment_id == "S01"


def test_s02_prompt_names_copied_refs_as_identity_anchors_not_episode_path(
    tmp_path: Path,
) -> None:
    request = _valid_s02_request(tmp_path)
    original = request.reference_images[0]
    seen: list[str] = []

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        prompt = (cwd / "prompt.md").read_text(encoding="utf-8")
        seen.append(prompt)
        refs, video = _write_listed_outputs(cwd)
        return _result(
            argv,
            stdout=_payload_json(segment_id="S02", refs=refs, video=video),
        )

    result = _provider(runner=fake_runner, ffprobe_runner=_ok_ffprobe()).generate(request)
    assert result.segment_id == "S02"
    prompt = seen[0]
    _assert_prompt_constraints(prompt, request)
    assert original.as_posix() not in prompt
    assert str(original) not in prompt
    lowered = prompt.casefold()
    assert "identity" in lowered
    assert "character continuity" in lowered or "character-continuity" in lowered
    assert "anchor" in lowered
    assert prompt.count("image_gen") == 1
    assert prompt.count("reference_to_video") == 1


@pytest.mark.parametrize(
    ("stderr", "error_code"),
    [
        ("not authenticated", "video_generation_auth_failed"),
        ("unauthenticated", "video_generation_auth_failed"),
        ("unauthorized", "video_generation_auth_failed"),
        ("not logged in", "video_generation_auth_failed"),
        ("credentials expired", "video_generation_auth_failed"),
        ("session expired", "video_generation_auth_failed"),
        ("token budget exceeded", "video_generation_failed"),
    ],
)
def test_process_mapping_auth_phrases_and_token_budget(
    stderr: str,
    error_code: str,
) -> None:
    GrokVideoError, _, _, _, map_grok_video_process_result = _load_video_api()

    with pytest.raises(GrokVideoError) as raised:
        map_grok_video_process_result(
            _result(["grok"], returncode=1, stderr=stderr)
        )

    assert raised.value.error_code == error_code
    assert str(raised.value) == error_code
    assert stderr not in str(raised.value)


@pytest.mark.parametrize(
    ("video_size", "accept"),
    [
        (24, True),
        (25, False),
    ],
)
def test_output_size_boundary_without_allocating_max_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_size: int,
    accept: bool,
) -> None:
    request = _valid_request(tmp_path)
    grok_cli = __import__("bv.video.grok_cli", fromlist=["_MAX_OUTPUT_BYTES"])
    limit = 24
    monkeypatch.setattr(grok_cli, "_MAX_OUTPUT_BYTES", limit, raising=False)

    def fake_runner(argv: list[str], **kwargs: object) -> CommandResult:
        cwd = Path(str(kwargs["cwd"]))
        ref = cwd / "generated-ref-0.png"
        video = cwd / "out.mp4"
        ref.write_bytes(b"R" * (limit - 8))
        video.write_bytes(b"V" * video_size)
        return _result(argv, stdout=_payload_json(refs=[str(ref)], video=str(video)))

    provider = _provider(runner=fake_runner, ffprobe_runner=_ok_ffprobe())
    if accept:
        result = provider.generate(request)
        assert result.source_path.stat().st_size == limit
        assert "accepted" in result.source_path.parts
        return

    with pytest.raises(Exception) as raised:
        provider.generate(request)
    _assert_stable_code(raised.value, expected="video_output_unsafe")
    assert _accepted_under(request.episode_root) == []
