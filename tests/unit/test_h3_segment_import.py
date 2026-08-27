from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from bv.core.process import CommandResult
from bv.storyboard.generate import H3PromptArtifact
from bv.video.import_h3 import (
    H3ImportError,
    all_required_segments_present,
    import_h3_segment,
    inspect_h3_segments,
)
from bv.video.probe import ProbeError, probe_media


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifacts(count: int = 4, *, duration_ms: int = 15_000) -> tuple[H3PromptArtifact, ...]:
    return tuple(
        H3PromptArtifact(
            segment_id=f"S{index:02d}",
            shot_id=f"S{index:02d}",
            required_duration_ms=duration_ms,
            prompt=f"prompt-{index}",
            prompt_sha256=_sha(f"prompt-{index}"),
            storyboard_sha256="a" * 64,
            semantic_lock_sha256="b" * 64,
            approved_script_sha256="c" * 64,
            audio_sha256="d" * 64,
            subtitle_sha256="e" * 64,
        )
        for index in range(1, count + 1)
    )


def _probe_payload(
    *, duration: str = "15.000", width: int = 1080, height: int = 1920,
    fps: str = "30/1", pixel_format: str = "yuv420p", audio: bool = False,
) -> dict[str, object]:
    streams: list[dict[str, object]] = [{
        "codec_type": "video", "codec_name": "vp9", "width": width,
        "height": height, "r_frame_rate": fps, "pix_fmt": pixel_format,
    }]
    if audio:
        streams.append({"codec_type": "audio", "codec_name": "opus"})
    return {"format": {"duration": duration}, "streams": streams}


def _runner(payload: object, *, returncode: int = 0):
    def run(argv: list[str], *, timeout_seconds: float, output_limit: int) -> CommandResult:
        assert argv[:3] == ["ffprobe", "-v", "error"]
        assert timeout_seconds > 0 and output_limit > 0
        return CommandResult(argv=["ffprobe"], returncode=returncode, stdout=json.dumps(payload))
    return run


def _source(tmp_path: Path, name: str = "download.mp4") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_bytes(b"temporary H3 fixture bytes")
    return path


def _import(
    tmp_path: Path, segment_id: str = "S01", *, artifacts: tuple[H3PromptArtifact, ...] | None = None,
    source: Path | None = None, payload: object | None = None,
):
    current = artifacts or _artifacts()
    original = source or _source(tmp_path)
    return import_h3_segment(
        book_id="book-01", episode_id="E001", segment_id=segment_id,
        prompt_artifacts=current, source_path=original,
        episode_media_root=tmp_path / "book" / "episodes" / "E001" / "media",
        ffprobe_runner=_runner(payload or _probe_payload()),
    )


def _episode_media_root(tmp_path: Path) -> Path:
    return tmp_path / "book" / "episodes" / "E001" / "media"


def _import_video(
    tmp_path: Path,
    segment_id: str = "S01",
    *,
    provider: str,
    artifacts: tuple[H3PromptArtifact, ...] | None = None,
    source: Path | None = None,
    payload: object | None = None,
):
    from bv.video.import_h3 import import_video_segment

    current = artifacts or _artifacts()
    original = source or _source(tmp_path)
    return import_video_segment(
        book_id="book-01",
        episode_id="E001",
        segment_id=segment_id,
        prompt_artifacts=current,
        source_path=original,
        episode_media_root=_episode_media_root(tmp_path),
        provider=provider,
        ffprobe_runner=_runner(payload or _probe_payload()),
    )


def _rewrite_legacy_h3_manifests(h3_root: Path, *, aggregate_status: str) -> None:
    """Drop provider and force a legacy aggregate status, keeping manifest hashes consistent."""
    for entry in h3_root.iterdir():
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not manifest_path.is_file():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload.pop("provider", None)
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    aggregate_path = h3_root / "manifest.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate.pop("provider", None)
    aggregate["status"] = aggregate_status
    rebuilt: list[dict[str, object]] = []
    for item in aggregate["imported_segments"]:
        digest = hashlib.sha256((h3_root / item["segment_id"] / "manifest.json").read_bytes()).hexdigest()
        rebuilt.append({**item, "manifest_sha256": digest})
    aggregate["imported_segments"] = rebuilt
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def test_probe_media_accepts_optional_audio_and_reports_only_decodable_facts(tmp_path: Path) -> None:
    facts = probe_media(_source(tmp_path), runner=_runner(_probe_payload(audio=True)))

    assert facts.duration_ms == 15_000
    assert (facts.width, facts.height, facts.frame_rate, facts.video_codec) == (1080, 1920, 30.0, "vp9")
    assert facts.pixel_format == "yuv420p"
    assert facts.audio_present is True and facts.audio_codec == "opus"


def test_probe_media_accepts_path_typed_ffprobe_command(tmp_path: Path) -> None:
    captured: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: object) -> CommandResult:
        captured.append(argv)
        return CommandResult(
            argv=argv,
            returncode=0,
            stdout=json.dumps(_probe_payload(audio=False)),
        )

    facts = probe_media(
        _source(tmp_path),
        runner=runner,
        ffprobe_command=Path("tools") / "ffprobe.exe",
    )

    assert facts.audio_present is False
    assert captured[0][0] == str(Path("tools") / "ffprobe.exe")


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (_probe_payload(duration="NaN"), "invalid_probe_facts"),
        (_probe_payload(duration="0"), "invalid_probe_facts"),
        (_probe_payload(fps="0/0"), "invalid_probe_facts"),
        (_probe_payload(width=0), "invalid_probe_facts"),
        ({"format": {"duration": "15"}, "streams": [{"codec_type": "audio", "codec_name": "aac"}]}, "no_decodable_video"),
        ("not json", "invalid_probe_output"),
    ],
)
def test_probe_media_fails_closed_for_missing_or_nonfinite_required_facts(
    tmp_path: Path, payload: object, code: str,
) -> None:
    with pytest.raises(ProbeError, match=code) as raised:
        probe_media(_source(tmp_path), runner=_runner(payload))

    assert str(raised.value) == code


def test_probe_media_rejects_bomb_sized_json_without_echoing_private_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bv.video.probe as module

    monkeypatch.setattr(module, "_MAX_PROBE_OUTPUT_BYTES", 32)
    private = "private-probe-data" * 10
    with pytest.raises(ProbeError, match="probe_output_too_large") as raised:
        probe_media(_source(tmp_path), runner=_runner({"private": private}))

    assert private not in str(raised.value)


def test_s01_copy_preserves_source_and_is_only_visual_sample_ready(tmp_path: Path) -> None:
    source = _source(tmp_path)
    before = source.read_bytes()

    result = _import(tmp_path, source=source)

    assert source.read_bytes() == before
    assert result.status == "visual_sample_ready"
    assert result.present_segment_ids == ("S01",)
    assert result.missing_segment_ids == ("S02", "S03", "S04")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["segment_id"] == "S01"
    assert "source_path" not in manifest
    assert manifest["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert (tmp_path / "book" / "episodes" / "E001" / "media" / "h3" / "S01" / f"original-{manifest['copied_sha256']}.mp4").is_file()


@pytest.mark.parametrize("count", [4, 5])
def test_exact_current_segment_set_transitions_to_video_imported(tmp_path: Path, count: int) -> None:
    artifacts = _artifacts(count, duration_ms=12_000 if count == 5 else 15_000)
    result = None
    for artifact in artifacts:
        result = _import(
            tmp_path, artifact.segment_id, artifacts=artifacts, source=_source(tmp_path, f"{artifact.segment_id}.mp4"),
            payload=_probe_payload(duration=f"{artifact.required_duration_ms / 1_000:.3f}"),
        )

    assert result is not None and result.status == "video_imported"
    assert result.present_segment_ids == tuple(f"S{index:02d}" for index in range(1, count + 1))
    aggregate = json.loads(result.aggregate_manifest_path.read_text(encoding="utf-8"))
    assert aggregate["status"] == "video_imported"
    assert [item["segment_id"] for item in aggregate["required_segments"]] == list(result.present_segment_ids)


def test_public_inspection_revalidates_current_manifests_without_importing(tmp_path: Path) -> None:
    artifacts = _artifacts()
    root = tmp_path / "book" / "episodes" / "E001" / "media"

    empty = inspect_h3_segments(
        book_id="book-01", episode_id="E001", prompt_artifacts=artifacts,
        episode_media_root=root,
    )
    assert empty.status == "awaiting_video_generation"
    assert empty.present_segment_ids == ()
    assert empty.missing_segment_ids == ("S01", "S02", "S03", "S04")
    assert all_required_segments_present(
        book_id="book-01", episode_id="E001", prompt_artifacts=artifacts,
        episode_media_root=root,
    ) is False

    for artifact in artifacts:
        _import(tmp_path, artifact.segment_id, artifacts=artifacts, source=_source(tmp_path, f"{artifact.segment_id}.mp4"))

    complete = inspect_h3_segments(
        book_id="book-01", episode_id="E001", prompt_artifacts=artifacts,
        episode_media_root=root,
    )
    assert complete.status == "video_imported"
    assert complete.present_segment_ids == ("S01", "S02", "S03", "S04")
    assert complete.missing_segment_ids == ()
    assert all_required_segments_present(
        book_id="book-01", episode_id="E001", prompt_artifacts=artifacts,
        episode_media_root=root,
    ) is True

    stale = list(artifacts)
    stale[0] = stale[0].model_copy(update={"prompt": "changed", "prompt_sha256": _sha("changed")})
    with pytest.raises(H3ImportError, match="existing_segment_integrity_invalid"):
        inspect_h3_segments(
            book_id="book-01", episode_id="E001", prompt_artifacts=tuple(stale),
            episode_media_root=root,
        )


def test_partial_set_after_s01_is_video_partial_and_lists_exact_present_missing_ids(tmp_path: Path) -> None:
    artifacts = _artifacts()
    _import(tmp_path, "S01", artifacts=artifacts, source=_source(tmp_path, "S01.mp4"))
    result = _import(tmp_path, "S02", artifacts=artifacts, source=_source(tmp_path, "S02.mp4"))

    assert result.status == "video_partial"
    assert result.present_segment_ids == ("S01", "S02")
    assert result.missing_segment_ids == ("S03", "S04")
    aggregate = json.loads(result.aggregate_manifest_path.read_text(encoding="utf-8"))
    assert aggregate["status"] == "video_partial"


@pytest.mark.parametrize("segment_id", ["S00", "S06", "S01-B01", "../S01", "S1"])
def test_import_rejects_unknown_beat_and_unsafe_segment_ids_before_copy(tmp_path: Path, segment_id: str) -> None:
    source = _source(tmp_path)

    with pytest.raises(H3ImportError, match="invalid_segment_id"):
        _import(tmp_path, segment_id, source=source)

    assert not (tmp_path / "book").exists()
    assert source.is_file()


def test_import_requires_exact_ordered_task18_prompt_set_not_existing_files(tmp_path: Path) -> None:
    artifacts = list(_artifacts())
    artifacts[1] = artifacts[1].model_copy(update={"segment_id": "S03", "shot_id": "S03"})

    with pytest.raises(H3ImportError, match="invalid_prompt_artifacts"):
        _import(tmp_path, artifacts=tuple(artifacts))


@pytest.mark.parametrize(
    "artifacts",
    [
        lambda: tuple(item.model_copy(update={"prompt_sha256": "0" * 64}) if item.segment_id == "S01" else item for item in _artifacts()),
        lambda: tuple(item.model_copy(update={"storyboard_sha256": "f" * 64}) if item.segment_id == "S02" else item for item in _artifacts()),
        lambda: _artifacts(4, duration_ms=10_000),
        lambda: _artifacts(4, duration_ms=15_001),
    ],
)
def test_import_requires_current_prompt_content_shared_identities_and_task18_duration_bounds(
    tmp_path: Path, artifacts,
) -> None:
    with pytest.raises(H3ImportError, match="invalid_prompt_artifacts"):
        _import(tmp_path, artifacts=artifacts())


@pytest.mark.parametrize("source_name", ["download.mov", "download.avi"])
def test_import_rejects_non_mp4_source_before_snapshot_or_probe(tmp_path: Path, source_name: str) -> None:
    source = _source(tmp_path, source_name)
    with pytest.raises(H3ImportError, match="unsafe_h3_path"):
        _import(tmp_path, source=source)
    assert not (tmp_path / "book").exists()


def test_import_rejects_oversized_source_before_snapshot_or_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bv.video.import_h3 as module

    source = _source(tmp_path)
    source.write_bytes(b"x" * 33)
    monkeypatch.setattr(module, "_MAX_H3_SOURCE_BYTES", 32)
    with pytest.raises(H3ImportError, match="h3_source_too_large"):
        _import(tmp_path, source=source)
    assert not (tmp_path / "book").exists()


@pytest.mark.parametrize("field", ["book_id", "episode_id"])
def test_import_rejects_unsafe_book_and_episode_identifiers_before_copy(tmp_path: Path, field: str) -> None:
    values = {"book_id": "book-01", "episode_id": "E001"}
    values[field] = "../unsafe"
    with pytest.raises(H3ImportError, match="unsafe_identifier"):
        import_h3_segment(
            **values, segment_id="S01", prompt_artifacts=_artifacts(), source_path=_source(tmp_path),
            episode_media_root=tmp_path / "book" / "episodes" / "E001" / "media",
            ffprobe_runner=_runner(_probe_payload()),
        )


def test_manifest_failure_rolls_back_only_new_h3_media_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bv.video.import_h3 as module

    source = _source(tmp_path)
    original = source.read_bytes()
    monkeypatch.setattr(module, "_write_new_manifest", lambda path, manifest: (_ for _ in ()).throw(H3ImportError("manifest_write_failed")))

    with pytest.raises(H3ImportError, match="manifest_write_failed"):
        _import(tmp_path, source=source)

    segment_root = tmp_path / "book" / "episodes" / "E001" / "media" / "h3" / "S01"
    assert source.read_bytes() == original
    assert not (segment_root / "manifest.json").exists()
    assert not list(segment_root.glob("original-*.mp4"))


def test_aggregate_failure_rolls_back_new_segment_and_preserves_previous_valid_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bv.video.import_h3 as module

    artifacts = _artifacts()
    first = _import(tmp_path, "S01", artifacts=artifacts, source=_source(tmp_path, "S01.mp4"))
    original_aggregate = first.aggregate_manifest_path.read_bytes()
    source = _source(tmp_path, "S02.mp4")
    monkeypatch.setattr(
        module,
        "_write_or_replace_aggregate",
        lambda path, manifest: (_ for _ in ()).throw(H3ImportError("aggregate_write_failed")),
    )

    with pytest.raises(H3ImportError, match="aggregate_write_failed"):
        _import(tmp_path, "S02", artifacts=artifacts, source=source)

    h3_root = first.aggregate_manifest_path.parent
    assert first.aggregate_manifest_path.read_bytes() == original_aggregate
    assert (h3_root / "S01" / "manifest.json").is_file()
    assert not (h3_root / "S02").exists()
    assert source.is_file()


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (_probe_payload(width=1920, height=1080), "invalid_video_aspect"),
        (_probe_payload(duration="15.051"), "segment_too_long"),
        (_probe_payload(duration="14.699"), "segment_too_short"),
    ],
)
def test_import_rejects_wrong_aspect_and_duration_outside_contract(
    tmp_path: Path, payload: object, code: str,
) -> None:
    source = _source(tmp_path)

    with pytest.raises(H3ImportError, match=code):
        _import(tmp_path, source=source, payload=payload)

    assert source.is_file()
    assert not (tmp_path / "book").exists()


def test_import_records_but_does_not_repair_0_to_300ms_shortfall(tmp_path: Path) -> None:
    result = _import(tmp_path, payload=_probe_payload(duration="14.700"))

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["probe"]["duration_ms"] == 14_700
    assert manifest["warning_codes"] == ["duration_shortfall_repair_required"]


def test_identical_retry_is_idempotent_but_conflicting_second_import_fails_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    first = _import(tmp_path, source=source)
    retry = _import(tmp_path, source=source)

    assert retry.idempotent is True and retry.manifest_path == first.manifest_path
    different = _source(tmp_path, "different.mp4")
    different.write_bytes(b"different bytes")
    with pytest.raises(H3ImportError, match="existing_segment_mismatch"):
        _import(tmp_path, source=different)


def test_import_fails_closed_for_stale_or_corrupt_existing_manifests(tmp_path: Path) -> None:
    result = _import(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_sha256"] = "0" * 64
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(H3ImportError, match="existing_segment_integrity_invalid"):
        _import(tmp_path)


@pytest.mark.parametrize(
    "field",
    ["storyboard_sha256", "semantic_lock_sha256", "approved_script_sha256", "audio_sha256", "subtitle_sha256"],
)
def test_import_fails_closed_when_any_current_task18_identity_has_staled(tmp_path: Path, field: str) -> None:
    _import(tmp_path)
    stale = tuple(item.model_copy(update={field: "f" * 64}) for item in _artifacts())

    with pytest.raises(H3ImportError, match="existing_segment_integrity_invalid"):
        _import(tmp_path, artifacts=stale, source=_source(tmp_path, f"{field}.mp4"))


def test_import_detects_source_mutation_after_snapshot_before_publication(tmp_path: Path) -> None:
    source = _source(tmp_path)

    def mutating_runner(argv: list[str], *, timeout_seconds: float, output_limit: int) -> CommandResult:
        source.write_bytes(b"mutated after snapshot")
        return CommandResult(argv=["ffprobe"], returncode=0, stdout=json.dumps(_probe_payload()))

    with pytest.raises(H3ImportError, match="source_changed_during_import"):
        import_h3_segment(
            book_id="book-01", episode_id="E001", segment_id="S01", prompt_artifacts=_artifacts(),
            source_path=source, episode_media_root=tmp_path / "book" / "episodes" / "E001" / "media",
            ffprobe_runner=mutating_runner,
        )

    assert not (tmp_path / "book").exists()


def test_import_rejects_redirected_source_or_destination_before_external_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bv.video.import_h3 as module

    source = _source(tmp_path)
    root = tmp_path / "book" / "episodes" / "E001" / "media"
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(module, "_is_redirected", lambda path: Path(path) in {source.parent, root})

    with pytest.raises(H3ImportError, match="unsafe_h3_path"):
        _import(tmp_path, source=source)

    assert list(outside.iterdir()) == []


def test_import_rejects_real_windows_junction_destination_and_leaves_external_target_empty(tmp_path: Path) -> None:
    source = _source(tmp_path)
    root = tmp_path / "book" / "episodes" / "E001" / "media"
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
        with pytest.raises(H3ImportError, match="unsafe_h3_path"):
            _import(tmp_path, source=source)
        assert list(outside.iterdir()) == []
    finally:
        if os.path.lexists(root):
            os.rmdir(root)


def test_aggregate_rejects_extra_foreign_artifact_and_stale_current_identity(tmp_path: Path) -> None:
    first = _import(tmp_path)
    foreign = first.manifest_path.parent.parent / "S99"
    foreign.mkdir()
    (foreign / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(H3ImportError, match="aggregate_integrity_invalid"):
        _import(tmp_path)

    clean = tmp_path / "clean"
    initial = _import(clean)
    stale = list(_artifacts())
    stale[0] = stale[0].model_copy(update={"prompt": "new current prompt", "prompt_sha256": _sha("new current prompt")})
    with pytest.raises(H3ImportError, match="existing_segment_integrity_invalid"):
        _import(clean, artifacts=tuple(stale), source=_source(clean, "retry.mp4"))
    assert initial.manifest_path.is_file()


def test_neutral_video_import_symbols_are_h3_aliases_and_primary_functions_require_provider() -> None:
    import inspect

    from bv.video.import_h3 import (
        H3AggregateManifest,
        H3ImportError,
        H3ImportResult,
        H3SegmentManifest,
        H3SegmentSetStatus,
        VideoAggregateManifest,
        VideoImportError,
        VideoImportResult,
        VideoSegmentManifest,
        VideoSegmentSetStatus,
        all_required_segments_present,
        all_required_video_segments_present,
        import_h3_segment,
        import_video_segment,
        inspect_h3_segments,
        inspect_video_segments,
    )

    assert VideoImportError is H3ImportError
    assert VideoSegmentManifest is H3SegmentManifest
    assert VideoAggregateManifest is H3AggregateManifest
    assert VideoImportResult is H3ImportResult
    assert VideoSegmentSetStatus is H3SegmentSetStatus

    for function in (
        inspect_video_segments,
        import_video_segment,
        all_required_video_segments_present,
    ):
        parameter = inspect.signature(function).parameters["provider"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty

    for function in (inspect_h3_segments, import_h3_segment, all_required_segments_present):
        assert "provider" not in inspect.signature(function).parameters


@pytest.mark.parametrize("provider", ["grok_cli", "grok_manual", "h3_manual"])
def test_import_and_inspect_persist_exact_requested_provider(tmp_path: Path, provider: str) -> None:
    from bv.video.import_h3 import inspect_video_segments

    result = _import_video(tmp_path, provider=provider)

    assert result.provider == provider
    assert result.status == "visual_sample_ready"
    segment = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    aggregate = json.loads(result.aggregate_manifest_path.read_text(encoding="utf-8"))
    assert segment["provider"] == provider
    assert aggregate["provider"] == provider
    assert aggregate["status"] == "visual_sample_ready"

    status = inspect_video_segments(
        book_id="book-01",
        episode_id="E001",
        prompt_artifacts=_artifacts(),
        episode_media_root=_episode_media_root(tmp_path),
        provider=provider,
    )
    assert status.provider == provider
    assert status.status == "visual_sample_ready"


def test_h3_wrappers_select_h3_manual_without_a_provider_argument(tmp_path: Path) -> None:
    result = _import(tmp_path)

    assert result.provider == "h3_manual"
    segment = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    aggregate = json.loads(result.aggregate_manifest_path.read_text(encoding="utf-8"))
    assert segment["provider"] == "h3_manual"
    assert aggregate["provider"] == "h3_manual"

    status = inspect_h3_segments(
        book_id="book-01",
        episode_id="E001",
        prompt_artifacts=_artifacts(),
        episode_media_root=_episode_media_root(tmp_path),
    )
    assert status.provider == "h3_manual"
    assert status.status == "visual_sample_ready"


def test_same_provider_retry_is_idempotent_but_different_provider_fails_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    first = _import_video(tmp_path, provider="grok_cli", source=source)
    retry = _import_video(tmp_path, provider="grok_cli", source=source)

    assert retry.idempotent is True
    assert retry.provider == "grok_cli"
    assert retry.manifest_path == first.manifest_path

    with pytest.raises(H3ImportError, match="existing_segment_integrity_invalid"):
        _import_video(tmp_path, provider="grok_manual", source=source)


def test_inspect_and_import_fail_closed_when_existing_media_has_a_different_provider(tmp_path: Path) -> None:
    from bv.video.import_h3 import (
        all_required_video_segments_present,
        inspect_video_segments,
    )

    artifacts = _artifacts()
    root = _episode_media_root(tmp_path)
    _import(tmp_path, "S01", artifacts=artifacts, source=_source(tmp_path, "S01.mp4"))

    with pytest.raises(H3ImportError, match="existing_segment_integrity_invalid"):
        inspect_video_segments(
            book_id="book-01",
            episode_id="E001",
            prompt_artifacts=artifacts,
            episode_media_root=root,
            provider="grok_cli",
        )

    with pytest.raises(H3ImportError, match="existing_segment_integrity_invalid"):
        all_required_video_segments_present(
            book_id="book-01",
            episode_id="E001",
            prompt_artifacts=artifacts,
            episode_media_root=root,
            provider="grok_cli",
        )

    with pytest.raises(H3ImportError, match="existing_segment_integrity_invalid"):
        _import_video(
            tmp_path,
            "S02",
            provider="grok_cli",
            artifacts=artifacts,
            source=_source(tmp_path, "S02.mp4"),
        )


def test_neutral_inspect_and_presence_use_provider_bound_statuses(tmp_path: Path) -> None:
    from bv.video.import_h3 import (
        all_required_video_segments_present,
        inspect_video_segments,
    )

    artifacts = _artifacts()
    root = _episode_media_root(tmp_path)

    empty = inspect_video_segments(
        book_id="book-01",
        episode_id="E001",
        prompt_artifacts=artifacts,
        episode_media_root=root,
        provider="grok_manual",
    )
    assert empty.status == "awaiting_video_generation"
    assert empty.provider == "grok_manual"
    assert empty.present_segment_ids == ()
    assert empty.missing_segment_ids == ("S01", "S02", "S03", "S04")
    assert all_required_video_segments_present(
        book_id="book-01",
        episode_id="E001",
        prompt_artifacts=artifacts,
        episode_media_root=root,
        provider="grok_manual",
    ) is False

    for artifact in artifacts:
        _import_video(
            tmp_path,
            artifact.segment_id,
            provider="grok_manual",
            artifacts=artifacts,
            source=_source(tmp_path, f"{artifact.segment_id}.mp4"),
        )

    complete = inspect_video_segments(
        book_id="book-01",
        episode_id="E001",
        prompt_artifacts=artifacts,
        episode_media_root=root,
        provider="grok_manual",
    )
    assert complete.status == "video_imported"
    assert complete.provider == "grok_manual"
    assert complete.present_segment_ids == ("S01", "S02", "S03", "S04")
    assert complete.missing_segment_ids == ()
    assert all_required_video_segments_present(
        book_id="book-01",
        episode_id="E001",
        prompt_artifacts=artifacts,
        episode_media_root=root,
        provider="grok_manual",
    ) is True


def test_invalid_runtime_provider_fails_before_probe_or_any_media_side_effect(tmp_path: Path) -> None:
    from bv.video.import_h3 import (
        all_required_video_segments_present,
        import_video_segment,
        inspect_video_segments,
    )

    artifacts = _artifacts()
    root = _episode_media_root(tmp_path)
    source = _source(tmp_path)
    original = source.read_bytes()

    def unexpected_probe(argv: list[str], *, timeout_seconds: float, output_limit: int) -> CommandResult:
        raise AssertionError("ffprobe must not run for an invalid provider")

    with pytest.raises(H3ImportError, match="invalid_video_provider"):
        inspect_video_segments(
            book_id="book-01",
            episode_id="E001",
            prompt_artifacts=artifacts,
            episode_media_root=root,
            provider="unknown_provider",  # type: ignore[arg-type]
        )
    with pytest.raises(H3ImportError, match="invalid_video_provider"):
        all_required_video_segments_present(
            book_id="book-01",
            episode_id="E001",
            prompt_artifacts=artifacts,
            episode_media_root=root,
            provider="unknown_provider",  # type: ignore[arg-type]
        )
    with pytest.raises(H3ImportError, match="invalid_video_provider"):
        import_video_segment(
            book_id="book-01",
            episode_id="E001",
            segment_id="S01",
            prompt_artifacts=artifacts,
            source_path=source,
            episode_media_root=root,
            provider="unknown_provider",  # type: ignore[arg-type]
            ffprobe_runner=unexpected_probe,
        )

    assert source.read_bytes() == original
    assert source.is_file()
    assert not root.exists()


@pytest.mark.parametrize(
    ("present", "legacy_status", "expected_status"),
    [
        (("S01",), "visual_sample_ready", "visual_sample_ready"),
        (("S01",), "awaiting_h3_segments", "visual_sample_ready"),
        (("S01", "S02"), "awaiting_h3_segments", "video_partial"),
        (("S01", "S02"), "h3_imported", "video_partial"),
        (("S01", "S02", "S03", "S04"), "h3_imported", "video_imported"),
        (("S01", "S02", "S03", "S04"), "awaiting_h3_segments", "video_imported"),
    ],
)
def test_legacy_h3_manifests_default_provider_and_normalize_status_from_present_ids(
    tmp_path: Path,
    present: tuple[str, ...],
    legacy_status: str,
    expected_status: str,
) -> None:
    artifacts = _artifacts()
    for segment_id in present:
        _import(tmp_path, segment_id, artifacts=artifacts, source=_source(tmp_path, f"{segment_id}.mp4"))

    h3_root = _episode_media_root(tmp_path) / "h3"
    _rewrite_legacy_h3_manifests(h3_root, aggregate_status=legacy_status)

    segment = json.loads((h3_root / present[0] / "manifest.json").read_text(encoding="utf-8"))
    aggregate = json.loads((h3_root / "manifest.json").read_text(encoding="utf-8"))
    assert "provider" not in segment
    assert "provider" not in aggregate
    assert aggregate["status"] == legacy_status

    status = inspect_h3_segments(
        book_id="book-01",
        episode_id="E001",
        prompt_artifacts=artifacts,
        episode_media_root=_episode_media_root(tmp_path),
    )
    assert status.status == expected_status
    assert status.provider == "h3_manual"
    assert status.present_segment_ids == present
    assert status.missing_segment_ids == tuple(
        segment_id for segment_id in ("S01", "S02", "S03", "S04") if segment_id not in present
    )
