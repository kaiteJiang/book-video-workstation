from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bv.delivery.exporter import (
    DeliveryError,
    DeliveryExporter,
    DeliveryRequest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixed_utc(value: str = "2026-08-22T16:30:00+00:00") -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _request(tmp_path: Path, *, title: str = "活着") -> DeliveryRequest:
    episode = tmp_path / "workspace" / "books" / "B001" / "episodes" / "E001"
    paths = {
        "script_path": episode / "script" / "approved.txt",
        "voice_path": episode / "media" / "voice" / "voice_master.wav",
        "subtitle_path": episode / "media" / "subtitles" / "subtitles.ass",
        "video_path": episode / "media" / "final" / "final.mp4",
        "cover_path": episode / "media" / "final" / "social_cover.png",
        "production_profile_path": episode / "production_profile.json",
        "voice_manifest_path": episode / "media" / "voice" / "voice_master.json",
        "render_manifest_path": episode / "media" / "final" / "final.render.json",
        "qc_manifest_path": episode / "media" / "final" / "final.qc.json",
        "social_cover_manifest_path": episode / "media" / "final" / "social_cover.json",
    }
    for index, path in enumerate(paths.values(), start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps({"fixture": index}), encoding="utf-8")
        else:
            path.write_bytes(f"fixture-{index}".encode())
    return DeliveryRequest(
        episode_root=episode,
        book_id="B001",
        episode_id="E001",
        title=title,
        author="余华",
        **paths,
    )


def test_delivery_uses_shanghai_date_and_expected_names(tmp_path: Path) -> None:
    exporter = DeliveryExporter(clock=lambda: _fixed_utc())

    bundle = exporter.export_candidate(_request(tmp_path))

    assert bundle.version == 1
    assert bundle.root.name == "2026-08-23"
    assert (bundle.root / "2026-08-23-活着-成片.mp4").is_file()
    assert (bundle.root / "2026-08-23-活着-作品封面.png").is_file()
    assert (bundle.root / "2026-08-23-活着-文稿.txt").is_file()
    assert (bundle.root / "2026-08-23-活着-旁白.wav").is_file()
    assert (bundle.root / "2026-08-23-活着-字幕.ass").is_file()
    assert bundle.manifest_path.name == "2026-08-23-活着-交付清单.json"
    assert bundle.manifest_path == bundle.files[-1]
    assert bundle.manifest_sha256 == _sha(bundle.manifest_path)


def test_any_same_day_collision_advances_the_whole_bundle_to_v2(
    tmp_path: Path,
) -> None:
    exporter = DeliveryExporter(clock=lambda: _fixed_utc())
    request = _request(tmp_path)

    first = exporter.export_candidate(request)
    second = exporter.export_candidate(request)

    assert first.version == 1
    assert second.version == 2
    assert all("-V2" in path.name for path in second.files)
    assert second.manifest_path.name == "2026-08-23-活着-交付清单-V2.json"
    assert all(path.is_file() for path in first.files)


def test_partial_collision_is_preserved_and_forces_next_version(tmp_path: Path) -> None:
    request = _request(tmp_path)
    date_root = request.episode_root / "deliveries" / "2026-08-23"
    date_root.mkdir(parents=True)
    partial = date_root / "2026-08-23-活着-成片.mp4"
    partial.write_bytes(b"do not overwrite")

    bundle = DeliveryExporter(clock=lambda: _fixed_utc()).export_candidate(request)

    assert bundle.version == 2
    assert partial.read_bytes() == b"do not overwrite"


def test_final_approval_does_not_rewrite_candidate_manifest(tmp_path: Path) -> None:
    exporter = DeliveryExporter(clock=lambda: _fixed_utc())
    bundle = exporter.export_candidate(_request(tmp_path))
    before = _sha(bundle.manifest_path)

    record = exporter.record_final_approval(
        bundle,
        approved_at=_fixed_utc("2026-08-23T02:00:00+00:00"),
    )

    assert _sha(bundle.manifest_path) == before
    assert record.delivery_manifest_sha256 == before
    assert record.video_sha256 == _sha(bundle.video_path)
    assert record.cover_sha256 == _sha(bundle.cover_path)
    assert record.status == "final_approved"
    assert record.path.is_file()
    with pytest.raises(DeliveryError, match="approval_exists"):
        exporter.record_final_approval(bundle, approved_at=_fixed_utc())


def test_delivery_rejects_source_outside_episode_and_reserved_title(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    with pytest.raises(DeliveryError, match="unsafe_delivery_source"):
        DeliveryExporter(clock=lambda: _fixed_utc()).export_candidate(
            request.model_copy(update={"video_path": outside})
        )

    with pytest.raises(DeliveryError, match="delivery_title_invalid"):
        DeliveryExporter(clock=lambda: _fixed_utc()).export_candidate(
            _request(tmp_path / "reserved", title="CON")
        )


def test_final_approval_rejects_matching_media_outside_bundle_root(
    tmp_path: Path,
) -> None:
    exporter = DeliveryExporter(clock=lambda: _fixed_utc())
    bundle = exporter.export_candidate(_request(tmp_path))
    outside = tmp_path / "outside-video.mp4"
    outside.write_bytes(bundle.video_path.read_bytes())

    with pytest.raises(DeliveryError, match="unsafe_approval_bundle"):
        exporter.record_final_approval(
            bundle.model_copy(update={"video_path": outside}),
            approved_at=_fixed_utc(),
        )


def test_final_approval_rejects_any_changed_candidate_artifact(tmp_path: Path) -> None:
    exporter = DeliveryExporter(clock=lambda: _fixed_utc())
    bundle = exporter.export_candidate(_request(tmp_path))
    script = next(path for path in bundle.files if path.name.endswith("文稿.txt"))
    script.write_text("changed after candidate export", encoding="utf-8")

    with pytest.raises(DeliveryError, match="delivery_artifact_stale"):
        exporter.record_final_approval(bundle, approved_at=_fixed_utc())
