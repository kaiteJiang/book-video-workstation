from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from bv.asr.alignment import AlignedCharacter, AlignedScript, PronunciationReport
from bv.production.profile import ProductionProfile
from bv.state.models import ArtifactRef, EpisodeState, StageManifest
from bv.state.store import StateStore
from bv.workflow.media_runtime import MediaProductionService, MediaWorkflowError


TEXT = "这是一段批准文本。"


class _UnusedIllustrationGateway:
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_alignment(root: Path) -> None:
    approved = root / "script" / "approved.txt"
    approved.parent.mkdir(parents=True, exist_ok=True)
    approved.write_text(TEXT, encoding="utf-8")
    aligned = AlignedScript(
        approved_text=TEXT,
        characters=tuple(
            AlignedCharacter(
                index=index,
                character=character,
                start_ms=index * 100,
                end_ms=(index + 1) * 100,
                relation="match",
            )
            for index, character in enumerate(TEXT)
        ),
        report=PronunciationReport(
            similarity=1.0,
            level="pass",
            violations=(),
            substitutions=0,
            deletions=0,
            insertions=0,
            protected_terms=(),
            approved_sha256=_sha(approved),
            audio_sha256="a" * 64,
            asr_sha256="b" * 64,
        ),
    )
    alignment = root / "media" / "alignment" / "alignment.json"
    alignment.parent.mkdir(parents=True, exist_ok=True)
    alignment.write_text(aligned.model_dump_json(), encoding="utf-8")


def _currentness_fixture(
    tmp_path: Path,
    *,
    sequence_mode: str,
    breaks_present: bool,
    stage_scoped: bool = False,
    configured_mode: str | None = None,
    joint_mode_evidence: str | None = None,
) -> tuple[MediaProductionService, StateStore, Path]:
    workspace = tmp_path / "workspace"
    root = workspace / "books" / "book-demo" / "episodes" / "E001"
    root.mkdir(parents=True)
    _write_alignment(root)
    if sequence_mode == "color-story-pair":
        (root / "production_profile.json").write_text(
            ProductionProfile.short_book_default().model_dump_json(indent=2),
            encoding="utf-8",
        )
    breaks = root / "script" / "subtitle_breaks.txt"
    if breaks_present:
        breaks.write_text("这是一段\n批准文本。\n", encoding="utf-8")
    artifact = root / "media" / "subtitles" / "subtitles.ass"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"current subtitle artifact")
    artifact_ref = ArtifactRef(
        path=str(artifact),
        sha256=_sha(artifact),
        size_bytes=artifact.stat().st_size,
    )
    inputs = {
        "media_mode": "full",
        "subtitle_sequence_mode": sequence_mode,
        "subtitle_breaks_presence": "present" if breaks_present else "absent",
    }
    if breaks_present:
        inputs["subtitle_breaks_sha256"] = _sha(breaks)
    config_hash = "1" * 64 if stage_scoped else "0" * 64
    config_identity = (
        f"stage-v1:{config_hash}" if stage_scoped else f"global-v1:{config_hash}"
    )
    completed = ["subtitles", "plan_illustrations", "visual_render", "render"]
    manifests = {
        name: StageManifest(
            stage=name,
            status="completed",
            inputs=dict(inputs) if name == "subtitles" else {},
            outputs={"artifact": artifact_ref},
            config_sha256=config_identity,
        )
        for name in completed
    }
    if joint_mode_evidence is not None:
        for name in ("tts", "asr"):
            evidence_path = root / "media" / "evidence" / f"{name}.json"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(f'{{"stage":"{name}"}}', encoding="utf-8")
            completed.append(name)
            manifests[name] = StageManifest(
                stage=name,
                status="completed",
                inputs={"media_mode": joint_mode_evidence},
                outputs={
                    "artifact": ArtifactRef(
                        path=str(evidence_path),
                        sha256=_sha(evidence_path),
                        size_bytes=evidence_path.stat().st_size,
                    )
                },
                config_sha256=config_identity,
            )
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        status="illustrations_ready",
        script_hash=_sha(root / "script" / "approved.txt"),
        completed_stages=completed,
        stage_manifests=manifests,
    )
    store = StateStore(workspace)
    store.save_episode(episode)
    kwargs = {"stage_config_sha256s": {"subtitles": config_hash}} if stage_scoped else {}
    service = MediaProductionService(
        store=store,
        stages={},
        images=_UnusedIllustrationGateway(),
        configured_mode=configured_mode,
        **kwargs,
    )
    return service, store, root


def _is_current(service: MediaProductionService, store: StateStore) -> tuple[bool, EpisodeState]:
    episode = store.load_episode("book-demo", "E001")
    current = service._manifest_current(episode, "subtitles", mode="full")
    return current, episode


def test_new_mode_unchanged_break_file_is_current(tmp_path: Path) -> None:
    service, store, _root = _currentness_fixture(
        tmp_path,
        sequence_mode="color-story-pair",
        breaks_present=True,
    )

    current, episode = _is_current(service, store)

    assert current is True
    assert episode.stale_stages == []


@pytest.mark.parametrize("mutation", ["changed", "empty", "removed"])
def test_new_mode_break_mutation_is_stale_and_invalidates_downstream(
    tmp_path: Path,
    mutation: str,
) -> None:
    service, store, root = _currentness_fixture(
        tmp_path,
        sequence_mode="color-story-pair",
        breaks_present=True,
    )
    breaks = root / "script" / "subtitle_breaks.txt"
    if mutation == "changed":
        breaks.write_text("篡改字幕。\n", encoding="utf-8")
    elif mutation == "empty":
        breaks.write_text(" \n\t\n", encoding="utf-8")
    else:
        breaks.unlink()

    current, episode = _is_current(service, store)

    assert current is False
    assert {"plan_illustrations", "visual_render", "render"}.issubset(
        episode.stale_stages
    )
    assert service._manifest_current(episode, "subtitles", mode="full") is False


def test_new_mode_missing_break_can_never_reuse_completed_manifest(tmp_path: Path) -> None:
    service, store, _root = _currentness_fixture(
        tmp_path,
        sequence_mode="color-story-pair",
        breaks_present=False,
    )

    current, _episode = _is_current(service, store)

    assert current is False


def test_legacy_missing_break_fallback_is_current_until_a_file_appears(
    tmp_path: Path,
) -> None:
    service, store, root = _currentness_fixture(
        tmp_path,
        sequence_mode="legacy-monochrome-reveal",
        breaks_present=False,
    )

    current, _episode = _is_current(service, store)
    assert current is True

    (root / "script" / "subtitle_breaks.txt").write_text(
        "这是一段\n批准文本。\n",
        encoding="utf-8",
    )
    current, episode = _is_current(service, store)
    assert current is False
    assert "plan_illustrations" in episode.stale_stages


def test_present_invalid_break_cannot_be_made_current_by_manifest_tamper(
    tmp_path: Path,
) -> None:
    service, store, root = _currentness_fixture(
        tmp_path,
        sequence_mode="color-story-pair",
        breaks_present=True,
    )
    breaks = root / "script" / "subtitle_breaks.txt"
    breaks.write_text("不等于批准文本。\n", encoding="utf-8")
    episode = store.load_episode("book-demo", "E001")
    episode.stage_manifests["subtitles"].inputs["subtitle_breaks_sha256"] = _sha(
        breaks
    )

    assert service._manifest_current(episode, "subtitles", mode="full") is False


def test_manifest_hash_or_sequence_mode_tamper_is_stale(tmp_path: Path) -> None:
    service, store, _root = _currentness_fixture(
        tmp_path,
        sequence_mode="color-story-pair",
        breaks_present=True,
    )
    episode = store.load_episode("book-demo", "E001")
    episode.stage_manifests["subtitles"].inputs["subtitle_breaks_sha256"] = "0" * 64
    assert service._manifest_current(episode, "subtitles", mode="full") is False

    episode = store.load_episode("book-demo", "E001")
    episode.stage_manifests["subtitles"].inputs["subtitle_sequence_mode"] = (
        "legacy-monochrome-reveal"
    )
    assert service._manifest_current(episode, "subtitles", mode="full") is False


def test_corrupt_profile_is_not_legacy_for_currentness(tmp_path: Path) -> None:
    service, store, root = _currentness_fixture(
        tmp_path,
        sequence_mode="color-story-pair",
        breaks_present=True,
    )
    (root / "production_profile.json").write_text("{", encoding="utf-8")

    current, _episode = _is_current(service, store)

    assert current is False


def test_leaf_link_is_stale_without_reading_or_changing_external_file(
    tmp_path: Path,
) -> None:
    service, store, root = _currentness_fixture(
        tmp_path,
        sequence_mode="legacy-monochrome-reveal",
        breaks_present=False,
    )
    external = tmp_path / "external-breaks.txt"
    external.write_text("这是一段批准文本。\n", encoding="utf-8")
    before = external.read_bytes()
    try:
        (root / "script" / "subtitle_breaks.txt").symlink_to(external)
    except OSError:
        pytest.skip("symlink creation is unavailable for this test account")

    current, _episode = _is_current(service, store)

    assert current is False
    assert external.read_bytes() == before


def test_parent_junction_is_stale_without_changing_external_file(tmp_path: Path) -> None:
    service, store, root = _currentness_fixture(
        tmp_path,
        sequence_mode="color-story-pair",
        breaks_present=True,
    )
    script = root / "script"
    external = tmp_path / "external-script"
    script.rename(external)
    before = (external / "subtitle_breaks.txt").read_bytes()
    result = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(script), str(external)],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("Windows junction creation is unavailable for this test account")

    current, _episode = _is_current(service, store)

    assert current is False
    assert (external / "subtitle_breaks.txt").read_bytes() == before


def test_stage_scoped_config_identity_remains_current(tmp_path: Path) -> None:
    service, store, _root = _currentness_fixture(
        tmp_path,
        sequence_mode="color-story-pair",
        breaks_present=True,
        stage_scoped=True,
    )

    current, _episode = _is_current(service, store)

    assert current is True


def test_render_restores_first_subtitle_dependent_gate_when_breaks_are_stale(
    tmp_path: Path,
) -> None:
    service, store, root = _currentness_fixture(
        tmp_path,
        sequence_mode="color-story-pair",
        breaks_present=True,
    )
    (root / "script" / "subtitle_breaks.txt").write_text(
        "篡改字幕。\n",
        encoding="utf-8",
    )

    with pytest.raises(MediaWorkflowError, match="subtitles_stale"):
        service.render("book-demo", "E001")

    episode = store.load_episode("book-demo", "E001")
    assert episode.status == "illustration_planning"
    assert "plan_illustrations" in episode.stale_stages
    assert episode.stage_manifests["plan_illustrations"].status == "stale"


@pytest.mark.parametrize("gate", ["render", "final"])
@pytest.mark.parametrize("tampered_mode", ["technical_sample", "invalid", None])
def test_public_downstream_gates_reject_untrusted_subtitle_media_mode(
    tmp_path: Path,
    gate: str,
    tampered_mode: str | None,
) -> None:
    service, store, root = _currentness_fixture(
        tmp_path,
        sequence_mode="color-story-pair",
        breaks_present=True,
        configured_mode="full",
    )
    episode = store.load_episode("book-demo", "E001")
    if gate == "final":
        episode.status = "awaiting_final_review"
    if tampered_mode is None:
        episode.stage_manifests["subtitles"].inputs.pop("media_mode")
    else:
        episode.stage_manifests["subtitles"].inputs["media_mode"] = tampered_mode
    store.save_episode(episode)
    breaks = root / "script" / "subtitle_breaks.txt"
    break_bytes = breaks.read_bytes()

    with pytest.raises(MediaWorkflowError, match="subtitles_stale"):
        (
            service.render("book-demo", "E001")
            if gate == "render"
            else service.approve_final("book-demo", "E001")
        )

    stale = store.load_episode("book-demo", "E001")
    assert service._manifest_current(stale, "subtitles", mode="full") is False
    assert stale.status == "illustration_planning"
    assert "plan_illustrations" in stale.stale_stages
    assert breaks.read_bytes() == break_bytes
    state_bytes = (root / "episode.json").read_bytes()
    with pytest.raises(MediaWorkflowError, match="subtitles_stale"):
        (
            service.render("book-demo", "E001")
            if gate == "render"
            else service.approve_final("book-demo", "E001")
        )
    assert (root / "episode.json").read_bytes() == state_bytes
    assert breaks.read_bytes() == break_bytes


@pytest.mark.parametrize("gate", ["render", "final"])
@pytest.mark.parametrize("trusted_mode", ["full", "technical_sample"])
def test_public_downstream_gates_accept_matching_configured_media_mode(
    tmp_path: Path,
    gate: str,
    trusted_mode: str,
) -> None:
    service, store, _root = _currentness_fixture(
        tmp_path,
        sequence_mode="color-story-pair",
        breaks_present=True,
        configured_mode=trusted_mode,
    )
    episode = store.load_episode("book-demo", "E001")
    episode.stage_manifests["subtitles"].inputs["media_mode"] = trusted_mode
    if gate == "final":
        episode.status = "awaiting_final_review"
    store.save_episode(episode)

    expected_error = "social_cover_not_ready" if gate == "render" else "final_gate_not_ready"
    with pytest.raises(MediaWorkflowError, match=expected_error):
        (
            service.render("book-demo", "E001")
            if gate == "render"
            else service.approve_final("book-demo", "E001")
        )

    assert store.load_episode("book-demo", "E001").status == episode.status


def test_unconfigured_legacy_service_uses_joint_upstream_mode_evidence(
    tmp_path: Path,
) -> None:
    service, store, _root = _currentness_fixture(
        tmp_path,
        sequence_mode="legacy-monochrome-reveal",
        breaks_present=False,
        joint_mode_evidence="full",
    )

    with pytest.raises(MediaWorkflowError, match="social_cover_not_ready"):
        service.render("book-demo", "E001")

    assert store.load_episode("book-demo", "E001").status == "illustrations_ready"


@pytest.mark.parametrize("gate", ["render", "final"])
@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_stage",
        "empty_outputs",
        "stale_tts",
        "stale_asr",
        "missing_output",
        "hash_tamper",
        "size_tamper",
        "config_tamper",
        "outside_path",
        "invalid_mode",
        "mode_conflict",
    ],
)
def test_unconfigured_legacy_rejects_noncurrent_joint_mode_evidence(
    tmp_path: Path,
    gate: str,
    mutation: str,
) -> None:
    service, store, root = _currentness_fixture(
        tmp_path,
        sequence_mode="legacy-monochrome-reveal",
        breaks_present=False,
        joint_mode_evidence="full",
    )
    episode = store.load_episode("book-demo", "E001")
    if gate == "final":
        episode.status = "awaiting_final_review"
    tts = episode.stage_manifests["tts"]
    if mutation == "wrong_stage":
        tts.stage = "asr"
    elif mutation == "empty_outputs":
        tts.outputs = {}
    elif mutation == "stale_tts":
        episode.stale_stages.append("tts")
    elif mutation == "stale_asr":
        episode.stale_stages.append("asr")
    elif mutation == "missing_output":
        Path(tts.outputs["artifact"].path).unlink()
    elif mutation == "hash_tamper":
        tts.outputs["artifact"].sha256 = "0" * 64
    elif mutation == "size_tamper":
        tts.outputs["artifact"].size_bytes += 1
    elif mutation == "config_tamper":
        tts.config_sha256 = f"global-v1:{'f' * 64}"
    elif mutation == "outside_path":
        outside = tmp_path / "outside-evidence.json"
        outside.write_text('{"stage":"tts"}', encoding="utf-8")
        tts.outputs["artifact"] = ArtifactRef(
            path=str(outside),
            sha256=_sha(outside),
            size_bytes=outside.stat().st_size,
        )
    elif mutation == "invalid_mode":
        episode.stage_manifests["asr"].inputs["media_mode"] = "invalid"
    else:
        episode.stage_manifests["asr"].inputs["media_mode"] = "technical_sample"
    store.save_episode(episode)

    with pytest.raises(MediaWorkflowError, match="subtitles_stale"):
        (
            service.render("book-demo", "E001")
            if gate == "render"
            else service.approve_final("book-demo", "E001")
        )

    stale = store.load_episode("book-demo", "E001")
    assert stale.status == "illustration_planning"
    assert "plan_illustrations" in stale.stale_stages
    assert not (root / "script" / "subtitle_breaks.txt").exists()
