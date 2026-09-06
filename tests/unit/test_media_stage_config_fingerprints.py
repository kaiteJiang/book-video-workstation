from __future__ import annotations

import json
from pathlib import Path

from bv.config import AppConfig
from bv.core.atomic import atomic_write_json
from bv.state.models import EpisodeState
from bv.workflow.runtime import (
    RuntimeAuthorization,
    _media_runtime_stage_config_sha256s,
    build_media_runtime_bindings,
)


CATALOG = Path("vendor/story_to_handdrawn_video/references/handdrawn-style-library.json")
PROMPTS = (
    Path("prompts/illustration/style_select.md"),
    Path("prompts/illustration/storyboard.md"),
)


def _hashes(config: AppConfig, catalog: Path) -> dict[str, str]:
    return _media_runtime_stage_config_sha256s(
        config,
        catalog_path=catalog,
        prompt_paths=PROMPTS,
        mode="full",
        sample_span=None,
    )


def test_catalog_change_invalidates_only_style_scoped_prepare_fingerprints(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes(CATALOG.read_bytes())
    config = AppConfig(workspace_dir=tmp_path / "workspace")
    before = _hashes(config, catalog)
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["styles"][0]["summary"] += " updated"
    atomic_write_json(catalog, payload)

    after = _hashes(config, catalog)

    assert {
        name for name in before if before[name] != after[name]
    } == {"select_style", "plan_illustrations", "prepare_representatives"}
    assert all(before[name] == after[name] for name in ("tts", "asr", "subtitles"))


def test_tts_asr_and_subtitle_config_changes_are_isolated_to_their_stage(
    tmp_path: Path,
) -> None:
    base = AppConfig(workspace_dir=tmp_path / "workspace")
    before = _hashes(base, CATALOG)

    tts = _hashes(
        base.model_copy(
            update={
                "indextts2": base.indextts2.model_copy(update={"voice_id": "黑金4"})
            }
        ),
        CATALOG,
    )
    asr = _hashes(
        base.model_copy(
            update={
                "volc_asr": base.volc_asr.model_copy(
                    update={"endpoint": "https://example.invalid/asr"}
                )
            }
        ),
        CATALOG,
    )
    subtitles = _hashes(
        base.model_copy(update={"subtitle_font_family": "A Different Font"}),
        CATALOG,
    )

    assert {name for name in before if before[name] != tts[name]} == {"tts"}
    assert {name for name in before if before[name] != asr[name]} == {"asr"}
    assert {name for name in before if before[name] != subtitles[name]} == {
        "subtitles"
    }


def test_stage_fingerprints_do_not_read_secret_environment_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = AppConfig(workspace_dir=tmp_path / "workspace")
    monkeypatch.setenv(config.volc_asr.api_key_env, "first-secret")
    monkeypatch.setenv(config.doubao_tts.api_key_env, "first-secret")
    before = _hashes(config, CATALOG)
    monkeypatch.setenv(config.volc_asr.api_key_env, "second-secret")
    monkeypatch.setenv(config.doubao_tts.api_key_env, "second-secret")

    assert _hashes(config, CATALOG) == before


def test_real_runtime_builder_catalog_refresh_preserves_upstream_stage_identity(
    tmp_path: Path,
) -> None:
    vendor = tmp_path / "vendor"
    catalog = vendor / "references" / "handdrawn-style-library.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes(CATALOG.read_bytes())
    workspace = tmp_path / "workspace"
    episode = EpisodeState(
        book_id="book-demo",
        episode_id="E001",
        status="representative_generation_running",
        script_hash="a" * 64,
    )
    episode_root = workspace / "books" / "book-demo" / "episodes" / "E001"
    episode_root.mkdir(parents=True)
    config = AppConfig(
        workspace_dir=workspace,
        subtitle_font_path=tmp_path / "font.ttf",
        handdrawn={"vendor_dir": vendor},
    )
    first = build_media_runtime_bindings(
        config, RuntimeAuthorization(allow_external=False), mode="full"
    ).service
    prepare_stages = (
        "tts",
        "asr",
        "subtitles",
        "select_style",
        "plan_illustrations",
        "prepare_representatives",
    )
    before = {
        name: first._expected_config_sha256(episode, name)
        for name in prepare_stages
    }
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["styles"][0]["summary"] += " runtime refresh"
    atomic_write_json(catalog, payload)

    second = build_media_runtime_bindings(
        config, RuntimeAuthorization(allow_external=False), mode="full"
    ).service
    after = {
        name: second._expected_config_sha256(episode, name) for name in before
    }

    assert all(before[name] == after[name] for name in ("tts", "asr", "subtitles"))
    assert all(
        before[name] != after[name]
        for name in ("select_style", "plan_illustrations", "prepare_representatives")
    )
