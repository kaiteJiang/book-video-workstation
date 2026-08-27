from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bv.config import (
    AppConfig,
    DoubaoTtsConfig,
    HanddrawnConfig,
    HumanWritingConfig,
    VideoConfig,
    load_config,
)

_DEFAULT_SKILL = Path.home() / ".codex/skills/human-writing/SKILL.md"
_DEFAULT_CHECKER = (
    Path.home() / ".codex/skills/human-writing/scripts/check_prose.py"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_CONFIG = _REPO_ROOT / "config.example.yaml"


def test_value_video_defaults_are_long_form() -> None:
    config = AppConfig()

    assert config.video.final_min_seconds == 45.0
    assert config.video.final_max_seconds == 60.0
    assert config.video.segment_target_seconds == 15.0
    assert config.video.segment_max_seconds == 15.5
    assert config.video_generation.provider == "grok_cli"
    assert config.video_generation.resolution == "480p"
    assert config.video_generation.command == "grok"
    assert config.video_generation.timeout_seconds == 900
    assert config.audio.hard_min_seconds == 45.0
    assert config.audio.ideal_min_seconds == 48.0
    assert config.audio.ideal_max_seconds == 56.0
    assert config.audio.hard_max_seconds == 60.0


def test_doubao_tts_config_contains_only_endpoint_and_environment_names() -> None:
    standalone = DoubaoTtsConfig()
    nested = AppConfig().doubao_tts

    assert standalone == nested
    assert standalone.submit_endpoint.endswith("/api/v3/tts/submit")
    assert standalone.query_endpoint.endswith("/api/v3/tts/query")
    assert standalone.api_key_env == "BV_DOUBAO_TTS_API_KEY"
    dumped = standalone.model_dump()
    assert "api_key" not in dumped
    assert set(dumped) == {
        "submit_endpoint",
        "query_endpoint",
        "api_key_env",
        "poll_interval_seconds",
        "timeout_seconds",
    }


def test_handdrawn_config_defaults_are_douyin_native() -> None:
    config = AppConfig()

    assert config.handdrawn.width == 1080
    assert config.handdrawn.height == 1920
    assert config.handdrawn.fps == 30
    assert config.handdrawn.safe_area.subtitle_bottom == 1660
    assert config.handdrawn.representative_count == 3
    assert config.handdrawn.image_size == "1080x1920"
    assert config.handdrawn.text_mode == "font"
    assert config.handdrawn.transition == "cross-dissolve"


def test_subtitle_defaults_use_a_real_bold_chinese_system_font() -> None:
    config = AppConfig()

    assert config.subtitle_font_path == Path(r"C:\Windows\Fonts\msyhbd.ttc")
    assert config.subtitle_font_family == "Microsoft YaHei"


@pytest.mark.parametrize("node_command", ["", "   "])
def test_handdrawn_config_rejects_blank_node_command(node_command: str) -> None:
    with pytest.raises(ValidationError, match="node_command.*nonblank"):
        HanddrawnConfig(node_command=node_command)


def test_load_config_resolves_relative_handdrawn_vendor_dir(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "handdrawn:\n"
        "  vendor_dir: vendor/story_to_handdrawn_video\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.handdrawn.vendor_dir == (
        tmp_path / "vendor" / "story_to_handdrawn_video"
    )


def test_video_generation_config_defaults_to_restricted_grok_cli() -> None:
    from bv.config import VideoGenerationConfig

    standalone = VideoGenerationConfig()
    nested = AppConfig().video_generation

    expected = {
        "provider": "grok_cli",
        "resolution": "480p",
        "command": "grok",
        "timeout_seconds": 900,
    }
    assert standalone.model_dump() == expected
    assert nested.model_dump() == expected
    assert type(nested) is VideoGenerationConfig


def test_human_writing_defaults_point_to_installed_skill() -> None:
    standalone = HumanWritingConfig()
    config = AppConfig()

    assert standalone.skill_path == _DEFAULT_SKILL
    assert standalone.checker_path == _DEFAULT_CHECKER
    assert standalone.skill_path.name == "SKILL.md"
    assert standalone.checker_path.name == "check_prose.py"
    assert config.human_writing.skill_path == _DEFAULT_SKILL
    assert config.human_writing.checker_path == _DEFAULT_CHECKER


def test_human_writing_defaults_follow_the_current_user_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    config = HumanWritingConfig()

    assert config.skill_path == (
        tmp_path / ".codex" / "skills" / "human-writing" / "SKILL.md"
    )
    assert config.checker_path == (
        tmp_path
        / ".codex"
        / "skills"
        / "human-writing"
        / "scripts"
        / "check_prose.py"
    )


def test_config_rejects_obsolete_video_duration_with_migration_guidance(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("video:\n  duration_seconds: 15.0\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            "video.duration_seconds.*final_min_seconds.*final_max_seconds.*"
            "video.segment_target_seconds.*video.segment_max_seconds"
        ),
    ):
        load_config(config_file)


def test_config_uses_explicit_indextts_paths_and_resolves_relative_paths(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "workspace_dir: workspace\n"
        "voice_path: workspace/voices/reference.wav\n"
        "subtitle_font_path: workspace/fonts/subtitle.ttf\n"
        "indextts2:\n"
        "  install_dir: 'D:\\\\Program Files (x86)\\\\index-tts'\n"
        "  model_dir: 'D:\\\\Program Files (x86)\\\\index-tts\\\\checkpoints'\n"
        "ffmpeg:\n"
        "  command: tools/ffmpeg.exe\n"
        "  ffprobe_command: tools/ffprobe.exe\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.workspace_dir == tmp_path / "workspace"
    assert config.voice_path == tmp_path / "workspace" / "voices" / "reference.wav"
    assert config.subtitle_font_path == tmp_path / "workspace" / "fonts" / "subtitle.ttf"
    assert config.indextts2.install_dir.name == "index-tts"
    assert config.indextts2.model_dir.name == "checkpoints"
    assert config.ffmpeg.command == tmp_path / "tools" / "ffmpeg.exe"
    assert config.ffmpeg.ffprobe_command == tmp_path / "tools" / "ffprobe.exe"
    assert config.video.final_min_seconds == 45.0
    assert config.video.final_max_seconds == 60.0
    assert config.video.segment_target_seconds == 15.0
    assert config.video.segment_max_seconds == 15.5
    assert config.video_generation.provider == "grok_cli"
    assert config.video_generation.resolution == "480p"
    assert config.video_generation.command == "grok"
    assert config.video_generation.timeout_seconds == 900
    assert config.audio.hard_min_seconds == 45.0
    assert config.audio.ideal_min_seconds == 48.0
    assert config.audio.ideal_max_seconds == 56.0
    assert config.audio.hard_max_seconds == 60.0
    assert config.codex.timeout_seconds == 1800
    assert config.grok.timeout_seconds == 600
    assert config.grok.blocking is False
    assert config.h3.mode == "manual_import"
    assert config.human_writing.skill_path == _DEFAULT_SKILL
    assert config.human_writing.checker_path == _DEFAULT_CHECKER


def test_config_preserves_absolute_windows_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "indextts2:\n"
        "  install_dir: 'D:\\\\Program Files (x86)\\\\index-tts'\n"
        "  model_dir: 'D:\\\\Program Files (x86)\\\\index-tts\\\\checkpoints'\n"
        "human_writing:\n"
        "  skill_path: 'C:\\\\Users\\\\Example\\\\.codex\\\\skills\\\\human-writing\\\\SKILL.md'\n"
        "  checker_path: "
        "'C:\\\\Users\\\\Example\\\\.codex\\\\skills\\\\human-writing\\\\scripts\\\\check_prose.py'\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.indextts2.install_dir == Path(
        r"D:\Program Files (x86)\index-tts"
    )
    assert config.indextts2.model_dir == Path(
        r"D:\Program Files (x86)\index-tts\checkpoints"
    )
    assert config.human_writing.skill_path == Path(
        r"C:\Users\Example\.codex\skills\human-writing\SKILL.md"
    )
    assert config.human_writing.checker_path == Path(
        r"C:\Users\Example\.codex\skills\human-writing\scripts\check_prose.py"
    )


def test_load_config_resolves_relative_human_writing_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "human_writing:\n"
        "  skill_path: skills/human-writing/SKILL.md\n"
        "  checker_path: skills/human-writing/scripts/check_prose.py\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.human_writing.skill_path == (
        tmp_path / "skills" / "human-writing" / "SKILL.md"
    )
    assert config.human_writing.checker_path == (
        tmp_path / "skills" / "human-writing" / "scripts" / "check_prose.py"
    )


@pytest.mark.parametrize("ambiguous_path", [r"D:models", r"\models"])
def test_config_rejects_ambiguous_windows_paths(
    tmp_path: Path,
    ambiguous_path: str,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "indextts2:\n"
        f"  install_dir: '{ambiguous_path}'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous Windows"):
        load_config(config_file)


@pytest.mark.parametrize(
    "field_name",
    ["skill_path", "checker_path"],
)
@pytest.mark.parametrize("ambiguous_path", [r"D:models", r"\models"])
def test_load_config_rejects_ambiguous_human_writing_paths(
    tmp_path: Path,
    field_name: str,
    ambiguous_path: str,
) -> None:
    other = "skill_path" if field_name == "checker_path" else "checker_path"
    other_value = "skills/ok.md"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "human_writing:\n"
        f"  {field_name}: '{ambiguous_path}'\n"
        f"  {other}: {other_value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous Windows"):
        load_config(config_file)


@pytest.mark.parametrize("provider", ["grok_cli", "grok_manual", "h3_manual"])
def test_video_generation_accepts_known_providers(provider: str) -> None:
    from bv.config import VideoGenerationConfig

    assert VideoGenerationConfig.model_validate({"provider": provider}).provider == provider


@pytest.mark.parametrize("resolution", ["480p", "720p"])
def test_video_generation_accepts_known_resolutions(resolution: str) -> None:
    from bv.config import VideoGenerationConfig

    assert (
        VideoGenerationConfig.model_validate({"resolution": resolution}).resolution
        == resolution
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"provider": "minimax_api"},
        {"resolution": "1080p"},
        {"command": ""},
        {"command": "   "},
        {"timeout_seconds": 0},
        {"timeout_seconds": -1},
        {"timeout_seconds": True},
        {"timeout_seconds": False},
        {"timeout_seconds": "900"},
    ],
    ids=[
        "unknown_provider",
        "unknown_resolution",
        "empty_command",
        "whitespace_command",
        "zero_timeout",
        "negative_timeout",
        "bool_true_timeout",
        "bool_false_timeout",
        "string_timeout",
    ],
)
def test_video_generation_rejects_invalid_provider_resolution_command_and_timeout(
    payload: dict[str, object],
) -> None:
    from bv.config import VideoGenerationConfig

    with pytest.raises(ValidationError):
        VideoGenerationConfig.model_validate(payload)


def test_load_config_accepts_explicit_video_generation_and_neutral_segment_keys(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "video:\n"
        "  segment_target_seconds: 14.0\n"
        "  segment_max_seconds: 14.5\n"
        "video_generation:\n"
        "  provider: grok_manual\n"
        "  resolution: 720p\n"
        "  command: grok\n"
        "  timeout_seconds: 1200\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.video.segment_target_seconds == 14.0
    assert config.video.segment_max_seconds == 14.5
    assert config.video_generation.provider == "grok_manual"
    assert config.video_generation.resolution == "720p"
    assert config.video_generation.command == "grok"
    assert config.video_generation.timeout_seconds == 1200


def test_video_config_serialization_exposes_only_neutral_segment_names() -> None:
    video = VideoConfig()
    dumped = video.model_dump()
    json_dumped = video.model_dump(mode="json")
    as_json = video.model_dump_json()
    app_video = AppConfig().model_dump()["video"]

    for payload in (dumped, json_dumped, app_video):
        assert payload["segment_target_seconds"] == 15.0
        assert payload["segment_max_seconds"] == 15.5
        assert "h3_segment_target_seconds" not in payload
        assert "h3_segment_max_seconds" not in payload

    assert "segment_target_seconds" in as_json
    assert "segment_max_seconds" in as_json
    assert "h3_segment_target_seconds" not in as_json
    assert "h3_segment_max_seconds" not in as_json


def test_load_config_maps_legacy_h3_segment_keys_and_serializes_neutral_names(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "video:\n"
        "  h3_segment_target_seconds: 14.0\n"
        "  h3_segment_max_seconds: 14.5\n",
        encoding="utf-8",
    )

    config = load_config(config_file)
    dumped = config.video.model_dump()
    as_json = config.video.model_dump_json()

    assert config.video.segment_target_seconds == 14.0
    assert config.video.segment_max_seconds == 14.5
    assert dumped["segment_target_seconds"] == 14.0
    assert dumped["segment_max_seconds"] == 14.5
    assert "h3_segment_target_seconds" not in dumped
    assert "h3_segment_max_seconds" not in dumped
    assert "h3_segment_target_seconds" not in as_json
    assert "h3_segment_max_seconds" not in as_json


@pytest.mark.parametrize(
    "yaml_body",
    [
        (
            "video:\n"
            "  segment_target_seconds: 15.0\n"
            "  h3_segment_target_seconds: 14.0\n"
        ),
        (
            "video:\n"
            "  segment_max_seconds: 15.5\n"
            "  h3_segment_max_seconds: 16.0\n"
        ),
    ],
    ids=["target_seconds", "max_seconds"],
)
def test_load_config_rejects_conflicting_legacy_and_neutral_segment_keys(
    tmp_path: Path,
    yaml_body: str,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_body, encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)conflict"):
        load_config(config_file)


def test_load_config_normalizes_equal_legacy_and_neutral_segment_keys(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "video:\n"
        "  segment_target_seconds: 14.0\n"
        "  h3_segment_target_seconds: 14.0\n"
        "  segment_max_seconds: 14.5\n"
        "  h3_segment_max_seconds: 14.5\n",
        encoding="utf-8",
    )

    config = load_config(config_file)
    dumped = config.video.model_dump()

    assert config.video.segment_target_seconds == 14.0
    assert config.video.segment_max_seconds == 14.5
    assert dumped["segment_target_seconds"] == 14.0
    assert dumped["segment_max_seconds"] == 14.5
    assert "h3_segment_target_seconds" not in dumped
    assert "h3_segment_max_seconds" not in dumped


def test_example_config_documents_video_generation_and_neutral_segment_keys() -> None:
    raw = yaml.safe_load(_EXAMPLE_CONFIG.read_text(encoding="utf-8"))

    assert isinstance(raw, dict)
    assert isinstance(raw.get("video_generation"), dict)
    assert isinstance(raw.get("video"), dict)
    video = raw["video"]
    assert video["segment_target_seconds"] == 15.0
    assert video["segment_max_seconds"] == 15.5
    assert "h3_segment_target_seconds" not in video
    assert "h3_segment_max_seconds" not in video

    config = load_config(_EXAMPLE_CONFIG)
    assert config.video.segment_target_seconds == 15.0
    assert config.video.segment_max_seconds == 15.5
    assert config.video_generation.provider == "grok_cli"
    assert config.video_generation.resolution == "480p"
    assert config.video_generation.command == "grok"
    assert config.video_generation.timeout_seconds == 900
