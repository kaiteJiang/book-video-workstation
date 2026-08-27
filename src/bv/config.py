from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Literal

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from .illustration.contracts import SafeArea


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CodexConfig(_ConfigModel):
    command: str = "codex"
    timeout_seconds: int = 1800


class GrokConfig(_ConfigModel):
    command: str = "grok"
    timeout_seconds: int = 600
    blocking: bool = False


class IndexTTS2Config(_ConfigModel):
    uv_command: str = "uv"
    install_dir: Path = Path(r"D:\Program Files (x86)\index-tts")
    model_dir: Path = Path(r"D:\Program Files (x86)\index-tts\checkpoints")
    device: str = "cuda"
    voice_id: str = "黑金3"


class DoubaoTtsConfig(_ConfigModel):
    submit_endpoint: str = (
        "https://openspeech.bytedance.com/api/v3/tts/submit"
    )
    query_endpoint: str = (
        "https://openspeech.bytedance.com/api/v3/tts/query"
    )
    api_key_env: str = "BV_DOUBAO_TTS_API_KEY"
    poll_interval_seconds: float = Field(default=2.0, ge=0.2, le=30.0)
    timeout_seconds: float = Field(default=900.0, ge=10.0, le=3600.0)


class FFmpegConfig(_ConfigModel):
    command: Path = Path(r"D:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe")
    ffprobe_command: Path = Path(
        r"D:\Program Files (x86)\ffmpeg\bin\ffprobe.exe"
    )


class VolcAsrConfig(_ConfigModel):
    endpoint: str = (
        "https://openspeech.bytedance.com/api/v3/auc/bigmodel/"
        "recognize/flash"
    )
    resource_id: str = "volc.bigasr.auc_turbo"
    api_key_env: str = "BV_VOLC_ASR_API_KEY"
    app_key_env: str = "BV_VOLC_ASR_APP_KEY"
    access_key_env: str = "BV_VOLC_ASR_ACCESS_KEY"


class VideoGenerationConfig(_ConfigModel):
    provider: Literal["grok_cli", "grok_manual", "h3_manual"] = "grok_cli"
    resolution: Literal["480p", "720p"] = "480p"
    command: str = "grok"
    timeout_seconds: StrictInt = Field(default=900, gt=0)

    @field_validator("command")
    @classmethod
    def _require_nonblank_command(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command must be nonblank")
        return value


class VideoConfig(_ConfigModel):
    width: int = 1080
    height: int = 1920
    frame_rate: int = 30
    final_min_seconds: float = 45.0
    final_max_seconds: float = 60.0
    segment_target_seconds: float = Field(
        default=15.0,
        validation_alias=AliasChoices(
            "segment_target_seconds", "h3_segment_target_seconds"
        ),
    )
    segment_max_seconds: float = Field(
        default=15.5,
        validation_alias=AliasChoices(
            "segment_max_seconds", "h3_segment_max_seconds"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_segment_keys(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        migrated = dict(data)
        pairs = (
            ("segment_target_seconds", "h3_segment_target_seconds"),
            ("segment_max_seconds", "h3_segment_max_seconds"),
        )
        for new_key, old_key in pairs:
            has_new = new_key in migrated
            has_old = old_key in migrated
            if has_new and has_old:
                if migrated[new_key] != migrated[old_key]:
                    raise ValueError(
                        f"conflicting video segment keys: {new_key} and {old_key}"
                    )
                migrated.pop(old_key)
        return migrated


class AudioConfig(_ConfigModel):
    hard_min_seconds: float = 45.0
    ideal_min_seconds: float = 48.0
    ideal_max_seconds: float = 56.0
    hard_max_seconds: float = 60.0
    voice_lufs: float = -18.0
    final_lufs: float = -16.0
    true_peak_db: float = -1.5
    silence_threshold_db: float = -45.0
    removable_edge_silence_ms: int = 300


class H3Config(_ConfigModel):
    mode: Literal["manual_import"] = "manual_import"


class HanddrawnConfig(_ConfigModel):
    vendor_dir: Path = Path("vendor/story_to_handdrawn_video")
    node_command: str = "npm"
    width: Literal[1080] = 1080
    height: Literal[1920] = 1920
    fps: Literal[30] = 30
    representative_count: Literal[3] = 3
    image_size: Literal["1080x1920"] = "1080x1920"
    text_mode: Literal["font"] = "font"
    transition: Literal["cut", "page-flip", "cross-dissolve"] = "cross-dissolve"
    safe_area: SafeArea = Field(default_factory=SafeArea)

    @field_validator("node_command")
    @classmethod
    def _require_nonblank_node_command(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("node_command must be nonblank")
        return value


class HumanWritingConfig(_ConfigModel):
    skill_path: Path = Field(
        default_factory=lambda: (
            Path.home() / ".codex/skills/human-writing/SKILL.md"
        )
    )
    checker_path: Path = Field(
        default_factory=lambda: (
            Path.home()
            / ".codex/skills/human-writing/scripts/check_prose.py"
        )
    )


class AppConfig(_ConfigModel):
    workspace_dir: Path = Path("workspace")
    voice_path: Path = Field(
        default=Path("workspace/voices/reference_voice.wav"),
        description="Path to the authorized IndexTTS2 reference voice.",
    )
    subtitle_font_path: Path = Field(
        default=Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        description="Path to the font used for rendered subtitles.",
    )
    subtitle_font_family: str = "Microsoft YaHei"
    codex: CodexConfig = Field(default_factory=CodexConfig)
    grok: GrokConfig = Field(default_factory=GrokConfig)
    indextts2: IndexTTS2Config = Field(default_factory=IndexTTS2Config)
    doubao_tts: DoubaoTtsConfig = Field(default_factory=DoubaoTtsConfig)
    ffmpeg: FFmpegConfig = Field(default_factory=FFmpegConfig)
    volc_asr: VolcAsrConfig = Field(default_factory=VolcAsrConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    video_generation: VideoGenerationConfig = Field(
        default_factory=VideoGenerationConfig
    )
    audio: AudioConfig = Field(default_factory=AudioConfig)
    h3: H3Config = Field(default_factory=H3Config)
    handdrawn: HanddrawnConfig = Field(default_factory=HanddrawnConfig)
    human_writing: HumanWritingConfig = Field(default_factory=HumanWritingConfig)


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    windows_path = PureWindowsPath(path)
    if windows_path.is_absolute():
        return path
    if windows_path.drive or windows_path.root:
        raise ValueError(
            "ambiguous Windows drive-relative or root-relative path; "
            "use a fully absolute path or a normal relative path"
        )
    return base_dir / path


def load_config(path: Path) -> AppConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    video = raw.get("video")
    if isinstance(video, dict) and "duration_seconds" in video:
        raise ValueError(
            "video.duration_seconds is obsolete; use video.final_min_seconds, "
            "video.final_max_seconds, video.segment_target_seconds, and "
            "video.segment_max_seconds instead"
        )

    config = AppConfig.model_validate(raw)
    base_dir = config_path.parent.resolve()
    indextts2 = config.indextts2.model_copy(
        update={
            "install_dir": _resolve_path(config.indextts2.install_dir, base_dir),
            "model_dir": _resolve_path(config.indextts2.model_dir, base_dir),
        }
    )
    ffmpeg = config.ffmpeg.model_copy(
        update={
            "command": _resolve_path(config.ffmpeg.command, base_dir),
            "ffprobe_command": _resolve_path(
                config.ffmpeg.ffprobe_command, base_dir
            ),
        }
    )
    human_writing = config.human_writing.model_copy(
        update={
            "skill_path": _resolve_path(config.human_writing.skill_path, base_dir),
            "checker_path": _resolve_path(
                config.human_writing.checker_path, base_dir
            ),
        }
    )
    handdrawn = config.handdrawn.model_copy(
        update={
            "vendor_dir": _resolve_path(config.handdrawn.vendor_dir, base_dir),
        }
    )
    return config.model_copy(
        update={
            "workspace_dir": _resolve_path(config.workspace_dir, base_dir),
            "voice_path": _resolve_path(config.voice_path, base_dir),
            "subtitle_font_path": _resolve_path(
                config.subtitle_font_path, base_dir
            ),
            "indextts2": indextts2,
            "ffmpeg": ffmpeg,
            "handdrawn": handdrawn,
            "human_writing": human_writing,
        }
    )
