from __future__ import annotations

import hashlib
import shutil
import subprocess
import wave
from pathlib import Path

import pymupdf
import pytest

from bv.asr.volcengine import AsrResult, VolcCredentials, WordTiming
from bv.config import IndexTTS2Config
from bv.subtitles.generate import SubtitleCue, render_ass
from bv.state.models import EpisodeState
from bv.voice.indextts2 import SynthesizedVoice
from bv.workflow.media_stages import AsrStage, SubtitleStage, TtsStage
from bv.workflow.runtime import RuntimeAuthorization
from bv.workflow.stages import StageContext


TEXT = "这本书让人看见，生活不必靠反复解释来证明。把判断放回自己手里，才有可能承担真正的选择。"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tone(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        frame = b"\x00\x10"
        stream.writeframes(frame * round(48_000 * seconds))


def test_fake_tts_asr_with_real_audio_processing_and_subtitles(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("local FFmpeg is required")
    root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    approved = root / "script" / "approved.txt"
    approved.parent.mkdir(parents=True)
    approved.write_text(TEXT, encoding="utf-8")
    context = StageContext(
        book_id="book-demo", episode_id="E001", episode_root=root,
        episode_state=EpisodeState(
            book_id="book-demo", episode_id="E001", status="script_approved",
            script_hash=_sha(approved),
        ),
    )
    reference = tmp_path / "black-gold-3.wav"
    _tone(reference, 1.0)

    def synthesize(**kwargs: object) -> SynthesizedVoice:
        output = Path(kwargs["raw_output_path"])
        _tone(output, 15.0)
        return SynthesizedVoice(raw_path=output, raw_sha256=_sha(output))

    tts = TtsStage(
        config=IndexTTS2Config(
            install_dir=tmp_path, model_dir=tmp_path, voice_id="黑金3"
        ),
        reference_voice_path=reference,
        ffmpeg_command=ffmpeg,
        mode="technical_sample",
        sample_span=(0, len(TEXT)),
        synthesizer=synthesize,
    )
    tts_outcome = tts.run(context)

    def recognize(_credentials: VolcCredentials, request) -> AsrResult:
        duration = 15_000
        words = tuple(
            WordTiming(
                text=character,
                start_time=duration * index // len(TEXT),
                end_time=max(
                    duration * (index + 1) // len(TEXT),
                    duration * index // len(TEXT) + 1,
                ),
                confidence=1.0,
            )
            for index, character in enumerate(TEXT)
        )
        return AsrResult(
            text=TEXT, words=words, duration_ms=duration,
            audio_sha256=request.audio_sha256, request_id=request.request_id,
            response_sha256="a" * 64,
        )

    asr = AsrStage(
        credentials=VolcCredentials(api_key="placeholder"),
        endpoint="https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash",
        authorization=RuntimeAuthorization(allow_external=True),
        recognizer=recognize,
    )
    asr_outcome = asr.run(context)
    font = Path("vendor/story_to_handdrawn_video/public/fonts/MaShanZheng-Regular.ttf")
    subtitle_outcome = SubtitleStage(
        font_path=font,
        font_family="Ma Shan Zheng",
    ).run(context)

    assert set(tts_outcome.outputs) == {"voice_master", "voice_manifest"}
    assert set(asr_outcome.outputs) == {"alignment", "pronunciation_report"}
    assert set(subtitle_outcome.outputs) == {
        "subtitles_srt", "subtitles_ass", "subtitle_cues", "subtitle_manifest"
    }
    assert all(path.is_file() for path in subtitle_outcome.outputs.values())


def test_real_ass_renders_the_longest_supported_caption_on_one_line(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("local FFmpeg is required")
    font = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
    if not font.is_file():
        pytest.skip("the production Noto Sans SC font is required")
    ass = tmp_path / "subtitles.ass"
    rendered = render_ass(
        (SubtitleCue(index=0, start_ms=0, end_ms=1_000, text="甲乙丙丁戊己庚辛壬癸子丑寅卯"),),
        font_path=font,
        font_sha256=_sha(font),
        font_family="Noto Sans SC",
    )
    assert "WrapStyle: 2" in rendered
    ass.write_text(rendered, encoding="utf-8")
    frame = tmp_path / "frame.png"
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1080x1920:d=1",
            "-vf",
            "subtitles=filename=subtitles.ass",
            "-frames:v",
            "1",
            str(frame),
        ],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    pixmap = pymupdf.Pixmap(str(frame))
    active_rows = [
        y
        for y in range(1_300, 1_800)
        if any(
            all(channel > 200 for channel in pixmap.pixel(x, y)[:3])
            for x in range(40, 1_040, 4)
        )
    ]
    active_columns = [
        x
        for x in range(1_080)
        if any(
            all(channel > 200 for channel in pixmap.pixel(x, y)[:3])
            for y in range(1_300, 1_800, 4)
        )
    ]

    assert active_rows
    assert max(active_rows) - min(active_rows) < 100
    assert active_columns
    assert min(active_columns) >= 84
    assert max(active_columns) <= 995
