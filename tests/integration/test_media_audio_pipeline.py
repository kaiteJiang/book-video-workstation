from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from datetime import UTC, datetime
from pathlib import Path

import pymupdf
import pytest

from bv.asr.volcengine import AsrResult, VolcCredentials, WordTiming
from bv.asr.alignment import AlignedCharacter, AlignedScript, PronunciationReport
from bv.config import IndexTTS2Config
from bv.production.profile import ProductionProfile
from bv.subtitles.generate import SubtitleCue, render_ass
from bv.state.models import EpisodeState
from bv.voice.indextts2 import SynthesizedVoice
from bv.voice.processing import NarrationDuration, VoiceManifest
from bv.workflow.media_stages import AsrStage, MediaStageError, SubtitleStage, TtsStage
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


def _subtitle_stage_inputs(tmp_path: Path) -> tuple[StageContext, Path, Path]:
    text = "这是一段批准文本。"
    root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    approved = root / "script" / "approved.txt"
    approved.parent.mkdir(parents=True)
    approved.write_text(text, encoding="utf-8")
    script_sha256 = _sha(approved)
    context = StageContext(
        book_id="book-demo",
        episode_id="E001",
        episode_root=root,
        episode_state=EpisodeState(
            book_id="book-demo",
            episode_id="E001",
            status="script_approved",
            script_hash=script_sha256,
        ),
    )
    report = PronunciationReport(
        similarity=1.0,
        level="pass",
        violations=(),
        substitutions=0,
        deletions=0,
        insertions=0,
        protected_terms=(),
        approved_sha256=script_sha256,
        audio_sha256="a" * 64,
        asr_sha256="b" * 64,
    )
    aligned = AlignedScript(
        approved_text=text,
        characters=tuple(
            AlignedCharacter(
                index=index,
                character=character,
                start_ms=index * 100,
                end_ms=(index + 1) * 100,
                relation="match",
            )
            for index, character in enumerate(text)
        ),
        report=report,
    )
    alignment_path = root / "media" / "alignment" / "alignment.json"
    alignment_path.parent.mkdir(parents=True)
    alignment_path.write_text(aligned.model_dump_json(), encoding="utf-8")
    voice = VoiceManifest(
        voice_id="fixture",
        script_sha256=script_sha256,
        reference_sha256="c" * 64,
        raw_sha256="d" * 64,
        master_sha256="a" * 64,
        processing_sha256="e" * 64,
        sample_rate=48_000,
        channels=1,
        sample_width=2,
        raw_duration_ms=len(text) * 100,
        master_duration_ms=len(text) * 100,
        qualifying_start_trim_ms=0,
        qualifying_end_trim_ms=0,
        duration=NarrationDuration(
            seconds=len(text) / 10,
            level="pass",
            band="ideal",
        ),
        created_at=datetime.now(UTC),
    )
    voice_path = root / "media" / "voice" / "voice_master.json"
    voice_path.parent.mkdir(parents=True)
    voice_path.write_text(voice.model_dump_json(), encoding="utf-8")
    font = tmp_path / "subtitle.ttf"
    font.write_bytes(b"\x00\x01\x00\x00subtitle stage test font")
    return context, font, approved.parent / "subtitle_breaks.txt"


def _run_subtitle_stage(context: StageContext, font: Path) -> None:
    SubtitleStage(font_path=font, font_family="Microsoft YaHei").run(context)


def _write_pair_profile(context: StageContext) -> None:
    (context.episode_root / "production_profile.json").write_text(
        ProductionProfile.short_book_default().model_dump_json(indent=2),
        encoding="utf-8",
    )


def test_color_story_pair_subtitle_stage_requires_break_file(tmp_path: Path) -> None:
    context, font, _breaks = _subtitle_stage_inputs(tmp_path)
    _write_pair_profile(context)

    with pytest.raises(MediaStageError, match="subtitle_input_invalid"):
        _run_subtitle_stage(context, font)


def test_pair_storyboard_without_profile_still_requires_break_file(
    tmp_path: Path,
) -> None:
    context, font, _breaks = _subtitle_stage_inputs(tmp_path)
    storyboard = (
        context.episode_root
        / "media"
        / "illustration"
        / "illustration_storyboard.json"
    )
    storyboard.parent.mkdir(parents=True)
    storyboard.write_text(
        '{"sequence_mode":"color-story-pair"}',
        encoding="utf-8",
    )

    with pytest.raises(MediaStageError, match="subtitle_input_invalid"):
        _run_subtitle_stage(context, font)


def test_color_story_pair_subtitle_stage_records_required_break_state(
    tmp_path: Path,
) -> None:
    context, font, breaks = _subtitle_stage_inputs(tmp_path)
    _write_pair_profile(context)
    breaks.write_text("这是一段批准文本。\n", encoding="utf-8")

    outcome = SubtitleStage(
        font_path=font,
        font_family="Microsoft YaHei",
    ).run(context)

    assert outcome.inputs["subtitle_sequence_mode"] == "color-story-pair"
    assert outcome.inputs["subtitle_breaks_presence"] == "present"
    assert outcome.inputs["subtitle_breaks_sha256"] == _sha(breaks)


def test_legacy_subtitle_stage_records_missing_break_fallback(tmp_path: Path) -> None:
    context, font, _breaks = _subtitle_stage_inputs(tmp_path)

    outcome = SubtitleStage(
        font_path=font,
        font_family="Microsoft YaHei",
    ).run(context)

    assert outcome.inputs["subtitle_sequence_mode"] == "legacy-monochrome-reveal"
    assert outcome.inputs["subtitle_breaks_presence"] == "absent"
    assert "subtitle_breaks_sha256" not in outcome.inputs


def test_corrupt_profile_cannot_be_treated_as_legacy_subtitle_fallback(
    tmp_path: Path,
) -> None:
    context, font, _breaks = _subtitle_stage_inputs(tmp_path)
    (context.episode_root / "production_profile.json").write_text(
        '{"schema_version":1,"visual":',
        encoding="utf-8",
    )

    with pytest.raises(MediaStageError, match="subtitle_input_invalid"):
        _run_subtitle_stage(context, font)


def test_subtitle_stage_rejects_an_empty_existing_break_file(tmp_path: Path) -> None:
    context, font, breaks = _subtitle_stage_inputs(tmp_path)
    breaks.write_text(" \n\t\n", encoding="utf-8")

    with pytest.raises(MediaStageError, match="subtitle_input_invalid"):
        _run_subtitle_stage(context, font)


def test_subtitle_stage_rejects_a_direct_break_file_symlink(tmp_path: Path) -> None:
    context, font, breaks = _subtitle_stage_inputs(tmp_path)
    target = tmp_path / "breaks-source.txt"
    target.write_text("这是一段批准文本。\n", encoding="utf-8")
    try:
        breaks.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable for this test account")

    with pytest.raises(MediaStageError, match="subtitle_input_invalid"):
        _run_subtitle_stage(context, font)


def test_subtitle_stage_rejects_break_file_under_a_parent_junction(tmp_path: Path) -> None:
    if not hasattr(Path, "symlink_to"):
        pytest.skip("Windows junction support is required")
    context, font, breaks = _subtitle_stage_inputs(tmp_path)
    breaks.write_text("这是一段批准文本。\n", encoding="utf-8")
    script = breaks.parent
    source = script.with_name("script-source")
    script.rename(source)
    result = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(script), str(source)],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("Windows junction creation is unavailable for this test account")

    with pytest.raises(MediaStageError, match="subtitle_input_invalid"):
        _run_subtitle_stage(context, font)


def test_fake_tts_asr_with_real_audio_processing_and_subtitles(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("local FFmpeg is required")
    root = tmp_path / "workspace" / "books" / "book-demo" / "episodes" / "E001"
    approved = root / "script" / "approved.txt"
    approved.parent.mkdir(parents=True)
    approved.write_text(TEXT, encoding="utf-8")
    semantic_chunks = (
        "这本书让人看见，",
        "生活不必靠反复解释来证明。",
        "把判断放回自己手里，",
        "才有可能承担真正的选择。",
    )
    (approved.parent / "subtitle_breaks.txt").write_text(
        "\n".join(semantic_chunks) + "\n",
        encoding="utf-8",
    )
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
    cue_payload = json.loads(
        subtitle_outcome.outputs["subtitle_cues"].read_text(encoding="utf-8")
    )
    assert [item["text"].replace("\n", "") for item in cue_payload] == list(
        semantic_chunks
    )
    assert subtitle_outcome.inputs["subtitle_breaks_sha256"] == _sha(
        approved.parent / "subtitle_breaks.txt"
    )

    (approved.parent / "subtitle_breaks.txt").unlink()
    fallback_outcome = SubtitleStage(
        font_path=font,
        font_family="Ma Shan Zheng",
    ).run(context)
    fallback_cues = json.loads(
        fallback_outcome.outputs["subtitle_cues"].read_text(encoding="utf-8")
    )
    assert "subtitle_breaks_sha256" not in fallback_outcome.inputs
    assert "".join(item["text"].replace("\n", "") for item in fallback_cues) == TEXT

    (approved.parent / "subtitle_breaks.txt").write_text("篡改的字幕。\n", encoding="utf-8")
    with pytest.raises(MediaStageError, match="subtitle_stage_failed"):
        SubtitleStage(font_path=font, font_family="Ma Shan Zheng").run(context)


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
