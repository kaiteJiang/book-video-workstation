from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bv.asr.alignment import AlignedCharacter, AlignedScript, PronunciationReport
from bv.subtitles.generate import (
    SubtitleCue,
    SubtitleError,
    build_cues,
    build_subtitle_manifest,
    render_ass,
    render_srt,
)


def _sha(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _aligned(
    text: str,
    *,
    timings: list[tuple[int | None, int | None]] | None = None,
    level: str = "pass",
    violations: tuple[str, ...] = (),
    approved_sha256: str | None = None,
) -> AlignedScript:
    character_timings = timings or [(index * 100, (index + 1) * 100) for index in range(len(text))]
    characters = tuple(
        AlignedCharacter(
            index=index,
            character=character,
            start_ms=start,
            end_ms=end,
            relation="match",
        )
        for index, (character, (start, end)) in enumerate(zip(text, character_timings, strict=True))
    )
    report = PronunciationReport(
        similarity=1.0,
        level=level,  # type: ignore[arg-type]
        violations=violations,
        substitutions=0,
        deletions=0,
        insertions=0,
        protected_terms=(),
        approved_sha256=approved_sha256 or _sha(text),
        audio_sha256="a" * 64,
        asr_sha256="b" * 64,
    )
    return AlignedScript(approved_text=text, characters=characters, report=report)


def _font(tmp_path: Path, name: str = "subtitle.ttf") -> tuple[Path, str]:
    path = tmp_path / name
    value = b"\x00\x01\x00\x00Task17 test font"
    path.write_bytes(value)
    return path, _sha(value)


def _template_bytes() -> bytes:
    return (Path(__file__).resolve().parents[2] / "styles" / "subtitle-default.ass").read_bytes()


def _visible(cues: tuple[SubtitleCue, ...]) -> str:
    return "".join(cue.text.replace("\n", "") for cue in cues)


def test_build_cues_requires_exact_approved_reconstruction_and_pass_only_gate() -> None:
    aligned = _aligned("这是批准文本。")
    altered = aligned.model_copy(
        update={"characters": aligned.characters[:-1] + (aligned.characters[-1].model_copy(update={"character": "！"}),)}
    )

    with pytest.raises(SubtitleError, match="approved_text_mismatch"):
        build_cues(altered)
    with pytest.raises(SubtitleError, match="pronunciation_not_pass"):
        build_cues(_aligned("这是批准文本。", level="review"))
    with pytest.raises(SubtitleError, match="pronunciation_violations"):
        build_cues(_aligned("这是批准文本。", violations=("number_mismatch:2026",)))


@pytest.mark.parametrize("field", ["approved_sha256", "audio_sha256", "asr_sha256"])
def test_build_cues_rejects_invalid_or_stale_hashes(field: str) -> None:
    aligned = _aligned("完整校验文本。")
    report = aligned.report.model_copy(update={field: "c" * 64 if field == "approved_sha256" else "C" * 64})

    with pytest.raises(SubtitleError, match="approved_hash_mismatch" if field == "approved_sha256" else "invalid_alignment_hash"):
        build_cues(aligned.model_copy(update={"report": report}))


def test_build_cues_rejects_missing_invalid_or_nonmonotonic_timing() -> None:
    text = "六个字文本。"
    missing = _aligned(text, timings=[(0, 100), (100, 200), (None, None), (300, 400), (400, 500), (500, 600)])
    backwards = _aligned(text, timings=[(0, 100), (100, 200), (50, 150), (300, 400), (400, 500), (500, 600)])

    with pytest.raises(SubtitleError, match="missing_alignment_timing"):
        build_cues(missing)
    with pytest.raises(SubtitleError, match="invalid_alignment_timing"):
        build_cues(backwards)


def test_build_cues_prefers_sentence_punctuation_then_pause_boundaries() -> None:
    text = "第一句话很完整。第二段内容写在这里停顿明显然后继续说明"
    timings = [(index * 100, (index + 1) * 100) for index in range(len(text))]
    pause_at = text.index("停顿")
    timings[pause_at] = (timings[pause_at][0] + 500, timings[pause_at][1] + 500)
    for index in range(pause_at + 1, len(text)):
        timings[index] = (timings[index][0] + 500, timings[index][1] + 500)

    cues = build_cues(_aligned(text, timings=timings), min_duration_ms=100)

    assert _visible(cues) == text
    assert cues[0].text.replace("\n", "").endswith("。")
    assert cues[1].text.replace("\n", "") == "第二段内容写在这里"


def test_build_cues_targets_six_to_fourteen_and_rebalances_one_character_orphans() -> None:
    text = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰"

    cues = build_cues(_aligned(text), min_duration_ms=100)

    lengths = [len(cue.text.replace("\n", "")) for cue in cues]
    assert all(6 <= length <= 14 for length in lengths)
    assert lengths != [14, 1]
    assert _visible(cues) == text


def test_build_cues_keeps_protected_terms_together_across_cues_and_lines() -> None:
    text = "请读一读百年孤独这本书再讨论它的开头。"

    cues = build_cues(_aligned(text), protected_terms=("百年孤独",), min_duration_ms=100)

    assert _visible(cues) == text
    assert sum("百年孤独" in cue.text.replace("\n", "") for cue in cues) == 1
    assert all("百年\n孤独" not in cue.text for cue in cues)


def test_build_cues_honors_verified_semantic_chunks_without_mechanical_splits() -> None:
    text = "阎真的《沧浪之水》，可以当作一张人格变化的账单来读。替自己的选择找理由。"
    chunks = (
        "阎真的《沧浪之水》，",
        "可以当作一张",
        "人格变化的账单来读。",
        "替自己的选择找理由。",
    )

    cues = build_cues(
        _aligned(text),
        semantic_chunks=chunks,
        min_duration_ms=100,
    )

    assert [cue.text.replace("\n", "") for cue in cues] == list(chunks)
    assert _visible(cues) == text


def test_build_cues_rejects_semantic_chunks_that_change_the_approved_script() -> None:
    aligned = _aligned("选择总有代价，人能决定自己愿意付哪一种。")

    with pytest.raises(SubtitleError, match="semantic_chunks_text_mismatch"):
        build_cues(
            aligned,
            semantic_chunks=("选择没有代价，", "人能决定自己愿意付哪一种。"),
            min_duration_ms=100,
        )


def test_build_cues_fails_closed_instead_of_merging_verified_semantic_chunks() -> None:
    aligned = _aligned("甲乙丙丁戊己")

    with pytest.raises(SubtitleError, match="insufficient_cue_duration"):
        build_cues(
            aligned,
            semantic_chunks=("甲乙丙", "丁戊己"),
        )


def test_build_cues_keeps_existing_fallback_when_no_semantic_chunks_exist() -> None:
    aligned = _aligned("第一句话很完整。第二段内容写在这里。")

    assert build_cues(aligned, min_duration_ms=100) == build_cues(
        aligned,
        semantic_chunks=(),
        min_duration_ms=100,
    )


def test_build_cues_allows_an_indivisible_protected_term_longer_than_target() -> None:
    text = "请记住这个超长不可拆分核心概念名称然后继续。"
    term = "超长不可拆分核心概念名称"

    cues = build_cues(_aligned(text), protected_terms=(term,), min_duration_ms=100)

    assert any(term in cue.text.replace("\n", "") for cue in cues)
    assert max(len(cue.text.replace("\n", "")) for cue in cues) > 14


def test_build_cues_preserves_whitespace_punctuation_and_two_line_source_text() -> None:
    text = " 你好，世界！ 这是\t批准的 文本。 "

    cues = build_cues(_aligned(text), min_duration_ms=100)

    assert _visible(cues) == "你好，世界！这是批准的文本。"
    assert all(cue.text.count("\n") <= 1 for cue in cues)
    assert "，" in _visible(cues) and "！" in _visible(cues) and "。" in _visible(cues)


def test_leading_anchor_punctuation_does_not_overlap_the_previous_cue() -> None:
    text = "甲乙丙丁戊己庚辛壬癸。\n《子丑寅卯辰巳午未申酉》"
    aligned = _aligned(text)
    period_index = text.index("。")
    newline_index = text.index("\n")
    bracket_index = text.index("《")
    characters = list(aligned.characters)
    for index in (newline_index, bracket_index):
        characters[index] = characters[index].model_copy(
            update={
                "start_ms": characters[period_index].start_ms,
                "end_ms": characters[period_index].end_ms,
                "relation": "anchor",
            }
        )

    cues = build_cues(
        aligned.model_copy(update={"characters": tuple(characters)}),
        min_duration_ms=100,
    )

    assert _visible(cues) == text.replace("\n", "")
    assert all(left.end_ms <= right.start_ms for left, right in zip(cues, cues[1:]))


def test_build_cues_uses_minimum_duration_without_overlap_and_fails_at_audio_end() -> None:
    text = "甲乙丙丁戊己庚辛壬癸子丑"
    cues = build_cues(_aligned(text), audio_duration_ms=1_300)

    assert all(cue.end_ms - cue.start_ms >= 600 for cue in cues)
    assert all(left.end_ms <= right.start_ms for left, right in zip(cues, cues[1:]))
    assert cues[-1].end_ms <= 1_300

    with pytest.raises(SubtitleError, match="insufficient_cue_duration"):
        build_cues(
            _aligned("甲乙丙丁戊己", timings=[(index * 50, (index + 1) * 50) for index in range(6)]),
            audio_duration_ms=300,
        )


def test_render_srt_is_deterministic_hour_safe_and_rejects_boundary_injection() -> None:
    cues = (
        SubtitleCue(index=0, start_ms=3_600_001, end_ms=3_600_999, text="你好，世界！"),
        SubtitleCue(index=1, start_ms=3_601_000, end_ms=3_601_600, text="第二行\n仍是批准文本。"),
    )

    assert render_srt(cues) == (
        "1\n01:00:00,001 --> 01:00:00,999\n你好世界\n\n"
        "2\n01:00:01,000 --> 01:00:01,600\n第二行仍是批准文本\n"
    )
    with pytest.raises(SubtitleError, match="unsafe_srt_text"):
        render_srt((SubtitleCue(index=0, start_ms=0, end_ms=600, text="安全\n\n2\n注入"),))


def test_render_srt_rejects_manual_cues_with_more_than_two_source_lines(tmp_path: Path) -> None:
    cue = (SubtitleCue(index=0, start_ms=0, end_ms=600, text="第一行\n第二行\n第三行"),)

    with pytest.raises(SubtitleError, match="unsafe_srt_text"):
        render_srt(cue)


def test_render_ass_collapses_manual_multiline_cues_to_one_line(tmp_path: Path) -> None:
    cue = (SubtitleCue(index=0, start_ms=0, end_ms=600, text="第一行\n第二行\n第三行"),)
    font_path, font_hash = _font(tmp_path)

    rendered = render_ass(cue, font_path=font_path, font_sha256=font_hash, font_family="Test")
    dialogue = next(line for line in rendered.splitlines() if line.startswith("Dialogue:"))

    assert dialogue.endswith("第一行第二行第三行")
    assert "\\N" not in dialogue


def test_renderers_reject_captions_too_wide_for_one_readable_line(tmp_path: Path) -> None:
    cue = (
        SubtitleCue(
            index=0,
            start_ms=0,
            end_ms=1_000,
            text="甲乙丙丁戊己庚辛壬癸子丑寅卯辰",
        ),
    )
    font_path, font_hash = _font(tmp_path)

    with pytest.raises(SubtitleError, match="subtitle_too_wide_for_single_line"):
        render_srt(cue)
    with pytest.raises(SubtitleError, match="subtitle_too_wide_for_single_line"):
        render_ass(
            cue,
            font_path=font_path,
            font_sha256=font_hash,
            font_family="Test",
        )


def test_render_ass_uses_live_template_safe_area_and_single_line_punctuation_free_text(tmp_path: Path) -> None:
    font_path, font_hash = _font(tmp_path)
    cues = (SubtitleCue(index=0, start_ms=0, end_ms=605, text="《我与地坛》：你好，世界！\n第二行。"),)

    rendered = render_ass(cues, font_path=font_path, font_sha256=font_hash, font_family="Test Subtitle Font")

    assert rendered.startswith("[Script Info]\n")
    assert "BV_TASK17_SUBTITLE_TEMPLATE_V1" in rendered
    assert "PlayResX: 1080" in rendered and "PlayResY: 1920" in rendered
    assert "Alignment,2" in rendered and "MarginV,420" in rendered
    assert "Test Subtitle Font" in rendered
    assert (
        "Style: BV_Task17_Default,Test Subtitle Font,68,&H00FFFFFF,"
        "&H000000FF,&H00101010,&HFF000000,-1,0,0,0,100,100,0,0,1,5,0,2,84,84,420,1"
        in rendered
    )
    assert "0:00:00.00,0:00:00.61" in rendered
    dialogue = next(line for line in rendered.splitlines() if line.startswith("Dialogue:"))
    assert dialogue.endswith("我与地坛你好世界第二行")
    assert "\\N" not in dialogue
    assert not any(mark in dialogue.rsplit(",,", 1)[-1] for mark in "《》：，！。")


def test_render_ass_quantizes_adjacent_manual_cues_without_centisecond_overlap(tmp_path: Path) -> None:
    font_path, font_hash = _font(tmp_path)
    cues = (
        SubtitleCue(index=0, start_ms=0, end_ms=605, text="第一条"),
        SubtitleCue(index=1, start_ms=606, end_ms=1_200, text="第二条"),
    )

    rendered = render_ass(cues, font_path=font_path, font_sha256=font_hash, font_family="Test")

    dialogue = [line for line in rendered.splitlines() if line.startswith("Dialogue:")]
    assert dialogue[0].startswith("Dialogue: 0,0:00:00.00,0:00:00.61,")
    assert dialogue[1].startswith("Dialogue: 0,0:00:00.61,0:00:01.20,")


def test_render_ass_handles_one_millisecond_gap_or_rejects_unrepresentable_order(tmp_path: Path) -> None:
    font_path, font_hash = _font(tmp_path)
    representable = (
        SubtitleCue(index=0, start_ms=0, end_ms=1, text="甲"),
        SubtitleCue(index=1, start_ms=2, end_ms=3, text="乙"),
    )

    rendered = render_ass(representable, font_path=font_path, font_sha256=font_hash, font_family="Test")
    dialogue = [line for line in rendered.splitlines() if line.startswith("Dialogue:")]
    assert dialogue[0].startswith("Dialogue: 0,0:00:00.00,0:00:00.01,")
    assert dialogue[1].startswith("Dialogue: 0,0:00:00.01,0:00:00.02,")

    unrepresentable = representable + (
        SubtitleCue(index=2, start_ms=3, end_ms=4, text="丙"),
    )
    with pytest.raises(SubtitleError, match="invalid_ass_timing"):
        render_ass(unrepresentable, font_path=font_path, font_sha256=font_hash, font_family="Test")


def test_render_ass_requires_verified_regular_font_and_rejects_reparse_paths(tmp_path: Path) -> None:
    font_path, font_hash = _font(tmp_path)
    cue = (SubtitleCue(index=0, start_ms=0, end_ms=600, text="批准文本。"),)

    with pytest.raises(SubtitleError, match="font_hash_mismatch"):
        render_ass(cue, font_path=font_path, font_sha256="d" * 64, font_family="Test")
    with pytest.raises(SubtitleError, match="invalid_font_file"):
        render_ass(cue, font_path=font_path, font_sha256=font_hash, font_family="Bad\nName")
    link = tmp_path / "font-link.ttf"
    try:
        link.symlink_to(font_path)
    except OSError:
        pytest.skip("Windows symlink creation is unavailable for this test account")
    with pytest.raises(SubtitleError, match="unsafe_font_path"):
        render_ass(cue, font_path=link, font_sha256=font_hash, font_family="Test")


def test_render_ass_accepts_verified_true_type_collection(tmp_path: Path) -> None:
    font_path = tmp_path / "subtitle.ttc"
    font_bytes = b"ttcf\x00\x02\x00\x00Task17 test font collection"
    font_path.write_bytes(font_bytes)
    cue = (SubtitleCue(index=0, start_ms=0, end_ms=600, text="批准文本。"),)

    rendered = render_ass(
        cue,
        font_path=font_path,
        font_sha256=_sha(font_bytes),
        font_family="Microsoft YaHei",
    )

    assert "Style: BV_Task17_Default,Microsoft YaHei" in rendered


def test_render_ass_uses_microsoft_yahei_without_a_background_box(
    tmp_path: Path,
) -> None:
    font_path = tmp_path / "subtitle.ttc"
    font_bytes = b"ttcf\x00\x02\x00\x00Task17 test font collection"
    font_path.write_bytes(font_bytes)

    rendered = render_ass(
        (SubtitleCue(index=0, start_ms=0, end_ms=600, text="批准文本。"),),
        font_path=font_path,
        font_sha256=_sha(font_bytes),
        font_family="Microsoft YaHei",
    )
    style = next(
        line.removeprefix("Style: ").split(",")
        for line in rendered.splitlines()
        if line.startswith("Style: BV_Task17_Default,")
    )

    assert style[1] == "Microsoft YaHei"
    assert style[6] == "&HFF000000"
    assert style[15] == "1"
    assert int(style[16]) > 0
    assert style[17] == "0"
    assert style[18] == "2"


def test_render_ass_rejects_near_miss_true_type_collection_signature(
    tmp_path: Path,
) -> None:
    font_path = tmp_path / "subtitle.ttc"
    font_bytes = b"ttcF\x00\x02\x00\x00Not a valid font collection"
    font_path.write_bytes(font_bytes)

    with pytest.raises(SubtitleError, match="invalid_font_file"):
        render_ass(
            (SubtitleCue(index=0, start_ms=0, end_ms=600, text="批准文本。"),),
            font_path=font_path,
            font_sha256=_sha(font_bytes),
            font_family="Microsoft YaHei",
        )


@pytest.mark.parametrize("event_type", ["Dialogue", "Comment", "Picture", "Sound", "Movie", "Command"])
def test_render_ass_rejects_hashed_templates_with_preseeded_events(tmp_path: Path, event_type: str) -> None:
    font_path, font_hash = _font(tmp_path)
    template_path = tmp_path / f"injected-{event_type}.ass"
    injected = _template_bytes() + f"{event_type}: 0,0:00:00.00,0:00:09.99,INJECTED\n".encode("utf-8")
    template_path.write_bytes(injected)
    cues = (SubtitleCue(index=0, start_ms=0, end_ms=600, text="批准文本。"),)

    with pytest.raises(SubtitleError, match="invalid_style_template"):
        render_ass(
            cues,
            font_path=font_path,
            font_sha256=font_hash,
            font_family="Test",
            style_template_path=template_path,
            style_template_sha256=_sha(injected),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        b"\n[Events]\n",
        b"\nStyle: BV_Task17_Default,__FONT_FAMILY__,72,&H00FFFFFF\n",
        b"\n[Malformed Section\n",
        b"\n; __UNREPLACED_PLACEHOLDER__\n",
        b"\n; __lower_placeholder__\n",
        b"\n; unsafe\x00control\n",
        b"\r; bare-carriage-return",
    ],
)
def test_render_ass_rejects_malformed_or_unsafe_hashed_templates(tmp_path: Path, mutation: bytes) -> None:
    font_path, font_hash = _font(tmp_path)
    template_path = tmp_path / "malformed.ass"
    malformed = _template_bytes() + mutation
    template_path.write_bytes(malformed)

    with pytest.raises(SubtitleError, match="invalid_style_template"):
        render_ass(
            (SubtitleCue(index=0, start_ms=0, end_ms=600, text="批准文本。"),),
            font_path=font_path,
            font_sha256=font_hash,
            font_family="Test",
            style_template_path=template_path,
            style_template_sha256=_sha(malformed),
        )


def test_font_and_template_reads_are_bounded(tmp_path: Path) -> None:
    cue = (SubtitleCue(index=0, start_ms=0, end_ms=600, text="批准文本。"),)
    normal_font, normal_font_hash = _font(tmp_path)
    oversized_template = _template_bytes() + b";" + b"x" * (256 * 1_024)
    template_path = tmp_path / "oversized.ass"
    template_path.write_bytes(oversized_template)

    with pytest.raises(SubtitleError, match="style_template_too_large"):
        render_ass(
            cue,
            font_path=normal_font,
            font_sha256=normal_font_hash,
            font_family="Test",
            style_template_path=template_path,
            style_template_sha256=_sha(oversized_template),
        )

    oversized_font = b"\x00\x01\x00\x00" + b"x" * (32 * 1_024 * 1_024)
    font_path = tmp_path / "oversized.ttf"
    font_path.write_bytes(oversized_font)
    with pytest.raises(SubtitleError, match="font_file_too_large"):
        render_ass(
            cue,
            font_path=font_path,
            font_sha256=_sha(oversized_font),
            font_family="Test",
        )


def test_build_manifest_binds_artifacts_without_leaking_text_or_paths(tmp_path: Path) -> None:
    aligned = _aligned("批准文本必须保密。")
    cues = build_cues(aligned, min_duration_ms=100)
    font_path, font_hash = _font(tmp_path)

    manifest = build_subtitle_manifest(
        aligned,
        cues,
        audio_duration_ms=2_000,
        font_path=font_path,
        font_sha256=font_hash,
        font_family="Private Font",
    )

    serialized = manifest.model_dump_json()
    assert manifest.approved_sha256 == _sha("批准文本必须保密。")
    assert manifest.audio_sha256 == "a" * 64 and manifest.asr_sha256 == "b" * 64
    assert manifest.cue_count == len(cues) and manifest.duration_ms == 2_000
    assert "批准文本" not in serialized
    assert str(font_path) not in serialized
    assert "Private Font" in serialized and font_hash in serialized


def test_build_manifest_requires_exact_approved_text_reconstruction(tmp_path: Path) -> None:
    aligned = _aligned(" 批准 文本。 ")
    font_path, font_hash = _font(tmp_path)
    valid = (SubtitleCue(index=0, start_ms=0, end_ms=900, text="批准\n文本。"),)

    manifest = build_subtitle_manifest(
        aligned,
        valid,
        audio_duration_ms=1_000,
        font_path=font_path,
        font_sha256=font_hash,
        font_family="Test",
    )
    assert manifest.cue_count == 1

    invalid_texts = ("完全错误", "批准文本", "批准文本。。")
    for invalid_text in invalid_texts:
        cues = (SubtitleCue(index=0, start_ms=0, end_ms=900, text=invalid_text),)
        with pytest.raises(SubtitleError, match="subtitle_text_mismatch") as error:
            build_subtitle_manifest(
                aligned,
                cues,
                audio_duration_ms=1_000,
                font_path=font_path,
                font_sha256=font_hash,
                font_family="Test",
            )
        assert str(error.value) == "subtitle_text_mismatch"


def test_build_manifest_rejects_empty_or_nonzero_based_cues(tmp_path: Path) -> None:
    aligned = _aligned("批准文本。")
    font_path, font_hash = _font(tmp_path)
    arguments = {
        "audio_duration_ms": 1_000,
        "font_path": font_path,
        "font_sha256": font_hash,
        "font_family": "Test",
    }

    with pytest.raises(SubtitleError, match="subtitle_text_mismatch"):
        build_subtitle_manifest(aligned, (), **arguments)
    with pytest.raises(SubtitleError, match="invalid_cue_timing"):
        build_subtitle_manifest(
            aligned,
            (SubtitleCue(index=1, start_ms=0, end_ms=600, text="批准文本。"),),
            **arguments,
        )


def test_public_models_forbid_unexpected_fields() -> None:
    with pytest.raises(Exception):
        SubtitleCue(index=0, start_ms=0, end_ms=600, text="文本", unexpected=True)
