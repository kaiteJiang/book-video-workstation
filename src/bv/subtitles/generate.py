"""Generate deterministic subtitle artifacts from approved, aligned text only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
import unicodedata
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from bv.asr.alignment import AlignedCharacter, AlignedScript


_DEFAULT_TEMPLATE = Path(__file__).resolve().parents[3] / "styles" / "subtitle-default.ass"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_STRONG_PUNCTUATION = frozenset("。！？!?…")
_SOFT_PUNCTUATION = frozenset("，、；,:;")
_PAUSE_BOUNDARY_MS = 250
_MAX_FONT_BYTES = 32 * 1_024 * 1_024
_MAX_STYLE_TEMPLATE_BYTES = 256 * 1_024
_MAX_SINGLE_LINE_DISPLAY_CHARACTERS = 14
_PLACEHOLDER = re.compile(r"__[A-Za-z0-9][A-Za-z0-9_]*__")
_SCRIPT_INFO_LINES = (
    "ScriptType: v4.00+",
    "PlayResX: __PLAY_RES_X__",
    "PlayResY: __PLAY_RES_Y__",
    "WrapStyle: 2",
    "ScaledBorderAndShadow: yes",
)
_STYLE_FORMAT = (
    "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
    "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
    "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding"
)
_EVENT_FORMAT = "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"


class SubtitleError(RuntimeError):
    """A stable, privacy-safe subtitle generation failure."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class SubtitleCue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str


class SubtitleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approved_sha256: str
    audio_sha256: str
    asr_sha256: str
    cue_content_sha256: str
    style_template_sha256: str
    font_family: str
    font_sha256: str
    width: int
    height: int
    cue_count: int
    duration_ms: int


class _CueGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    units: tuple[AlignedCharacter, ...]
    text: str
    start_ms: int
    end_ms: int


def build_cues(
    aligned: AlignedScript,
    *,
    audio_duration_ms: int | None = None,
    protected_terms: tuple[str, ...] = (),
    semantic_chunks: tuple[str, ...] = (),
    min_duration_ms: int = 600,
) -> tuple[SubtitleCue, ...]:
    """Build timed readable cues using only Task 14 approved text and Task 16 timing."""

    characters, duration_ms = _validate_aligned(aligned, audio_duration_ms)
    if not isinstance(min_duration_ms, int) or isinstance(min_duration_ms, bool) or min_duration_ms <= 0:
        raise SubtitleError("invalid_minimum_duration")
    visible = [character for character in characters if not character.character.isspace()]
    if not visible:
        raise SubtitleError("missing_visible_approved_text")
    protected_spans = _protected_spans(aligned.approved_text, protected_terms)
    groups = (
        _groups_from_semantic_chunks(visible, semantic_chunks, protected_spans)
        if semantic_chunks
        else _segment(visible, protected_spans)
    )
    allocated = _allocate_durations(
        groups,
        duration_ms,
        min_duration_ms,
        protected_spans,
        preserve_boundaries=bool(semantic_chunks),
    )
    return tuple(
        SubtitleCue(index=index, start_ms=group.start_ms, end_ms=group.end_ms, text=group.text)
        for index, group in enumerate(allocated)
    )


def _groups_from_semantic_chunks(
    visible: list[AlignedCharacter],
    semantic_chunks: tuple[str, ...],
    protected_spans: tuple[tuple[int, int], ...],
) -> list[_CueGroup]:
    normalized = tuple(
        "".join(character for character in chunk if not character.isspace())
        for chunk in semantic_chunks
        if isinstance(chunk, str) and chunk.strip()
    )
    source = "".join(character.character for character in visible)
    if len(normalized) != len(semantic_chunks) or "".join(normalized) != source:
        raise SubtitleError("semantic_chunks_text_mismatch")

    groups: list[_CueGroup] = []
    offset = 0
    for chunk in normalized:
        end = offset + len(chunk)
        groups.append(_make_group(visible[offset:end], protected_spans))
        offset = end
    return groups


def render_srt(cues: Sequence[SubtitleCue]) -> str:
    """Render deterministic UTF-8 SRT text without filesystem side effects."""

    _validate_cues(cues)
    blocks: list[str] = []
    for number, cue in enumerate(cues, start=1):
        _validate_srt_text(cue.text)
        display_text = _subtitle_display_text(cue.text)
        blocks.append(
            f"{number}\n{_srt_time(cue.start_ms)} --> {_srt_time(cue.end_ms)}\n{display_text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_ass(
    cues: Sequence[SubtitleCue],
    *,
    font_path: Path,
    font_sha256: str,
    font_family: str,
    width: int = 1080,
    height: int = 1920,
    style_template_path: Path | None = None,
    style_template_sha256: str | None = None,
) -> str:
    """Render a complete ASS v4+ subtitle document from a verified local font."""

    _validate_cues(cues)
    quantized_intervals = _quantize_ass_intervals(cues)
    _validate_frame_size(width, height)
    _validate_font(font_path, font_sha256, font_family)
    template, _ = _load_template(style_template_path, style_template_sha256)
    document = template.replace("__FONT_FAMILY__", font_family).replace("__PLAY_RES_X__", str(width)).replace("__PLAY_RES_Y__", str(height))
    if _PLACEHOLDER.search(document):
        raise SubtitleError("invalid_style_template")
    lines = [document.rstrip("\r\n")]
    for cue, (start, end) in zip(cues, quantized_intervals, strict=True):
        display_text = _subtitle_display_text(cue.text)
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},"
            f"BV_Task17_Default,,0,0,0,,{_escape_ass_text(display_text)}"
        )
    return "\n".join(lines) + "\n"


def _subtitle_display_text(text: str) -> str:
    without_punctuation = "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith("P")
    )
    single_line = "".join(without_punctuation.splitlines())
    display_text = re.sub(r"[^\S\r\n]+", " ", single_line).strip()
    if sum(not character.isspace() for character in display_text) > _MAX_SINGLE_LINE_DISPLAY_CHARACTERS:
        raise SubtitleError("subtitle_too_wide_for_single_line")
    return display_text


def build_subtitle_manifest(
    aligned: AlignedScript,
    cues: Sequence[SubtitleCue],
    *,
    audio_duration_ms: int,
    font_path: Path,
    font_sha256: str,
    font_family: str,
    width: int = 1080,
    height: int = 1920,
    style_template_path: Path | None = None,
    style_template_sha256: str | None = None,
) -> SubtitleManifest:
    """Return a hash-bound, privacy-safe manifest; it contains no subtitle text or paths."""

    _, duration_ms = _validate_aligned(aligned, audio_duration_ms)
    _validate_cues(cues, duration_ms=duration_ms)
    _validate_manifest_text(aligned.approved_text, cues)
    _validate_frame_size(width, height)
    _validate_font(font_path, font_sha256, font_family)
    _, template_hash = _load_template(style_template_path, style_template_sha256)
    payload = [
        {"index": cue.index, "start_ms": cue.start_ms, "end_ms": cue.end_ms, "text": cue.text}
        for cue in cues
    ]
    cue_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SubtitleManifest(
        approved_sha256=aligned.report.approved_sha256,
        audio_sha256=aligned.report.audio_sha256,
        asr_sha256=aligned.report.asr_sha256,
        cue_content_sha256=cue_hash,
        style_template_sha256=template_hash,
        font_family=font_family,
        font_sha256=font_sha256,
        width=width,
        height=height,
        cue_count=len(cues),
        duration_ms=duration_ms,
    )


def _validate_aligned(
    aligned: AlignedScript, audio_duration_ms: int | None,
) -> tuple[tuple[AlignedCharacter, ...], int]:
    if aligned.report.level != "pass":
        raise SubtitleError("pronunciation_not_pass")
    if aligned.report.violations:
        raise SubtitleError("pronunciation_violations")
    if not _is_sha256(aligned.report.approved_sha256):
        raise SubtitleError("invalid_alignment_hash")
    if hashlib.sha256(aligned.approved_text.encode("utf-8")).hexdigest() != aligned.report.approved_sha256:
        raise SubtitleError("approved_hash_mismatch")
    if not _is_sha256(aligned.report.audio_sha256) or not _is_sha256(aligned.report.asr_sha256):
        raise SubtitleError("invalid_alignment_hash")
    if not aligned.approved_text or len(aligned.characters) != len(aligned.approved_text):
        raise SubtitleError("approved_text_mismatch")
    if "".join(item.character for item in aligned.characters) != aligned.approved_text:
        raise SubtitleError("approved_text_mismatch")
    for index, item in enumerate(aligned.characters):
        if item.index != index:
            raise SubtitleError("approved_text_mismatch")
    latest_end = 0
    previous_start = -1
    for item in aligned.characters:
        if item.start_ms is None or item.end_ms is None:
            raise SubtitleError("missing_alignment_timing")
        if item.start_ms < 0 or item.end_ms <= item.start_ms or item.start_ms < previous_start:
            raise SubtitleError("invalid_alignment_timing")
        previous_start = item.start_ms
        latest_end = max(latest_end, item.end_ms)
    duration = latest_end if audio_duration_ms is None else audio_duration_ms
    if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
        raise SubtitleError("invalid_audio_duration")
    if latest_end > duration:
        raise SubtitleError("invalid_alignment_timing")
    return aligned.characters, duration


def _protected_spans(text: str, protected_terms: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    unique_terms = sorted(set(protected_terms), key=lambda value: (-len(value), value))
    for term in unique_terms:
        if not isinstance(term, str) or not term or term not in text:
            raise SubtitleError("invalid_protected_term")
        offset = 0
        while True:
            start = text.find(term, offset)
            if start < 0:
                break
            spans.append((start, start + len(term)))
            offset = start + 1
    return tuple(sorted(set(spans)))


def _segment(
    visible: list[AlignedCharacter], protected_spans: tuple[tuple[int, int], ...],
) -> list[_CueGroup]:
    groups: list[_CueGroup] = []
    start = 0
    while start < len(visible):
        end = _choose_end(visible, start, protected_spans)
        groups.append(_make_group(visible[start:end], protected_spans))
        start = end
    return groups


def _choose_end(
    visible: list[AlignedCharacter], start: int, protected_spans: tuple[tuple[int, int], ...],
) -> int:
    total = len(visible)
    remaining = total - start
    if remaining <= 14:
        return total
    upper = min(total, start + 14)
    upper = _extend_past_protected_term(visible, upper, protected_spans)
    candidates = [
        end for end in range(start + 6, upper + 1)
        if end == total or _safe_break(visible, end, protected_spans)
    ]
    if not candidates:
        candidates = [
            end for end in range(start + 1, upper + 1)
            if end == total or _safe_break(visible, end, protected_spans)
        ]
    if not candidates:
        raise SubtitleError("protected_term_prevents_segmentation")
    selected = min(candidates, key=lambda end: _boundary_score(visible, start, end))
    if 0 < total - selected < 6:
        balanced = start + (remaining + 1) // 2
        if _safe_break(visible, balanced, protected_spans):
            return balanced
    return selected


def _extend_past_protected_term(
    visible: list[AlignedCharacter], upper: int, protected_spans: tuple[tuple[int, int], ...],
) -> int:
    limit = upper
    changed = True
    while changed:
        changed = False
        left_index = visible[limit - 1].index
        for start, end in protected_spans:
            if start <= left_index < end - 1:
                for position, character in enumerate(visible):
                    if character.index >= end:
                        if position > limit:
                            limit = position
                            changed = True
                        break
                else:
                    limit = len(visible)
    return limit


def _boundary_score(visible: list[AlignedCharacter], start: int, end: int) -> tuple[int, int, int]:
    previous = visible[end - 1]
    following = visible[end] if end < len(visible) else None
    if previous.character in _STRONG_PUNCTUATION:
        kind = 0
    elif previous.character in _SOFT_PUNCTUATION:
        kind = 1
    elif following is not None and following.start_ms - previous.end_ms >= _PAUSE_BOUNDARY_MS:
        kind = 2
    else:
        kind = 3
    return kind, abs((end - start) - 10), -end


def _safe_break(
    visible: list[AlignedCharacter], end: int, protected_spans: tuple[tuple[int, int], ...],
) -> bool:
    if end <= 0 or end >= len(visible):
        return end == len(visible)
    left = visible[end - 1].index
    right = visible[end].index
    return not any(start <= left and right < finish for start, finish in protected_spans)


def _make_group(units: Iterable[AlignedCharacter], protected_spans: tuple[tuple[int, int], ...]) -> _CueGroup:
    values = tuple(units)
    if not values:
        raise SubtitleError("empty_subtitle_cue")
    timing_values = tuple(value for value in values if value.relation != "anchor") or values
    return _CueGroup(
        units=values,
        text=_display_text(values, protected_spans),
        start_ms=(
            timing_values[0].start_ms
            if timing_values[0].start_ms is not None
            else -1
        ),
        end_ms=(
            timing_values[-1].end_ms
            if timing_values[-1].end_ms is not None
            else -1
        ),
    )


def _display_text(units: tuple[AlignedCharacter, ...], protected_spans: tuple[tuple[int, int], ...]) -> str:
    if len(units) < 9:
        return "".join(item.character for item in units)
    midpoint = len(units) // 2
    candidates = [
        position for position in range(1, len(units))
        if _safe_break(list(units), position, protected_spans)
    ]
    if not candidates:
        return "".join(item.character for item in units)
    split = min(candidates, key=lambda position: (abs(position - midpoint), position))
    return "".join(item.character for item in units[:split]) + "\n" + "".join(item.character for item in units[split:])


def _allocate_durations(
    groups: list[_CueGroup], duration_ms: int, min_duration_ms: int,
    protected_spans: tuple[tuple[int, int], ...],
    *,
    preserve_boundaries: bool = False,
) -> list[_CueGroup]:
    values = list(groups)
    index = 0
    while index < len(values):
        group = values[index]
        if group.end_ms - group.start_ms >= min_duration_ms:
            index += 1
            continue
        if preserve_boundaries:
            raise SubtitleError("insufficient_cue_duration")
        if index + 1 < len(values) and _may_merge(group, values[index + 1]):
            values[index:index + 2] = [_make_group(group.units + values[index + 1].units, protected_spans)]
            continue
        if index > 0 and _may_merge(values[index - 1], group):
            values[index - 1:index + 1] = [_make_group(values[index - 1].units + group.units, protected_spans)]
            index -= 1
            continue
        limit = values[index + 1].start_ms if index + 1 < len(values) else duration_ms
        proposed_end = min(group.start_ms + min_duration_ms, limit, duration_ms)
        if proposed_end - group.start_ms < min_duration_ms or proposed_end <= group.end_ms:
            raise SubtitleError("insufficient_cue_duration")
        values[index] = group.model_copy(update={"end_ms": proposed_end})
        index += 1
    _validate_group_timing(values, duration_ms)
    return values


def _may_merge(left: _CueGroup, right: _CueGroup) -> bool:
    return len(left.units) + len(right.units) <= 14


def _validate_group_timing(groups: Sequence[_CueGroup], duration_ms: int) -> None:
    previous_end = 0
    for group in groups:
        if group.start_ms < previous_end or group.end_ms <= group.start_ms or group.end_ms > duration_ms:
            raise SubtitleError("invalid_cue_timing")
        previous_end = group.end_ms


def _validate_cues(cues: Sequence[SubtitleCue], *, duration_ms: int | None = None) -> None:
    previous_end = 0
    for expected_index, cue in enumerate(cues):
        if cue.index != expected_index or cue.start_ms < previous_end or cue.end_ms <= cue.start_ms:
            raise SubtitleError("invalid_cue_timing")
        if duration_ms is not None and cue.end_ms > duration_ms:
            raise SubtitleError("invalid_cue_timing")
        previous_end = cue.end_ms


def _validate_manifest_text(approved_text: str, cues: Sequence[SubtitleCue]) -> None:
    approved_visible = "".join(character for character in approved_text if not character.isspace())
    if not approved_visible or not cues:
        raise SubtitleError("subtitle_text_mismatch")
    cue_parts: list[str] = []
    for cue in cues:
        _validate_srt_text(cue.text)
        visible = "".join(character for character in cue.text if not character.isspace())
        if not visible:
            raise SubtitleError("subtitle_text_mismatch")
        cue_parts.append(visible)
    if "".join(cue_parts) != approved_visible:
        raise SubtitleError("subtitle_text_mismatch")


def _validate_srt_text(text: str) -> None:
    if (
        not isinstance(text, str) or "\r" in text or text.count("\n") > 1
        or "\n\n" in text or text.startswith("\n") or text.endswith("\n")
    ):
        raise SubtitleError("unsafe_srt_text")
    if any(ord(character) < 32 and character != "\n" for character in text):
        raise SubtitleError("unsafe_srt_text")


def _srt_time(value: int) -> str:
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _validate_font(font_path: Path, font_sha256: str, font_family: str) -> None:
    path = Path(font_path)
    if not _is_sha256(font_sha256):
        raise SubtitleError("font_hash_mismatch")
    if (
        not isinstance(font_family, str) or not font_family
        or any(value in font_family for value in ("\r", "\n", ","))
        or _PLACEHOLDER.search(font_family)
    ):
        raise SubtitleError("invalid_font_file")
    try:
        unsafe = (
            not path.is_file() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
            or path.is_symlink() or _redirect_in_existing_chain(path)
        )
    except OSError:
        unsafe = True
    if unsafe:
        raise SubtitleError("unsafe_font_path")
    data = _read_bounded_file(
        path,
        max_bytes=_MAX_FONT_BYTES,
        too_large_error="font_file_too_large",
        read_error="unsafe_font_path",
    )
    if not data.startswith((b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf")):
        raise SubtitleError("invalid_font_file")
    if hashlib.sha256(data).hexdigest() != font_sha256:
        raise SubtitleError("font_hash_mismatch")


def _load_template(path: Path | None, expected_hash: str | None) -> tuple[str, str]:
    template_path = _DEFAULT_TEMPLATE if path is None else Path(path)
    try:
        if not template_path.is_file() or template_path.is_symlink() or _redirect_in_existing_chain(template_path):
            raise SubtitleError("unsafe_style_template")
        raw = _read_bounded_file(
            template_path,
            max_bytes=_MAX_STYLE_TEMPLATE_BYTES,
            too_large_error="style_template_too_large",
            read_error="invalid_style_template",
        )
        text = raw.decode("utf-8")
    except SubtitleError:
        raise
    except (OSError, UnicodeDecodeError):
        raise SubtitleError("invalid_style_template") from None
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and (not _is_sha256(expected_hash) or expected_hash != digest):
        raise SubtitleError("style_template_hash_mismatch")
    normalized = _validate_template_structure(text)
    return normalized, digest


def _validate_template_structure(text: str) -> str:
    if "\r" in text.replace("\r\n", ""):
        raise SubtitleError("invalid_style_template")
    normalized = text.replace("\r\n", "\n")
    if any(ord(character) < 32 and character != "\n" for character in normalized):
        raise SubtitleError("invalid_style_template")
    placeholders = _PLACEHOLDER.findall(normalized)
    if sorted(placeholders) != ["__FONT_FAMILY__", "__PLAY_RES_X__", "__PLAY_RES_Y__"]:
        raise SubtitleError("invalid_style_template")
    if normalized.count("BV_TASK17_SUBTITLE_TEMPLATE_V1") != 1:
        raise SubtitleError("invalid_style_template")

    expected_sections = ("Script Info", "V4+ Styles", "Events")
    sections: list[str] = []
    content: dict[str, list[str]] = {name: [] for name in expected_sections}
    current: str | None = None
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") or line.endswith("]"):
            match = re.fullmatch(r"\[([^\[\]]+)\]", line)
            if match is None or match.group(1) not in content or match.group(1) in sections:
                raise SubtitleError("invalid_style_template")
            current = match.group(1)
            sections.append(current)
            continue
        if current is None:
            raise SubtitleError("invalid_style_template")
        content[current].append(line)
    if tuple(sections) != expected_sections:
        raise SubtitleError("invalid_style_template")
    if tuple(content["Script Info"]) != _SCRIPT_INFO_LINES:
        raise SubtitleError("invalid_style_template")
    if len(content["V4+ Styles"]) != 2 or content["V4+ Styles"][0] != _STYLE_FORMAT:
        raise SubtitleError("invalid_style_template")
    _validate_style_line(content["V4+ Styles"][1])
    if content["Events"] != [_EVENT_FORMAT]:
        raise SubtitleError("invalid_style_template")
    return normalized


def _validate_style_line(line: str) -> None:
    if not line.startswith("Style: "):
        raise SubtitleError("invalid_style_template")
    fields = line.removeprefix("Style: ").split(",")
    format_fields = _STYLE_FORMAT.removeprefix("Format: ").split(",")
    if (
        len(fields) != len(format_fields)
        or fields[0] != "BV_Task17_Default"
        or fields[1] != "__FONT_FAMILY__"
        or fields[18] != "2"
        or any(character in line for character in ("{", "}", "\\"))
    ):
        raise SubtitleError("invalid_style_template")


def _read_bounded_file(
    path: Path,
    *,
    max_bytes: int,
    too_large_error: str,
    read_error: str,
) -> bytes:
    try:
        with path.open("rb") as stream:
            value = stream.read(max_bytes + 1)
    except OSError:
        raise SubtitleError(read_error) from None
    if len(value) > max_bytes:
        raise SubtitleError(too_large_error)
    return value


def _validate_frame_size(width: int, height: int) -> None:
    if (
        not isinstance(width, int) or isinstance(width, bool) or width <= 0
        or not isinstance(height, int) or isinstance(height, bool) or height <= 0
    ):
        raise SubtitleError("invalid_frame_size")


def _quantize_ass_intervals(cues: Sequence[SubtitleCue]) -> tuple[tuple[int, int], ...]:
    intervals: list[tuple[int, int]] = []
    previous_end = 0
    for cue in cues:
        start = (cue.start_ms + 9) // 10
        end = max((cue.end_ms + 9) // 10, start + 1)
        if start < previous_end:
            raise SubtitleError("invalid_ass_timing")
        intervals.append((start, end))
        previous_end = end
    return tuple(intervals)


def _ass_time(centiseconds: int) -> str:
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{fraction:02d}"


def _escape_ass_text(text: str) -> str:
    _validate_srt_text(text)
    escaped: list[str] = []
    for character in text:
        if character == "\\":
            escaped.append("\\\\")
        elif character == "{":
            escaped.append("\\{")
        elif character == "}":
            escaped.append("\\}")
        elif character == "\n":
            escaped.append("\\N")
        else:
            escaped.append(character)
    return "".join(escaped)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _redirect_in_existing_chain(path: Path) -> bool:
    current = path
    while True:
        try:
            if current.exists() or current.is_symlink():
                attributes = current.stat(follow_symlinks=False).st_file_attributes
                if current.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                    return True
        except (AttributeError, OSError):
            if current.is_symlink():
                return True
        parent = current.parent
        if parent == current:
            return False
        current = parent
