"""Deterministic alignment of approved narration to private ASR word timing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict

from bv.asr.volcengine import AsrResult, WordTiming


class AlignmentError(RuntimeError):
    """A stable, privacy-safe alignment failure."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class AlignedCharacter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    character: str
    start_ms: int | None
    end_ms: int | None
    relation: Literal["match", "substitution", "deletion", "interpolated", "anchor"]


class PronunciationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    similarity: float
    level: Literal["pass", "review", "fail"]
    violations: tuple[str, ...]
    substitutions: int
    deletions: int
    insertions: int
    protected_terms: tuple[tuple[str, bool], ...]
    approved_sha256: str
    audio_sha256: str
    asr_sha256: str


class AlignedScript(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approved_text: str
    characters: tuple[AlignedCharacter, ...]
    report: PronunciationReport


def align_approved_text(
    approved_text: str, asr_result: AsrResult, *, audio_sha256: str,
    protected_terms: tuple[str, ...] = (),
) -> AlignedScript:
    """Align only timing evidence; returned text always remains the approved input."""

    if not isinstance(approved_text, str) or not approved_text:
        raise AlignmentError("invalid_approved_text")
    if not _is_sha256(audio_sha256) or not _is_sha256(asr_result.audio_sha256):
        raise AlignmentError("invalid_audio_hash")
    if not _is_sha256(asr_result.response_sha256):
        raise AlignmentError("invalid_response_hash")
    if not asr_result.words:
        raise AlignmentError("missing_word_timings")
    if asr_result.audio_sha256 != audio_sha256:
        raise AlignmentError("audio_hash_mismatch")
    _validate_timings(asr_result.words, asr_result.duration_ms)
    approved_units = _approved_units(approved_text)
    recognized_units = _timed_asr_units(asr_result.words)
    if not approved_units or not recognized_units:
        raise AlignmentError("missing_word_timings")
    operations = _edit_operations(
        [unit[1] for unit in approved_units], [unit[0] for unit in recognized_units],
    )
    substitutions = sum(operation == "substitution" for operation, _, _ in operations)
    deletions = sum(operation == "deletion" for operation, _, _ in operations)
    insertions = sum(operation == "insertion" for operation, _, _ in operations)
    distance = substitutions + deletions + insertions
    similarity = 1 - distance / max(len(approved_units), len(recognized_units))
    level: Literal["pass", "review", "fail"]
    if similarity >= 0.94:
        level = "pass"
    elif similarity >= 0.90:
        level = "review"
    else:
        level = "fail"
    assigned = _assign_comparison_units(approved_units, recognized_units, operations)
    _interpolate_deletions(assigned, approved_text, level)
    characters = _render_characters(approved_text, assigned)
    _validate_output_timings(characters, asr_result.duration_ms)
    violations, outcomes = _protected_violations(
        approved_text, asr_result.text, protected_terms, operations,
    )
    if violations:
        level = "fail"
    report = PronunciationReport(
        similarity=similarity, level=level, violations=tuple(violations),
        substitutions=substitutions, deletions=deletions, insertions=insertions,
        protected_terms=outcomes,
        approved_sha256=hashlib.sha256(approved_text.encode("utf-8")).hexdigest(),
        audio_sha256=audio_sha256,
        asr_sha256=hashlib.sha256(asr_result.text.encode("utf-8")).hexdigest(),
    )
    return AlignedScript(approved_text=approved_text, characters=tuple(characters), report=report)


def _comparison_characters(character: str) -> str:
    normalized = unicodedata.normalize("NFKC", character).casefold()
    return "".join(
        value for value in normalized
        if not value.isspace() and not unicodedata.category(value).startswith("P")
    )


def _approved_units(text: str) -> list[tuple[int, str]]:
    return [(index, value) for index, character in enumerate(text) for value in _comparison_characters(character)]


def _timed_asr_units(words: tuple[WordTiming, ...]) -> list[tuple[str, int, int]]:
    units: list[tuple[str, int, int]] = []
    for word in words:
        characters = [character for value in word.text for character in _comparison_characters(value)]
        if not characters:
            continue
        duration = word.end_time - word.start_time
        for index, character in enumerate(characters):
            start = word.start_time + duration * index // len(characters)
            end = word.start_time + duration * (index + 1) // len(characters)
            if end <= start:
                raise AlignmentError("invalid_alignment_timing")
            units.append((character, start, end))
    return units


def _edit_operations(approved: list[str], recognized: list[str]) -> list[tuple[str, int | None, int | None]]:
    rows, columns = len(approved), len(recognized)
    matrix = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        matrix[row][0] = row
    for column in range(1, columns + 1):
        matrix[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            replace = matrix[row - 1][column - 1] + (approved[row - 1] != recognized[column - 1])
            matrix[row][column] = min(replace, matrix[row - 1][column] + 1, matrix[row][column - 1] + 1)
    operations: list[tuple[str, int | None, int | None]] = []
    row, column = rows, columns
    while row or column:
        if row and column:
            cost = int(approved[row - 1] != recognized[column - 1])
            if matrix[row][column] == matrix[row - 1][column - 1] + cost:
                operations.append(("match" if cost == 0 else "substitution", row - 1, column - 1))
                row, column = row - 1, column - 1
                continue
        if row and matrix[row][column] == matrix[row - 1][column] + 1:
            operations.append(("deletion", row - 1, None))
            row -= 1
        else:
            operations.append(("insertion", None, column - 1))
            column -= 1
    operations.reverse()
    return operations


def _assign_comparison_units(
    approved: list[tuple[int, str]], recognized: list[tuple[str, int, int]],
    operations: list[tuple[str, int | None, int | None]],
) -> dict[int, list[tuple[str, int | None, int | None]]]:
    assigned: dict[int, list[tuple[str, int | None, int | None]]] = {}
    for relation, approved_index, recognized_index in operations:
        if approved_index is None:
            continue
        original_index = approved[approved_index][0]
        if recognized_index is None:
            assigned.setdefault(original_index, []).append((relation, None, None))
        else:
            _, start, end = recognized[recognized_index]
            assigned.setdefault(original_index, []).append((relation, start, end))
    return assigned


def _interpolate_deletions(
    assigned: dict[int, list[tuple[str, int | None, int | None]]], text: str,
    level: Literal["pass", "review", "fail"],
) -> None:
    comparable_indices = [index for index, character in enumerate(text) if _comparison_characters(character)]
    for position, index in enumerate(comparable_indices):
        values = assigned.get(index)
        if values is None or all(start is None or end is None for _, start, end in values):
            left = next((assigned.get(candidate) for candidate in reversed(comparable_indices[:position]) if _has_timing(assigned.get(candidate))), None)
            right = next((assigned.get(candidate) for candidate in comparable_indices[position + 1:] if _has_timing(assigned.get(candidate))), None)
            if level == "fail" or left is None or right is None:
                continue
            left_end = max(end for _, _, end in left if end is not None)
            right_start = min(start for _, start, _ in right if start is not None)
            if right_start <= left_end:
                continue
            assigned[index] = [("interpolated", left_end, right_start)]


def _has_timing(values: list[tuple[str, int | None, int | None]] | None) -> bool:
    return values is not None and any(start is not None and end is not None for _, start, end in values)


def _render_characters(text: str, assigned: dict[int, list[tuple[str, int | None, int | None]]]) -> list[AlignedCharacter]:
    rendered: list[AlignedCharacter] = []
    previous: AlignedCharacter | None = None
    for index, character in enumerate(text):
        values = assigned.get(index)
        if values:
            relation = "interpolated" if any(value[0] == "interpolated" for value in values) else values[0][0]
            starts = [start for _, start, _ in values if start is not None]
            ends = [end for _, _, end in values if end is not None]
            item = AlignedCharacter(
                index=index, character=character, start_ms=min(starts) if starts else None,
                end_ms=max(ends) if ends else None, relation=relation,
            )
        else:
            item = _anchored_character(index, character, previous)
        rendered.append(item)
        if item.start_ms is not None and item.end_ms is not None:
            previous = item
    for index in range(len(rendered) - 2, -1, -1):
        if rendered[index].start_ms is None and rendered[index].end_ms is None:
            following = rendered[index + 1]
            if following.start_ms is not None and following.end_ms is not None:
                rendered[index] = AlignedCharacter(
                    index=rendered[index].index, character=rendered[index].character,
                    start_ms=following.start_ms, end_ms=following.end_ms, relation="anchor",
                )
    return rendered


def _anchored_character(index: int, character: str, previous: AlignedCharacter | None) -> AlignedCharacter:
    if previous is None:
        return AlignedCharacter(index=index, character=character, start_ms=None, end_ms=None, relation="anchor")
    return AlignedCharacter(
        index=index, character=character, start_ms=previous.start_ms,
        end_ms=previous.end_ms, relation="anchor",
    )


def _validate_output_timings(characters: list[AlignedCharacter], duration_ms: int) -> None:
    previous_start = -1
    for character in characters:
        if character.start_ms is None or character.end_ms is None:
            if not character.character.isspace():
                raise AlignmentError("unusable_alignment_timing")
            continue
        if character.start_ms < previous_start or character.start_ms < 0 or character.end_ms <= character.start_ms or character.end_ms > duration_ms:
            raise AlignmentError("invalid_alignment_timing")
        previous_start = character.start_ms


def _validate_timings(words: tuple[WordTiming, ...], duration_ms: int) -> None:
    previous_end = 0
    if not isinstance(duration_ms, int) or duration_ms <= 0:
        raise AlignmentError("invalid_alignment_timing")
    for word in words:
        if word.start_time < previous_end or word.start_time < 0 or word.end_time <= word.start_time or word.end_time > duration_ms:
            raise AlignmentError("invalid_alignment_timing")
        previous_end = word.end_time


@dataclass(frozen=True)
class _SensitiveOccurrence:
    token: str
    start: int
    end: int


def _protected_violations(
    approved: str, recognized: str, protected_terms: tuple[str, ...],
    operations: list[tuple[str, int | None, int | None]],
) -> tuple[list[str], tuple[tuple[str, bool], ...]]:
    comparable_approved = _comparison_text(approved)
    comparable_recognized = _comparison_text(recognized)
    violations: list[str] = []
    outcomes: list[tuple[str, bool]] = []
    for term in protected_terms:
        normalized = _comparison_text(term)
        approved_occurrences = _literal_occurrences(comparable_approved, normalized, term)
        recognized_occurrences = _literal_occurrences(comparable_recognized, normalized, term)
        matches = (
            bool(normalized)
            and bool(approved_occurrences)
            and len(approved_occurrences) == len(recognized_occurrences)
            and all(_is_contiguous_match(occurrence, operations) for occurrence in approved_occurrences)
        )
        outcomes.append((term, matches))
        if not matches:
            _append_unique(violations, f"key_term_mismatch:{term}")

    approved_number_occurrences = _number_occurrences(approved)
    recognized_number_occurrences = _number_occurrences(recognized)
    approved_number_tokens = [occurrence.token for occurrence in approved_number_occurrences]
    recognized_number_tokens = [occurrence.token for occurrence in recognized_number_occurrences]
    if approved_number_tokens != recognized_number_tokens and [
        _canonical_number_token(occurrence.token)
        for occurrence in approved_number_occurrences
    ] == [
        _canonical_number_token(occurrence.token)
        for occurrence in recognized_number_occurrences
    ]:
        approved_number_occurrences = []
        recognized_number_occurrences = []
    approved_numbers = Counter(occurrence.token for occurrence in approved_number_occurrences)
    recognized_numbers = Counter(occurrence.token for occurrence in recognized_number_occurrences)
    for number in sorted(approved_numbers.keys() | recognized_numbers.keys()):
        if approved_numbers[number] != recognized_numbers[number] or any(
            occurrence.token == number and not _is_contiguous_match(occurrence, operations)
            for occurrence in approved_number_occurrences
        ):
            _append_unique(violations, f"number_mismatch:{number}")

    negation_tokens = ("没有", "不", "无", "未")
    approved_negations = _nonoverlapping_occurrences(comparable_approved, negation_tokens)
    recognized_negations = _nonoverlapping_occurrences(comparable_recognized, negation_tokens)
    if (
        Counter(occurrence.token for occurrence in approved_negations)
        != Counter(occurrence.token for occurrence in recognized_negations)
        or any(not _is_contiguous_match(occurrence, operations) for occurrence in approved_negations)
    ):
        violations.append("negation_mismatch")
    return violations, tuple(outcomes)


def _literal_occurrences(text: str, normalized: str, token: str) -> list[_SensitiveOccurrence]:
    if not normalized:
        return []
    occurrences: list[_SensitiveOccurrence] = []
    cursor = 0
    while True:
        start = text.find(normalized, cursor)
        if start < 0:
            return occurrences
        occurrences.append(_SensitiveOccurrence(token=token, start=start, end=start + len(normalized)))
        cursor = start + len(normalized)


def _nonoverlapping_occurrences(text: str, tokens: tuple[str, ...]) -> list[_SensitiveOccurrence]:
    occupied: set[int] = set()
    occurrences: list[_SensitiveOccurrence] = []
    normalized_tokens = sorted(
        ((_comparison_text(token), token) for token in tokens),
        key=lambda item: (-len(item[0]), item[1]),
    )
    for normalized, token in normalized_tokens:
        for occurrence in _literal_occurrences(text, normalized, token):
            span = set(range(occurrence.start, occurrence.end))
            if span.isdisjoint(occupied):
                occupied.update(span)
                occurrences.append(occurrence)
    return sorted(occurrences, key=lambda occurrence: (occurrence.start, occurrence.end, occurrence.token))


def _number_occurrences(text: str) -> list[_SensitiveOccurrence]:
    normalized, comparison_spans = _nfkc_with_comparison_spans(text)
    pattern = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[零〇一二两三四五六七八九十百千万亿]+")
    occurrences: list[_SensitiveOccurrence] = []
    for match in pattern.finditer(normalized):
        spans = [
            comparison_spans[index]
            for index in range(match.start(), match.end())
            if comparison_spans[index][1] > comparison_spans[index][0]
        ]
        if spans:
            occurrences.append(
                _SensitiveOccurrence(token=match.group(0), start=spans[0][0], end=spans[-1][1])
            )
    return occurrences


def _canonical_number_token(token: str) -> str:
    normalized = unicodedata.normalize("NFKC", token).replace(",", "")
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", normalized):
        return normalized
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    small_units = {"十": 10, "百": 100, "千": 1_000}
    large_units = {"万": 10_000, "亿": 100_000_000}
    # Digit-by-digit years and identifiers are not place-value expressions.
    # 一六〇一 is 1601, while 一千六百零一 follows the unit parser below.
    if normalized and all(character in digits for character in normalized):
        return "".join(str(digits[character]) for character in normalized)
    if not normalized or any(
        character not in digits and character not in small_units and character not in large_units
        for character in normalized
    ):
        return normalized
    total = 0
    section = 0
    number = 0
    for character in normalized:
        if character in digits:
            number = digits[character]
        elif character in small_units:
            unit = small_units[character]
            section += (number or 1) * unit
            number = 0
        else:
            section += number
            total += (section or 1) * large_units[character]
            section = 0
            number = 0
    return str(total + section + number)


def _nfkc_with_comparison_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    normalized_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    comparison_offset = 0
    for character in text:
        normalized = unicodedata.normalize("NFKC", character)
        normalized_parts.append(normalized)
        for normalized_character in normalized:
            comparable = _comparison_characters(normalized_character)
            start = comparison_offset
            comparison_offset += len(comparable)
            spans.append((start, comparison_offset))
    return "".join(normalized_parts), spans


def _is_contiguous_match(
    occurrence: _SensitiveOccurrence,
    operations: list[tuple[str, int | None, int | None]],
) -> bool:
    approved_map = {
        approved_index: (relation, recognized_index)
        for relation, approved_index, recognized_index in operations
        if approved_index is not None
    }
    recognized_indices: list[int] = []
    for approved_index in range(occurrence.start, occurrence.end):
        relation, recognized_index = approved_map.get(approved_index, ("", None))
        if relation != "match" or recognized_index is None:
            return False
        recognized_indices.append(recognized_index)
    return all(
        current == previous + 1
        for previous, current in zip(recognized_indices, recognized_indices[1:])
    )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _comparison_text(value: str) -> str:
    return "".join(character for character in value for character in _comparison_characters(character))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
