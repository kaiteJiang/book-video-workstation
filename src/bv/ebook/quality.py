from .models import SourceQualityReport


DEFAULT_MIN_COMPLETE_CHARS = 500
REPLACEMENT_CHARACTER_RATE_LIMIT = 0.005
REPEATED_WINDOW_CHARS = 200
REPEATED_WINDOW_LIMIT = 10
_FINGERPRINT_MASK = (1 << 64) - 1
_FINGERPRINT_BASE = 0x9E3779B185EBCA87
_BUCKET_BITS = 24
_BUCKET_MASK = (1 << _BUCKET_BITS) - 1


def assess_source_quality(
    text: str,
    *,
    min_complete_chars: int = DEFAULT_MIN_COMPLETE_CHARS,
) -> SourceQualityReport:
    """Return deterministic quality gates for a decoded ebook source."""

    non_whitespace = "".join(text.split())
    codes: list[str] = []
    if not non_whitespace:
        codes.append("empty_normalized_text")
    elif len(non_whitespace) < min_complete_chars:
        codes.append("too_short_for_complete_book")

    replacement_rate = text.count("\ufffd") / len(text) if text else 0.0
    if replacement_rate > REPLACEMENT_CHARACTER_RATE_LIMIT:
        codes.append("replacement_character_rate")

    if _has_repeated_window(text):
        codes.append("repeated_window")

    return SourceQualityReport(blocking=bool(codes), codes=codes)


def _has_repeated_window(text: str) -> bool:
    minimum_repeated_length = REPEATED_WINDOW_CHARS + REPEATED_WINDOW_LIMIT - 1
    if len(text) < minimum_repeated_length:
        return False

    buckets = bytearray(_BUCKET_MASK + 1)
    for fingerprint in _rolling_fingerprints(text):
        bucket = fingerprint & _BUCKET_MASK
        if buckets[bucket] < REPEATED_WINDOW_LIMIT:
            buckets[bucket] += 1

    for start, fingerprint in enumerate(_rolling_fingerprints(text)):
        if buckets[fingerprint & _BUCKET_MASK] < REPEATED_WINDOW_LIMIT:
            continue
        window = text[start : start + REPEATED_WINDOW_CHARS]
        if _occurs_at_least(text, window, REPEATED_WINDOW_LIMIT):
            return True
    return False


def _rolling_fingerprints(text: str):
    fingerprint = 0
    for character in text[:REPEATED_WINDOW_CHARS]:
        fingerprint = (
            fingerprint * _FINGERPRINT_BASE + ord(character)
        ) & _FINGERPRINT_MASK
    yield fingerprint

    outgoing_factor = pow(
        _FINGERPRINT_BASE,
        REPEATED_WINDOW_CHARS,
        _FINGERPRINT_MASK + 1,
    )
    for index in range(REPEATED_WINDOW_CHARS, len(text)):
        fingerprint = (
            fingerprint * _FINGERPRINT_BASE
            + ord(text[index])
            - ord(text[index - REPEATED_WINDOW_CHARS]) * outgoing_factor
        ) & _FINGERPRINT_MASK
        yield fingerprint


def _occurs_at_least(text: str, window: str, limit: int) -> bool:
    start = -1
    for _ in range(limit):
        start = text.find(window, start + 1)
        if start == -1:
            return False
    return True
