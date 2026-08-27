from bv.core.redaction import redact_text


def test_redaction_hides_known_secret_values() -> None:
    text = "Authorization: Bearer abc123 X-Api-Key: secret456"

    redacted = redact_text(text, {"abc123", "secret456"})

    assert "abc123" not in redacted
    assert "secret456" not in redacted
    assert "<REDACTED>" in redacted


def test_redaction_handles_overlapping_and_empty_values() -> None:
    text = "prefix-super-secret-suffix"

    redacted = redact_text(text, {"secret", "super-secret", ""})

    assert redacted == "prefix-<REDACTED>-suffix"
