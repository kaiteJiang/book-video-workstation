from collections.abc import Iterable


def redact_text(text: str, secrets: Iterable[str]) -> str:
    redacted = text
    values = sorted({secret for secret in secrets if secret}, key=len, reverse=True)
    for secret in values:
        redacted = redacted.replace(secret, "<REDACTED>")
    return redacted
