"""Private Volcengine flash-ASR request helpers."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import stat
from typing import Callable, Mapping
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import wave

from pydantic import BaseModel, ConfigDict, Field, SecretStr


_RESOURCE_ID = "volc.bigasr.auc_turbo"
_FLASH_ENDPOINT = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
_MAX_AUDIO_BYTES = 20 * 1_024 * 1_024
_MAX_REQUEST_BYTES = 28 * 1_024 * 1_024
_MAX_RESPONSE_BYTES = 4 * 1_024 * 1_024
_MIN_MASTER_DURATION_MS = 45_000
_MAX_MASTER_DURATION_MS = 150_000


class VolcengineAsrError(RuntimeError):
    """A stable, privacy-safe ASR failure."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class VolcCredentials(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr | None = None
    app_key: SecretStr | None = None
    access_key: SecretStr | None = None


class FlashRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    audio_path: Path
    audio_sha256: str
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    endpoint: str = _FLASH_ENDPOINT


class WordTiming(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    start_time: int
    end_time: int
    confidence: float | None = None


class AsrResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    words: tuple[WordTiming, ...]
    duration_ms: int
    audio_sha256: str
    request_id: str
    response_sha256: str


Transport = Callable[..., tuple[int, Mapping[str, str], bytes]]


def build_headers(
    credentials: VolcCredentials, *, request_id: str, resource_id: str = _RESOURCE_ID
) -> tuple[dict[str, str], frozenset[str]]:
    """Build official flash-ASR headers and the secret redaction set."""

    _validate_request_id(request_id)
    if resource_id != _RESOURCE_ID:
        raise VolcengineAsrError("invalid_resource_id")
    api_key = credentials.api_key.get_secret_value() if credentials.api_key is not None else None
    app_key = credentials.app_key.get_secret_value() if credentials.app_key is not None else None
    access_key = credentials.access_key.get_secret_value() if credentials.access_key is not None else None
    if isinstance(api_key, str) and api_key.strip() and app_key is None and access_key is None:
        auth_headers = {"X-Api-Key": api_key}
        secrets = frozenset({api_key})
    elif api_key is None and isinstance(app_key, str) and app_key.strip() and isinstance(access_key, str) and access_key.strip():
        auth_headers = {"X-Api-App-Key": app_key, "X-Api-Access-Key": access_key}
        secrets = frozenset({app_key, access_key})
    else:
        raise VolcengineAsrError("invalid_authentication")
    return (
        {
            "Content-Type": "application/json",
            **auth_headers,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": request_id,
            "X-Api-Sequence": "-1",
        },
        secrets,
    )


def build_flash_payload(request: FlashRequest) -> bytes:
    """Return the bounded in-memory official flash-ASR body."""

    body, _ = _build_flash_payload(request)
    return body


def _build_flash_payload(request: FlashRequest) -> tuple[bytes, int]:
    audio_bytes, duration_ms = _read_master_snapshot(request)
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    payload = {
        "user": {"uid": "bv-workstation"},
        "audio": {"format": "wav", "data": encoded_audio},
        "request": {"model_name": "bigmodel"},
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(body) > _MAX_REQUEST_BYTES:
        raise VolcengineAsrError("asr_request_too_large")
    return body, duration_ms


def parse_flash_response(
    *, status_code: int, headers: Mapping[str, str], body: bytes, audio_duration_ms: int,
    audio_sha256: str = "", request_id: str = "",
) -> AsrResult:
    """Validate and flatten the flash-ASR response without retaining raw payloads."""

    if not isinstance(status_code, int) or not 200 <= status_code < 300:
        raise VolcengineAsrError("asr_http_error")
    provider_status = _header(headers, "X-Api-Status-Code")
    if provider_status != "20000000":
        raise VolcengineAsrError("asr_provider_error")
    if not isinstance(audio_duration_ms, int) or audio_duration_ms <= 0:
        raise VolcengineAsrError("invalid_master_audio")
    if not isinstance(body, bytes):
        raise VolcengineAsrError("invalid_asr_response")
    if len(body) > _MAX_RESPONSE_BYTES:
        raise VolcengineAsrError("asr_response_too_large")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise VolcengineAsrError("invalid_asr_response") from None
    if not isinstance(payload, dict):
        raise VolcengineAsrError("invalid_asr_response")
    result = payload.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("text"), str) or not result["text"].strip():
        raise VolcengineAsrError("invalid_asr_response")
    utterances = result.get("utterances")
    if not isinstance(utterances, list):
        raise VolcengineAsrError("invalid_asr_response")
    words: list[WordTiming] = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            raise VolcengineAsrError("invalid_asr_response")
        raw_words = utterance.get("words")
        if not isinstance(raw_words, list):
            raise VolcengineAsrError("invalid_asr_response")
        for raw_word in raw_words:
            words.append(_parse_word(raw_word, audio_duration_ms, words[-1] if words else None))
    if not words:
        raise VolcengineAsrError("empty_word_timings")
    if _comparison_text("".join(word.text for word in words)) != _comparison_text(result["text"]):
        raise VolcengineAsrError("inconsistent_asr_text")
    return AsrResult(
        text=result["text"], words=tuple(words), duration_ms=audio_duration_ms,
        audio_sha256=audio_sha256, request_id=request_id,
        response_sha256=hashlib.sha256(body).hexdigest(),
    )


def recognize_flash(
    credentials: VolcCredentials, request: FlashRequest, *, transport: Transport | None = None,
    timeout: float = 30.0,
) -> AsrResult:
    """Submit one verified master WAV using an injected or stdlib HTTP transport."""

    headers, secrets = build_headers(credentials, request_id=request.request_id)
    _validate_endpoint(request.endpoint)
    body, duration_ms = _build_flash_payload(request)
    sender = _stdlib_transport if transport is None else transport
    try:
        status_code, response_headers, response_body = sender(
            url=request.endpoint, headers=headers, body=body, timeout=timeout, secrets=secrets,
        )
    except VolcengineAsrError:
        raise
    except Exception:
        raise VolcengineAsrError("asr_transport_error") from None
    _recheck_master_hash(request)
    return parse_flash_response(
        status_code=status_code, headers=response_headers, body=response_body,
        audio_duration_ms=duration_ms, audio_sha256=request.audio_sha256,
        request_id=request.request_id,
    )


def _parse_word(raw_word: object, duration_ms: int, previous: WordTiming | None) -> WordTiming:
    if not isinstance(raw_word, dict):
        raise VolcengineAsrError("invalid_word_timing")
    text = raw_word.get("text")
    start = raw_word.get("start_time")
    end = raw_word.get("end_time")
    confidence = raw_word.get("confidence")
    if (
        not isinstance(text, str) or not text.strip()
        or not isinstance(start, int) or isinstance(start, bool)
        or not isinstance(end, int) or isinstance(end, bool)
        or start < 0 or end <= start or end > duration_ms
        or previous is not None and start < previous.end_time
    ):
        raise VolcengineAsrError("invalid_word_timing")
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(float(confidence)):
            raise VolcengineAsrError("invalid_word_timing")
        confidence = float(confidence)
    return WordTiming(text=text, start_time=start, end_time=end, confidence=confidence)


def _read_master_snapshot(request: FlashRequest) -> tuple[bytes, int]:
    path = Path(request.audio_path)
    try:
        invalid_path = (
            not path.is_file() or path.is_symlink() or _redirect_in_existing_chain(path)
            or path.suffix.lower() != ".wav" or path.stat().st_size <= 0
        )
    except OSError:
        invalid_path = True
    if invalid_path:
        raise VolcengineAsrError("unsafe_master_audio")
    if not _is_sha256(request.audio_sha256):
        raise VolcengineAsrError("master_hash_mismatch")
    audio_bytes = _read_bounded_file(path, _MAX_AUDIO_BYTES, "master_audio_too_large")
    if hashlib.sha256(audio_bytes).hexdigest() != request.audio_sha256:
        raise VolcengineAsrError("master_hash_mismatch")
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as stream:
            if (
                stream.getcomptype() != "NONE" or stream.getframerate() != 48_000
                or stream.getnchannels() != 1 or stream.getsampwidth() != 2
                or stream.getnframes() <= 0
            ):
                raise VolcengineAsrError("invalid_master_audio")
            duration_ms = round(stream.getnframes() * 1_000 / stream.getframerate())
            if not _MIN_MASTER_DURATION_MS <= duration_ms <= _MAX_MASTER_DURATION_MS:
                raise VolcengineAsrError("invalid_master_audio")
            return audio_bytes, duration_ms
    except VolcengineAsrError:
        raise
    except (OSError, EOFError, wave.Error):
        raise VolcengineAsrError("invalid_master_audio") from None


def _read_bounded_file(path: Path, max_bytes: int, error_code: str) -> bytes:
    try:
        with Path(path).open("rb") as stream:
            value = stream.read(max_bytes + 1)
    except OSError:
        raise VolcengineAsrError("invalid_master_audio") from None
    if len(value) > max_bytes:
        raise VolcengineAsrError(error_code)
    return value


def _recheck_master_hash(request: FlashRequest) -> None:
    try:
        current = _read_bounded_file(Path(request.audio_path), _MAX_AUDIO_BYTES, "master_hash_changed")
    except VolcengineAsrError:
        raise VolcengineAsrError("master_hash_changed") from None
    if hashlib.sha256(current).hexdigest() != request.audio_sha256:
        raise VolcengineAsrError("master_hash_changed")


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https" or parsed.hostname != "openspeech.bytedance.com"
        or parsed.port is not None or parsed.path != "/api/v3/auc/bigmodel/recognize/flash"
        or parsed.query or parsed.fragment or parsed.username or parsed.password
    ):
        raise VolcengineAsrError("invalid_endpoint")


def _validate_request_id(request_id: str) -> None:
    try:
        uuid.UUID(request_id)
    except (AttributeError, ValueError, TypeError):
        raise VolcengineAsrError("invalid_request_id") from None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    return next((str(value) for key, value in headers.items() if key.lower() == name.lower()), None)


def _comparison_text(value: str) -> str:
    return "".join(
        character for character in unicodedata.normalize("NFKC", value).casefold()
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _redirect_in_existing_chain(path: Path) -> bool:
    current = Path(path)
    while True:
        if current.exists() or current.is_symlink():
            if _is_redirected(current):
                return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _is_redirected(path: Path) -> bool:
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        attributes = 0
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request: Request, fp: object, code: int, msg: str, headers: object, newurl: str) -> None:
        return None


def _stdlib_transport(
    *, url: str, headers: Mapping[str, str], body: bytes, timeout: float, secrets: frozenset[str],
) -> tuple[int, Mapping[str, str], bytes]:
    """Use stdlib HTTP without redirects; callers should inject a fake in tests."""

    del secrets
    try:
        response = build_opener(_NoRedirect()).open(Request(url, data=body, headers=dict(headers), method="POST"), timeout=timeout)
        with response:
            return response.status, dict(response.headers.items()), response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        return error.code, dict(error.headers.items()) if error.headers else {}, error.read(_MAX_RESPONSE_BYTES + 1)
    except (OSError, URLError, ValueError):
        raise VolcengineAsrError("asr_transport_error") from None
