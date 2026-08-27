from __future__ import annotations

import base64
import hashlib
import json
import uuid
import wave
from pathlib import Path

import pytest

from bv.asr.volcengine import (
    AsrResult,
    FlashRequest,
    VolcCredentials,
    VolcengineAsrError,
    build_flash_payload,
    build_headers,
    parse_flash_response,
    recognize_flash,
)


def _write_master(path: Path, *, frames: int = 48_000 * 45) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x00\x00" * frames)


def _request(path: Path) -> FlashRequest:
    return FlashRequest(
        audio_path=path,
        audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        request_id="123e4567-e89b-12d3-a456-426614174000",
    )


@pytest.mark.parametrize(
    "credentials",
    [
        VolcCredentials(),
        VolcCredentials(api_key=""),
        VolcCredentials(app_key="legacy-app"),
        VolcCredentials(access_key="legacy-access"),
        VolcCredentials(api_key="new", app_key="legacy-app", access_key="legacy-access"),
        VolcCredentials(api_key="new", app_key="legacy-app"),
    ],
)
def test_build_headers_rejects_any_incomplete_or_mixed_auth_mode(credentials: VolcCredentials) -> None:
    with pytest.raises(VolcengineAsrError, match="invalid_authentication") as raised:
        build_headers(credentials, request_id="123e4567-e89b-12d3-a456-426614174000")

    assert str(raised.value) == "invalid_authentication"


def test_credentials_never_leak_any_auth_field_through_repr_or_serialization() -> None:
    credentials = VolcCredentials(
        api_key="TOP_SECRET_API", app_key="TOP_SECRET_APP", access_key="TOP_SECRET_ACCESS",
    )

    rendered = (
        repr(credentials), str(credentials), repr(credentials.model_dump()),
        str(credentials.model_dump(mode="json")), credentials.model_dump_json(),
    )

    for secret in ("TOP_SECRET_API", "TOP_SECRET_APP", "TOP_SECRET_ACCESS"):
        assert all(secret not in value for value in rendered)


def test_flash_request_uses_exact_official_headers_and_private_base64_audio(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    _write_master(master)
    request = _request(master)

    headers, secrets = build_headers(VolcCredentials(api_key="new-secret"), request_id=request.request_id)
    payload = json.loads(build_flash_payload(request).decode("utf-8"))

    assert headers == {
        "Content-Type": "application/json",
        "X-Api-Key": "new-secret",
        "X-Api-Resource-Id": "volc.bigasr.auc_turbo",
        "X-Api-Request-Id": request.request_id,
        "X-Api-Sequence": "-1",
    }
    assert secrets == frozenset({"new-secret"})
    assert payload == {
        "user": {"uid": "bv-workstation"},
        "audio": {"format": "wav", "data": base64.b64encode(master.read_bytes()).decode("ascii")},
        "request": {"model_name": "bigmodel"},
    }
    assert str(master) not in build_flash_payload(request).decode("utf-8")
    assert uuid.UUID(request.request_id).version is not None


def test_flash_request_accepts_full_length_book_video_narration(tmp_path: Path) -> None:
    master = tmp_path / "longform-master.wav"
    _write_master(master, frames=48_000 * 150)

    payload = json.loads(build_flash_payload(_request(master)).decode("utf-8"))

    assert payload["audio"]["format"] == "wav"
    assert base64.b64decode(payload["audio"]["data"]) == master.read_bytes()


def test_flash_request_uses_exact_legacy_auth_and_rejects_bad_uuid_resource_or_master(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    _write_master(master)
    request = _request(master)

    headers, secrets = build_headers(
        VolcCredentials(app_key="legacy-app", access_key="legacy-access"), request_id=request.request_id,
    )

    assert headers["X-Api-App-Key"] == "legacy-app"
    assert headers["X-Api-Access-Key"] == "legacy-access"
    assert "X-Api-Key" not in headers
    assert secrets == frozenset({"legacy-app", "legacy-access"})
    for request_id, resource_id, expected_code in [
        ("not-a-uuid", "volc.bigasr.auc_turbo", "invalid_request_id"),
        (request.request_id, "wrong-resource", "invalid_resource_id"),
    ]:
        with pytest.raises(VolcengineAsrError, match=expected_code):
            build_headers(VolcCredentials(api_key="secret"), request_id=request_id, resource_id=resource_id)

    _write_master(master, frames=48_000)
    with pytest.raises(VolcengineAsrError, match="invalid_master_audio"):
        recognize_flash(VolcCredentials(api_key="secret"), _request(master), transport=lambda **_: (_ for _ in ()).throw(AssertionError()))


def test_parse_flash_response_accepts_text_punctuation_but_rejects_inconsistent_or_nonfinite_words() -> None:
    punctuation = json.dumps({"result": {"text": "你，好！", "utterances": [{"words": [
        {"text": "你", "start_time": 0, "end_time": 100}, {"text": "好", "start_time": 100, "end_time": 200},
    ]}]}}).encode()
    parsed = parse_flash_response(
        status_code=200, headers={"X-Api-Status-Code": "20000000"}, body=punctuation, audio_duration_ms=200,
    )
    assert parsed.text == "你，好！"

    for body, code in [
        (json.dumps({"result": {"text": "你好", "utterances": [{"words": [{"text": "再见", "start_time": 0, "end_time": 100}]}]}}).encode(), "inconsistent_asr_text"),
        (json.dumps({"result": {"text": "你", "utterances": [{"words": [{"text": "你", "start_time": 0, "end_time": 100, "confidence": float("nan")}]}]}}).encode(), "invalid_word_timing"),
    ]:
        with pytest.raises(VolcengineAsrError, match=code):
            parse_flash_response(status_code=200, headers={"X-Api-Status-Code": "20000000"}, body=body, audio_duration_ms=200)


def test_request_building_rejects_invalid_master_and_untrusted_endpoint_before_transport(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    _write_master(master, frames=48_000)
    short_request = _request(master)
    called = False

    def transport(**kwargs: object) -> tuple[int, dict[str, str], bytes]:
        nonlocal called
        called = True
        return 200, {"X-Api-Status-Code": "20000000"}, _success_body()

    with pytest.raises(VolcengineAsrError, match="invalid_master_audio"):
        build_flash_payload(short_request)
    with pytest.raises(VolcengineAsrError, match="invalid_master_audio"):
        recognize_flash(VolcCredentials(api_key="secret"), short_request, transport=transport)
    assert called is False

    _write_master(master)
    hostile = _request(master).model_copy(update={"endpoint": "https://evil.example/api/v3/auc/bigmodel/recognize/flash"})
    with pytest.raises(VolcengineAsrError, match="invalid_endpoint"):
        recognize_flash(VolcCredentials(api_key="secret"), hostile, transport=transport)
    assert called is False


def _success_body() -> bytes:
    return json.dumps({
        "result": {
            "text": "你好",
            "utterances": [{"words": [
                {"text": "你", "start_time": 0, "end_time": 100, "confidence": 0.99},
                {"text": "好", "start_time": 100, "end_time": 220},
            ]}],
        },
    }).encode("utf-8")


@pytest.mark.parametrize(
    ("status_code", "headers", "body", "expected_code"),
    [
        (401, {"X-Api-Status-Code": "20000000"}, _success_body(), "asr_http_error"),
        (200, {"X-Api-Status-Code": "10000000"}, _success_body(), "asr_provider_error"),
        (200, {"X-Api-Status-Code": "20000000"}, b"[]", "invalid_asr_response"),
        (200, {"X-Api-Status-Code": "20000000"}, b'{"result":{"text":"","utterances":[]}}', "invalid_asr_response"),
        (200, {"X-Api-Status-Code": "20000000"}, b'{"result":{"text":"x","utterances":[]}}', "empty_word_timings"),
    ],
)
def test_parse_flash_response_rejects_status_and_malformed_private_data(
    status_code: int, headers: dict[str, str], body: bytes, expected_code: str
) -> None:
    with pytest.raises(VolcengineAsrError, match=expected_code) as raised:
        parse_flash_response(
            status_code=status_code, headers=headers, body=body, audio_duration_ms=1_000,
        )

    assert str(raised.value) == expected_code
    assert body.decode("utf-8", errors="replace") not in str(raised.value)


@pytest.mark.parametrize(
    "words",
    [
        [{"text": "你", "start_time": 100, "end_time": 100}],
        [{"text": "你", "start_time": 200, "end_time": 100}],
        [{"text": "你", "start_time": 0, "end_time": 200}, {"text": "好", "start_time": 100, "end_time": 300}],
        [{"text": "你", "start_time": 0, "end_time": 1_001}],
    ],
)
def test_parse_flash_response_rejects_impossible_word_timings(words: list[dict[str, object]]) -> None:
    body = json.dumps({"result": {"text": "你好", "utterances": [{"words": words}]}}).encode()

    with pytest.raises(VolcengineAsrError, match="invalid_word_timing"):
        parse_flash_response(
            status_code=200, headers={"X-Api-Status-Code": "20000000"}, body=body,
            audio_duration_ms=1_000,
        )


def test_recognize_flash_sends_secrets_only_to_redaction_and_rechecks_master_hash(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    _write_master(master)
    request = _request(master)
    seen: dict[str, object] = {}

    def transport(**kwargs: object) -> tuple[int, dict[str, str], bytes]:
        seen.update(kwargs)
        return 200, {"X-Api-Status-Code": "20000000"}, _success_body()

    result = recognize_flash(
        VolcCredentials(api_key="new-secret"), request, transport=transport,
    )

    assert isinstance(result, AsrResult)
    assert result.text == "你好"
    assert [word.text for word in result.words] == ["你", "好"]
    assert seen["url"] == "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
    assert seen["secrets"] == frozenset({"new-secret"})
    assert seen["headers"] == {
        "Content-Type": "application/json", "X-Api-Key": "new-secret",
        "X-Api-Resource-Id": "volc.bigasr.auc_turbo",
        "X-Api-Request-Id": request.request_id, "X-Api-Sequence": "-1",
    }
    assert "new-secret" not in str(result)


def test_recognize_flash_rejects_bad_inputs_before_fake_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bv.asr.volcengine as module

    master = tmp_path / "master.wav"
    _write_master(master)
    called = False

    def transport(**kwargs: object) -> tuple[int, dict[str, str], bytes]:
        nonlocal called
        called = True
        return 200, {"X-Api-Status-Code": "20000000"}, _success_body()

    with pytest.raises(VolcengineAsrError, match="invalid_authentication"):
        recognize_flash(VolcCredentials(), _request(master), transport=transport)
    assert called is False

    with pytest.raises(VolcengineAsrError, match="master_hash_mismatch"):
        recognize_flash(VolcCredentials(api_key="secret"), _request(master).model_copy(update={"audio_sha256": "0" * 64}), transport=transport)
    assert called is False

    monkeypatch.setattr(module, "_is_redirected", lambda path: Path(path) == master)
    with pytest.raises(VolcengineAsrError, match="unsafe_master_audio"):
        recognize_flash(VolcCredentials(api_key="secret"), _request(master), transport=transport)
    assert called is False


def test_recognize_flash_detects_hash_race_and_does_not_retry_provider_errors(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    _write_master(master)
    calls = 0

    def mutating_transport(**kwargs: object) -> tuple[int, dict[str, str], bytes]:
        master.write_bytes(b"changed")
        return 200, {"X-Api-Status-Code": "20000000"}, _success_body()

    with pytest.raises(VolcengineAsrError, match="master_hash_changed"):
        recognize_flash(VolcCredentials(api_key="secret"), _request(master), transport=mutating_transport)

    _write_master(master)

    def client_error_transport(**kwargs: object) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        calls += 1
        return 429, {"X-Api-Status-Code": "42900000"}, b'{"private":"failure"}'

    with pytest.raises(VolcengineAsrError, match="asr_http_error") as raised:
        recognize_flash(VolcCredentials(api_key="secret"), _request(master), transport=client_error_transport)

    assert calls == 1
    assert "private" not in str(raised.value)


def test_recognize_flash_rejects_mismatched_single_read_snapshot_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bv.asr.volcengine as module

    master = tmp_path / "master.wav"
    _write_master(master)
    request = _request(master)
    raced_snapshot = bytearray(master.read_bytes())
    raced_snapshot[-1] ^= 1
    called = False

    def raced_read(path: Path, max_bytes: int, error_code: str) -> bytes:
        assert Path(path) == master
        assert max_bytes > len(raced_snapshot)
        return bytes(raced_snapshot)

    def transport(**kwargs: object) -> tuple[int, dict[str, str], bytes]:
        nonlocal called
        called = True
        return 200, {"X-Api-Status-Code": "20000000"}, _success_body()

    monkeypatch.setattr(module, "_read_bounded_file", raced_read, raising=False)

    with pytest.raises(VolcengineAsrError, match="master_hash_mismatch"):
        recognize_flash(VolcCredentials(api_key="secret"), request, transport=transport)

    assert called is False


def test_build_flash_payload_rejects_request_body_over_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bv.asr.volcengine as module

    master = tmp_path / "master.wav"
    _write_master(master)
    monkeypatch.setattr(module, "_MAX_REQUEST_BYTES", 100, raising=False)

    with pytest.raises(VolcengineAsrError, match="asr_request_too_large"):
        build_flash_payload(_request(master))


def test_parse_flash_response_rejects_oversized_body_with_stable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bv.asr.volcengine as module

    monkeypatch.setattr(module, "_MAX_RESPONSE_BYTES", 32, raising=False)

    with pytest.raises(VolcengineAsrError, match="asr_response_too_large") as raised:
        parse_flash_response(
            status_code=200, headers={"X-Api-Status-Code": "20000000"},
            body=b"x" * 33, audio_duration_ms=1_000,
        )

    assert str(raised.value) == "asr_response_too_large"


def test_stdlib_transport_reads_only_response_limit_plus_one_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bv.asr.volcengine as module

    master = tmp_path / "master.wav"
    _write_master(master)
    seen: dict[str, int] = {}

    class FakeResponse:
        status = 200
        headers = {"X-Api-Status-Code": "20000000"}

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, amount: int) -> bytes:
            seen["read_amount"] = amount
            return b"x" * amount

    class FakeOpener:
        def open(self, request: object, timeout: float) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(module, "_MAX_RESPONSE_BYTES", 32, raising=False)
    monkeypatch.setattr(module, "build_opener", lambda *args: FakeOpener())

    with pytest.raises(VolcengineAsrError, match="asr_response_too_large"):
        recognize_flash(VolcCredentials(api_key="secret"), _request(master))

    assert seen == {"read_amount": 33}
