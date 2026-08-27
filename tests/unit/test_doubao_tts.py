from __future__ import annotations

import hashlib
import io
import json
import logging
import wave
from pathlib import Path

import httpx
import pytest

from bv.config import DoubaoTtsConfig
from bv.voice.doubao import (
    DoubaoCredentials,
    DoubaoNarrationSynthesizer,
    DoubaoTtsError,
)
from bv.voice.providers import (
    NarrationRequest,
    ProviderReceipt,
    narration_request_sha256,
)
from bv.workflow.runtime import RuntimeAuthorization


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wav_bytes(seconds: float = 0.1) -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x01\x00" * round(48_000 * seconds))
    return target.getvalue()


def _request(tmp_path: Path) -> NarrationRequest:
    episode = tmp_path / "episode"
    script = episode / "script" / "approved.txt"
    script.parent.mkdir(parents=True)
    script.write_text("这是已经批准的完整旁白文稿", encoding="utf-8")
    return NarrationRequest(
        book_id="book-demo",
        episode_id="E001",
        approved_script_path=script,
        approved_script_sha256=_sha256(script),
        provider_resource_id="seed-tts-2.0",
        provider_voice_id="reader-a",
        speed=1.0,
        output_dir=episode / ".private" / "tts",
        kind="formal",
    )


def _scoped_authorization(request: NarrationRequest) -> RuntimeAuthorization:
    return RuntimeAuthorization(
        allow_external=True,
        book_id=request.book_id,
        episode_id=request.episode_id,
        script_sha256=request.approved_script_sha256,
        provider_voice_id=request.provider_voice_id,
        max_formal_submissions=1,
        allow_fallback=False,
    )


def _config() -> DoubaoTtsConfig:
    return DoubaoTtsConfig(poll_interval_seconds=0.2, timeout_seconds=10.0)


def _credentials() -> DoubaoCredentials:
    return DoubaoCredentials(api_key="secret-api-key")


def test_formal_submit_uses_current_contract_without_persisting_secret_or_url(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/submit"):
            return httpx.Response(
                200,
                json={
                    "code": 20_000_000,
                    "message": "ok",
                    "data": {
                        "task_id": "task-123",
                        "task_status": 1,
                        "req_text_length": 13,
                    },
                },
            )
        if request.url.path.endswith("/query"):
            return httpx.Response(
                200,
                json={
                    "code": 20_000_000,
                    "message": "ok",
                    "data": {
                        "task_id": "task-123",
                        "task_status": 2,
                        "req_text_length": 13,
                        "audio_url": "https://signed.example/audio.wav?secret=query",
                        "url_expire_time": 2_000_000_000,
                    },
                },
            )
        if request.url.host == "signed.example":
            return httpx.Response(200, content=_wav_bytes())
        raise AssertionError(f"unexpected request: {request.url}")

    request = _request(tmp_path)
    synthesizer = DoubaoNarrationSynthesizer(
        config=_config(),
        credentials=_credentials(),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )
    with caplog.at_level(logging.DEBUG):
        artifact = synthesizer.synthesize(request, _scoped_authorization(request))

    submit = seen[0]
    assert submit.method == "POST"
    assert submit.headers["x-api-key"] == "secret-api-key"
    assert "X-Api-App-Id" not in submit.headers
    assert "X-Api-Access-Key" not in submit.headers
    assert submit.headers["X-Api-Resource-Id"] == "seed-tts-2.0"
    body = json.loads(submit.content)
    assert body == {
        "unique_id": body["unique_id"],
        "req_params": {
            "text": "这是已经批准的完整旁白文稿",
            "speaker": "reader-a",
            "audio_params": {
                "format": "wav",
                "sample_rate": 48_000,
                "speech_rate": 0,
                "enable_timestamp": True,
            },
        },
    }
    assert len(body["unique_id"]) == 36
    assert submit.headers["X-Api-Request-Id"] == body["unique_id"]
    query = seen[1]
    assert query.method == "POST"
    assert query.headers["x-api-key"] == "secret-api-key"
    assert "X-Api-App-Id" not in query.headers
    assert "X-Api-Access-Key" not in query.headers
    assert query.headers["X-Api-Resource-Id"] == "seed-tts-2.0"
    assert json.loads(query.content) == {"task_id": "task-123"}
    assert artifact.raw_audio_path.is_file()
    receipt_text = artifact.receipt_path.read_text(encoding="utf-8")
    assert "secret-api-key" not in receipt_text
    assert "signed.example" not in receipt_text
    assert "secret-api-key" not in caplog.text
    assert "signed.example" not in caplog.text
    receipt = ProviderReceipt.model_validate_json(receipt_text)
    assert receipt.task_id == "task-123"
    assert receipt.status == "success"


def test_existing_working_receipt_queries_without_second_submit(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.output_dir.mkdir(parents=True)
    request_hash = DoubaoNarrationSynthesizer.request_sha256(request)
    receipt_path = request.output_dir / "doubao-formal-receipt.json"
    receipt_path.write_text(
        ProviderReceipt(
            provider="doubao",
            request_sha256=request_hash,
            task_id="task-existing",
            task_id_sha256=hashlib.sha256(b"task-existing").hexdigest(),
            response_sha256="a" * 64,
            status="working",
            error_code=None,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    methods: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        methods.append(http_request.method)
        if http_request.url.path.endswith("/query"):
            assert json.loads(http_request.content) == {"task_id": "task-existing"}
            return httpx.Response(
                200,
                json={
                    "code": 20_000_000,
                    "message": "ok",
                    "data": {
                        "task_id": "task-existing",
                        "task_status": 2,
                        "req_text_length": 13,
                        "audio_url": "https://signed.example/audio.wav",
                        "url_expire_time": 2_000_000_000,
                    },
                },
            )
        if http_request.url.host == "signed.example":
            return httpx.Response(200, content=_wav_bytes())
        raise AssertionError("submit must not be called")

    synthesizer = DoubaoNarrationSynthesizer(
        config=_config(),
        credentials=_credentials(),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )

    artifact = synthesizer.synthesize(
        request,
        RuntimeAuthorization(allow_external=True),
    )

    assert methods == ["POST", "GET"]
    assert artifact.raw_audio_path.is_file()


def test_query_transport_failure_keeps_working_receipt_and_never_resubmits(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    request.output_dir.mkdir(parents=True)
    receipt_path = request.output_dir / "doubao-formal-receipt.json"
    receipt_path.write_text(
        ProviderReceipt(
            provider="doubao",
            request_sha256=DoubaoNarrationSynthesizer.request_sha256(request),
            task_id="task-existing",
            task_id_sha256=hashlib.sha256(b"task-existing").hexdigest(),
            response_sha256="a" * 64,
            status="working",
            error_code=None,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    paths: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        paths.append(http_request.url.path)
        raise httpx.ConnectError("PRIVATE_TRANSPORT_DETAIL", request=http_request)

    synthesizer = DoubaoNarrationSynthesizer(
        config=_config(),
        credentials=_credentials(),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )

    with pytest.raises(DoubaoTtsError, match="doubao_query_failed") as captured:
        synthesizer.synthesize(request, RuntimeAuthorization(allow_external=True))

    assert str(captured.value) == "doubao_query_failed"
    assert paths == ["/api/v3/tts/query"]
    assert ProviderReceipt.model_validate_json(
        receipt_path.read_text("utf-8")
    ).status == "working"


def test_provider_failure_without_text_length_is_persisted_without_fallback(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    paths: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        paths.append(http_request.url.path)
        if http_request.url.path.endswith("/submit"):
            return httpx.Response(
                200,
                json={
                    "code": 20_000_000,
                    "message": "ok",
                    "data": {
                        "task_id": "task-failed",
                        "task_status": 1,
                        "req_text_length": 13,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "code": 20_000_000,
                "message": "ok",
                "data": {
                    "task_id": "task-failed",
                    "task_status": 3,
                },
            },
        )

    synthesizer = DoubaoNarrationSynthesizer(
        config=_config(),
        credentials=_credentials(),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )

    with pytest.raises(DoubaoTtsError, match="doubao_task_failed"):
        synthesizer.synthesize(request, _scoped_authorization(request))

    receipt = ProviderReceipt.model_validate_json(
        (request.output_dir / "doubao-formal-receipt.json").read_text("utf-8")
    )
    assert receipt.status == "failed"
    assert receipt.error_code == "doubao_task_failed"
    assert sum(path.endswith("/submit") for path in paths) == 1
    assert sum(path.endswith("/query") for path in paths) == 1


@pytest.mark.parametrize("status_code", [400, 403, 500])
def test_submit_failure_has_one_request_and_no_fallback(
    tmp_path: Path,
    status_code: int,
) -> None:
    request = _request(tmp_path)
    requests: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        requests.append(http_request)
        return httpx.Response(
            status_code,
            json={"code": status_code * 100, "message": "PRIVATE_PROVIDER_BODY"},
        )

    synthesizer = DoubaoNarrationSynthesizer(
        config=_config(),
        credentials=_credentials(),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )

    with pytest.raises(DoubaoTtsError, match="doubao_submit_failed") as captured:
        synthesizer.synthesize(request, _scoped_authorization(request))

    assert str(captured.value) == "doubao_submit_failed"
    assert len(requests) == 1
    assert not (request.output_dir / "voice_raw.wav").exists()


def test_working_timeout_is_resumable_and_never_resubmits(tmp_path: Path) -> None:
    request = _request(tmp_path)
    paths: list[str] = []
    clock_values = iter([0.0, 1.0, 11.0])

    def handler(http_request: httpx.Request) -> httpx.Response:
        paths.append(http_request.url.path)
        return httpx.Response(
            200,
            json={
                "code": 20_000_000,
                "message": "ok",
                "data": {
                    "task_id": "task-timeout",
                    "task_status": 1,
                    "req_text_length": 13,
                },
            },
        )

    synthesizer = DoubaoNarrationSynthesizer(
        config=_config(),
        credentials=_credentials(),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        monotonic=lambda: next(clock_values),
        sleep=lambda _: None,
    )

    with pytest.raises(DoubaoTtsError, match="doubao_task_working"):
        synthesizer.synthesize(request, _scoped_authorization(request))

    receipt = ProviderReceipt.model_validate_json(
        (request.output_dir / "doubao-formal-receipt.json").read_text("utf-8")
    )
    assert receipt.status == "working"
    assert sum(path.endswith("/submit") for path in paths) == 1


def test_download_failure_keeps_queryable_task_without_signed_url(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    paths: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        paths.append(http_request.url.path)
        if http_request.url.path.endswith("/submit"):
            return httpx.Response(
                200,
                json={
                    "code": 20_000_000,
                    "message": "ok",
                    "data": {
                        "task_id": "task-download",
                        "task_status": 1,
                        "req_text_length": 13,
                    },
                },
            )
        if http_request.url.path.endswith("/query"):
            return httpx.Response(
                200,
                json={
                    "code": 20_000_000,
                    "message": "ok",
                    "data": {
                        "task_id": "task-download",
                        "task_status": 2,
                        "req_text_length": 13,
                        "audio_url": "https://signed.example/expired.wav?signature=private",
                        "url_expire_time": 1,
                    },
                },
            )
        return httpx.Response(403, content=b"expired")

    synthesizer = DoubaoNarrationSynthesizer(
        config=_config(),
        credentials=_credentials(),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )

    with pytest.raises(DoubaoTtsError, match="doubao_download_failed"):
        synthesizer.synthesize(request, _scoped_authorization(request))

    receipt_text = (
        request.output_dir / "doubao-formal-receipt.json"
    ).read_text("utf-8")
    receipt = ProviderReceipt.model_validate_json(receipt_text)
    assert receipt.status == "working"
    assert receipt.error_code == "doubao_download_failed"
    assert receipt.task_id == "task-download"
    assert "signed.example" not in receipt_text
    assert sum(path.endswith("/submit") for path in paths) == 1
    assert sum(path.endswith("/query") for path in paths) == 1


def test_request_hash_binds_provider_resource_id(tmp_path: Path) -> None:
    request = _request(tmp_path)
    other_resource = request.model_copy(
        update={"provider_resource_id": "seed-tts-1.0"}
    )

    assert narration_request_sha256(request) != narration_request_sha256(
        other_resource
    )


def test_query_rejects_a_different_task_id_without_download(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.output_dir.mkdir(parents=True)
    receipt_path = request.output_dir / "doubao-formal-receipt.json"
    receipt_path.write_text(
        ProviderReceipt(
            provider="doubao",
            request_sha256=DoubaoNarrationSynthesizer.request_sha256(request),
            task_id="task-expected",
            task_id_sha256=hashlib.sha256(b"task-expected").hexdigest(),
            response_sha256="a" * 64,
            status="working",
            error_code=None,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    requests: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        requests.append(http_request)
        return httpx.Response(
            200,
            json={
                "code": 20_000_000,
                "message": "ok",
                "data": {
                    "task_id": "task-other",
                    "task_status": 2,
                    "req_text_length": 13,
                    "audio_url": "https://signed.example/wrong.wav",
                    "url_expire_time": 2_000_000_000,
                },
            },
        )

    synthesizer = DoubaoNarrationSynthesizer(
        config=_config(),
        credentials=_credentials(),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )

    with pytest.raises(DoubaoTtsError, match="doubao_query_task_mismatch"):
        synthesizer.synthesize(request, RuntimeAuthorization(allow_external=True))

    assert len(requests) == 1
    assert ProviderReceipt.model_validate_json(
        receipt_path.read_text("utf-8")
    ).task_id == "task-expected"


def test_credentials_report_state_without_values() -> None:
    credentials = DoubaoCredentials.from_environment(
        _config(),
        environ={
            "BV_DOUBAO_TTS_API_KEY": "private-api-key",
        },
    )

    assert credentials.state() == {"api_key": "SET"}
    assert "private-api-key" not in repr(credentials)


def test_missing_credentials_can_be_reported_but_cannot_build_client() -> None:
    credentials = DoubaoCredentials.from_environment(_config(), environ={})

    assert credentials.state() == {"api_key": "UNSET"}
    with pytest.raises(DoubaoTtsError, match="doubao_credentials_missing"):
        DoubaoNarrationSynthesizer(
            config=_config(),
            credentials=credentials,
            http=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))),
            sleep=lambda _: None,
        )
