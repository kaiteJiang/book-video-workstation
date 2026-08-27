from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bv.config import DoubaoTtsConfig
from bv.voice.providers import (
    NarrationArtifact,
    NarrationRequest,
    ProviderReceipt,
    narration_request_sha256,
)
from bv.workflow.runtime import (
    RuntimeAuthorization,
    require_external_authorization,
)


_HTTPX_LOG_LOCK = threading.RLock()


class DoubaoTtsError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


@dataclass(frozen=True, repr=False)
class DoubaoCredentials:
    api_key: str

    def require_set(self) -> None:
        if not self.api_key.strip():
            raise DoubaoTtsError("doubao_credentials_missing")

    def __repr__(self) -> str:
        state = self.state()
        return f"DoubaoCredentials(api_key={state['api_key']})"

    def state(self) -> dict[str, Literal["SET", "UNSET"]]:
        return {
            "api_key": "SET" if self.api_key else "UNSET",
        }

    @classmethod
    def from_environment(
        cls,
        config: DoubaoTtsConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> DoubaoCredentials:
        source = os.environ if environ is None else environ
        return cls(api_key=source.get(config.api_key_env, ""))


class _DoubaoModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DoubaoTask(_DoubaoModel):
    task_id: str
    task_status: Literal[1, 2, 3]
    text_length: int = Field(ge=0)
    response_sha256: str
    audio_url: str | None = Field(default=None, repr=False)
    url_expire_time: int | None = None


class DoubaoAsyncClient:
    def __init__(
        self,
        *,
        config: DoubaoTtsConfig,
        credentials: DoubaoCredentials,
        http: httpx.Client,
    ) -> None:
        credentials.require_set()
        self.config = config
        self.credentials = credentials
        self.http = http

    def submit(self, request: NarrationRequest, text: str) -> DoubaoTask:
        request_id = str(uuid4())
        payload = {
            "unique_id": request_id,
            "req_params": {
                "text": text,
                "speaker": request.provider_voice_id,
                "audio_params": {
                    "format": "wav",
                    "sample_rate": 48_000,
                    "speech_rate": round((request.speed - 1.0) * 100),
                    "enable_timestamp": True,
                },
            },
        }
        response = self._request(
            "POST",
            self.config.submit_endpoint,
            resource_id=request.provider_resource_id,
            request_id=request_id,
            error_code="doubao_submit_failed",
            json_payload=payload,
        )
        return self._parse_task(response, "doubao_submit_failed")

    def query(self, *, task_id: str, resource_id: str) -> DoubaoTask:
        response = self._request(
            "POST",
            self.config.query_endpoint,
            resource_id=resource_id,
            request_id=str(uuid4()),
            error_code="doubao_query_failed",
            json_payload={"task_id": task_id},
        )
        task = self._parse_task(response, "doubao_query_failed")
        if task.task_id != task_id:
            raise DoubaoTtsError("doubao_query_task_mismatch")
        return task

    def download(self, audio_url: str, destination: Path) -> str:
        try:
            with _suppress_httpx_url_logging():
                response = self.http.get(audio_url)
        except Exception:
            raise DoubaoTtsError("doubao_download_failed") from None
        if response.status_code != 200 or not response.content:
            raise DoubaoTtsError("doubao_download_failed")
        _atomic_write_bytes(destination, response.content)
        return _sha256_file(destination)

    def _request(
        self,
        method: Literal["POST"],
        url: str,
        *,
        resource_id: str,
        request_id: str,
        error_code: Literal["doubao_submit_failed", "doubao_query_failed"],
        json_payload: dict[str, object] | None = None,
    ) -> httpx.Response:
        headers = {
            "x-api-key": self.credentials.api_key,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": request_id,
            "Content-Type": "application/json",
        }
        try:
            with _suppress_httpx_url_logging():
                return self.http.request(
                    method,
                    url,
                    headers=headers,
                    json=json_payload,
                )
        except Exception:
            raise DoubaoTtsError(error_code) from None

    @staticmethod
    def _parse_task(response: httpx.Response, error_code: str) -> DoubaoTask:
        response_hash = hashlib.sha256(response.content).hexdigest()
        if response.status_code != 200:
            raise DoubaoTtsError(error_code)
        try:
            payload = response.json()
            if (
                not isinstance(payload, dict)
                or payload.get("code") != 20_000_000
                or not isinstance(payload.get("data"), dict)
            ):
                raise ValueError
            data = payload["data"]
            return DoubaoTask(
                task_id=data["task_id"],
                task_status=data["task_status"],
                text_length=data.get("req_text_length", 0),
                response_sha256=response_hash,
                audio_url=data.get("audio_url"),
                url_expire_time=data.get("url_expire_time"),
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            raise DoubaoTtsError(error_code) from None


class DoubaoNarrationSynthesizer:
    provider: Literal["doubao"] = "doubao"

    def __init__(
        self,
        *,
        config: DoubaoTtsConfig,
        credentials: DoubaoCredentials,
        http: httpx.Client,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.client = DoubaoAsyncClient(
            config=config,
            credentials=credentials,
            http=http,
        )
        self.monotonic = monotonic
        self.sleep = sleep

    @staticmethod
    def request_sha256(request: NarrationRequest) -> str:
        return narration_request_sha256(request)

    def synthesize(
        self,
        request: NarrationRequest,
        authorization: RuntimeAuthorization,
    ) -> NarrationArtifact:
        episode_root = _validate_request_paths(request)
        text = _read_approved_script(request)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = request.output_dir / f"doubao-{request.kind}-receipt.json"
        raw_path = request.output_dir / "voice_raw.wav"
        request_hash = self.request_sha256(request)
        receipt = _load_receipt(receipt_path, request_hash)

        if receipt is not None and receipt.status == "success":
            if (
                receipt.raw_audio_sha256 is not None
                and raw_path.is_file()
                and _sha256_file(raw_path) == receipt.raw_audio_sha256
            ):
                return _artifact(raw_path, receipt_path, receipt)
            raise DoubaoTtsError("doubao_success_artifact_missing")
        if receipt is not None and receipt.status == "failed":
            raise DoubaoTtsError(receipt.error_code or "doubao_task_failed")

        if receipt is None:
            authorization.assert_tts_submission(
                book_id=request.book_id,
                episode_id=request.episode_id,
                script_sha256=request.approved_script_sha256,
                provider_voice_id=request.provider_voice_id,
                kind=request.kind,
            )
            task = self.client.submit(request, text)
            receipt = _receipt_from_task(
                task,
                request_hash=request_hash,
                status=(
                    "submitted"
                    if task.task_status == 1
                    else _receipt_status_before_download(task)
                ),
            )
            _atomic_write_receipt(receipt_path, receipt, episode_root)
        else:
            require_external_authorization(authorization)
            if receipt.task_id is None:
                raise DoubaoTtsError("doubao_receipt_invalid")
            task = DoubaoTask(
                task_id=receipt.task_id,
                task_status=1,
                text_length=len(text),
                response_sha256=receipt.response_sha256 or "0" * 64,
            )

        deadline = self.monotonic() + self.config.timeout_seconds
        while task.task_status == 1:
            if self.monotonic() >= deadline:
                _atomic_write_receipt(
                    receipt_path,
                    _receipt_from_task(
                        task,
                        request_hash=request_hash,
                        status="working",
                    ),
                    episode_root,
                )
                raise DoubaoTtsError("doubao_task_working")
            self.sleep(self.config.poll_interval_seconds)
            task = self.client.query(
                task_id=task.task_id,
                resource_id=request.provider_resource_id,
            )
            _atomic_write_receipt(
                receipt_path,
                _receipt_from_task(
                    task,
                    request_hash=request_hash,
                    status=_receipt_status_before_download(task),
                ),
                episode_root,
            )

        if task.task_status == 3:
            failed = _receipt_from_task(
                task,
                request_hash=request_hash,
                status="failed",
                error_code="doubao_task_failed",
            )
            _atomic_write_receipt(receipt_path, failed, episode_root)
            raise DoubaoTtsError("doubao_task_failed")
        if not task.audio_url:
            raise DoubaoTtsError("doubao_audio_url_missing")

        temporary = request.output_dir / f"voice_raw-{uuid4().hex}.tmp"
        try:
            try:
                raw_hash = self.client.download(task.audio_url, temporary)
            except DoubaoTtsError as error:
                if error.error_code == "doubao_download_failed":
                    _atomic_write_receipt(
                        receipt_path,
                        _receipt_from_task(
                            task,
                            request_hash=request_hash,
                            status="working",
                            error_code="doubao_download_failed",
                        ),
                        episode_root,
                    )
                raise
            if raw_path.exists() or raw_path.is_symlink():
                raise DoubaoTtsError("doubao_raw_output_exists")
            os.replace(temporary, raw_path)
            success = _receipt_from_task(
                task,
                request_hash=request_hash,
                status="success",
                raw_audio_sha256=raw_hash,
            )
            _atomic_write_receipt(receipt_path, success, episode_root)
            return _artifact(raw_path, receipt_path, success)
        finally:
            if temporary.exists() and temporary.is_file():
                temporary.unlink()


def _read_approved_script(request: NarrationRequest) -> str:
    try:
        raw = request.approved_script_path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        raise DoubaoTtsError("doubao_script_invalid") from None
    if (
        hashlib.sha256(raw).hexdigest() != request.approved_script_sha256
        or not text.strip()
        or len(text) >= 100_000
    ):
        raise DoubaoTtsError("doubao_script_invalid")
    return text


def _validate_request_paths(request: NarrationRequest) -> Path:
    try:
        episode_root = request.approved_script_path.parents[1].absolute()
    except IndexError:
        raise DoubaoTtsError("unsafe_doubao_path") from None
    script = request.approved_script_path.absolute()
    output = request.output_dir.absolute()
    if not script.is_relative_to(episode_root) or not output.is_relative_to(episode_root):
        raise DoubaoTtsError("unsafe_doubao_path")
    for path in (episode_root, request.approved_script_path, request.output_dir):
        if _redirect_in_existing_chain(path):
            raise DoubaoTtsError("unsafe_doubao_path")
    return episode_root


def _load_receipt(path: Path, request_hash: str) -> ProviderReceipt | None:
    if not path.exists() and not path.is_symlink():
        return None
    if _redirect_in_existing_chain(path) or not path.is_file():
        raise DoubaoTtsError("doubao_receipt_invalid")
    try:
        receipt = ProviderReceipt.model_validate_json(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError):
        raise DoubaoTtsError("doubao_receipt_invalid") from None
    if receipt.provider != "doubao" or receipt.request_sha256 != request_hash:
        raise DoubaoTtsError("doubao_receipt_mismatch")
    if receipt.task_id is None or receipt.task_id_sha256 != _sha256_text(receipt.task_id):
        raise DoubaoTtsError("doubao_receipt_invalid")
    return receipt


def _receipt_from_task(
    task: DoubaoTask,
    *,
    request_hash: str,
    status: Literal["submitted", "working", "success", "failed"],
    error_code: str | None = None,
    raw_audio_sha256: str | None = None,
) -> ProviderReceipt:
    return ProviderReceipt(
        provider="doubao",
        request_sha256=request_hash,
        task_id=task.task_id,
        task_id_sha256=_sha256_text(task.task_id),
        response_sha256=task.response_sha256,
        raw_audio_sha256=raw_audio_sha256,
        status=status,
        error_code=error_code,
    )


def _receipt_status_before_download(
    task: DoubaoTask,
) -> Literal["working", "failed"]:
    return "failed" if task.task_status == 3 else "working"


def _artifact(
    raw_path: Path,
    receipt_path: Path,
    receipt: ProviderReceipt,
) -> NarrationArtifact:
    raw_hash = _sha256_file(raw_path)
    if receipt.raw_audio_sha256 != raw_hash:
        raise DoubaoTtsError("doubao_raw_hash_mismatch")
    return NarrationArtifact(
        raw_audio_path=raw_path,
        raw_audio_sha256=raw_hash,
        receipt_path=receipt_path,
        receipt_sha256=_sha256_file(receipt_path),
    )


def _atomic_write_receipt(
    path: Path,
    receipt: ProviderReceipt,
    episode_root: Path,
) -> None:
    if _redirect_in_existing_chain(path) or not path.absolute().is_relative_to(
        episode_root.absolute()
    ):
        raise DoubaoTtsError("unsafe_doubao_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(receipt.model_dump_json(indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _redirect_in_existing_chain(path: Path) -> bool:
    candidate = Path(path)
    while True:
        if candidate.exists() or candidate.is_symlink():
            if candidate.is_symlink() or _is_windows_reparse(candidate):
                return True
        parent = candidate.parent
        if parent == candidate:
            return False
        candidate = parent


def _is_windows_reparse(path: Path) -> bool:
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & 0x400)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _suppress_httpx_url_logging() -> Iterator[None]:
    logger = logging.getLogger("httpx")
    with _HTTPX_LOG_LOCK:
        previous = logger.disabled
        logger.disabled = True
        try:
            yield
        finally:
            logger.disabled = previous
