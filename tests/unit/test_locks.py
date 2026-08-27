import json
import os
import ctypes
from ctypes import wintypes
from pathlib import Path

import pytest

from bv.state import locks
from bv.state.locks import EpisodeLock, LockHeldError, recover_stale_lock


def test_second_lock_is_rejected(tmp_path: Path) -> None:
    lock_path = tmp_path / ".run.lock"

    with EpisodeLock(lock_path):
        with pytest.raises(LockHeldError):
            with EpisodeLock(lock_path):
                raise AssertionError("unreachable")


def test_lock_records_pid_and_timestamp_and_releases_own_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / ".run.lock"

    with EpisodeLock(lock_path):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert isinstance(payload["timestamp"], str)

    assert not lock_path.exists()


def test_lock_does_not_delete_replaced_lock_on_exit(tmp_path: Path) -> None:
    lock_path = tmp_path / ".run.lock"

    with EpisodeLock(lock_path):
        lock_path.write_text(
            json.dumps({"pid": os.getpid(), "timestamp": "other-owner"}),
            encoding="utf-8",
        )

    assert lock_path.exists()


def test_stale_recovery_is_explicit_and_keeps_live_owner_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / ".run.lock"

    with EpisodeLock(lock_path):
        assert recover_stale_lock(lock_path) is False
        assert lock_path.exists()

    lock_path.write_text(
        json.dumps({"pid": 2_147_483_647, "timestamp": "old-owner"}),
        encoding="utf-8",
    )

    assert recover_stale_lock(lock_path) is True
    assert not lock_path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle identity regression")
def test_exit_preserves_replacement_after_same_handle_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".run.lock"
    replacement_payload = {"pid": os.getpid(), "timestamp": "replacement-owner"}

    def replace_lock_path() -> None:
        replacement = tmp_path / ".replacement.lock"
        original = tmp_path / ".original.lock"
        replacement.write_text(
            json.dumps(replacement_payload),
            encoding="utf-8",
        )
        os.rename(lock_path, original)
        os.replace(replacement, lock_path)

    monkeypatch.setattr(locks, "_before_lock_delete", replace_lock_path, raising=False)

    with EpisodeLock(lock_path):
        pass

    assert lock_path.exists()
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement_payload


@pytest.mark.skipif(os.name != "nt", reason="Windows handle identity regression")
def test_stale_recovery_preserves_replacement_after_same_handle_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".run.lock"
    stale_payload = {"pid": 2_147_483_647, "timestamp": "old-owner"}
    replacement_payload = {"pid": os.getpid(), "timestamp": "replacement-owner"}
    lock_path.write_text(json.dumps(stale_payload), encoding="utf-8")

    def replace_lock_path() -> None:
        replacement = tmp_path / ".replacement.lock"
        original = tmp_path / ".original.lock"
        replacement.write_text(
            json.dumps(replacement_payload),
            encoding="utf-8",
        )
        os.rename(lock_path, original)
        os.replace(replacement, lock_path)

    monkeypatch.setattr(locks, "_before_lock_delete", replace_lock_path, raising=False)

    assert recover_stale_lock(lock_path) is True
    assert lock_path.exists()
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement_payload


@pytest.mark.skipif(os.name != "nt", reason="Windows ctypes contract")
def test_windows_lock_api_prototypes_are_explicit() -> None:
    assert locks._OPEN_PROCESS.argtypes == [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    assert locks._OPEN_PROCESS.restype is wintypes.HANDLE
    assert locks._GET_EXIT_CODE_PROCESS.argtypes == [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    assert locks._GET_EXIT_CODE_PROCESS.restype is wintypes.BOOL
    assert locks._CLOSE_HANDLE.argtypes == [wintypes.HANDLE]
    assert locks._CLOSE_HANDLE.restype is wintypes.BOOL
    assert locks._CREATE_FILE_W.argtypes[0] is wintypes.LPCWSTR
    assert locks._CREATE_FILE_W.restype is wintypes.HANDLE
    assert locks._READ_FILE.argtypes[0] is wintypes.HANDLE
    assert locks._READ_FILE.restype is wintypes.BOOL
    assert locks._SET_FILE_INFORMATION_BY_HANDLE.argtypes[0] is wintypes.HANDLE
    assert locks._SET_FILE_INFORMATION_BY_HANDLE.restype is wintypes.BOOL


def test_invalid_handle_helper_rejects_null_and_invalid_handles() -> None:
    assert locks._is_invalid_handle(None)
    assert locks._is_invalid_handle(0)
    assert locks._is_invalid_handle(locks._INVALID_HANDLE_VALUE)
