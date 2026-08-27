import ctypes
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from ctypes import wintypes


_GENERIC_READ = 0x80000000
_DELETE = 0x00010000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_FILE_INFO_BY_HANDLE_CLASS_DISPOSITION = 4
_ERROR_INVALID_PARAMETER = 87
_STILL_ACTIVE = 259
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class _FILE_DISPOSITION_INFO(ctypes.Structure):
    _fields_ = [("DeleteFile", wintypes.BOOL)]


if os.name == "nt":
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _OPEN_PROCESS = _KERNEL32.OpenProcess
    _OPEN_PROCESS.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _OPEN_PROCESS.restype = wintypes.HANDLE

    _GET_EXIT_CODE_PROCESS = _KERNEL32.GetExitCodeProcess
    _GET_EXIT_CODE_PROCESS.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _GET_EXIT_CODE_PROCESS.restype = wintypes.BOOL

    _CLOSE_HANDLE = _KERNEL32.CloseHandle
    _CLOSE_HANDLE.argtypes = [wintypes.HANDLE]
    _CLOSE_HANDLE.restype = wintypes.BOOL

    _CREATE_FILE_W = _KERNEL32.CreateFileW
    _CREATE_FILE_W.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _CREATE_FILE_W.restype = wintypes.HANDLE

    _READ_FILE = _KERNEL32.ReadFile
    _READ_FILE.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    _READ_FILE.restype = wintypes.BOOL

    _SET_FILE_INFORMATION_BY_HANDLE = _KERNEL32.SetFileInformationByHandle
    _SET_FILE_INFORMATION_BY_HANDLE.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _SET_FILE_INFORMATION_BY_HANDLE.restype = wintypes.BOOL
else:
    _KERNEL32 = None
    _OPEN_PROCESS = None
    _GET_EXIT_CODE_PROCESS = None
    _CLOSE_HANDLE = None
    _CREATE_FILE_W = None
    _READ_FILE = None
    _SET_FILE_INFORMATION_BY_HANDLE = None


class LockHeldError(RuntimeError):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        super().__init__(f"lock already held: {self.path}")


def is_process_alive(pid: int) -> bool:
    """Probe process liveness without treating an access failure as stale."""

    if pid <= 0:
        return False

    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    process_query_limited_information = 0x1000
    handle = _OPEN_PROCESS(process_query_limited_information, False, pid)
    if _is_invalid_handle(handle):
        # ERROR_INVALID_PARAMETER means the PID is gone. Other failures are
        # conservatively treated as live so recovery cannot remove a lock we
        # could not inspect.
        return ctypes.get_last_error() != _ERROR_INVALID_PARAMETER

    try:
        exit_code = wintypes.DWORD()
        if not _GET_EXIT_CODE_PROCESS(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == _STILL_ACTIVE
    finally:
        _CLOSE_HANDLE(handle)


class EpisodeLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._owner: dict[str, Any] | None = None

    def __enter__(self) -> "EpisodeLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        owner = {
            "pid": os.getpid(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            fd = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise LockHeldError(self.path) from exc

        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(owner, stream, ensure_ascii=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            _delete_owned_lock(self.path, owner)
            raise

        self._owner = owner
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        owner = self._owner
        self._owner = None
        if owner is not None:
            _delete_owned_lock(self.path, owner)


def recover_stale_lock(path: Path) -> bool:
    """Remove a lock only after an explicit dead-owner liveness check."""

    target = Path(path)
    owner = _read_lock_payload(target)
    if owner is None:
        return False

    pid = owner.get("pid")
    if not isinstance(pid, int) or is_process_alive(pid):
        return False

    return _delete_owned_lock(target, owner)


def _before_lock_delete() -> None:
    """Test seam immediately before a validated handle is marked deleted."""


def _delete_owned_lock(path: Path, expected_owner: dict[str, Any]) -> bool:
    if os.name == "nt":
        return _delete_owned_lock_windows(path, expected_owner)
    return _delete_owned_lock_portable(path, expected_owner)


def _delete_owned_lock_windows(
    path: Path,
    expected_owner: dict[str, Any],
) -> bool:
    handle = _CREATE_FILE_W(
        str(path),
        _GENERIC_READ | _DELETE,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        _FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if _is_invalid_handle(handle):
        return False

    try:
        if _read_lock_payload_from_handle(handle) != expected_owner:
            return False

        _before_lock_delete()
        disposition = _FILE_DISPOSITION_INFO(True)
        return bool(
            _SET_FILE_INFORMATION_BY_HANDLE(
                handle,
                _FILE_INFO_BY_HANDLE_CLASS_DISPOSITION,
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            )
        )
    finally:
        _CLOSE_HANDLE(handle)


def _delete_owned_lock_portable(
    path: Path,
    expected_owner: dict[str, Any],
) -> bool:
    if _read_lock_payload(path) != expected_owner:
        return False

    _before_lock_delete()
    # A second check is the most conservative portable fallback available
    # without an OS-specific identity-bound delete primitive.
    if _read_lock_payload(path) != expected_owner:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _is_invalid_handle(handle: Any) -> bool:
    value = getattr(handle, "value", handle)
    return value in (None, 0, _INVALID_HANDLE_VALUE)


def _read_lock_payload_from_handle(handle: wintypes.HANDLE) -> dict[str, Any] | None:
    buffer = ctypes.create_string_buffer(64 * 1024)
    bytes_read = wintypes.DWORD()
    if not _READ_FILE(
        handle,
        buffer,
        len(buffer) - 1,
        ctypes.byref(bytes_read),
        None,
    ):
        return None
    try:
        value = json.loads(buffer.raw[: bytes_read.value].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_lock_payload(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
