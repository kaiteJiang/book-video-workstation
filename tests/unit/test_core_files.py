import json
from pathlib import Path

import pytest

from bv.core.atomic import atomic_write_json
from bv.core.errors import BVError
from bv.core.hashing import sha256_file


def test_atomic_write_produces_valid_json_and_cleans_temp_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "state.json"

    atomic_write_json(target, {"状态": "pending"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"状态": "pending"}
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_write_keeps_existing_file_when_serialization_fails(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state.json"
    target.write_text('{"status": "old"}', encoding="utf-8")

    with pytest.raises(TypeError):
        atomic_write_json(target, {"invalid": object()})

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "old"}
    assert not list(tmp_path.glob("*.tmp"))


def test_sha256_is_stable(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("abc", encoding="utf-8")

    assert sha256_file(target) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_bv_error_carries_safe_failure_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "failure.log"

    error = BVError(
        code="CONFIG_INVALID",
        stage="config",
        user_message="配置文件无效",
        next_command="bv doctor",
        log_path=log_path,
    )

    assert error.code == "CONFIG_INVALID"
    assert error.stage == "config"
    assert error.user_message == "配置文件无效"
    assert error.next_command == "bv doctor"
    assert error.log_path == log_path
    assert str(error) == "配置文件无效"
