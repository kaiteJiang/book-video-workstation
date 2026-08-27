import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=target.name,
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        json.loads(Path(temp_name).read_text(encoding="utf-8"))
        os.replace(temp_name, target)
    finally:
        temp_path = Path(temp_name)
        if temp_path.exists():
            temp_path.unlink()
