import json
import os
from pathlib import Path

from pydantic import BaseModel, Field


class Event(BaseModel):
    event: str
    book_id: str
    episode_id: str | None
    timestamp: str
    artifact_hash: str | None = None
    stage: str | None = None
    detail: dict[str, str] = Field(default_factory=dict)


def append_event(path: Path, event: Event) -> None:
    """Append one durable UTF-8 JSONL event record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
