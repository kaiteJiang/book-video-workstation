from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess


_POOL_ROW = re.compile(
    r"^\|\s*(?P<number>\d+)\s*\|\s*(?P<state>[^|]+?)\s*\|"
    r"\s*《(?P<title>[^》]+)》\s*\|"
)
_INVALID_WINDOWS_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True, slots=True)
class BookMenuEntry:
    book_id: str
    title: str
    display_name: str
    folder: Path


def build_display_name(
    *,
    title: str,
    produced_at: datetime,
    pool_number: int | None,
    trial_number: int | None,
    final_approved: bool,
) -> str:
    safe_title = _INVALID_WINDOWS_NAME.sub("_", title).strip().rstrip(" .")
    if not safe_title:
        safe_title = "未命名图书"
    prefix = (
        f"{pool_number:03d}"
        if pool_number is not None
        else f"试作{trial_number:02d}"
    )
    suffix = "" if final_approved else "-待终审"
    return f"{prefix}-{safe_title}-{produced_at:%Y-%m-%d-%H-%M}{suffix}"


def refresh_book_folder_menu(
    *,
    books_root: Path,
    pool_path: Path,
    apply_shell_attributes: bool = True,
) -> tuple[BookMenuEntry, ...]:
    books_root = Path(books_root)
    pool_numbers = _read_pool_numbers(Path(pool_path))
    records: list[dict[str, object]] = []
    for folder in sorted(path for path in books_root.iterdir() if path.is_dir()):
        book_path = folder / "book.json"
        if not book_path.is_file():
            continue
        payload = json.loads(book_path.read_text(encoding="utf-8"))
        title = payload.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        produced_at = _production_time(folder)
        records.append(
            {
                "book_id": folder.name,
                "title": title.strip(),
                "folder": folder,
                "produced_at": produced_at,
                "pool_number": pool_numbers.get(title.strip()),
                "final_approved": _episode_status(folder) == "final_approved",
            }
        )

    trials = sorted(
        (record for record in records if record["pool_number"] is None),
        key=lambda record: (record["produced_at"], record["title"]),
    )
    trial_numbers = {
        str(record["book_id"]): index for index, record in enumerate(trials, start=1)
    }
    entries: list[BookMenuEntry] = []
    for record in records:
        pool_number = record["pool_number"]
        display_name = build_display_name(
            title=str(record["title"]),
            produced_at=record["produced_at"],
            pool_number=pool_number if isinstance(pool_number, int) else None,
            trial_number=(
                None
                if isinstance(pool_number, int)
                else trial_numbers[str(record["book_id"])]
            ),
            final_approved=bool(record["final_approved"]),
        )
        folder = Path(record["folder"])
        _write_desktop_ini(
            folder,
            display_name,
            apply_shell_attributes=apply_shell_attributes,
        )
        entries.append(
            BookMenuEntry(
                book_id=str(record["book_id"]),
                title=str(record["title"]),
                display_name=display_name,
                folder=folder,
            )
        )
    _mark_final_approved_pool_rows(
        Path(pool_path),
        {
            str(record["title"])
            for record in records
            if bool(record["final_approved"])
        },
    )
    return tuple(
        sorted(
            entries,
            key=lambda entry: (
                1 if entry.display_name.startswith("试作") else 0,
                entry.display_name,
            ),
        )
    )


def refresh_configured_book_folder_menu(
    *,
    workspace_root: Path,
    apply_shell_attributes: bool = True,
) -> tuple[BookMenuEntry, ...]:
    workspace_root = Path(workspace_root)
    config_path = workspace_root / "book-menu.json"
    if not config_path.is_file():
        return ()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    pool_value = payload.get("pool_path")
    if not isinstance(pool_value, str) or not pool_value.strip():
        raise ValueError("book_menu_pool_path_invalid")
    pool_path = Path(pool_value)
    if not pool_path.is_absolute():
        pool_path = workspace_root / pool_path
    return refresh_book_folder_menu(
        books_root=workspace_root / "books",
        pool_path=pool_path,
        apply_shell_attributes=apply_shell_attributes,
    )


def _read_pool_numbers(path: Path) -> dict[str, int]:
    numbers: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _POOL_ROW.match(line)
        if match is None:
            continue
        numbers[match.group("title").strip()] = int(match.group("number"))
    return numbers


def _mark_final_approved_pool_rows(path: Path, titles: set[str]) -> None:
    original = path.read_bytes().decode("utf-8")
    lines = original.splitlines(keepends=True)
    for index, line in enumerate(lines):
        match = _POOL_ROW.match(line)
        if (
            match is None
            or match.group("title").strip() not in titles
            or match.group("state").strip() != "⬜"
        ):
            continue
        start, end = match.span("state")
        lines[index] = f"{line[:start]}✅{line[end:]}"
    updated = "".join(lines)
    if updated == original:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(updated.encode("utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _episode_status(folder: Path) -> str:
    preferred = folder / "episodes" / "E001" / "episode.json"
    candidates = (preferred,) if preferred.is_file() else tuple(
        sorted((folder / "episodes").glob("*/episode.json"))
    )
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = payload.get("status")
        if isinstance(status, str):
            return status
    return "unknown"


def _production_time(folder: Path) -> datetime:
    approvals = sorted(
        folder.glob("episodes/*/deliveries/**/*最终批准*.json"),
        reverse=True,
    )
    for path in approvals:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "final_approved":
            continue
        approved_at = payload.get("approved_at")
        if isinstance(approved_at, str):
            return datetime.fromisoformat(approved_at)
    videos = sorted(folder.glob("episodes/*/media/final/final.mp4"))
    source = max(videos, key=lambda path: path.stat().st_mtime) if videos else folder
    return datetime.fromtimestamp(source.stat().st_mtime).astimezone()


def _write_desktop_ini(
    folder: Path,
    display_name: str,
    *,
    apply_shell_attributes: bool,
) -> None:
    desktop_ini = folder / "desktop.ini"
    if apply_shell_attributes and os.name == "nt" and desktop_ini.exists():
        subprocess.run(
            ["attrib", "-h", "-s", str(desktop_ini)],
            check=True,
            capture_output=True,
        )
    desktop_ini.write_text(
        "[.ShellClassInfo]\n" f"LocalizedResourceName={display_name}\n",
        encoding="utf-16",
        newline="\n",
    )
    if apply_shell_attributes and os.name == "nt":
        subprocess.run(
            ["attrib", "+r", str(folder)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["attrib", "+h", "+s", str(desktop_ini)],
            check=True,
            capture_output=True,
        )
