from datetime import datetime
import json
from pathlib import Path

from bv.books.menu import (
    build_display_name,
    refresh_book_folder_menu,
    refresh_configured_book_folder_menu,
)


def test_build_display_name_keeps_pool_number_and_marks_pending() -> None:
    produced_at = datetime.fromisoformat("2026-08-31T16:53:00+08:00")

    assert build_display_name(
        title="认知觉醒",
        produced_at=produced_at,
        pool_number=5,
        trial_number=None,
        final_approved=True,
    ) == "005-认知觉醒-2026-08-31-16-53"
    assert build_display_name(
        title="活着",
        produced_at=produced_at,
        pool_number=10,
        trial_number=None,
        final_approved=False,
    ) == "010-活着-2026-08-31-16-53-待终审"


def test_refresh_book_folder_menu_writes_shell_names_without_renaming_ids(
    tmp_path: Path,
) -> None:
    books_root = tmp_path / "books"
    pool_path = tmp_path / "pool.md"
    pool_path.write_text(
        "| 序号 | 完成 | 书名 | 作者 |\n"
        "|---:|:---:|---|---|\n"
        "| 5 | ✅ | 《认知觉醒》 | 周岭 |\n",
        encoding="utf-8",
    )
    fixtures = (
        (
            "book-title-cognition",
            "认知觉醒",
            "2026-08-31T16:53:00+08:00",
        ),
        (
            "book-title-trial",
            "安徒生童话",
            "2026-08-28T23:45:00+08:00",
        ),
    )
    for folder_name, title, approved_at in fixtures:
        folder = books_root / folder_name
        delivery = folder / "episodes" / "E001" / "deliveries" / "2026-08-31"
        delivery.mkdir(parents=True)
        (folder / "book.json").write_text(
            json.dumps({"title": title}, ensure_ascii=False),
            encoding="utf-8",
        )
        (folder / "episodes" / "E001" / "episode.json").write_text(
            json.dumps({"status": "final_approved"}),
            encoding="utf-8",
        )
        (delivery / f"{title}-最终批准.json").write_text(
            json.dumps(
                {"status": "final_approved", "approved_at": approved_at},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    entries = refresh_book_folder_menu(
        books_root=books_root,
        pool_path=pool_path,
        apply_shell_attributes=False,
    )

    assert [entry.display_name for entry in entries] == [
        "005-认知觉醒-2026-08-31-16-53",
        "试作01-安徒生童话-2026-08-28-23-45",
    ]
    assert sorted(path.name for path in books_root.iterdir()) == [
        "book-title-cognition",
        "book-title-trial",
    ]
    assert (
        books_root / "book-title-cognition" / "desktop.ini"
    ).read_text(encoding="utf-16") == (
        "[.ShellClassInfo]\n"
        "LocalizedResourceName=005-认知觉醒-2026-08-31-16-53\n"
    )


def test_configured_refresh_reads_local_workspace_menu_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    folder = workspace / "books" / "book-title-cognition"
    delivery = folder / "episodes" / "E001" / "deliveries" / "2026-08-31"
    delivery.mkdir(parents=True)
    pool_path = tmp_path / "pool.md"
    pool_path.write_text(
        "| 5 | ✅ | 《认知觉醒》 | 周岭 |\n",
        encoding="utf-8",
    )
    (workspace / "book-menu.json").write_text(
        json.dumps({"pool_path": str(pool_path)}, ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / "book.json").write_text(
        json.dumps({"title": "认知觉醒"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / "episodes" / "E001" / "episode.json").write_text(
        json.dumps({"status": "final_approved"}),
        encoding="utf-8",
    )
    (delivery / "认知觉醒-最终批准.json").write_text(
        json.dumps(
            {
                "status": "final_approved",
                "approved_at": "2026-08-31T16:53:00+08:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entries = refresh_configured_book_folder_menu(
        workspace_root=workspace,
        apply_shell_attributes=False,
    )

    assert [entry.display_name for entry in entries] == [
        "005-认知觉醒-2026-08-31-16-53"
    ]


def test_refresh_book_folder_menu_marks_only_final_approved_pool_rows(
    tmp_path: Path,
) -> None:
    books_root = tmp_path / "books"
    pool_path = tmp_path / "pool.md"
    pool_path.write_text(
        "| 序号 | 完成 | 书名 | 作者 |\n"
        "|---:|:---:|---|---|\n"
        "| 11 | ⬜ | 《平凡的世界》 | 路遥 |\n"
        "| 12 | ⬜ | 《百年孤独》 | 加西亚·马尔克斯 |\n",
        encoding="utf-8",
    )
    for folder_name, title, status in (
        ("book-title-pingfan", "平凡的世界", "final_approved"),
        ("book-title-bainian", "百年孤独", "awaiting_final_review"),
    ):
        folder = books_root / folder_name
        episode = folder / "episodes" / "E001"
        episode.mkdir(parents=True)
        (folder / "book.json").write_text(
            json.dumps({"title": title}, ensure_ascii=False),
            encoding="utf-8",
        )
        (episode / "episode.json").write_text(
            json.dumps({"status": status}),
            encoding="utf-8",
        )

    refresh_book_folder_menu(
        books_root=books_root,
        pool_path=pool_path,
        apply_shell_attributes=False,
    )

    assert pool_path.read_text(encoding="utf-8") == (
        "| 序号 | 完成 | 书名 | 作者 |\n"
        "|---:|:---:|---|---|\n"
        "| 11 | ✅ | 《平凡的世界》 | 路遥 |\n"
        "| 12 | ⬜ | 《百年孤独》 | 加西亚·马尔克斯 |\n"
    )
