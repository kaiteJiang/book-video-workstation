import json
from pathlib import Path


def test_vendored_handdrawn_snapshot_is_pinned_and_attributed() -> None:
    root = Path("vendor/story_to_handdrawn_video")
    metadata = json.loads((root / "UPSTREAM.json").read_text("utf-8"))
    assert metadata["repository_url"] == "https://github.com/gnipbao/story-to-handdrawn-video"
    assert metadata["commit"] == "fbab5b27f4f0db61739d86f78000a39eeaa692d3"
    assert metadata["license"] == "MIT"
    assert (root / "LICENSE").read_text("utf-8").startswith("MIT License")
    styles = json.loads(
        (root / "references/handdrawn-style-library.json").read_text("utf-8")
    )
    assert len(styles["styles"]) == 20


def test_vendored_renderer_records_and_pins_security_patch_versions() -> None:
    root = Path("vendor/story_to_handdrawn_video")
    metadata = json.loads((root / "UPSTREAM.json").read_text("utf-8"))
    lock = json.loads((root / "package-lock.json").read_text("utf-8"))

    assert "transitive dependency security patches" in metadata["local_changes"]
    packages = lock["packages"]
    assert packages["node_modules/fast-uri"]["version"] == "3.1.5"
    assert packages["node_modules/nanoid"]["version"] == "3.3.18"
    assert packages["node_modules/postcss"]["version"] == "8.5.26"
