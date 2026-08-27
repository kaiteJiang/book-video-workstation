# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-08-08

### Added

- Built-in library of 20 selectable hand-drawn styles with aliases, intended
  use cases, prompt recipes, negative constraints, and source attribution.
- One square comparison sample per style plus a labeled contact sheet.
- `--style` selection by order, id, Chinese name, English name, or alias,
  and `--list-styles` catalog output with example paths.

### Changed

- Keep the approved colored-pencil diary look as the immutable default while
  allowing non-default styles to use independent prompt locks and palettes.
- Include the selected style and reference fingerprint in generated asset
  caches so switching styles cannot silently reuse stale images.
- Expand the Skill contract, UI metadata, and bilingual README with visual
  selection guidance and the complete style table.

## [1.0.0] - 2026-07-21

### Added

- Initial open-source release of `story-to-handdrawn-video`.
- Remotion renderer (`src/`): story-text beats, uploaded-page cropping,
  left-to-right `text → bw → color` reveals, page-flip transition,
  safe contained framing, silent MP4 output (final 1080×1440, preview 720×960).
- Unified CLI entry `scripts/run_story_video.py` with plan / generate /
  import / render / preview / full modes and Codex Image2 + OpenAI API generators.
- Distributable agent skill in `skill-package/story-to-handdrawn-video/`
  with `STORY_VIDEO_PROJECT` discovery and upward directory walk.
- Example story, style references, and the Ma Shan Zheng font (OFL).

[Unreleased]: https://github.com/gnipbao/story-to-handdrawn-video/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/gnipbao/story-to-handdrawn-video/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/gnipbao/story-to-handdrawn-video/releases/tag/v1.0.0
