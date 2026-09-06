---
name: bv-book-video-style-contract
description: Use when producing or restyling Chinese book videos in BV_Workstation with color story pairs, ASR-aligned ink transitions, title placement, subtitles, transitions, or style replacement that must remain consistent.
---

# BV Book Video Style Contract

Keep the approved story-pair meaning and presentation mechanics stable while letting illustration style change independently. A style swap changes how each pair looks; it does not rewrite narration, retime ASR alignment, reframe the pair, or restructure the video.

## Required workflow

1. Read [the visual contract](references/visual-contract.md) before planning illustrations, rendering a sample, restyling an episode, or approving final media.
2. Treat the final narration as the global clock. Use its final ASR alignment to map each approved narration semantic turn to a scene frame; preserve the script, audio, subtitle timing, scene boundaries, story-pair meanings, title overlay, transitions, and encoding contract.
3. **REQUIRED SUB-SKILL:** Use `jl-oral-linebreaks` in short-subtitle mode before subtitle generation. Save `script/subtitle_breaks.txt`, run its line checker with `--max-cjk 14`, and require the reconstructed text to match the approved narration.
4. Plan 3--4 scenes as 6--8 full-color story-pair masters. For every scene, generate color A first from local frame 0, then generate color B with the exact current A path and SHA-256 as its first-priority image reference. B keeps A's identity, clothing, setting, camera direction, and style while advancing exactly one narration-supported action.
5. For a style replacement, preserve the approved storyboard after excluding only style fingerprints, prompts, asset paths, and asset hashes. Regenerate both A and B in every pair while preserving its semantic turn and story-pair meaning; never rerun writing, TTS, ASR, subtitle timing, or semantic scene planning.
6. For four scenes, show S01-A/B, S03-A/B, and S04-A/B together as six representative images; only their approval authorizes S02-A then S02-B. For three scenes, all three pairs are representatives. This approval does not authorize changes to narration, voice, subtitles, cover, QC, delivery, or final approval.
7. Render and inspect the required A, ink midpoint, B, transition, title, and subtitle frames. Keep the result as a candidate until the user approves it.

## Style boundary

Replaceable: style ID, medium, brushwork, texture, palette, character rendering language, style prompt atoms, and the 6--8 full-color A/B assets derived from them.

Stable: story-pair meaning, semantic-turn span and ASR mapping, continuity constraints, layout, typography, subtitle treatment, 45-frame multi-point ink bloom, 15-frame cross-dissolve, and final media specifications.

If a requested style conflicts with the stable presentation layer, report the conflict and ask whether the user wants a new presentation contract. Do not silently mutate both layers.

## Legacy compatibility

`legacy-monochrome-reveal` is read/render compatibility for already-existing legacy Episodes only. It retains its historical BW/color assets and timing. New or restyled work uses full-color A/B story pairs and must not create a BW layer or use a fixed midpoint.
