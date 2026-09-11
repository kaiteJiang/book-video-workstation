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
4. For long-form, use [bv-story-unit-ab](../bv-story-unit-ab/SKILL.md) for narrative semantics. Each A/B pair spans a complete story unit from initial situation through narrated development to climax/result; never reduce it to a gesture change. Keep identity and style, allowing story-supported changes in time, location, age, clothing and camera. Generate A first and bind B to its current path/SHA-256 as identity reference, not a frozen shot. ASR controls when the result is revealed. Formal videos use 3--48 units; a user-requested standalone prototype may show one complete unit without manufacturing three pairs.
5. For a style replacement, preserve the approved storyboard after excluding only style fingerprints, prompts, asset paths, and asset hashes. Regenerate both A and B in every pair while preserving its semantic turn and story-pair meaning; never rerun writing, TTS, ASR, subtitle timing, or semantic scene planning.
6. Show exactly three complete A/B pairs together: the opening, one user-approved meaningful middle event, and the ending. For the legacy four-scene layout this remains S01, S03, and S04; for three scenes all pairs remain representatives. Follow the current user authorization; do not request the same approval again when final production is already explicitly authorized. A representative-only approval authorizes only the remaining pairs. This approval does not authorize changes to narration, voice, subtitles, cover, QC, delivery, or final approval.
7. Render and inspect the required A, ink midpoint, B, transition, title, and subtitle frames. Keep the result as a candidate until the user approves it.

## Style boundary

Replaceable: style ID, medium, brushwork, texture, palette, character rendering language, style prompt atoms, and all full-color A/B assets derived from them.

Stable: story-pair meaning, semantic-turn span and ASR mapping, continuity constraints, layout, typography, subtitle treatment, 45-frame multi-point ink bloom, 15-frame cross-dissolve, and final media specifications.

A user-requested story-unit restructure is distinct from style-only replacement. Follow the user's authorized scope, preserve historical artifacts, and regenerate affected dependencies; do not claim the old approval applies to changed narration or story structure.

## Legacy compatibility

`legacy-monochrome-reveal` is read/render compatibility for already-existing legacy Episodes only. It retains its historical BW/color assets and timing. New or restyled work uses full-color A/B story pairs and must not create a BW layer or use a fixed midpoint.
