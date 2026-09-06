# Visual Contract

## Stable presentation layer

| Area | Contract |
|---|---|
| Canvas | 1080x1920, 30 fps, H.264, yuv420p |
| Duration | 30-45 seconds, driven by final narration |
| Scenes | 3 or 4 stable scene IDs, each with one full-color A/B story pair (6 or 8 masters) |
| Title | Book title and author centered at the top for the full video |
| Title geometry | center x=540, top=104, title 88 px, author 44 px, 18 px gap |
| Scene start | Color A is fully visible from local frame 0 |
| Semantic turn | The approved narration semantic-turn span is mapped through final-audio ASR to `semantic_turn_frame`; fixed percentages are not a substitute |
| Ink reveal | At the semantic turn, reveal color B over exactly 45 frames through five or more overlapping soft radial ink blooms; no straight wipe edge, paper-white flash, or empty canvas |
| Continuity | B uses the exact current A path and SHA-256 as first-priority reference and advances one narration-supported action with the same identity, clothing, setting, camera direction, and style |
| B hold | B is fully visible before the final 15 frames and remains visible until the transition |
| Transition | Complete B cross-dissolves to complete next-scene A for 15 frames with a genuine blended midpoint and no white flash |
| Subtitles | Microsoft YaHei, 68 px bold, white, dark outline, bottom centered, no background box |
| Subtitle ASS | BorderStyle=1, transparent BackColour, Outline > 0, Shadow=0, Alignment=2 |
| Subtitle text | One line per cue, punctuation-free final display, natural closed spoken chunks |

Each semantic turn must be ASR-alignable and leave at least 30 complete A frames before the reveal, 45 reveal frames, and at least 30 complete B frames before the 15-frame transition. If it cannot, repair the approved scene boundary/alignment; do not move the turn to a fixed midpoint.

## Replaceable illustration layer

A new visual style may change only:

- `style_id` and style fingerprint;
- medium, brushwork, texture, palette, color rules, and character rendering language;
- style prompt atoms and their hashes;
- the 6--8 color A/B masters and their asset hashes.

It must preserve:

- approved narration, voice, ASR, subtitle text, and timing;
- 3--4 scene IDs/order, start/end time, story-pair meaning, semantic-turn span, setting, characters, relationships, composition intent, and A-to-B action progression;
- title and author content and geometry;
- reveal timing, bloom behavior, transition, subtitle styling, and render graph.

## Style-only replacement check

Before generating replacement images, compare the old and new storyboard after removing only style metadata, prompt hashes, asset paths, and asset hashes. The remaining canonical JSON must match exactly. Regenerate both A and B for every pair, preserving the semantic turn and story-pair meaning. Store new assets in a versioned or style-specific directory; never overwrite approved or rejected evidence.

Example: changing `retro-gouache-concept` to `warm-flat-storybook` may change paint texture, palette, and character drawing. It may not turn a hospital interaction into a symbolic landscape, move the title, add a scene, change the semantic turn/subtitle timing, or replace the ink-bloom reveal.

## Representative-pair gate

Generate A before B. For a four-scene Episode, show S01-A/B, S03-A/B, and S04-A/B together as six representative images; each pair must include both current-hash-bound assets. Their approval authorizes only S02-A then S02-B. For a three-scene Episode, all three pairs are representatives. A changed A invalidates its B and downstream representative approval; a changed B invalidates the representative approval and downstream visual work while keeping A.

## Required evidence

- `subtitle_breaks.txt` and a passing `jl-oral-linebreaks` line check;
- local frame 0 and the pre-turn frame showing complete color A;
- ink midpoint containing localized pixels from both A and B, with five or more soft blooms;
- post-turn frame showing complete color B before the cross-scene transition;
- start, midpoint, and end frames for all three transitions;
- representative frames showing title placement and background-free subtitles;
- ffprobe facts, loudness measurement, manifests, hashes, and human final review.

## Legacy compatibility

`legacy-monochrome-reveal` remains supported only to read and render existing legacy Episodes with historical `{bw, color}` assets. It keeps its historical local-frame BW reveal behavior. New and restyled work uses `color-story-pair` with `{anchor, continuation}` assets, full-color A from frame 0, and no BW derivative.
