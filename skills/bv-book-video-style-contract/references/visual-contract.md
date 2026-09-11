# Visual Contract

## Stable presentation layer

| Area | Contract |
|---|---|
| Canvas | 1080x1920, 30 fps, H.264, yuv420p |
| Duration | Driven by final narration; story mode recommends 6-8 minutes and flags over 10 minutes for review without rejecting good shorter or longer stories |
| Scenes | Formal films use 3-48 complete narrative units; pair count follows the story. An explicitly requested standalone prototype may use one complete unit. |
| Title | Book title and author centered at the top for the full video |
| Title geometry | center x=540, top=160, title 88 px, author 44 px, 18 px gap; keep the full title block below the top 150 px phone-UI danger zone |
| Scene start | Color A is fully visible from local frame 0 |
| Semantic turn | The approved narration semantic-turn span is mapped through final-audio ASR to `semantic_turn_frame`; fixed percentages are not a substitute |
| Ink reveal | At the semantic turn, reveal color B over exactly 45 frames through five or more overlapping soft radial ink blooms; no straight wipe edge, paper-white flash, or empty canvas |
| Continuity | Follow bv-story-unit-ab: A establishes the unit; B shows its climax/result after narrated development. Bind the exact current A path/SHA-256 for identity/style; allow justified changes in time, place, age, clothes and camera. A micro-gesture is not a narrative unit. |
| B hold | B is fully visible before the final 15 frames and remains visible until the transition |
| Transition | Complete B cross-dissolves to complete next-scene A for 15 frames with a genuine blended midpoint and no white flash |
| Subtitles | Microsoft YaHei, 68 px bold, white, dark outline, bottom centered, no background box |
| Subtitle ASS | BorderStyle=1, transparent BackColour, Outline > 0, Shadow=0, Alignment=2, MarginV=420; keep captions above bottom labels and descriptions |
| Subtitle text | One line per cue, punctuation-free final display, natural closed spoken chunks |

Each semantic turn must be ASR-alignable and leave at least 30 complete A frames before the reveal, 45 reveal frames, and at least 30 complete B frames before the 15-frame transition. If it cannot, repair the approved scene boundary/alignment; do not move the turn to a fixed midpoint.

## Replaceable illustration layer

A new visual style may change only:

- `style_id` and style fingerprint;
- medium, brushwork, texture, palette, color rules, and character rendering language;
- style prompt atoms and their hashes;
- all color A/B masters and their asset hashes.

It must preserve:

- approved narration, voice, ASR, subtitle text, and timing;
- scene IDs/order, start/end time, story-pair meaning, semantic-turn span, setting, characters, relationships, composition intent, and A-to-B action progression;
- title and author content and geometry;
- reveal timing, bloom behavior, transition, subtitle styling, and render graph.

## Style-only replacement check

Before generating replacement images, compare the old and new storyboard after removing only style metadata, prompt hashes, asset paths, and asset hashes. The remaining canonical JSON must match exactly. Regenerate both A and B for every pair, preserving the semantic turn and story-pair meaning. Store new assets in a versioned or style-specific directory; never overwrite approved or rejected evidence.

Example: changing `retro-gouache-concept` to `warm-flat-storybook` may change paint texture, palette, and character drawing. It may not turn a hospital interaction into a symbolic landscape, move the title, add a scene, change the semantic turn/subtitle timing, or replace the ink-bloom reveal.

## Representative-pair gate

Generate A before B. Show exactly three complete pairs chosen from the opening, a meaningful user-approved middle event, and the ending; each pair must include both current-hash-bound assets. The legacy four-scene choice remains S01/S03/S04 and a three-scene episode uses all pairs. Approval authorizes only the remaining pairs. A changed A invalidates its B and downstream representative approval; a changed B invalidates the representative approval and downstream visual work while keeping A.

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
