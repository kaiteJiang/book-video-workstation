import {existsSync, readFileSync} from 'node:fs';
import {dirname, isAbsolute, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const publicRoot = resolve(root, 'public');
const minimumRevealFrames = 60;
const storyPairInkRevealFrames = 45;
const storyPairContinuationSideFrames = storyPairInkRevealFrames + 30 + 15;

const inputFiles = () => {
  const args = process.argv.slice(2);
  if (args.length >= 2 && args[0] === '--input') return args.slice(1);
  if (args.length > 0 && !args.some((arg) => arg.startsWith('--'))) return args;
  return ['storyboard.9x16.fixture.json'];
};

const finiteInteger = (value) => Number.isInteger(value) && value >= 0;
const characterCount = (value) => [...value.trim()].length;

const assetPath = (value, label, kind, errors) => {
  if (typeof value !== 'string' || value.length === 0 || isAbsolute(value)) {
    errors.push(`${label}: ${kind} asset must be a relative path`);
    return;
  }
  const absolute = resolve(publicRoot, value);
  const fromPublic = relative(publicRoot, absolute);
  if (fromPublic.startsWith('..') || isAbsolute(fromPublic)) {
    errors.push(`${label}: ${kind} asset escapes public directory`);
    return;
  }
  if (!existsSync(absolute)) {
    errors.push(`${label}: missing ${kind} asset at public/${value}`);
  }
};

const validateSafeArea = (safeArea, errors) => {
  const required = [
    'top_reserved',
    'keyline_top',
    'keyline_bottom',
    'illustration_top',
    'illustration_bottom',
    'subtitle_top',
    'subtitle_bottom',
    'bottom_reserved',
    'right_reserved',
  ];
  if (!safeArea || required.some((key) => !finiteInteger(safeArea[key]))) {
    errors.push('safe_area fields must be non-negative integers');
    return;
  }
  if (
    safeArea.top_reserved !== safeArea.keyline_top ||
    safeArea.keyline_top >= safeArea.keyline_bottom ||
    safeArea.keyline_bottom !== safeArea.illustration_top ||
    safeArea.illustration_top >= safeArea.illustration_bottom ||
    safeArea.illustration_bottom !== safeArea.subtitle_top ||
    safeArea.subtitle_top >= safeArea.subtitle_bottom ||
    safeArea.subtitle_bottom + safeArea.bottom_reserved !== 1920 ||
    safeArea.right_reserved <= 0 ||
    safeArea.right_reserved >= 1080
  ) {
    errors.push('safe_area regions must be ordered inside 1080x1920');
  }
};

const validate = (file) => {
  const absoluteFile = resolve(root, file);
  const errors = [];
  if (!existsSync(absoluteFile)) return [`missing storyboard: ${file}`];

  let storyboard;
  try {
    storyboard = JSON.parse(readFileSync(absoluteFile, 'utf8'));
  } catch (error) {
    return [`invalid JSON: ${error.message}`];
  }

  const project = storyboard.project;
  const scenes = storyboard.scenes;
  if (!project || !Array.isArray(scenes) || scenes.length === 0) {
    return ['storyboard must contain project and at least one scene'];
  }
  if (project.ratio !== '9:16') errors.push('project.ratio must be 9:16');
  if (project.width !== 1080 || project.height !== 1920) {
    errors.push('project dimensions must be 1080x1920');
  }
  if (project.fps !== 30) errors.push('project.fps must be 30');
  if (!Number.isInteger(project.total_frames) || project.total_frames <= 0) {
    errors.push('project.total_frames must be a positive integer');
  }
  if (!['cut', 'page-flip', 'cross-dissolve'].includes(project.transition)) {
    errors.push('project.transition must be cut, page-flip, or cross-dissolve');
  }
  if (!finiteInteger(project.transition_frames)) {
    errors.push('project.transition_frames must be a non-negative integer');
  }
  if (
    project.show_key_line !== undefined &&
    typeof project.show_key_line !== 'boolean'
  ) {
    errors.push('project.show_key_line must be a boolean when present');
  }
  if (
    project.ink_reveal_frames !== undefined &&
    !finiteInteger(project.ink_reveal_frames)
  ) {
    errors.push('project.ink_reveal_frames must be a non-negative integer when present');
  }
  validateSafeArea(storyboard.safe_area, errors);
  if (
    storyboard.title_overlay !== undefined &&
    (
      typeof storyboard.title_overlay?.title !== 'string' ||
      storyboard.title_overlay.title.trim().length === 0 ||
      typeof storyboard.title_overlay?.author !== 'string' ||
      storyboard.title_overlay.author.trim().length === 0
    )
  ) {
    errors.push('title_overlay title and author must be nonblank strings');
  }

  const ids = new Set();
  const hasStoryPairs = scenes.some(
    (scene) => scene?.sequence_mode === 'color-story-pair',
  );
  if (hasStoryPairs && project.ink_reveal_frames !== storyPairInkRevealFrames) {
    errors.push(`project.ink_reveal_frames must be ${storyPairInkRevealFrames} for story pairs`);
  }
  let previous = null;
  for (const scene of scenes) {
    const label = scene?.id || '(unknown scene)';
    if (!scene || typeof scene.id !== 'string' || !scene.assets) {
      errors.push(`${label}: id and assets are required`);
      continue;
    }
    if (ids.has(scene.id)) errors.push(`duplicate scene id: ${scene.id}`);
    ids.add(scene.id);
    if (
      !finiteInteger(scene.start_ms) ||
      !finiteInteger(scene.end_ms) ||
      scene.end_ms <= scene.start_ms
    ) {
      errors.push(`${label}: invalid millisecond range`);
    }
    if (
      !finiteInteger(scene.from_frame) ||
      !finiteInteger(scene.to_frame) ||
      scene.to_frame <= scene.from_frame
    ) {
      errors.push(`${label}: invalid frame range`);
    } else if (scene.to_frame - scene.from_frame < minimumRevealFrames) {
      errors.push(`${label}: scene is shorter than reveal minimum`);
    }
    if (typeof scene.key_line !== 'string') {
      errors.push(`${label}: key_line must be a string`);
    } else if (
      characterCount(scene.key_line) < 6 ||
      characterCount(scene.key_line) > 14
    ) {
      errors.push(`${label}: key_line must contain 6 to 14 characters`);
    }
    if (typeof scene.narration !== 'string' || !scene.narration.trim()) {
      errors.push(`${label}: narration must be nonblank`);
    }
    const sequenceMode = scene.sequence_mode ?? 'legacy-monochrome-reveal';
    if (!['legacy-monochrome-reveal', 'color-story-pair'].includes(sequenceMode)) {
      errors.push(`${label}: unsupported sequence_mode`);
    } else if (sequenceMode === 'color-story-pair') {
      const assetKeys = Object.keys(scene.assets).sort();
      if (assetKeys.join(',') !== 'anchor,continuation') {
        errors.push(`${label}: story-pair assets must contain only anchor and continuation`);
      }
      assetPath(scene.assets.anchor, label, 'anchor', errors);
      assetPath(scene.assets.continuation, label, 'continuation', errors);
      if (
        !finiteInteger(scene.semantic_turn_frame) ||
        scene.semantic_turn_frame - scene.from_frame < 30 ||
        scene.to_frame - scene.semantic_turn_frame < storyPairContinuationSideFrames
      ) {
        errors.push(`${label}: semantic_turn_frame must leave anchor, ink, and transition time`);
      }
    } else {
      const assetKeys = Object.keys(scene.assets).sort();
      if (assetKeys.join(',') !== 'bw,color') {
        errors.push(`${label}: legacy assets must contain only bw and color`);
      }
      assetPath(scene.assets.bw, label, 'bw', errors);
      assetPath(scene.assets.color, label, 'color', errors);
    }

    if (previous === null) {
      if (scene.from_frame !== 0) errors.push('first scene must start at frame 0');
      if (scene.start_ms !== 0) errors.push('first scene must start at 0ms');
    } else {
      if (scene.from_frame > previous.to_frame) errors.push('scene frame gap');
      if (scene.from_frame < previous.to_frame) errors.push('scene frame overlap');
      if (scene.start_ms !== previous.end_ms) errors.push('scene millisecond gap or overlap');
    }
    previous = scene;
  }

  const last = scenes.at(-1);
  if (last?.to_frame !== project.total_frames) {
    errors.push('final scene frame must equal project.total_frames');
  }
  if (
    last &&
    Math.round((last.end_ms * project.fps) / 1000) !== project.total_frames
  ) {
    errors.push('master duration does not match project.total_frames');
  }
  if (
    project.transition !== 'cut' &&
    project.transition_frames >=
      Math.min(...scenes.map((scene) => scene.to_frame - scene.from_frame))
  ) {
    errors.push('transition_frames must fit inside every scene');
  }

  if (errors.length === 0) {
    console.log(
      `✓ ${file} · ${scenes.length} scenes · ${project.total_frames} frames · 1080x1920`,
    );
  }
  return errors;
};

const files = inputFiles();
const errors = files.flatMap((file) =>
  validate(file).map((error) => `${file}: ${error}`),
);

if (errors.length > 0) {
  console.error(errors.map((error) => `✗ ${error}`).join('\n'));
  process.exit(1);
}

console.log('✓ all storyboards valid · silent picture tracks');
