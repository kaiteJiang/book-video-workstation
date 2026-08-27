import type {SceneData, Storyboard} from './types';

export const sceneDurationFrames = (scene: SceneData) =>
  scene.to_frame - scene.from_frame;

export const transitionFramesFor = (value: Storyboard) => {
  if (value.project.transition === 'cut') return 0;
  const shortestScene = Math.min(
    ...value.scenes.map((scene) => sceneDurationFrames(scene)),
  );
  return Math.min(
    value.project.transition_frames,
    Math.max(1, Math.floor(shortestScene * 0.45)),
  );
};

export const totalFramesFor = (value: Storyboard) =>
  value.project.total_frames;
