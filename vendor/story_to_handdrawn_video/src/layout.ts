import type {SafeArea} from './types';

export type LayoutBox = {
  top: number;
  height: number;
  left: number;
  right: number;
};

export const PAPER_COLOR = '#fffdf7';

const CONTENT_LEFT = 72;

export const sceneLayout = (safeArea: SafeArea) => ({
  keyLine: {
    top: safeArea.keyline_top,
    height: safeArea.keyline_bottom - safeArea.keyline_top,
    left: CONTENT_LEFT,
    right: safeArea.right_reserved,
  } satisfies LayoutBox,
  illustration: {
    top: 0,
    height: 1920,
    left: 0,
    right: 0,
  } satisfies LayoutBox,
  subtitle: {
    top: safeArea.subtitle_top,
    height: safeArea.subtitle_bottom - safeArea.subtitle_top,
    left: CONTENT_LEFT,
    right: safeArea.right_reserved,
  } satisfies LayoutBox,
});
