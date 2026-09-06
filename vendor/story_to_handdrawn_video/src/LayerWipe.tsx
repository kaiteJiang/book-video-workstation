import {Img, staticFile, useCurrentFrame} from 'remotion';
import {revealProgress} from './easing';
import type {LayoutBox} from './layout';

type LayerWipeProps = {
  src: string;
  startFrame: number;
  durationFrames: number;
  zIndex: number;
  treatment: 'bw' | 'color' | 'story-color';
  box: LayoutBox;
  visibleFromStart?: boolean;
};

const treatmentFilter = {
  bw: 'grayscale(1) contrast(1.72) brightness(1.12)',
  color: 'brightness(1.035) contrast(1.04)',
  'story-color': 'brightness(1.035) contrast(1.04)',
} as const;

const watercolorBlooms = [
  {x: 46, y: 67, delay: 0, width: 52, height: 43},
  {x: 22, y: 49, delay: 0.12, width: 48, height: 39},
  {x: 78, y: 57, delay: 0.18, width: 50, height: 42},
  {x: 58, y: 34, delay: 0.3, width: 46, height: 36},
  {x: 35, y: 82, delay: 0.4, width: 45, height: 34},
] as const;

const clamp = (value: number) => Math.min(1, Math.max(0, value));

const watercolorMask = (progress: number) => {
  if (progress >= 0.999) {
    return 'linear-gradient(#000 0 0)';
  }

  const blooms = watercolorBlooms.map((bloom) => {
    const local = clamp((progress - bloom.delay) / (1 - bloom.delay));
    const opacity = clamp(local * 1.65);
    const width = 2 + bloom.width * local;
    const height = 2 + bloom.height * local;
    return (
      `radial-gradient(ellipse ${width}% ${height}% at ${bloom.x}% ${bloom.y}%, ` +
      `rgba(0,0,0,${opacity}) 0%, rgba(0,0,0,${opacity * 0.9}) 48%, ` +
      'rgba(0,0,0,0) 82%)'
    );
  });
  const finalWash = clamp((progress - 0.72) / 0.28);
  blooms.push(
    `linear-gradient(rgba(0,0,0,${finalWash}), rgba(0,0,0,${finalWash}))`,
  );
  return blooms.join(', ');
};

export const LayerWipe: React.FC<LayerWipeProps> = ({
  src,
  startFrame,
  durationFrames,
  zIndex,
  treatment,
  box,
  visibleFromStart = false,
}) => {
  const frame = useCurrentFrame();
  const easedProgress = revealProgress(frame, startFrame, durationFrames);
  const progress = visibleFromStart
    ? 1
    : frame > startFrame
      ? Math.max(0.03, easedProgress)
      : easedProgress;
  const maskImage = watercolorMask(progress);

  return (
    <div
      style={{
        position: 'absolute',
        zIndex,
        top: box.top,
        height: box.height,
        left: box.left,
        right: box.right,
        WebkitMaskImage: maskImage,
        maskImage,
        WebkitMaskRepeat: 'no-repeat',
        maskRepeat: 'no-repeat',
        overflow: 'hidden',
      }}
    >
      <Img
        src={staticFile(src)}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          objectPosition: 'center center',
          filter: treatmentFilter[treatment],
        }}
      />
    </div>
  );
};
