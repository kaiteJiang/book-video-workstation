import {Img, staticFile, useCurrentFrame} from 'remotion';
import {revealProgress} from './easing';
import type {LayoutBox} from './layout';

type LayerWipeProps = {
  src: string;
  startFrame: number;
  durationFrames: number;
  zIndex: number;
  treatment: 'bw' | 'color';
  box: LayoutBox;
  visibleFromStart?: boolean;
};

const treatmentFilter = {
  bw: 'grayscale(1) contrast(1.72) brightness(1.12)',
  color: 'brightness(1.035) contrast(1.04)',
} as const;

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
  const progress = visibleFromStart
    ? 1
    : revealProgress(frame, startFrame, durationFrames);

  return (
    <div
      style={{
        position: 'absolute',
        zIndex,
        top: box.top,
        height: box.height,
        left: box.left,
        right: box.right,
        clipPath: `inset(0 ${100 - progress * 100}% 0 0)`,
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
