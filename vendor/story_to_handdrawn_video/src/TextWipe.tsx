import type {CSSProperties} from 'react';
import {useCurrentFrame} from 'remotion';
import {revealProgress} from './easing';
import type {LayoutBox} from './layout';

type KeyLineWipeProps = {
  text: string;
  startFrame: number;
  durationFrames: number;
  box: LayoutBox;
};

const textStyle: CSSProperties = {
  fontFamily: 'OriginalDiaryHand, STKaiti, serif',
  fontSize: 76,
  fontWeight: 400,
  lineHeight: 1.12,
  letterSpacing: '0.035em',
  color: '#ffffff',
  WebkitTextStroke: '0.7px #171714',
  margin: 0,
  textAlign: 'left',
  whiteSpace: 'nowrap',
  transform: 'rotate(-0.35deg)',
};

export const KeyLineWipe: React.FC<KeyLineWipeProps> = ({
  text,
  startFrame,
  durationFrames,
  box,
}) => {
  const frame = useCurrentFrame();
  const progress = revealProgress(frame, startFrame, durationFrames);

  return (
    <div
      style={{
        position: 'absolute',
        zIndex: 40,
        top: box.top,
        height: box.height,
        left: box.left,
        right: box.right,
        display: 'flex',
        alignItems: 'center',
        clipPath: `inset(0 ${100 - progress * 100}% 0 0)`,
        overflow: 'hidden',
      }}
    >
      <p style={textStyle}>{text}</p>
    </div>
  );
};
