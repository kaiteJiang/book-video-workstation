import {
  AbsoluteFill,
  interpolate,
  Sequence,
  useCurrentFrame,
} from 'remotion';
import {revealProgress} from './easing';
import {PAPER_COLOR} from './layout';
import {Scene} from './Scene';
import {sceneDurationFrames, transitionFramesFor} from './timeline';
import type {SafeArea, SceneData, Storyboard} from './types';

const PageFlipScene: React.FC<{
  scene: SceneData;
  safeArea: SafeArea;
  durationInFrames: number;
  transitionFrames: number;
  isLast: boolean;
}> = ({scene, safeArea, durationInFrames, transitionFrames, isLast}) => {
  const frame = useCurrentFrame();
  const progress = isLast
    ? 0
    : revealProgress(
        frame,
        durationInFrames - transitionFrames,
        transitionFrames,
      );

  return (
    <AbsoluteFill style={{backgroundColor: PAPER_COLOR, overflow: 'hidden'}}>
      <AbsoluteFill
        style={{
          transformOrigin: 'left center',
          transform: `perspective(1800px) rotateY(${progress * 86}deg)`,
          filter: `brightness(${1 - progress * 0.08})`,
          boxShadow:
            progress > 0
              ? `${18 + progress * 24}px 0 ${28 + progress * 36}px rgba(48,43,37,${0.08 + progress * 0.16})`
              : 'none',
          backfaceVisibility: 'hidden',
        }}
      >
        <Scene scene={scene} safeArea={safeArea} />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const CutStoryVideo: React.FC<{value: Storyboard}> = ({value}) => (
  <AbsoluteFill style={{backgroundColor: PAPER_COLOR}}>
    {value.scenes.map((scene) => (
      <Sequence
        key={scene.id}
        from={scene.from_frame}
        durationInFrames={sceneDurationFrames(scene)}
        name={`Scene ${scene.id}`}
      >
        <Scene scene={scene} safeArea={value.safe_area} />
      </Sequence>
    ))}
  </AbsoluteFill>
);

const PageFlipStoryVideo: React.FC<{value: Storyboard}> = ({value}) => {
  const transitionFrames = transitionFramesFor(value);

  return (
    <AbsoluteFill style={{backgroundColor: PAPER_COLOR}}>
      {value.scenes.map((scene, index) => (
        <Sequence
          key={scene.id}
          from={scene.from_frame}
          durationInFrames={sceneDurationFrames(scene)}
          name={`Page ${scene.id}`}
        >
          <PageFlipScene
            scene={scene}
            safeArea={value.safe_area}
            durationInFrames={sceneDurationFrames(scene)}
            transitionFrames={transitionFrames}
            isLast={index === value.scenes.length - 1}
          />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

const CrossDissolveScene: React.FC<{
  scene: SceneData;
  safeArea: SafeArea;
  transitionFrames: number;
  isFirst: boolean;
}> = ({scene, safeArea, transitionFrames, isFirst}) => {
  const frame = useCurrentFrame();
  const opacity = isFirst
    ? 1
    : interpolate(frame, [0, transitionFrames], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      });

  return (
    <AbsoluteFill style={{opacity}}>
      <Scene
        scene={scene}
        safeArea={safeArea}
        immediateIllustration
      />
    </AbsoluteFill>
  );
};

const CrossDissolveStoryVideo: React.FC<{value: Storyboard}> = ({value}) => {
  const transitionFrames = transitionFramesFor(value);

  return (
    <AbsoluteFill style={{backgroundColor: PAPER_COLOR}}>
      {value.scenes.map((scene, index) => (
        <Sequence
          key={scene.id}
          from={scene.from_frame}
          durationInFrames={
            sceneDurationFrames(scene) +
            (index === value.scenes.length - 1 ? 0 : transitionFrames)
          }
          name={`Dissolve ${scene.id}`}
          style={{zIndex: index}}
        >
          <CrossDissolveScene
            scene={scene}
            safeArea={value.safe_area}
            transitionFrames={transitionFrames}
            isFirst={index === 0}
          />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

export const StoryboardVideo: React.FC<{value: Storyboard}> = ({value}) =>
  value.project.transition === 'cross-dissolve' && value.scenes.length > 1 ? (
    <CrossDissolveStoryVideo value={value} />
  ) : value.project.transition === 'page-flip' && value.scenes.length > 1 ? (
    <PageFlipStoryVideo value={value} />
  ) : (
    <CutStoryVideo value={value} />
  );

export const StoryVideo: React.FC<Storyboard> = (value) => (
  <StoryboardVideo value={value} />
);
