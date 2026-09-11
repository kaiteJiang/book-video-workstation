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

const TitleOverlay: React.FC<{
  value: Storyboard['title_overlay'];
}> = ({value}) =>
  value ? (
    <AbsoluteFill
      style={{
        zIndex: 100,
        pointerEvents: 'none',
        alignItems: 'center',
        paddingTop: 160,
        color: '#123858',
        fontFamily:
          'Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif',
        textAlign: 'center',
      }}
    >
      <div
        style={{
          fontSize: 88,
          fontWeight: 700,
          lineHeight: 1.12,
          letterSpacing: 8,
          textShadow: '0 2px 10px rgba(255,253,247,0.9)',
        }}
      >
        {value.title}
      </div>
      <div
        style={{
          marginTop: 18,
          fontSize: 44,
          fontWeight: 500,
          lineHeight: 1.2,
          letterSpacing: 10,
          textShadow: '0 2px 8px rgba(255,253,247,0.9)',
        }}
      >
        {value.author}
      </div>
    </AbsoluteFill>
  ) : null;

const PageFlipScene: React.FC<{
  scene: SceneData;
  safeArea: SafeArea;
  durationInFrames: number;
  transitionFrames: number;
  isLast: boolean;
  showKeyLine: boolean;
}> = ({scene, safeArea, durationInFrames, transitionFrames, isLast, showKeyLine}) => {
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
        <Scene scene={scene} safeArea={safeArea} showKeyLine={showKeyLine} />
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
        <Scene
          scene={scene}
          safeArea={value.safe_area}
          showKeyLine={value.project.show_key_line !== false}
        />
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
            showKeyLine={value.project.show_key_line !== false}
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
  showKeyLine: boolean;
}> = ({scene, safeArea, transitionFrames, isFirst, showKeyLine}) => {
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
        showKeyLine={showKeyLine}
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
            showKeyLine={value.project.show_key_line !== false}
          />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

export const StoryboardVideo: React.FC<{value: Storyboard}> = ({value}) => (
  <AbsoluteFill style={{backgroundColor: PAPER_COLOR}}>
    {value.project.transition === 'cross-dissolve' && value.scenes.length > 1 ? (
      <CrossDissolveStoryVideo value={value} />
    ) : value.project.transition === 'page-flip' && value.scenes.length > 1 ? (
      <PageFlipStoryVideo value={value} />
    ) : (
      <CutStoryVideo value={value} />
    )}
    <TitleOverlay value={value.title_overlay} />
  </AbsoluteFill>
  );

export const StoryVideo: React.FC<Storyboard> = (value) => (
  <StoryboardVideo value={value} />
);
