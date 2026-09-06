import {AbsoluteFill, useVideoConfig} from 'remotion';
import {LayerWipe} from './LayerWipe';
import {PAPER_COLOR, sceneLayout} from './layout';
import {KeyLineWipe} from './TextWipe';
import {sceneDurationFrames} from './timeline';
import type {SafeArea, SceneData} from './types';

export const Scene: React.FC<{
  scene: SceneData;
  safeArea: SafeArea;
  immediateIllustration?: boolean;
  showKeyLine?: boolean;
}> = ({scene, safeArea, immediateIllustration = false, showKeyLine = true}) => {
  const {fps} = useVideoConfig();
  const total = sceneDurationFrames(scene);
  const at = (ratio: number) => Math.max(1, Math.round(total * ratio));
  const colorStartFrame = fps * 2;
  const colorDurationFrames = Math.max(
    1,
    Math.min(at(0.36), total - colorStartFrame - 15),
  );
  const layout = sceneLayout(safeArea);

  if (fps !== 30) {
    throw new Error('storyboard_fps_must_be_30');
  }

  const illustration =
    scene.sequence_mode === 'color-story-pair' ? (
      <>
        <LayerWipe
          src={scene.assets.anchor}
          startFrame={0}
          durationFrames={1}
          zIndex={10}
          treatment="story-color"
          box={layout.illustration}
          visibleFromStart
        />
        <LayerWipe
          src={scene.assets.continuation}
          startFrame={scene.semantic_turn_frame - scene.from_frame}
          durationFrames={45}
          zIndex={30}
          treatment="story-color"
          box={layout.illustration}
        />
      </>
    ) : (
      <>
        <LayerWipe
          src={scene.assets.bw}
          startFrame={0}
          durationFrames={1}
          zIndex={10}
          treatment="bw"
          box={layout.illustration}
          visibleFromStart
        />
        <LayerWipe
          src={scene.assets.color}
          startFrame={colorStartFrame}
          durationFrames={colorDurationFrames}
          zIndex={30}
          treatment="color"
          box={layout.illustration}
        />
      </>
    );

  return (
    <AbsoluteFill style={{backgroundColor: PAPER_COLOR, overflow: 'hidden'}}>
      {showKeyLine ? (
        <KeyLineWipe
          text={scene.key_line}
          startFrame={0}
          durationFrames={at(0.2)}
          box={layout.keyLine}
        />
      ) : null}
      {illustration}
    </AbsoluteFill>
  );
};
