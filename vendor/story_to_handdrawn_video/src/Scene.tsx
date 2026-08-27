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
}> = ({scene, safeArea, immediateIllustration = false}) => {
  const {fps} = useVideoConfig();
  const total = sceneDurationFrames(scene);
  const at = (ratio: number) => Math.max(1, Math.round(total * ratio));
  const layout = sceneLayout(safeArea);

  if (fps !== 30) {
    throw new Error('storyboard_fps_must_be_30');
  }

  return (
    <AbsoluteFill style={{backgroundColor: PAPER_COLOR, overflow: 'hidden'}}>
      <KeyLineWipe
        text={scene.key_line}
        startFrame={0}
        durationFrames={at(0.2)}
        box={layout.keyLine}
      />
      <LayerWipe
        src={scene.assets.bw}
        startFrame={at(0.18)}
        durationFrames={at(0.42)}
        zIndex={10}
        treatment="bw"
        box={layout.illustration}
        visibleFromStart={immediateIllustration}
      />
      <LayerWipe
        src={scene.assets.color}
        startFrame={at(0.52)}
        durationFrames={at(0.36)}
        zIndex={30}
        treatment="color"
        box={layout.illustration}
      />
    </AbsoluteFill>
  );
};
