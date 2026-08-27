import type {CalculateMetadataFunction} from 'remotion';
import {Composition} from 'remotion';
import {StoryVideo} from './StoryVideo';
import {storyboard} from './storyboard';
import type {Storyboard} from './types';

const calculateMetadata: CalculateMetadataFunction<Storyboard> = ({props}) => ({
  durationInFrames: props.project.total_frames,
  fps: props.project.fps,
  width: props.project.width,
  height: props.project.height,
  props,
});

export const RemotionRoot: React.FC = () => (
  <Composition
    id="PictureSilent"
    component={StoryVideo}
    defaultProps={storyboard}
    calculateMetadata={calculateMetadata}
  />
);
