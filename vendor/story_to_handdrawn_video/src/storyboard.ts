import rawStoryboard from '../storyboard.9x16.fixture.json';
import {totalFramesFor, transitionFramesFor} from './timeline';
import type {Storyboard} from './types';

export const storyboard = rawStoryboard as Storyboard;
export {totalFramesFor, transitionFramesFor};
export const totalFrames = totalFramesFor(storyboard);
