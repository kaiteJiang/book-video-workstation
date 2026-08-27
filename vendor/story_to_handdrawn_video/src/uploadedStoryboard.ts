import rawUploadedStoryboard from '../storyboard.uploaded.json';
import {totalFramesFor} from './storyboard';
import type {Storyboard} from './types';

// Legacy 3:4 uploads stay readable but are no longer registered for V0.2
// rendering. Keep the cast isolated until that optional path is migrated.
export const uploadedStoryboard = rawUploadedStoryboard as unknown as Storyboard;
export const uploadedTotalFrames = totalFramesFor(uploadedStoryboard);
