export type SafeArea = {
  top_reserved: number;
  keyline_top: number;
  keyline_bottom: number;
  illustration_top: number;
  illustration_bottom: number;
  subtitle_top: number;
  subtitle_bottom: number;
  bottom_reserved: number;
  right_reserved: number;
};

export type LegacySceneAssets = {
  bw: string;
  color: string;
};

export type StoryPairAssets = {
  anchor: string;
  continuation: string;
};

type SceneBase = {
  id: string;
  start_ms: number;
  end_ms: number;
  from_frame: number;
  to_frame: number;
  key_line: string;
  narration: string;
};

export type LegacySceneData = SceneBase & {
  sequence_mode?: 'legacy-monochrome-reveal';
  assets: LegacySceneAssets;
};

export type StoryPairSceneData = SceneBase & {
  sequence_mode: 'color-story-pair';
  semantic_turn_frame: number;
  assets: StoryPairAssets;
};

export type SceneData = LegacySceneData | StoryPairSceneData;

export type Storyboard = {
  project: {
    width: 1080;
    height: 1920;
    fps: 30;
    ratio: '9:16';
    total_frames: number;
    transition: 'cut' | 'page-flip' | 'cross-dissolve';
    transition_frames: number;
    ink_reveal_frames?: number;
    show_key_line?: boolean;
  };
  safe_area: SafeArea;
  title_overlay?: {
    title: string;
    author: string;
  };
  scenes: SceneData[];
};
