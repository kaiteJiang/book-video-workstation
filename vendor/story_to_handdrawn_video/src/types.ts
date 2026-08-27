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

export type SceneAssets = {
  bw: string;
  color: string;
};

export type SceneData = {
  id: string;
  start_ms: number;
  end_ms: number;
  from_frame: number;
  to_frame: number;
  key_line: string;
  narration: string;
  assets: SceneAssets;
};

export type Storyboard = {
  project: {
    width: 1080;
    height: 1920;
    fps: 30;
    ratio: '9:16';
    total_frames: number;
    transition: 'cut' | 'page-flip' | 'cross-dissolve';
    transition_frames: number;
  };
  safe_area: SafeArea;
  scenes: SceneData[];
};
