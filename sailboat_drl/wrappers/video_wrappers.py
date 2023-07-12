import tqdm
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo
from stable_baselines3.common.logger import Video

from ..utils import extract_env_instance
from ..logger import Logger

class CustomRecordVideo(RecordVideo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pbar = None
        self.idx = 0

    def reset(self, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = None
        return super().reset(**kwargs)

    def step(self, action):
        if self.pbar is None:
            timelimit_env = extract_env_instance(self.env, TimeLimit)
            episode_length = timelimit_env._max_episode_steps if timelimit_env else None
            self.pbar = tqdm.tqdm(total=episode_length, desc='[LogRecordVideo]')
        self.pbar.update(1)
        return super().step(action)

    def close_video_recorder(self):
        was_recording = self.recording
        super().close_video_recorder()
        if was_recording and self.video_recorder:
            self.idx += 1
            fps = self.env.metadata.get('render.fps', 30)
            print(f'[LogRecordVideo] logging video (idx={self.idx})')
            Logger.record({f'video/{self.idx}': self.video_recorder.path})
