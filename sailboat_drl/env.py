from typing import Callable
import wandb
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward
from gymnasium.wrappers.rescale_action import RescaleAction
from stable_baselines3.common.monitor import Monitor
from sailboat_gym import EPISODE_LENGTH

from .rewards import available_rewards, available_renderer
from .wrappers import available_obs_wrappers, available_act_wrappers
from .cli import args


class WandBRecordVideo(RecordVideo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.env.metadata)
        print(
            f'[WandBRecordVideo] fps={self.env.metadata.get("render_fps")}')

    def close_video_recorder(self):
        was_recording = self.recording
        super().close_video_recorder()
        if was_recording and wandb.run is not None:
            print('saving video...')
            fps = self.env.metadata.get('render_fps', 30)
            wandb.log({
                'video': wandb.Video(self.video_recorder.path, fps=fps, format='mp4'),
                'step': self.step_id,
            })


class CbWrapper(gym.Wrapper):
    def __init__(self, env, reset_cb=None, step_cb=None, close_cb=None):
        super().__init__(env)
        self.reset_cb = reset_cb
        self.step_cb = step_cb
        self.close_cb = close_cb

    def reset(self, **kwargs):
        if self.reset_cb is not None:
            self.reset_cb()
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.step_cb is not None:
            self.step_cb(action)
        return self.env.step(action)

    def close(self):
        if self.close_cb is not None:
            self.close_cb()
        return self.env.close()


def wind_generator_fn(seed: int | None):
    return [-1, 0]


def prepare_env(name='default', eval=False, wandb_args=None):
    def __init():
        if wandb_args is not None:
            run = wandb.init(**wandb_args)
            run_id = run.id
        else:
            run_id = 'default'

        Reward = available_rewards[args.reward]
        Renderer = available_renderer[args.reward]
        ObsWrapper = available_obs_wrappers[args.obs]
        ActWrapper = available_act_wrappers[args.act]

        reward = Reward(**args.reward_args)

        env = gym.make('SailboatLSAEnv-v0',
                       renderer=Renderer(reward),
                       reward_fn=reward,
                       wind_generator_fn=wind_generator_fn,
                       container_tag=args.container_tag,
                       keep_sim_alive=True,
                       video_speed=6,
                       name=name)
        env = TimeLimit(env, max_episode_steps=EPISODE_LENGTH*3)

        if eval:
            # env = CbWrapper(env,
            #                 reset_cb=lambda: print(f'[{name}] Resetting...'),
            #                 step_cb=lambda _: print(f'[{name}] Step...'),
            #                 close_cb=lambda: print(f'[{name}] Closing...'))
            env = WandBRecordVideo(env,
                                   video_folder=f'videos/{run_id}', episode_trigger=lambda x: True, video_length=0)

        env = ObsWrapper(env, reward)
        env = ActWrapper(env, theta_sail=0)
        env = FlattenObservation(env)
        env = NormalizeObservation(env)
        if not eval:
            env = NormalizeReward(env)

        env = Monitor(env)

        return env
    return __init
