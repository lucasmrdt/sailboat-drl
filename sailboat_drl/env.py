import wandb
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward
from stable_baselines3.common.monitor import Monitor
from sailboat_gym import get_best_sail, SailboatLSAEnv

from .rewards import available_rewards, available_renderer
from .wrappers import available_obs_wrappers, available_act_wrappers
from .cli import args


class WandBRecordVideo(RecordVideo):
    def __init__(self, *args, wandb_args={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_args = wandb_args
        print(f'[WandBRecordVideo] fps={self.env.metadata.get("render_fps")}')

    def start_video_recorder(self):
        was_recording = self.recording
        super().start_video_recorder()
        if not was_recording:
            run = wandb.init(**self.wandb_args,
                             reinit=True,
                             job_type='eval',
                             name=f'eval-{self.step_id}')
            assert run is not None, 'failed to initialize wandb'

    def close_video_recorder(self):
        was_recording = self.recording
        super().close_video_recorder()
        if was_recording and wandb.run and self.video_recorder:
            fps = self.env.metadata.get('render_fps', 30)
            wandb.log({
                'video': wandb.Video(self.video_recorder.path, fps=fps, format='mp4'),
            })


def prepare_env(name='default', eval=False, record=False, wandb_args=None, env_name='SailboatLSAEnv-v0'):
    theta_wind = np.deg2rad(-(90+180))
    wind_speed = 2

    def create_new_wandb_run():
        if wandb_args is not None:
            job_type = 'eval' if eval else 'train'
            run = wandb.init(**wandb_args,
                             reinit=True,
                             job_type=job_type,
                             name=f'{job_type}-{name}')
            assert run is not None, 'failed to initialize wandb'

    def wind_generator_fn(seed: int | None):
        return np.array([np.cos(theta_wind), np.sin(theta_wind)])*wind_speed

    def __init():
        group_id = wandb_args['group'] if wandb_args and 'group' in wandb_args else None

        Reward = available_rewards[args.reward]
        Renderer = available_renderer[args.reward]
        ObsWrapper = available_obs_wrappers[args.obs]
        ActWrapper = available_act_wrappers[args.act]

        reward = Reward(**args.reward_args)

        if not record:
            create_new_wandb_run()

        env = gym.make(env_name,
                       renderer=Renderer(reward),
                       reward_fn=reward,
                       wind_generator_fn=wind_generator_fn,
                       container_tag=args.container_tag,
                       keep_sim_alive=True,
                       video_speed=20,
                       name=name)
        env = TimeLimit(env,
                        max_episode_steps=args.episode_duration * SailboatLSAEnv.NB_STEPS_PER_SECONDS)

        if record:
            video_folder = f'videos/{wandb_args["group"]}' if wandb_args and 'group' in wandb_args else f'videos/{group_id}'
            env = WandBRecordVideo(env,
                                   wandb_args=wandb_args, video_folder=video_folder, episode_trigger=lambda _: True, video_length=0)

        env = ObsWrapper(env, reward)
        env = ActWrapper(env, theta_sail=get_best_sail(env_name, theta_wind))
        env = FlattenObservation(env)
        env = NormalizeObservation(env)

        if not eval:
            env = NormalizeReward(env)

        env = Monitor(env)

        return env
    return __init
