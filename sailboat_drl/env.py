import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.normalize import NormalizeObservation
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from stable_baselines3.common.monitor import Monitor
from sailboat_gym import Action, Observation

from .rewards import available_rewards, available_renderer
from .cli import args


class CbWrapper(gym.Wrapper):
    def __init__(self, env, reset_cb=None, step_cb=None):
        super().__init__(env)
        self.reset_cb = reset_cb
        self.step_cb = step_cb

    def reset(self, **kwargs):
        if self.reset_cb is not None:
            self.reset_cb()
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.step_cb is not None:
            self.step_cb(action)
        return self.env.step(action)


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env, theta_sail: float):
        super().__init__(env)
        self.theta_sail = np.array([theta_sail])
        self.action_space = gym.spaces.Box(
            low=np.array([-1], dtype=np.float32),
            high=np.array([1], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )

    def action(self, action: np.ndarray[1]) -> Action:
        return {
            'theta_rudder': np.clip(action[0] * np.pi, -np.pi, np.pi),
            'theta_sail': self.theta_sail
        }


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_dim: int):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf]*obs_dim, dtype=np.float32),
            high=np.array([np.inf]*obs_dim, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32
        )

    def __map_obs_to_numpy(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs[key] for key in obs.keys()])

    def observation(self, observation: Observation) -> np.ndarray:
        return self.__map_obs_to_numpy(observation)


def wind_generator_fn(seed: int | None):
    return [-2, 0]


def prepare_env(name='default', record=False, run_id='default'):
    def __init():
        Reward = available_rewards[args.reward]
        Renderer = available_renderer[args.reward]

        reward = Reward(**args.reward_args)

        env = gym.make('SailboatLSAEnv-v0',
                       renderer=Renderer(reward),
                       reward_fn=reward,
                       wind_generator_fn=wind_generator_fn,
                       container_tag=args.container_tag,
                       keep_sim_alive=True,
                       name=name)
        if record:
            env = RecordVideo(
                env, video_folder=f'output/videos/{run_id}', episode_trigger=lambda _: True)

        env = TransformObservation(env, f=reward.transform_obs)
        env = FilterObservation(env, args.selected_obs)
        env = ActionWrapper(env,
                            theta_sail=np.pi / 4)
        env = FlattenObservation(env)
        env = NormalizeObservation(env)
        env = Monitor(env)

        env = CbWrapper(env,
                        reset_cb=lambda: print(f'[{name}] Resetting...'))

        return env
    return __init
