from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper
from gymnasium.wrappers.record_video import RecordVideo
from sailboat_gym import get_best_sail, Action, SailboatEnv

from ..utils import is_env_instance, extract_env_instance
from ..logger import Logger


class BestFixedSail(gym.Wrapper):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        assert not is_env_instance(env, ObservationWrapper), 'env must not be wrapped by an ObservationWrapper as it depends on the raw original observation'
        self.theta_sail = None

    def reset(self, **kwargs) -> Any:
        obs, info = super().reset(**kwargs)
        assert self.env.unwrapped.spec is not None, 'env must be registered'
        env_name = self.env.unwrapped.spec.id
        wind = obs['wind']
        wind_angle = np.arctan2(wind[1], wind[0])
        self.theta_sail = get_best_sail(env_name, wind_angle)
        return obs, info


class RudderAngleAction(BestFixedSail, ActionWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Box(
            low=np.array([-1], dtype=np.float32),
            high=np.array([1], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )

    def action(self, action: np.ndarray) -> Action:
        assert self.theta_sail is not None, 'theta_sail must be set by reset'

        theta_rudder = np.clip(action[0], -np.pi/2, np.pi/2)
        Logger.record({'act/mlp': action[0], 'act/theta_rudder': theta_rudder})
        return {
            'theta_rudder': theta_rudder,
            'theta_sail': self.theta_sail
        }


class RudderForceAction(BestFixedSail, ActionWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sailboat_env = extract_env_instance(self.env, SailboatEnv)
        assert sailboat_env is not None, 'env must inherit from SailboatEnv'

        # in 5 second we can turn the rudder at most pi
        self.dt = np.pi/(sailboat_env.NB_STEPS_PER_SECONDS*5)

        self.theta_rudder = np.array([0])  # in rad
        self.directions = [-1, 1, 0]
        # -1: boat is turning left
        # 1: boat is turning right
        # 0: boat is going straight
        self.action_space = gym.spaces.Discrete(len(self.directions))

    def reset(self, **kwargs):
        self.theta_rudder = np.array([0])  # reset rudder angle
        return super().reset(**kwargs)
    
    def action(self, action: int) -> Action:
        assert self.theta_sail is not None, 'theta_sail must be set by reset'

        direction = self.directions[action]
        self.theta_rudder = np.clip(self.theta_rudder + direction * self.dt,
                                    -np.pi/2,
                                    np.pi/2)
        Logger.record({'act/mlp': action, 'act/theta_rudder': self.theta_rudder})
        return {
            'theta_rudder': self.theta_rudder,
            'theta_sail': self.theta_sail
        }
