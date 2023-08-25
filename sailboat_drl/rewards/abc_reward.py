import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod
from sailboat_gym import Observation, Action

from ..utils import norm, smallest_signed_angle, rotate_vector


class AbcReward(ABC):
    def __init__(self, path: list):
        assert len(path) == 2 and len(path[0]) == 2 and len(
            path[1]) == 2, 'Path must be a list of 2D vectors'
        self.path = np.array(path, dtype=np.float32)

    def _compute_xte(self, obs: Observation):
        path = self.path
        p_boat = obs['p_boat'][0:2]  # X and Y axis
        d = (path[1] - path[0])
        n = np.array([-d[1], d[0]], dtype=float)  # Normal vector to the path
        n /= norm(n)
        xte = np.dot(path[0] - p_boat, n)
        return xte

    def _compute_tae(self, obs: Observation):
        # get absolute velocity angle
        v = obs['dt_p_boat'][0:2]  # X and Y axis
        theta_boat = obs['theta_boat'][2]
        v = rotate_vector(v, theta_boat)
        v_angle = np.arctan2(v[1], v[0])

        # get absolute target angle
        path = self.path
        d = path[1] - path[0]
        target_angle = np.arctan2(d[1], d[0])

        tae = smallest_signed_angle(target_angle - v_angle)
        return tae

    def _compute_vmc(self, obs: Observation):
        # get absolute velocity
        v = obs['dt_p_boat'][0:2]  # X and Y axis
        theta_boat = obs['theta_boat'][2]
        v = rotate_vector(v, theta_boat)

        # get absolute path
        path = self.path
        d = path[1] - path[0]

        vmc = np.dot(v, d) / norm(d)
        return vmc

    @property
    def observation_space(self):
        return spaces.Dict({
            'xte': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'tae': spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            'vmc': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        return {
            'xte': np.array([self._compute_xte(obs)]),
            'tae': np.array([self._compute_tae(obs)]),
            'vmc': np.array([self._compute_vmc(obs)]),
        }

    @abstractmethod
    def reward_fn(self, obs: Observation, act: Action, next_obs: Observation) -> float:
        raise NotImplementedError

    @abstractmethod
    def stop_condition_fn(self, obs: Observation, act: Action, next_obs: Observation) -> bool:
        raise NotImplementedError
