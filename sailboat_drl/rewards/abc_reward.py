import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod
from sailboat_gym import Observation, Action

from ..utils import norm, smallest_signed_angle, rotate_vector


class AbcReward(ABC):
    def __init__(self, path: list, xte_threshold: float = 0.1, *args, **kwargs):
        assert len(path) == 2 and len(path[0]) == 2 and len(
            path[1]) == 2, 'Path must be a list of 2D vectors'
        self.path = np.array(path, dtype=np.float32)
        self.path_length = norm(self.path[1] - self.path[0])
        self.normalized_path = self.path / self.path_length
        self.xte_threshold = self.path_length * xte_threshold
        self._prev_obs = None
        self._current_obs = None

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

    def _compute_heading_error(self, obs: Observation):
        # get absolute velocity angle
        theta_boat = obs['theta_boat'][2]

        # get absolute target angle
        path = self.path
        d = path[1] - path[0]
        target_angle = np.arctan2(d[1], d[0])

        heading_error = smallest_signed_angle(target_angle - theta_boat)
        return heading_error

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

    def _compute_gain_dist(self, p_boat, next_p_boat):
        path = self.normalized_path
        gain_dist = (next_p_boat - p_boat).dot(path[1] - path[0])
        return gain_dist

    def _is_in_failure_state(self, obs: Observation):
        if abs(self._compute_xte(obs)) > self.xte_threshold or abs(self._compute_heading_error(obs)) > np.pi / 2:
            return True
        return False

    @property
    def observation_space(self):
        return spaces.Dict({
            'xte': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'tae': spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            'vmc': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'delta': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        return {
            'xte': np.array([self._compute_xte(obs)]),
            'tae': np.array([self._compute_tae(obs)]),
            'vmc': np.array([self._compute_vmc(obs)]),
            'delta': np.array([self.xte_threshold - abs(self._compute_xte(obs))]),
        }

    def on_reset(self, obs, info):
        self._prev_obs = None
        self._current_obs = obs

    def on_step(self, obs, reward, done, truncated, info):
        self._prev_obs = self._current_obs
        self._current_obs = obs

    def stop_condition_fn(self, obs: Observation, act: Action, next_obs: Observation) -> bool:
        return self._is_in_failure_state(next_obs)

    @abstractmethod
    def reward_fn(self, obs: Observation, act: Action, next_obs: Observation) -> float:
        raise NotImplementedError


class EvalReward(AbcReward):
    def __init__(self, RewardCls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = RewardCls(*args, **kwargs)

    @property
    def observation_space(self):
        return self.reward.observation_space

    def observation(self, obs):
        return self.reward.observation(obs)

    def stop_condition_fn(self, obs: Observation, act: Action, next_obs: Observation) -> bool:
        return self._is_in_failure_state(next_obs)

    def reward_fn(self, obs: Observation, act: Action, next_obs: Observation) -> float:
        if self._is_in_failure_state(next_obs):
            return 0
        p_boat = obs['p_boat'][0:2]
        next_p_boat = next_obs['p_boat'][0:2]
        gain_dist = self._compute_gain_dist(p_boat, next_p_boat)
        return gain_dist
