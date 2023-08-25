import numpy as np
from gymnasium import spaces
from sailboat_gym import Action, Observation

from .abc_reward import AbcReward
from ..utils import norm, smallest_signed_angle, rotate_vector


class MaxDistReward(AbcReward):
    def __init__(self, path: list, full_obs: bool = False):
        super().__init__(path)
        self.full_obs = full_obs
        self._path_length = norm(self.path[1] - self.path[0])
        self._normalized_path = self.path / self._path_length
        self._previous_p_boat = np.zeros(2)
        # 10% of the path length
        self._xte_threshold = self._path_length * 0.1

    def _compute_gain_dist(self, p_boat, next_p_boat):
        path = self._normalized_path
        gain_dist = (next_p_boat - p_boat).dot(path[1] - path[0])
        return gain_dist

    def _is_in_failure_state(self, obs: Observation):
        if self._compute_xte(obs) > self._xte_threshold:
            return True
        return False

    @property
    def observation_space(self):
        return spaces.Dict({
            **(super().observation_space.spaces if self.full_obs else {}),
            'gain_dist': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        p_boat = obs['p_boat'][0:2]
        gain_dist = self._compute_gain_dist(self._previous_p_boat, p_boat)
        self._previous_p_boat = p_boat
        return {
            **(super().observation(obs) if self.full_obs else {}),
            'gain_dist': np.array([gain_dist]),
        }

    def reward_fn(self, obs, act, next_obs):
        if self._is_in_failure_state(obs):
            return -self._path_length
        p_boat = obs['p_boat'][0:2]
        next_p_boat = next_obs['p_boat'][0:2]
        gain_dist = self._compute_gain_dist(p_boat, next_p_boat)
        return gain_dist

    def stop_condition_fn(self, obs: Observation, act: Action, next_obs: Observation) -> bool:
        return self._is_in_failure_state(obs)
