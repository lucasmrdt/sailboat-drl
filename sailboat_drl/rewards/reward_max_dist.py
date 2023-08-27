import numpy as np
from gymnasium import spaces

from .abc_reward import AbcReward


class MaxDistReward_V1(AbcReward):
    @property
    def observation_space(self):
        return spaces.Dict({
            **super().observation_space.spaces,
            'gain_dist': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        prev_obs = self._prev_obs
        p_boat = obs['p_boat'][0:2]
        prev_p_boat = prev_obs['p_boat'][0:2] if prev_obs is not None else p_boat
        return {
            **super().observation(obs),
            'gain_dist': np.array([self._compute_gain_dist(prev_p_boat, p_boat)]),
        }

    def reward_fn(self, obs, act, next_obs):
        if self._is_in_failure_state(next_obs):
            return -self.path_length
        p_boat = obs['p_boat'][0:2]
        next_p_boat = next_obs['p_boat'][0:2]
        gain_dist = self._compute_gain_dist(p_boat, next_p_boat)
        return gain_dist
