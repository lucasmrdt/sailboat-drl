import numpy as np
from gymnasium import spaces

from .abc_reward import AbcReward


class MaxVMCReward(AbcReward):
    def __init__(self, path: list, full_obs: bool = False):
        super().__init__(path)
        self.full_obs = full_obs

    @property
    def observation_space(self):
        return spaces.Dict({
            **(super().observation_space.spaces if self.full_obs else {}),
            'vmc': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        return {
            **(super().observation(obs) if self.full_obs else {}),
            'vmc': np.array([self._compute_vmc(obs)])
        }

    def __call__(self, obs, act, next_obs):
        vmc = self._compute_vmc(next_obs)
        return vmc
