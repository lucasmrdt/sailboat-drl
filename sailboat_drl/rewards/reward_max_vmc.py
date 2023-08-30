import numpy as np
from gymnasium import spaces

from .abc_reward import AbcReward


class MaxVMCWithPenality(AbcReward):
    def __init__(self, rudder_change_penalty, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rudder_change_penalty = rudder_change_penalty

    @property
    def observation_space(self):
        return spaces.Dict({
            'xte': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'heading_error': spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            'vmc': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        return {
            'xte': np.array([self._compute_xte(obs)]),
            'heading_error': np.array([self._compute_heading_error(obs)]),
            'vmc': np.array([self._compute_vmc(obs)])
        }

    def reward_fn(self, obs, act, next_obs):
        if self._is_in_failure_state(next_obs):
            return -self.path_length
        vmc = self._compute_vmc(next_obs)
        dt_theta_rudder = next_obs['dt_theta_rudder'][0]
        # bound of dt_theta_rudder is [-6, 6]
        return vmc - self.rudder_change_penalty * (dt_theta_rudder / 6)**2


class MaxVMCWithPenalityAndDelta(AbcReward):
    def __init__(self, rudder_change_penalty, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rudder_change_penalty = rudder_change_penalty

    @property
    def observation_space(self):
        return spaces.Dict({
            'xte': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'heading_error': spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            'vmc': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'delta': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        return {
            'xte': np.array([self._compute_xte(obs)]),
            'heading_error': np.array([self._compute_heading_error(obs)]),
            'vmc': np.array([self._compute_vmc(obs)]),
            'delta': np.array([self.xte_threshold - abs(self._compute_xte(obs))]),
        }

    def reward_fn(self, obs, act, next_obs):
        if self._is_in_failure_state(next_obs):
            return -self.path_length
        vmc = self._compute_vmc(next_obs)
        dt_theta_rudder = next_obs['dt_theta_rudder'][0]
        xte = self._compute_xte(next_obs)
        # bound of dt_theta_rudder is [-6, 6]
        return vmc - self.rudder_change_penalty * (dt_theta_rudder / 6)**2
