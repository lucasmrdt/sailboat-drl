import numpy as np
from gymnasium import spaces
from sailboat_gym import Action, Observation

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


class MaxVMCWith2PenalityAndDelta(MaxVMCWithPenalityAndDelta):
    def __init__(self, rudder_change_penalty, xte_penality, *args, **kwargs):
        super().__init__(rudder_change_penalty, *args, **kwargs)
        self.rudder_change_penalty = rudder_change_penalty
        self.xte_penality = xte_penality

    def reward_fn(self, obs, act, next_obs):
        if self._is_in_failure_state(next_obs):
            return -self.path_length
        vmc = self._compute_vmc(next_obs)
        dt_theta_rudder = next_obs['dt_theta_rudder'][0]
        xte = self._compute_xte(next_obs)
        # bound of dt_theta_rudder is [-6, 6]
        # bound of xte is [-10, 10]
        # bound of VMC is [-.4, .4]
        return vmc / .4 - self.rudder_change_penalty * (dt_theta_rudder / 6)**2 - self.xte_penality * (xte / 10)**2


class MaxVMCMinXTE(AbcReward):
    def __init__(self, vmc_coef, xte_coef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vmc_coef = vmc_coef
        self.xte_coef = xte_coef

    def stop_condition_fn(self, obs: Observation, act: Action, next_obs: Observation) -> bool:
        return False

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
        vmc = self._compute_vmc(next_obs)
        xte = self._compute_xte(next_obs)

        vmc = vmc / .4
        xte = xte / 10

        r_vmc = 2 * (np.exp(vmc + 1) - 1) / (np.exp(2) - 1) - 1
        r_xte = (np.exp(-(xte**2 - 1)) - 1) / (np.e - 1)

        return self.vmc_coef * r_vmc + self.xte_coef * r_xte


class MaxVMCMinXTEMinDtRudder(MaxVMCMinXTE):
    def __init__(self, rudder_coef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rudder_coef = rudder_coef

    def reward_fn(self, obs, act, next_obs):
        vmc = self._compute_vmc(next_obs)
        xte = self._compute_xte(next_obs)
        dt_theta_rudder = next_obs['dt_theta_rudder'][0]

        vmc = vmc / .4
        xte = xte / 10
        dt_theta_rudder = dt_theta_rudder / 6

        r_vmc = 2 * (np.exp(vmc + 1) - 1) / (np.exp(2) - 1) - 1
        r_xte = (np.exp(-(xte**2 - 1)) - 1) / (np.e - 1)
        r_rudder = dt_theta_rudder**2

        return self.vmc_coef * r_vmc + self.xte_coef * r_xte - self.rudder_coef * r_rudder


class MaxVMCMinXTEPenalizeXTE(AbcReward):
    """changes:
        - r_xte \in [0,1] -> r_xte \in [-1,1]
    """

    def __init__(self, vmc_coef, xte_coef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vmc_coef = vmc_coef
        self.xte_coef = xte_coef

    def stop_condition_fn(self, obs: Observation, act: Action, next_obs: Observation) -> bool:
        return False

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
        vmc = self._compute_vmc(next_obs)
        xte = self._compute_xte(next_obs)

        vmc = vmc / .4
        xte = xte / 10

        r_vmc = 2 * (np.exp(vmc + 1) - 1) / (np.exp(2) - 1) - 1
        r_xte = 2 * (np.exp(-(xte**2 - 1)) - 1) / (np.e - 1) - 1

        return self.vmc_coef * r_vmc + self.xte_coef * r_xte


class MaxVMCPenalizeXTEMPenalizeDtRudder(MaxVMCMinXTEPenalizeXTE):
    def __init__(self, rudder_coef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rudder_coef = rudder_coef

    def reward_fn(self, obs, act, next_obs):
        vmc = self._compute_vmc(next_obs)
        xte = self._compute_xte(next_obs)
        dt_theta_rudder = next_obs['dt_theta_rudder'][0]

        vmc = vmc / .4
        xte = xte / 10
        dt_theta_rudder = dt_theta_rudder / 6

        r_vmc = 2 * (np.exp(vmc + 1) - 1) / (np.exp(2) - 1) - 1
        r_xte = 2 * (np.exp(-(xte**2 - 1)) - 1) / (np.e - 1) - 1
        r_rudder = dt_theta_rudder**2

        return self.vmc_coef * r_vmc + self.xte_coef * r_xte - self.rudder_coef * r_rudder
