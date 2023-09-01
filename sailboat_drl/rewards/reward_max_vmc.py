import numpy as np
from gymnasium import spaces
from sailboat_gym import Action, Observation

from .abc_reward import AbcReward


class MaxVMCWithPenality(AbcReward):
    def __init__(self, rudder_change_penality, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rudder_change_penality = rudder_change_penality

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
        return vmc - self.rudder_change_penality * (dt_theta_rudder / 6)**2


class MaxVMCWithPenalityAndDelta(AbcReward):
    def __init__(self, rudder_change_penality, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rudder_change_penality = rudder_change_penality

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
        return vmc - self.rudder_change_penality * (dt_theta_rudder / 6)**2


class MaxVMCWith2PenalityAndDelta(MaxVMCWithPenalityAndDelta):
    def __init__(self, rudder_change_penality, xte_penality, *args, **kwargs):
        super().__init__(rudder_change_penality, *args, **kwargs)
        self.rudder_change_penality = rudder_change_penality
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
        return vmc / .4 - self.rudder_change_penality * (dt_theta_rudder / 6)**2 - self.xte_penality * (xte / 10)**2


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


class MaxVMCPenalizeXTEMPenalizeDeltaRudder(MaxVMCPenalizeXTEMPenalizeDtRudder):
    def reward_fn(self, obs, act, next_obs):
        vmc = self._compute_vmc(next_obs)
        xte = self._compute_xte(next_obs)
        prev_theta_rudder = obs['theta_rudder'][0]
        theta_rudder = next_obs['theta_rudder'][0]

        vmc = vmc / .4
        xte = xte / 10
        delta_theta_rudder = (theta_rudder - prev_theta_rudder) / .2

        r_vmc = 2 * (np.exp(vmc + 1) - 1) / (np.exp(2) - 1) - 1
        r_xte = 2 * (np.exp(-(xte**2 - 1)) - 1) / (np.e - 1) - 1
        r_rudder = delta_theta_rudder**2

        return self.vmc_coef * r_vmc + self.xte_coef * r_xte - self.rudder_coef * r_rudder


def f_sigm_inf(a, x, delta=1, eps=1e-2):
    A = np.exp(-a)
    A_x = np.exp(-a * x)
    A_delta = np.exp(-a * delta)
    B = (eps - A_delta) / (A_delta * (1 - eps))
    return (A_x * (B + 1)) / (A_x * B + 1)


def f_sigm_bounded(a, x, bound=[-1, 1]):
    def sig(x):
        return 1 / (1 + np.exp(-a * x))
    return (1 - sig(x) - sig(bound[0])) / (sig(bound[1]) - sig(bound[0]))


class MaxVMCCustomShape(AbcReward):
    def __init__(self, xte_a, xte_coef, vmc_a, vmc_coef, rudder_coef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xte_a = xte_a
        self.xte_coef = xte_coef
        self.vmc_a = vmc_a
        self.vmc_coef = vmc_coef
        self.rudder_coef = rudder_coef

    def reward_fn(self, obs, act, next_obs):
        vmc = self._compute_vmc(next_obs)
        xte = self._compute_xte(next_obs)
        prev_theta_rudder = obs['theta_rudder'][0]
        theta_rudder = next_obs['theta_rudder'][0]

        vmc = vmc / .4
        xte = xte / 10
        delta_theta_rudder = (theta_rudder - prev_theta_rudder) / .2

        # XTE penality:
        xte_penality = f_sigm_inf(self.xte_a, xte**2) - 1

        # VMC reward
        vmc_reward = 2 * (1 - f_sigm_bounded(self.vmc_a, vmc)) - 1

        # Rudder penality
        rudder_penality = -delta_theta_rudder**2

        return self.vmc_coef * vmc_reward + self.xte_coef * xte_penality + self.rudder_coef * rudder_penality


def sig_norm(x, center=0, a=None, x_delta_to_center=.1, y_delta_from_1=.1, bound=[-1, 1]):
    def sig(x):
        nonlocal a
        if a is None:
            a = np.log(y_delta_from_1 / (1 - y_delta_from_1)) / \
                x_delta_to_center
        b = -center
        return 1 / (1 + np.exp(-a * (x + b)))
    return (sig(x) - sig(bound[0])) / (sig(bound[1]) - sig(bound[0]))


class MaxVMCCustomShapeV2(AbcReward):
    def __init__(self, xte_params, xte_coef, vmc_params, vmc_coef, rudder_coef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xte_params = xte_params
        self.xte_coef = xte_coef
        self.vmc_params = vmc_params
        self.vmc_coef = vmc_coef
        self.rudder_coef = rudder_coef

    def xte_reward(self, xte):
        steepness = self.vmc_params['steepness']
        s = sig_norm(
            np.abs(xte),
            a=steepness,
            center=1,
            bound=[0, 2]
        )
        return -s

    def vmc_reward(self, vmc):
        start_penality = self.vmc_params['start_penality']
        steepness = self.vmc_params['steepness']
        s = sig_norm(
            vmc,
            center=start_penality,
            a=steepness,
        )
        return 2 * s - 1

    def reward_fn(self, obs, act, next_obs):
        vmc = self._compute_vmc(next_obs)
        xte = self._compute_xte(next_obs)
        prev_theta_rudder = obs['theta_rudder'][0]
        theta_rudder = next_obs['theta_rudder'][0]

        vmc = vmc / .4
        xte = xte / 10
        delta_theta_rudder = (theta_rudder - prev_theta_rudder) / .2

        vmc_reward = self.vmc_reward(vmc)
        xte_penaltiy = self.xte_reward(xte)
        rudder_penality = -delta_theta_rudder**2

        return self.vmc_coef * vmc_reward + self.xte_coef * xte_penaltiy + self.rudder_coef * rudder_penality
