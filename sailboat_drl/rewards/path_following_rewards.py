import cv2
import numpy as np
from gymnasium import spaces
from sailboat_gym import CV2DRenderer, Observation

from .abc_reward import AbcReward
from ..utils import norm, normalize


def smallest_signed_angle(angle):
    """Transform an angle to be between -pi and pi"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class AbcPFReward(AbcReward):  # PF: Path Following
    def __init__(self, path: list):
        assert len(path) == 2 and len(path[0]) == 2 and len(
            path[1]) == 2, 'Path must be a list of 2D vectors'
        self.path = np.array(path)

    def _compute_xte(self, obs: Observation):
        d = self.path[1] - self.path[0]
        p_boat = obs['p_boat'][0:2]  # X and Y axis
        n = np.array([-d[1], d[0]])  # Normal vector to the path
        n /= norm(n)
        xte = np.dot(p_boat - self.path[0], n)
        return xte

    def _compute_tae(self, obs: Observation):
        d = self.path[1] - self.path[0]
        target_angle = np.arctan2(d[1], d[0])
        theta_boat = obs['theta_boat'][2]  # Z axis
        tae = smallest_signed_angle(target_angle - theta_boat)
        return tae

    def _compute_vmc(self, obs: Observation):
        d = self.path[1] - self.path[0]
        v = obs['dt_p_boat'][0:2]  # X and Y axis
        theta_boat = obs['theta_boat'][2]
        rot_mat = np.array([[np.cos(theta_boat), -np.sin(theta_boat)],
                            [np.sin(theta_boat), np.cos(theta_boat)]])
        v = rot_mat@v
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


class PFRenderer(CV2DRenderer):
    def __init__(self, reward: AbcPFReward, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward

    def _draw_reward(self, img, obs):
        path = self.reward.path * (self.size - 2*self.padding) + self.padding
        cv2.line(img,
                 tuple(path[0].astype(int)),
                 tuple(path[1].astype(int)),
                 (0, 255, 0),
                 2)

    def render(self, obs, draw_extra_fct=None):
        return super().render(obs, draw_extra_fct=self._draw_reward)  # type: ignore


class PFSparseReward(AbcPFReward):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, obs, act, next_obs):
        xte = self._compute_xte(next_obs)
        vmc = self._compute_vmc(next_obs)
        return -(self.k1 * xte**2 + self.k2 * vmc**2)**.5


class PFDerajEtAl2022Reward(AbcPFReward):
    def __init__(self, k1=1, k2=1, **kwargs):
        super().__init__(**kwargs)
        self.k1 = k1
        self.k2 = k2

    def __call__(self, obs, act, next_obs):
        xte = self._compute_xte(next_obs)
        vmc = self._compute_vmc(next_obs)
        vmc_bar = (vmc - 1)/2
        return np.exp(-self.k1 * xte**2) + np.exp(self.k2 * vmc_bar**2)


class PFMaxVMC(AbcPFReward):
    @property
    def observation_space(self):
        return spaces.Dict({
            'vmc': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        return {'vmc': np.array([self._compute_vmc(obs)])}

    def __call__(self, obs, act, next_obs):
        vmc = self._compute_vmc(next_obs)
        return vmc
