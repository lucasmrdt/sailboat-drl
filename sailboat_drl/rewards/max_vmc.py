import cv2
import numpy as np
from gymnasium import spaces
from sailboat_gym import Action, CV2DRenderer, Observation

from .abc_reward import AbcReward
from ..utils import norm, smallest_signed_angle, rotate_vector


class AbcPFReward(AbcReward):  # PF: Path Following
    def __init__(self, path: list):
        assert len(path) == 2 and len(path[0]) == 2 and len(
            path[1]) == 2, 'Path must be a list of 2D vectors'
        self.path = np.array(path, dtype=np.float32)
        # self.map_bounds = np.array(map_bounds)
        # self.absolute_path = self.path * (self.map_bounds[1] - self.map_bounds[0]) + self.map_bounds[0]

    def _compute_xte(self, obs: Observation):
        path = self.path
        p_boat = obs['p_boat'][0:2]  # X and Y axis
        d = (path[1] - path[0])
        n = np.array([-d[1], d[0]])  # Normal vector to the path
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


class PFRenderer(CV2DRenderer):
    def __init__(self, reward: AbcPFReward, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward

    def _draw_reward(self, img, obs):
        path = self._translate_and_scale_to_fit_in_map(self.reward.path)
        cv2.line(img,
                 tuple(path[0].astype(int)),
                 tuple(path[1].astype(int)),
                 (0, 127, 0),
                 2)

    def render(self, obs, draw_extra_fct=None):
        return super().render(obs, draw_extra_fct=self._draw_reward)  # type: ignore


class PFMaxVMCWithAllRewardObs(AbcPFReward):
    def __call__(self, obs, act, next_obs):
        vmc = self._compute_vmc(next_obs)
        return vmc


class PFMaxVMC(PFMaxVMCWithAllRewardObs):
    @property
    def observation_space(self):
        return spaces.Dict({
            'vmc': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        return {'vmc': np.array([self._compute_vmc(obs)])}
