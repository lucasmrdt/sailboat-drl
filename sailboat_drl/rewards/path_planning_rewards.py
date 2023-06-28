from typing import Any, Dict
import numpy as np
import cv2
from gymnasium import spaces
from sailboat_gym import CV2DRenderer, Observation

from .abc_reward import AbcReward
from ..utils import norm, normalize


def smallest_signed_angle(angle):
    """Transform an angle to be between -pi and pi"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class AbcPPReward(AbcReward):  # PP: Path Planning
    def __init__(self, target: list, radius: float):
        assert len(target) == 2, 'Target must be a 2D vector'
        self.target = np.array(target)
        self.radius = radius

    @property
    def observation_space(self):
        return spaces.Dict({
            'dist_to_target': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'tae': spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
        })

    def observation(self, obs):
        p_boat = obs['p_boat'][0:2]  # X and Y axis
        d = p_boat - self.target
        angle_to_target = np.arctan2(d[1], d[0])
        theta_boat = obs['theta_boat'][2]  # Z axis
        tae = smallest_signed_angle(angle_to_target - theta_boat)
        return {
            'dist_to_target': np.array([norm(d)]),
            'tae': np.array([tae]),
        }


class PPRenderer(CV2DRenderer):
    def __init__(self, reward: AbcPPReward, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward

    def __draw_reward(self, img, obs):
        target = self.reward.target * \
            (self.size - 2*self.padding) + self.padding
        cv2.circle(img,
                   tuple(target.astype(int)),
                   self.reward.radius,
                   (0, 255, 0),
                   -1)

    def render(self, obs, draw_extra_fct=None):
        return super().render(obs, draw_extra_fct=self.__draw_reward)


class PPSparseReward(AbcPPReward):
    def __call__(self, obs, act, next_obs):
        p_boat = obs['p_boat'][0:2]
        if norm(p_boat - self.target) < self.radius:
            return 1
        return 0


class PPDistToTargetReward(AbcPPReward):
    def __call__(self, obs, act, next_obs):
        p_boat = next_obs['p_boat'][0:2]
        return -norm(p_boat - self.target)


class PPGainedDistToTargetReward(AbcPPReward):
    def __call__(self, obs, act, next_obs):
        p_boat = obs['p_boat'][0:2]
        p_boat_next = next_obs['p_boat'][0:2]
        dist = norm(p_boat - self.target)
        dist_next = norm(p_boat_next - self.target)
        return dist_next - dist


class PPVelocityReward(AbcPPReward):
    def __call__(self, obs, act, next_obs):
        v_boat_next = next_obs['dt_p_boat'][0:2]  # X and Y axis
        return norm(v_boat_next)
