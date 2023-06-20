import numpy as np
import cv2
from sailboat_gym import CV2DRenderer

from .abc_reward import AbcReward
from ..utils import norm, normalize


class AbcPathPlanningReward(AbcReward):
    nb_extra_obs = 2

    def __init__(self, target: list, radius: float):
        assert len(target) == 2, 'Target must be a 2D vector'
        self.target = np.array(target)
        self.radius = radius

    def transform_obs(self, obs):
        assert 'p_boat' in obs, 'p_boat must be in obs'
        p_boat = obs['p_boat'][0:2]
        d = p_boat - self.target
        angle_to_target = np.arctan2(d[1], d[0])
        return {
            **obs,
            'dist_to_target': np.array([norm(d)]),
            'angle_to_target': np.array([angle_to_target]),
        }


class PathPlanningRenderer(CV2DRenderer):
    def __init__(self, reward: AbcPathPlanningReward, *args, **kwargs):
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


class PathPlanningSparse1Reward(AbcPathPlanningReward):
    def __call__(self, obs, act):
        p_boat = obs['p_boat'][0:2]
        if norm(p_boat - self.target) < self.radius:
            return 1
        return 0


class PathPlanningDense1Reward(AbcPathPlanningReward):
    def __call__(self, obs, act):
        p_boat = obs['p_boat'][0:2]
        return -norm(p_boat - self.target)
