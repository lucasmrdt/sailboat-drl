import cv2
import numpy as np
from sailboat_gym import CV2DRenderer

from .abc_reward import AbcReward


class RewardRenderer(CV2DRenderer):
    def __init__(self, reward: AbcReward, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward

    def _draw_reward(self, img, obs):
        path = self._translate_and_scale_to_fit_in_map(self.reward.path)
        xte_threshold = self._scale_to_fit_in_img(self.reward.xte_threshold)
        delta = np.array([0, xte_threshold])
        cv2.line(img,
                 tuple(path[0].astype(int)),
                 tuple(path[1].astype(int)),
                 (0, 127, 0),
                 2)
        cv2.line(img,
                 tuple((path[0] + delta).astype(int)),
                 tuple((path[1] + delta).astype(int)),
                 (255, 0, 0),
                 1)
        cv2.line(img,
                 tuple((path[0] - delta).astype(int)),
                 tuple((path[1] - delta).astype(int)),
                 (255, 0, 0),
                 1)

    def render(self, obs, draw_extra_fct=None):
        return super().render(obs, draw_extra_fct=self._draw_reward)  # type: ignore
