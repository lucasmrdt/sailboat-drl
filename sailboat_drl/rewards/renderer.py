import cv2
import numpy as np
from gymnasium import spaces
from sailboat_gym import Action, CV2DRenderer, Observation

from .abc_reward import AbcReward
from ..utils import norm, smallest_signed_angle, rotate_vector


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
