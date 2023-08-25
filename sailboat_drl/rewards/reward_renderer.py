import cv2
from sailboat_gym import CV2DRenderer

from .abc_reward import AbcReward


class RewardRenderer(CV2DRenderer):
    def __init__(self, reward: AbcReward, *args, **kwargs):
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
