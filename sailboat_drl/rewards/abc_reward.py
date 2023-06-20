import numpy as np
from typing import Dict
from sailboat_gym import Observation, Action
from abc import ABC, abstractmethod


class AbcReward(ABC):
    nb_extra_obs: int = 0

    @abstractmethod
    def __call__(self, obs: Observation, act: Action) -> float:
        raise NotImplementedError

    def transform_obs(self, obs: Observation) -> Dict[str, np.ndarray]:
        return obs
