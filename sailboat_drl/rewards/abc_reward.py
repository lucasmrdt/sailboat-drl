from typing import Dict, Any
from gymnasium import spaces, ObservationWrapper
from sailboat_gym import Observation, Action
from abc import ABC, abstractmethod


class AbcReward(ABC, ObservationWrapper):
    @abstractmethod
    def __call__(self, obs: Observation, act: Action, next_obs: Observation) -> float:
        raise NotImplementedError

    @property
    def observation_space(self):
        return spaces.Dict({})

    def observation(self, obs: Observation) -> Dict[str, Any]:
        return obs  # type: ignore
