from typing import Any
from collections import defaultdict
import numpy as np
from gymnasium import ObservationWrapper, spaces, Wrapper

from ..rewards import AbcReward
from ..utils import norm
from ..logger import Logger

class CustomObservationWrapper(ObservationWrapper):
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

class RewardObs(CustomObservationWrapper):
    def __init__(self, env, reward: AbcReward, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.reward = reward

    @property
    def observation_space(self):
        return self.reward.observation_space
    
    def observation(self, obs):
        obs = self.reward.observation(obs)
        return obs


class Basic2DObs(RewardObs, Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cum_sum = defaultdict(float)

    def reset(self, *args, **kwargs):
        self.cum_sum.clear()
        return super().reset(*args, **kwargs)
    
    @property
    def observation_space(self):
        return spaces.Dict({
            **super().observation_space,
            'v_angle': spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            'v_norm': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'theta_boat': spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            'dt_theta_boat': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'theta_rudder': spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            'dt_theta_rudder': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'wind_angle': spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            'wind_norm': spaces.Box(low=0, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        v = obs['dt_p_boat'][0:2]
        v_angle = np.arctan2(v[1], v[0])
        v_norm = norm(v)

        wind = obs['wind']
        wind_angle = np.arctan2(wind[1], wind[0])
        wind_norm = norm(wind)

        obs = {
            **super().observation(obs),
            'v_angle': v_angle,
            'v_norm': v_norm,
            'theta_boat': obs['theta_boat'][2],  # Z axis
            'dt_theta_boat': obs['dt_theta_boat'][2],  # Z axis
            'theta_rudder': obs['theta_rudder'][0],
            'dt_theta_rudder': obs['dt_theta_rudder'][0],
            'wind_angle': wind_angle,
            'wind_norm': wind_norm,
        }

        log_obs = {f'obs/{k}': v for k, v in obs.items()}
        for k, v in list(log_obs.items()):
            self.cum_sum[k] += v
            log_obs[f'obs/cum_{k}'] = self.cum_sum[k]
        Logger.record(log_obs)

        return obs

