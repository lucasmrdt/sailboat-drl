from typing import Any
from collections import defaultdict
from gymnasium import ObservationWrapper, spaces, Wrapper
from sailboat_gym import GymObservation
import numpy as np

from ..rewards import AbcReward
from ..utils import norm, rotate_vector
from ..logger import Logger


class CustomObservationWrapper(ObservationWrapper, Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cum_sum = defaultdict(float)

    def reset(self, **kwargs) -> Any:
        self.cum_sum.clear()
        return super().reset(**kwargs)

    def log(self, obs):
        log_obs = {f'obs/{k}': v for k, v in obs.items()}
        for k, v in list(log_obs.items()):
            self.cum_sum[k] += v
            log_obs[f'obs/cum_{k}'] = self.cum_sum[k]
        Logger.record(log_obs)


class RewardObs(CustomObservationWrapper):
    def __init__(self, env, reward: AbcReward, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.reward = reward

    @property
    def observation_space(self):
        return self.reward.observation_space

    def observation(self, obs):
        return self.reward.observation(obs)


class Basic2DObs(RewardObs):
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

        reward_obs = super().observation(obs)

        self.log({**reward_obs, **obs})

        obs = {
            **reward_obs,
            'v_angle': v_angle,
            'v_norm': v_norm,
            'theta_boat': obs['theta_boat'][2],  # Z axis
            'dt_theta_boat': obs['dt_theta_boat'][2],  # Z axis
            'theta_rudder': obs['theta_rudder'][0],
            'dt_theta_rudder': obs['dt_theta_rudder'][0],
            'wind_angle': wind_angle,
            'wind_norm': wind_norm,
        }
        return obs


class Basic2DObs_V2(RewardObs):
    """Changes:
        - instead of using angle, use normalized cos/sin which allows to represent angle in a circular/continuous way instead of discontinuous representation caused by modulo
    """

    @property
    def observation_space(self):
        return spaces.Dict({
            **super().observation_space,
            'v_angle': spaces.Box(low=-1, high=1, shape=(2,)),
            'v_norm': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'theta_boat': spaces.Box(low=-1, high=1, shape=(2,)),
            'dt_theta_boat': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'theta_rudder': spaces.Box(low=-1, high=1, shape=(2,)),
            'dt_theta_rudder': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'wind_angle': spaces.Box(low=-1, high=1, shape=(2,)),
            'wind_norm': spaces.Box(low=0, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        v = obs['dt_p_boat'][0:2]
        v_angle = np.arctan2(v[1], v[0])
        v_norm = norm(v)

        theta_boat = obs['theta_boat'][2]  # Z axis

        theta_rudder = obs['theta_rudder'][0]

        wind = obs['wind']
        wind_angle = np.arctan2(wind[1], wind[0])
        wind_norm = norm(wind)

        reward_obs = super().observation(obs)

        self.log({**reward_obs, **obs})

        obs = {
            **reward_obs,
            'v_angle': np.array([np.cos(v_angle), np.sin(v_angle)]),
            'v_norm': v_norm,
            'theta_boat': np.array([np.cos(theta_boat), np.sin(theta_boat)]),
            'dt_theta_boat': obs['dt_theta_boat'][2],  # Z axis
            'theta_rudder': np.array([np.cos(theta_rudder), np.sin(theta_rudder)]),
            'dt_theta_rudder': obs['dt_theta_rudder'].item(),
            'wind_angle': np.array([np.cos(wind_angle), np.sin(wind_angle)]),
            'wind_norm': wind_norm,
        }
        return obs


class Basic2DObs_V3(RewardObs):
    """Changes:
        - the velocity vector is relative to the heading direction of the boat, we should use the velocity vector in the world frame instead
        - change theta_rudder representation to angle only as the rudder angle is relative to the boat heading direction and its bounded between -pi/2 and pi/2
    """

    @property
    def observation_space(self):
        return spaces.Dict({
            **super().observation_space,
            'v_angle': spaces.Box(low=-1, high=1, shape=(2,)),
            'v_norm': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'theta_boat': spaces.Box(low=-1, high=1, shape=(2,)),
            'dt_theta_boat': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'theta_rudder': spaces.Box(low=-1, high=1, shape=(1,)),
            'dt_theta_rudder': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'wind_angle': spaces.Box(low=-1, high=1, shape=(2,)),
            'wind_norm': spaces.Box(low=0, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        theta_boat = obs['theta_boat'][2]  # Z axis

        v = obs['dt_p_boat'][0:2]
        v = rotate_vector(v, theta_boat)
        v_angle = np.arctan2(v[1], v[0])
        v_norm = norm(v)

        theta_rudder = obs['theta_rudder'][0]

        wind = obs['wind']
        wind_angle = np.arctan2(wind[1], wind[0])
        wind_norm = norm(wind)

        reward_obs = super().observation(obs)

        self.log({**reward_obs, **obs})

        obs = {
            **reward_obs,
            'v_angle': np.array([np.cos(v_angle), np.sin(v_angle)]),
            'v_norm': v_norm,
            'theta_boat': np.array([np.cos(theta_boat), np.sin(theta_boat)]),
            'dt_theta_boat': obs['dt_theta_boat'][2],  # Z axis
            'theta_rudder': theta_rudder,
            'dt_theta_rudder': obs['dt_theta_rudder'].item(),
            'wind_angle': np.array([np.cos(wind_angle), np.sin(wind_angle)]),
            'wind_norm': wind_norm,
        }
        return obs


class Basic2DObs_V4(RewardObs):
    """Changes:
        - simplify the observation space by removing raw velocity, raw heading
    """

    @property
    def observation_space(self):
        return spaces.Dict({
            **super().observation_space,
            'theta_rudder': spaces.Box(low=-1, high=1, shape=(1,)),
            'dt_theta_rudder': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'wind_angle': spaces.Box(low=-1, high=1, shape=(2,)),
            'wind_norm': spaces.Box(low=0, high=np.inf, shape=(1,)),
        })

    def observation(self, obs):
        theta_rudder = obs['theta_rudder'][0]

        wind = obs['wind']
        wind_angle = np.arctan2(wind[1], wind[0])
        wind_norm = norm(wind)

        reward_obs = super().observation(obs)

        self.log({**reward_obs, **obs})

        obs = {
            **reward_obs,
            'theta_rudder': theta_rudder,
            'dt_theta_rudder': obs['dt_theta_rudder'].item(),
            'wind_angle': np.array([np.cos(wind_angle), np.sin(wind_angle)]),
            'wind_norm': wind_norm,
        }
        return obs


class RawObs(RewardObs):
    @property
    def observation_space(self):
        return spaces.Dict({
            **super().observation_space,
            **GymObservation,
        })

    def observation(self, obs):
        obs = {
            **super().observation(obs),
            **obs,
        }
        self.log(obs)
        return obs
