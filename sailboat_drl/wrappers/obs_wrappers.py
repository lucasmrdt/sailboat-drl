import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium.wrappers.record_video import RecordVideo

from ..rewards import AbcReward
from ..utils import norm, is_env_instance
from ..logger import log


class ObservationWrapperUsingReward(ObservationWrapper):
    def __init__(self, env, reward: AbcReward):
        super().__init__(env)
        self.reward = reward
        self.is_eval_env = is_env_instance(env, RecordVideo)

    @property
    def observation_space(self):
        return self.reward.observation_space

    def observation(self, obs):
        obs = self.reward.observation(obs)
        log(obs, prefix=f'{"eval" if self.is_eval_env else "train"}/obs')
        return obs


class Basic2DObs(ObservationWrapperUsingReward):
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
        })

    def observation(self, obs):
        v = obs['dt_p_boat'][0:2]
        v_angle = np.arctan2(v[1], v[0])
        v_norm = norm(v)
        obs = {
            **super().observation(obs),
            'v_angle': v_angle,
            'v_norm': v_norm,
            'theta_boat': obs['theta_boat'][2],  # Z axis
            'dt_theta_boat': obs['dt_theta_boat'][2],  # Z axis
            'theta_rudder': obs['theta_rudder'][0],
            'dt_theta_rudder': obs['dt_theta_rudder'][0],
        }
        log(obs, prefix=f'{"eval" if self.is_eval_env else "train"}/obs')
        return obs
