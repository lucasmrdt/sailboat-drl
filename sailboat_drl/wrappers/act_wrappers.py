import numpy as np
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from sailboat_gym import Action, SailboatLSAEnv

from ..utils import is_env_instance
from ..logger import log


class ActionWithFixedSail(gym.ActionWrapper):
    def __init__(self, env, theta_sail: float):
        super().__init__(env)
        self.theta_sail = np.array([theta_sail])
        self.is_eval_env = is_env_instance(env, RecordVideo)


class RudderAngleAction(ActionWithFixedSail):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Box(
            low=np.array([-1], dtype=np.float32),
            high=np.array([1], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )

    def action(self, action: np.ndarray) -> Action:
        theta_rudder = np.clip(action[0], -np.pi/2, np.pi/2)
        log({'mlp': action[0], 'theta_rudder': theta_rudder},
            prefix=f'{"eval" if self.is_eval_env else "train"}/act')
        return {
            'theta_rudder': theta_rudder,
            'theta_sail': self.theta_sail
        }


class RudderForceAction(ActionWithFixedSail, gym.Wrapper):
    # in 5 second we can turn the rudder at most pi
    dt = np.pi/(SailboatLSAEnv.NB_STEPS_PER_SECONDS*5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_rudder = np.array([0])  # in rad
        self.directions = [-1, 1, 0]
        # -1: boat is turning left
        # 1: boat is turning right
        # 0: boat is going straight
        self.action_space = gym.spaces.Discrete(len(self.directions))

    def reset(self, **kwargs):
        self.theta_rudder = np.array([0])  # reset rudder angle
        return super().reset(**kwargs)

    def action(self, action: int) -> Action:
        direction = self.directions[action]
        self.theta_rudder = np.clip(self.theta_rudder + direction * self.dt,
                                    -np.pi/2,
                                    np.pi/2)
        log({'mlp': action, 'theta_rudder': self.theta_rudder},
            prefix=f'{"eval" if self.is_eval_env else "train"}/act')
        return {
            'theta_rudder': self.theta_rudder,
            'theta_sail': self.theta_sail
        }
