import wandb
import numpy as np
import gymnasium as gym
from sailboat_gym import Action, SailboatLSAEnv


class ActionWithFixedSail(gym.ActionWrapper):
    def __init__(self, env, theta_sail: float):
        super().__init__(env)
        self.theta_sail = np.array([theta_sail])

    def action(self, action: np.ndarray[1]) -> Action:
        theta_rudder = np.clip(action[0], -np.pi/2, np.pi/2)
        if wandb.run is not None:
            wandb.log({'train/action': action[0]})
            wandb.log({'train/theta_rudder': theta_rudder})
        return {
            'theta_rudder': theta_rudder,
            'theta_sail': self.theta_sail
        }


class RudderAngleAction(ActionWithFixedSail):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Box(
            low=np.array([-1], dtype=np.float32),
            high=np.array([1], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )


class RudderForceAction(ActionWithFixedSail):
    # in 5 second we can turn the rudder at most pi
    dt = np.pi/(SailboatLSAEnv.SIM_RATE*5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_rudder = np.array([0])  # in rad
        self.directions = [-1, 1, 0]
        # -1: boat is turning left
        # 1: boat is turning right
        # 0: boat is going straight
        self.action_space = gym.spaces.Discrete(len(self.directions))

    def action(self, action: int) -> Action:
        direction = self.directions[action]
        if wandb.run is not None:
            wandb.log({'train/direction': direction})
        self.theta_rudder = np.clip(self.theta_rudder + direction * self.dt,
                                    -np.pi/2,
                                    np.pi/2)
        return super().action(self.theta_rudder)
