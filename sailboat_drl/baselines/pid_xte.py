import numpy as np
from stable_baselines3.common import type_aliases
from simple_pid import PID


class PIDXTEAlgo(type_aliases.PolicyPredictor):
    def __init__(self, Kp, Ki, Kd, dt):
        self.pid = PID(Kp, Ki, Kd, setpoint=0)
        self.dt = dt
        self.first_pos = None
        self.last_pos = None

    def predict(self, obs, **_):
        assert 'xte' in obs, f'xte not in obs: {obs}'
        if self.first_pos is None:
            self.first_pos = obs['p_boat'][0]
        self.last_pos = obs['p_boat'][0]
        xte = obs['xte'][0]
        control = self.pid(xte, dt=self.dt)
        action = np.array([control])
        return [action], None
