import numpy as np
from stable_baselines3.common import type_aliases
from simple_pid import PID


class PIDTAEAlgo(type_aliases.PolicyPredictor):
    def __init__(self, Kp, Ki, Kd, dt):
        self.pid = PID(Kp, Ki, Kd, setpoint=0)
        self.dt = dt
        self.first_pos = None
        self.last_pos = None

    def predict(self, obs, **_):
        assert 'tae' in obs, f'tae not in obs: {obs}'
        if self.first_pos is None:
            self.first_pos = obs['p_boat'][0]
        self.last_pos = obs['p_boat'][0]
        tae = obs['tae'][0]
        control = self.pid(tae, dt=self.dt)
        action = np.array([control])
        return [action], None
