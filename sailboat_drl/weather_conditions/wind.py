import numpy as np


class ABCWindGenerator:
    def __init__(self, wind_dir, wind_speed):
        assert 0 <= wind_dir < 2 * np.pi, \
            'wind_dir must be in [0, 2*pi['
        self.theta_wind = wind_dir + np.pi  # where the wind is coming from
        self.wind_speed = wind_speed


class ConstantWindGenerator(ABCWindGenerator):
    def __call__(self, t):
        return np.array([np.cos(self.theta_wind), np.sin(self.theta_wind)], dtype=float) * self.wind_speed
