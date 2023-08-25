import numpy as np


class ABCWaterCurrentGenerator:
    def __init__(self, water_current_dir, water_current_speed):
        assert 0 <= water_current_dir < 2 * np.pi, \
            'water_current_dir must be in [0, 2*pi['
        self.theta_water_current = water_current_dir
        self.water_current_speed = water_current_speed


class NoWaterCurrentGenerator(ABCWaterCurrentGenerator):
    def __call__(self, t):
        return np.array([np.cos(self.theta_water_current), np.sin(self.theta_water_current)], dtype=float) * self.water_current_speed

    # def water_current_generator_fn(t):
    #     # nonlocal _water_theta, _next_water_theta
    #     # if t % 100 == 0:
    #     #     _water_theta = _next_water_theta
    #     #     _next_water_theta = water_theta + np.random.normal(0, 1)
    #     # # theta = _water_theta + (_next_water_theta -
    #     # #                         _water_theta) * (t % 100) / 100
    #     # theta = _next_water_theta
    #     # print(np.rad2deg(theta))
    #     return np.array([np.cos(water_theta), np.sin(water_theta)], dtype=float) * np.linalg.norm(water_current)
    #     return np.array([0, 0]).astype(float)
