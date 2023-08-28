import numpy as np


class OscillationForceGenerator:
    def __init__(self, force_theta, force_speed, sigma_dir, sigma_speed, nb_oscilations=1, nb_steps_per_episode=2000):
        assert 0 <= force_theta < 2 * np.pi, \
            'force_theta must be in [0, 2*pi['
        self.force_theta = force_theta
        self.force_speed = force_speed
        self.sigma_dir = sigma_dir
        self.sigma_speed = sigma_speed

        self.current_theta = force_theta
        self.current_speed = 0

        self.target_theta = np.random.normal(self.force_theta, self.sigma_dir)
        self.target_speed = np.random.normal(self.force_speed, self.sigma_speed)
        print(f'target_theta = {self.target_theta}')
        print(f'target_speed = {self.target_speed}')

        self.initialisation_duration = int(
            nb_steps_per_episode * .05)  # 5% of the episode
        self.transition_duration = nb_steps_per_episode // nb_oscilations

    def get_force(self, t):
        if t <= self.initialisation_duration:
            duration = self.initialisation_duration
        else:
            duration = self.transition_duration
        if t % duration == 0 and t >= self.initialisation_duration:
            self.current_theta = self.target_theta
            self.current_speed = self.target_speed
            self.target_theta = np.random.normal(
                self.force_theta, self.sigma_dir)
            self.target_speed = np.random.normal(
                self.force_speed, self.sigma_speed)
            print(f'target_theta = {self.target_theta}')
            print(f'target_speed = {self.target_speed}')

        t = t % duration
        theta = self.current_theta + \
            (self.target_theta - self.current_theta) * \
            (t / duration)
        speed = self.current_speed + \
            (self.target_speed - self.current_speed) * \
            (t / duration)
        return np.array([np.cos(theta), np.sin(theta)], dtype=float) * speed
