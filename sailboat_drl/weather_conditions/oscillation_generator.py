import numpy as np


class OscillationGenerator:
    def __init__(self, force_theta, force_speed, sigma_dir, sigma_speed, nb_oscilations=1, nb_steps_per_episode=2000, initialisation_duration=100):
        assert 0 <= force_theta < 2 * np.pi, \
            'force_theta must be in [0, 2*pi['
        self.force_theta = force_theta
        self.force_speed = force_speed
        self.sigma_dir = sigma_dir
        self.sigma_speed = sigma_speed

        # 5% of the episode
        self.initialisation_duration = initialisation_duration
        self.transition_duration = nb_steps_per_episode // nb_oscilations

    def _initialise(self):
        self.current_theta = self.force_theta
        self.current_speed = 0
        self.target_theta = np.random.normal(self.force_theta,
                                             self.sigma_dir)
        self.target_speed = np.random.normal(self.force_speed,
                                             self.sigma_speed)

    def _get_duration(self, t):
        if t <= self.initialisation_duration:
            return self.initialisation_duration
        else:
            return self.transition_duration

    def _generate_new_target(self):
        self.current_theta = self.target_theta
        self.current_speed = self.target_speed
        self.target_theta = np.random.normal(self.force_theta,
                                             self.sigma_dir)
        self.target_speed = np.random.normal(self.force_speed,
                                             self.sigma_speed)

    def _interpolate(self, a, b, t, duration):
        return a + (b - a) * (t / duration)

    def get_force(self, t):
        if t == 0:
            self._initialise()

        if t <= self.initialisation_duration and self.initialisation_duration > 0:
            duration = self.initialisation_duration
        else:
            t -= self.initialisation_duration
            duration = self.transition_duration

        if t % duration == 0 and t >= self.initialisation_duration:
            self._generate_new_target()

        t = t % duration
        theta = self._interpolate(self.current_theta,
                                  self.target_theta,
                                  t,
                                  duration)
        speed = self._interpolate(self.current_speed,
                                  self.target_speed,
                                  t,
                                  duration)
        return np.array([np.cos(theta), np.sin(theta)], dtype=float) * speed
