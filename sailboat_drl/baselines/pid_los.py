import numpy as np
from stable_baselines3.common import type_aliases
from simple_pid import PID

from ..utils import rotate_vector, smallest_signed_angle


class PIDLOSAlgo(type_aliases.PolicyPredictor):
    def __init__(self, Kp, Ki, Kd, dt, path, radius):
        self.pid = PID(Kp, Ki, Kd, setpoint=0)
        self.dt = dt
        self.path = path
        self.radius = radius

    def compute_los(self, pos):
        p1, p2 = self.path - pos  # center on pos
        dx, dy = p2 - p1 + 1e-8  # avoid div by 0
        dr = np.sqrt(dx**2 + dy**2)
        D = p1[0] * p2[1] - p2[0] * p1[1]
        delta = self.radius**2 * dr**2 - D**2

        def compute_intersect(sign=1):
            return np.array([
                D * dy + sign * np.sign(dy) * dx * np.sqrt(delta),
                -D * dx + sign * np.abs(dy) * np.sqrt(delta),
            ]) / dr**2

        if delta < 0:
            return []
        elif delta == 0:
            return [compute_intersect()]
        else:
            return [compute_intersect(-1), compute_intersect(1)]

    def predict(self, obs, **_):
        assert 'p_boat' in obs, f'p_boat not in obs: {obs}'
        assert 'xte' in obs, f'xte not in obs: {obs}'
        assert 'theta_boat' in obs, f'theta_boat not in obs: {obs}'

        pos = obs['p_boat'][0, 0:2]
        vel = obs['dt_p_boat'][0, 0:2]
        xte = obs['xte'][0, 0]
        theta_boat = obs['theta_boat'][0, 2]  # z axis

        # transform relative velocity to world frame
        vel = rotate_vector(vel, theta_boat)

        los = np.array(self.compute_los(pos))

        if len(los) > 0:
            path_vec = self.path[1] - self.path[0]
            # keep only points in the path direction
            los = los[np.dot(los, path_vec) > 0]

        # if we are too far from the path (ie. no los intersections), point to the path orthogonally
        if len(los) == 0:
            los_angles = np.array([-np.sign(xte) * np.pi / 2])
        else:
            los_angles = np.arctan2(los[:, 1], los[:, 0])

        vel_angle = np.arctan2(vel[1], vel[0])
        vel_norm = np.linalg.norm(vel)

        # if vel is too small, use boat angle
        if vel_norm < 0.1:
            ref_angle = theta_boat
        else:
            ref_angle = vel_angle

        ref_angle = vel_angle if vel_norm > 0.1 else theta_boat
        angle_diffs = smallest_signed_angle(los_angles - ref_angle)

        angle_error = min(angle_diffs, key=abs)
        control = self.pid(angle_error, dt=self.dt)
        action = np.array([control])
        return [action], None
