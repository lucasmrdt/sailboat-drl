from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import type_aliases
from simple_pid import PID
from torch import nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sailboat_gym import env_by_name
import numpy as np

from ..env import create_env, available_rewards
from ..logger import Logger
from ..utils import rotate_vector, smallest_signed_angle


class TAEPIDAlgo(type_aliases.PolicyPredictor):
    def __init__(self, Kp, Ki, Kd, dt):
        self.pid = PID(Kp, Ki, Kd, setpoint=0)
        self.dt = dt

    def predict(self, obs, **_):
        assert 'tae' in obs, f'tae not in obs: {obs}'
        tae = obs['tae'][0]
        control = self.pid(tae, dt=self.dt)
        action = np.array([control])
        return [action], None


class LOSPIDAlgo(type_aliases.PolicyPredictor):
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

#  Info: {'map_bounds': array([[   0., -100.,    0.],
#        [ 200.,  100.,    1.]], dtype=float32)}


pid_algo_by_name = {
    'tae': TAEPIDAlgo,
    'los': LOSPIDAlgo,
}


def get_args(overwrite_args={}):
    def extended_eval(s):
        return eval(s, {'pi': np.pi, 'nn': nn})

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name',
                        type=str, required=True, help='experiment name')
    parser.add_argument('--pid-algo', choices=list(pid_algo_by_name.keys()),
                        required=True, help='PID algorithm')
    parser.add_argument('--Kp', type=float, default=1, help='Kp')
    parser.add_argument('--Ki', type=float, default=0.1, help='Ki')
    parser.add_argument('--Kd', type=float, default=0.05, help='Kd')
    parser.add_argument('--los-radius', type=float,
                        default=20, help='LOS radius')
    parser.add_argument('--env-name', choices=list(env_by_name.keys()),
                        default=list(env_by_name.keys())[0], help='environment name')
    parser.add_argument('--reward', choices=list(available_rewards.keys()),
                        default=list(available_rewards.keys())[0], help='reward function')
    parser.add_argument('--reward-kwargs', type=extended_eval,
                        default={}, help='reward function arguments')
    parser.add_argument('--episode-duration', type=int,
                        default=100, help='episode duration (in seconds)')
    parser.add_argument('--wind-speed', type=float, default=1,
                        help='wind speed')
    parser.add_argument('--keep-sim-running', action='store_true',
                        help='keep the simulator running after training')
    args = parser.parse_args()

    args.__dict__ = {k: v for k, v in vars(args).items()
                     if k not in overwrite_args}
    args.__dict__.update(overwrite_args)
    return args


def prepare_env(args, is_eval=False, theta_wind=np.pi / 2):
    def _init():
        return create_env(env_idx=args.name,
                          is_eval=is_eval,
                          wind_speed=args.wind_speed,
                          theta_wind=theta_wind,
                          reward=args.reward,
                          reward_kwargs=args.reward_kwargs,
                          act='rudder_angle_act',
                          env_name=args.env_name,
                          seed=0,
                          keep_sim_running=args.keep_sim_running,
                          episode_duration=args.episode_duration,
                          prepare_env_for_nn=False,
                          logger_prefix=args.name)
    return _init


def evaluate_pid_algo(args, theta_wind):
    args = get_args(args)

    nb_steps_per_seconds = env_by_name[args.env_name].NB_STEPS_PER_SECONDS
    dt = 1 / nb_steps_per_seconds

    PIDAlgo = pid_algo_by_name[args.pid_algo]
    if PIDAlgo == TAEPIDAlgo:
        pid_algo = PIDAlgo(args.Kp, args.Ki, args.Kd, dt)
    elif PIDAlgo == LOSPIDAlgo:
        path = np.array(args.reward_kwargs['path'], dtype=np.float32)
        radius = args.los_radius
        pid_algo = LOSPIDAlgo(args.Kp, args.Ki, args.Kd, dt, path, radius)
    else:
        raise ValueError(f'Unknown PID algorithm: {PIDAlgo}')

    Logger.configure(f'{args.name}/eval.py')
    env = prepare_env(args, is_eval=True, theta_wind=theta_wind)()
    mean_reward, std_reward = evaluate_policy(pid_algo,
                                              env,
                                              n_eval_episodes=1)
    env.close()
    return mean_reward, std_reward
