import allow_local_package_imports

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import type_aliases
from simple_pid import PID
from torch import nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sailboat_gym import env_by_name

from sailboat_drl.env import create_env, available_rewards, available_wind_generators, available_water_current_generators
from sailboat_drl.logger import Logger
from sailboat_drl.utils import rotate_vector, smallest_signed_angle
from sailboat_drl.baselines.pid_tae import PIDTAEAlgo
from sailboat_drl.baselines.pid_los import PIDLOSAlgo


#  Info: {'map_bounds': array([[   0., -100.,    0.],
#        [ 200.,  100.,    1.]], dtype=float32)}


pid_algo_by_name = {
    'tae': PIDTAEAlgo,
    'los': PIDLOSAlgo,
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
                        default='SailboatLSAEnv-v0', help='environment name')
    parser.add_argument('--reward', choices=list(available_rewards.keys()),
                        default='pf_max_dist_with_all_reward_obs', help='reward function')
    parser.add_argument('--reward-kwargs', type=extended_eval,
                        default={'path': [[0, 0], [200, 0]]}, help='reward function arguments')
    parser.add_argument('--water-current', choices=list(available_water_current_generators.keys()),
                        default='none', help='water current generator')
    parser.add_argument('--wind', choices=list(available_wind_generators.keys()),
                        default='constant', help='wind generator')
    parser.add_argument('--wind-dir', type=float, default=90,
                        help='wind direction (in deg)')
    parser.add_argument('--wind-speed', type=float, default=1.5,
                        help='wind speed')
    parser.add_argument('--water-current-dir', type=float,
                        default=90, help='water current direction (in deg)')
    parser.add_argument('--water-current-speed', type=float,
                        default=0.01, help='water current speed')
    parser.add_argument('--episode-duration', type=int,
                        default=100, help='episode duration (in seconds)')
    parser.add_argument('--keep-sim-running', action='store_true',
                        help='keep the simulator running after training')
    parser.add_argument('--n', type=int, default=1,
                        help='number of trials')
    args, unknown = parser.parse_known_args()

    args.__dict__ = {k: v for k, v in vars(args).items()
                     if k not in overwrite_args}
    args.__dict__.update(overwrite_args)
    return args


def prepare_env(args):
    def deg2rad(deg):
        return np.deg2rad(deg) % (2 * np.pi)

    return create_env(env_idx=f'{args.wind_dir}deg',
                      is_eval=True,
                      water_current_generator=args.water_current,
                      water_current_speed=args.water_current_speed,
                      water_current_dir=deg2rad(args.water_current_dir),
                      wind_generator=args.wind,
                      wind_speed=args.wind_speed,
                      wind_dir=deg2rad(args.wind_dir),
                      reward=args.reward,
                      reward_kwargs=args.reward_kwargs,
                      act='rudder_angle_act',
                      obs='raw_obs',
                      env_name=args.env_name,
                      keep_sim_running=args.keep_sim_running,
                      episode_duration=args.episode_duration,
                      prepare_env_for_nn=False,
                      logger_prefix=args.name)


def eval_pid(overwrite_args={}):
    args = get_args(overwrite_args)

    nb_steps_per_seconds = env_by_name[args.env_name].NB_STEPS_PER_SECONDS
    dt = 1 / nb_steps_per_seconds

    PIDAlgo = pid_algo_by_name[args.pid_algo]
    if PIDAlgo == PIDTAEAlgo:
        pid_algo = PIDAlgo(args.Kp, args.Ki, args.Kd, dt)
    elif PIDAlgo == PIDLOSAlgo:
        path = np.array(args.reward_kwargs['path'], dtype=np.float32)
        radius = args.los_radius
        pid_algo = PIDAlgo(args.Kp, args.Ki, args.Kd, dt, path, radius)
    else:
        raise ValueError(f'Unknown PID algorithm: {PIDAlgo}')

    Logger.configure(f'{args.name}/eval.py')
    Logger.log_hyperparams(args.__dict__)

    env = prepare_env(args)
    mean_reward, std_reward = evaluate_policy(pid_algo,
                                              env,
                                              n_eval_episodes=args.n)
    env.close()

    print(pid_algo.first_pos, pid_algo.last_pos)
    print(pid_algo.last_pos - pid_algo.first_pos)
    return mean_reward, std_reward


if __name__ == '__main__':
    mean_reward, std_reward = eval_pid()
    print(f'mean_reward = {mean_reward}')
    print(f'std_reward = {std_reward}')
