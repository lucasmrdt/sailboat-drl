import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from sailboat_drl import evaluate_pid_algo


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, required=True,
                        help='experiment name')
    parser.add_argument('--theta-wind', type=float, required=True,
                        help='wind direction (in deg)')
    args, unknown = parser.parse_known_args()
    return args


def run_pid():
    args = parse_args()

    overwrite_args = {
        'name': args.name,
        'keep_sim_running': True,
    }

    theta_wind = np.deg2rad(args.theta_wind)

    mean_reward, std_reward = evaluate_pid_algo(overwrite_args, theta_wind)

    print(f'mean_reward={mean_reward}, std_reward={std_reward}')


if __name__ == '__main__':
    run_pid()
